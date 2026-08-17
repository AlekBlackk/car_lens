"""Цикл обучения классификатора (M-1.2).

Оркестратор: сам ничего не вычисляет сверх шага оптимизации. Расписания
живут в `training.schedule`, метрики — в `training.metrics`, чтение сплита —
в `training.samples`, и каждый из них проверяется в CI отдельно.

EMA берётся готовая (`AveragedModel` + `get_ema_multi_avg_fn`, Р-19):
собственная реализация — лишний код и лишнее место, где веса тихо
перестанут обновляться.

bf16 включается через `torch.autocast` без `GradScaler`: масштабирование
градиентов нужно только fp16, у bf16 диапазон экспоненты как у fp32. Под
autocast идёт только бэкбон — голова ArcFace считается в fp32, см. комментарий
в `train_one_epoch`.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from pipeline.catalog import STANFORD_CATALOG_PATH, class_index, load_catalog
from pipeline.paths import crops_dir, runs_dir, splits_dir
from training.config import SKELETON_CONFIG_PATH, TrainConfig, load_config, validate_epoch_budgets
from training.dataset import build_transforms, make_loader
from training.evaluate import evaluate
from training.metrics import Metrics
from training.model import ArcFaceHead, CarClassifier, create_model
from training.samples import read_samples
from training.schedule import backbone_mult, lr_at, margin_at


def make_optimizer(
    model: CarClassifier, head: ArcFaceHead, *, lr: float, weight_decay: float
) -> AdamW:
    """AdamW с двумя группами параметров: бэкбон и всё остальное.

    Две группы нужны ради `backbone_lr_mult` из §4.2: множитель применяется
    только к первой.
    """
    return AdamW(
        [
            {"params": list(model.backbone.parameters()), "name": "backbone"},
            {
                "params": list(model.embedding.parameters()) + list(head.parameters()),
                "name": "head",
            },
        ],
        lr=lr,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: CarClassifier,
    head: ArcFaceHead,
    loader: DataLoader[tuple[Tensor, int]],
    optimizer: AdamW,
    ema: AveragedModel,
    *,
    epoch: int,
    config: TrainConfig,
    device: torch.device,
    steps_per_epoch: int,
    limit_batches: int | None,
) -> tuple[float, float]:
    """Одна эпоха. Возвращает средний loss и последний применённый lr."""
    model.train()
    head.train()
    head.set_margin(
        margin_at(
            epoch,
            start=config.arcface.margin_start,
            end=config.arcface.margin_end,
            warmup_epochs=config.arcface.margin_warmup_epochs,
        )
    )
    множитель = backbone_mult(
        epoch, mult=config.backbone_lr_mult, epochs=config.backbone_lr_mult_epochs
    )
    потери = nn.CrossEntropyLoss()
    всего_шагов = config.epochs * steps_per_epoch
    разогрев = config.warmup_epochs * steps_per_epoch

    сумма = 0.0
    сколько = 0
    последний_lr = 0.0
    for номер, (картинки, метки) in enumerate(loader):
        if limit_batches is not None and номер >= limit_batches:
            break
        шаг = epoch * steps_per_epoch + номер
        текущий = lr_at(шаг, base_lr=config.lr, total_steps=всего_шагов, warmup_steps=разогрев)
        for группа in optimizer.param_groups:
            группа["lr"] = текущий * (множитель if группа["name"] == "backbone" else 1.0)
        последний_lr = текущий

        картинки = картинки.to(device, non_blocking=True)
        метки = метки.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            эмбеддинг = model(картинки)
        # Голова считается в fp32, вне autocast: `ArcFaceHead` берёт `acos(cos)`
        # рядом с ±1, а у bf16 всего 8 бит мантиссы — там `acos` теряет точность
        # настолько, что градиент по угловому марджину превращается в шум.
        with torch.autocast(device_type=device.type, enabled=False):
            логиты = head(эмбеддинг.float(), метки)
            значение = потери(логиты, метки)
        значение.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), config.grad_clip
        )
        optimizer.step()
        ema.update_parameters(model)

        сумма += float(значение.detach())
        сколько += 1
    if сколько == 0:
        raise RuntimeError("эпоха не сделала ни одного шага: пустой загрузчик")
    return сумма / сколько, последний_lr


def render_report(
    config: TrainConfig,
    эпохи: list[dict[str, Any]],
    val: Metrics,
    test: Metrics,
    *,
    val_ema: Metrics,
    test_ema: Metrics,
    best_epoch: int,
    best_val_top1: float,
    device: str,
    seconds: float,
    peak_vram_mb: float,
    started_at: datetime,
) -> str:
    """Карточка прогона на русском (§10 дизайна).

    `started_at` — метка старта, из которой собрано имя каталога прогона.
    Рендеринг идёт в конце прогона, и `datetime.now()` здесь поставил бы дату
    финиша: для 33 минут это косметика, для прогона через полночь — уже враньё.
    """
    худшие = sorted(val.per_class_recall.items(), key=lambda пара: пара[1])[:10]
    строки = [
        "# Карточка прогона M-1.2",
        "",
        f"Дата: {started_at.date().isoformat()}",
        "",
        f"- Бэкбон: `{config.backbone}`",
        f"- Классов: {len(val.per_class_recall)} из встреченных в val",
        f"- Эпох: {config.epochs}, batch {config.batch_size}, img_size {config.img_size}",
        f"- Устройство: {device}, время прогона: {seconds / 60:.1f} мин",
        f"- Пик VRAM: {peak_vram_mb:.0f} МБ",
        "",
        "## Метрики",
        "",
        "| Сплит | top1_model | top5_model | top1_brand |",
        "|---|---|---|---|",
        f"| val | {val.top1_model:.4f} | {val.top5_model:.4f} | {val.top1_brand:.4f} |",
        f"| test | {test.top1_model:.4f} | {test.top5_model:.4f} | {test.top1_brand:.4f} |",
        f"| val (EMA) | {val_ema.top1_model:.4f} | {val_ema.top5_model:.4f} | "
        f"{val_ema.top1_brand:.4f} |",
        f"| test (EMA) | {test_ema.top1_model:.4f} | {test_ema.top5_model:.4f} | "
        f"{test_ema.top1_brand:.4f} |",
        "",
        f"Лучшая эпоха: {best_epoch}, val top1_model = {best_val_top1:.4f}. Именно её веса",
        "лежат в `best.pt`, и в M-1.3 поедут они, а не финальные из таблицы выше —",
        "порог ниже сверяйте с этим числом.",
        "",
        "Строки `(EMA)` считаются один раз в конце прогона, и читать их как ответ на",
        "вопрос «сырые веса или усреднённые» нельзя. В `AveragedModel` завёрнут только",
        "бэкбон: голова ArcFace остаётся сырой, а BN-буферы torch при `use_buffers=False`",
        "копирует из живой модели на каждом `update_parameters`. Измерен, значит, гибрид —",
        "усреднённый бэкбон с сырой головой и сырыми буферами, а не усреднённая модель.",
        "Числа посчитаны и воспроизводимы, но аргументом за или против усреднения не",
        "являются: веса для M-1.3 берутся сырые, из `best.pt`. Подробности — долг 7 в §11",
        "дизайна; честная оценка требует усреднять голову вместе с бэкбоном и стоит ещё",
        "одного прогона, поэтому на M-1.2 не делалась.",
        "",
        "Отдельно: числа `(EMA)` посчитаны на финальном усреднении, то есть на",
        "`ema_state` из `last.pt`. В `best.pt` лежит `ema_state` лучшей по сырым весам",
        "эпохи — другие тензоры, и к числам выше они отношения не имеют.",
        "",
        "Пороги §5.2 к этому прогону не применяются (Р-13): они про 150 классов на",
        "уличных данных. Единственный порог здесь — `val top1_model >= 0.50`, пол",
        "здравомыслия, отделяющий работающий пайплайн от бага в разметке или",
        "нормализации.",
        "",
        "## Десять худших классов по recall на val",
        "",
    ]
    строки += [f"- `{model_id}`: {значение:.3f}" for model_id, значение in худшие]
    строки += [
        "",
        "## По эпохам",
        "",
        "| Эпоха | loss | lr | val top1 | сек |",
        "|---|---|---|---|---|",
    ]
    строки += [
        f"| {запись['epoch']} | {запись['loss']:.4f} | {запись['lr']:.2e} | "
        f"{запись['val_top1_model']:.4f} | {запись['seconds']:.0f} |"
        for запись in эпохи
    ]
    return "\n".join(строки) + "\n"


def _пик_vram_мб(device: torch.device) -> float:
    """Пик выделенной torch памяти на устройстве, в МБ. Вне cuda — ноль.

    Считается по учёту torch, а не по `nvidia-smi`: кеш аллокатора и контекст
    драйвера сюда не входят, так что число ниже занятого в системе.
    """
    if device.type != "cuda":
        return 0.0
    # `float(...)` не ради арифметики (байты и так целые), а ради mypy: у него
    # импорты torch пропущены, и `max_memory_allocated` возвращает `Any`.
    return float(torch.cuda.max_memory_allocated()) / 1024 / 1024


def _git_sha() -> str:
    """Хеш текущего коммита или `unknown` вне репозитория."""
    try:
        готово = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return готово.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа. Возвращает код выхода процесса."""
    парсер = argparse.ArgumentParser(description="Обучение классификатора (M-1.2)")
    парсер.add_argument("--config", type=Path, default=SKELETON_CONFIG_PATH)
    парсер.add_argument("--catalog", type=Path, default=STANFORD_CATALOG_PATH)
    парсер.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    парсер.add_argument("--epochs", type=int, default=None, help="переопределить epochs конфига")
    парсер.add_argument(
        "--limit-batches", type=int, default=None, help="шагов на эпоху, для отладки"
    )
    парсер.add_argument(
        "--no-pretrained",
        action="store_true",
        help="случайная инициализация бэкбона: смоук-тест не должен ходить в сеть за весами",
    )
    аргументы = парсер.parse_args(argv)

    конфиг = load_config(аргументы.config)
    if аргументы.epochs is not None:
        # `replace`, а не `TrainConfig(**asdict(...))`: `asdict` рекурсивно
        # разворачивает вложенный `ArcFaceConfig` в словарь, и конструктор
        # получил бы `dict` вместо датакласса.
        конфиг = replace(конфиг, epochs=аргументы.epochs)
        # Override едет после валидации, поэтому эпоховые бюджеты перепроверяются
        # здесь: иначе `--epochs 2` при `warmup_epochs: 3` прошло бы молча, и весь
        # прогон остался бы в разогреве — тот же приём, что у `--limit-batches`.
        validate_epoch_budgets(конфиг, source="--epochs")
    # Флаги, поменявшие прогон, едут в манифест: `config.yaml` рядом с ним — копия
    # файла до override'ов, и по одному ему нельзя понять, почему в журнале две
    # эпохи вместо шестидесяти. Пустой объект означает «прогон ровно по конфигу».
    переопределения: dict[str, Any] = {}
    if аргументы.epochs is not None:
        переопределения["epochs"] = аргументы.epochs
    if аргументы.limit_batches is not None:
        переопределения["limit_batches"] = аргументы.limit_batches
    if аргументы.no_pretrained:
        переопределения["no_pretrained"] = True
    # Глобальные генераторы засеваются ради torch (инициализация головы, dropout)
    # и ради того, что дёргает `random`/`numpy` в самом проекте. Аугментации сюда
    # не входят: у трансформов albumentations 2.x свои `py_random` и
    # `random_generator` от энтропии ОС, они засеваются через
    # `build_transforms(seed=...)` и пересеваются в воркерах.
    random.seed(конфиг.seed)
    np.random.seed(конфиг.seed)
    torch.manual_seed(конфиг.seed)
    torch.backends.cudnn.benchmark = True  # Р-20: скорость важнее побитовой воспроизводимости

    индекс = class_index(load_catalog(аргументы.catalog))
    обратный = {значение: ключ for ключ, значение in индекс.items()}
    устройство = torch.device(аргументы.device)
    кропы = crops_dir() / "stanford"

    модель, голова = create_model(
        конфиг.backbone,
        num_classes=len(индекс),
        embedding_dim=конфиг.embedding_dim,
        pretrained=not аргументы.no_pretrained,
        s=конфиг.arcface.s,
        m=конфиг.arcface.margin_start,
    )
    модель.to(устройство)
    голова.to(устройство)

    обучение = make_loader(
        read_samples(
            splits_dir() / "train.parquet", кропы, num_classes=len(индекс), skip_multi_car=True
        ),
        build_transforms(
            конфиг.img_size, train=True, mean=модель.mean, std=модель.std, seed=конфиг.seed
        ),
        batch_size=конфиг.batch_size,
        shuffle=True,
        num_workers=конфиг.num_workers,
        seed=конфиг.seed,
    )
    проверочные = {
        # Здесь `seed` не нужен: только `Resize` и нормализация, `shuffle=False` —
        # порядок и содержимое батчей детерминированы без всякого генератора.
        имя: make_loader(
            read_samples(
                splits_dir() / f"{имя}.parquet",
                кропы,
                num_classes=len(индекс),
                skip_multi_car=False,
            ),
            build_transforms(конфиг.img_size, train=False, mean=модель.mean, std=модель.std),
            batch_size=конфиг.batch_size,
            shuffle=False,
            num_workers=конфиг.num_workers,
        )
        for имя in ("val", "test")
    }

    оптимизатор = make_optimizer(модель, голова, lr=конфиг.lr, weight_decay=конфиг.weight_decay)
    ema = AveragedModel(модель, multi_avg_fn=get_ema_multi_avg_fn(конфиг.ema_decay))
    шагов_в_эпохе = max(1, len(обучение))
    if аргументы.limit_batches is not None:
        # Знаменатель расписания обязан ехать вместе с лимитом шагов: иначе
        # отладочный прогон целиком остаётся в разогреве и проверяет не тот
        # режим, в котором пойдёт боевой.
        шагов_в_эпохе = max(1, min(шагов_в_эпохе, аргументы.limit_batches))
    # SHA за прогон не меняется, а `git rev-parse` на 60 эпохах — 61 запуск.
    хеш_коммита = _git_sha()

    старт = datetime.now(UTC)
    прогон = runs_dir() / f"m1_2_train_{старт.strftime('%Y%m%d_%H%M%S')}"
    прогон.mkdir(parents=True, exist_ok=True)
    (прогон / "config.yaml").write_text(
        аргументы.config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    журнал: list[dict[str, Any]] = []
    лучший = -1.0
    лучшая_эпоха = -1

    def записать_метрики(**итоги: Any) -> None:
        """Перезаписать `metrics.json` текущим журналом плюс переданные итоги.

        Вызывается в конце каждой эпохи, а не только после цикла: падение на
        55-й эпохе из 60 иначе теряет весь журнал, хотя чекпоинты уже на диске.
        Единственное место сборки файла — чтобы формат не разъехался.

        `best_epoch`, `best_val_top1` и `peak_vram_mb` пишутся на каждом вызове, а
        не один раз в конце: по ним сверяют, что в M-1.3 поедут веса из `best.pt`,
        и после падения на 55-й эпохе они нужны ровно так же, как журнал.
        """
        (прогон / "metrics.json").write_text(
            json.dumps(
                {
                    "epochs": журнал,
                    "best_epoch": лучшая_эпоха,
                    "best_val_top1": лучший,
                    "peak_vram_mb": _пик_vram_мб(устройство),
                    **итоги,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    начало = time.monotonic()
    for эпоха in range(конфиг.epochs):
        отсечка = time.monotonic()
        loss, lr = train_one_epoch(
            модель,
            голова,
            обучение,
            оптимизатор,
            ema,
            epoch=эпоха,
            config=конфиг,
            device=устройство,
            steps_per_epoch=шагов_в_эпохе,
            limit_batches=аргументы.limit_batches,
        )
        метрики_val = evaluate(
            модель, голова, проверочные["val"], idx_to_model_id=обратный, device=устройство
        )
        журнал.append(
            {
                "epoch": эпоха,
                "loss": loss,
                "lr": lr,
                "margin": голова.m,
                "val_top1_model": метрики_val.top1_model,
                "val_top5_model": метрики_val.top5_model,
                "val_top1_brand": метрики_val.top1_brand,
                "seconds": time.monotonic() - отсечка,
            }
        )
        чекпоинт = {
            "model_state": модель.state_dict(),
            "ema_state": ema.module.state_dict(),
            "head_state": голова.state_dict(),
            "class_index": индекс,
            "img_size": конфиг.img_size,
            "mean": модель.mean,
            "std": модель.std,
            "config": asdict(конфиг),
            "epoch": эпоха,
            "metrics": asdict(метрики_val),
            "git_sha": хеш_коммита,
        }
        torch.save(чекпоинт, прогон / "last.pt")
        if метрики_val.top1_model > лучший:
            лучший = метрики_val.top1_model
            лучшая_эпоха = эпоха
            torch.save(чекпоинт, прогон / "best.pt")
        записать_метрики()
        print(f"эпоха {эпоха}: loss {loss:.4f}, lr {lr:.2e}, val top1 {метрики_val.top1_model:.4f}")

    метрики_val = evaluate(
        модель, голова, проверочные["val"], idx_to_model_id=обратный, device=устройство
    )
    метрики_test = evaluate(
        модель, голова, проверочные["test"], idx_to_model_id=обратный, device=устройство
    )
    # EMA оценивается один раз в конце прогона: на 171 классе поэпохальная оценка
    # удваивает время валидации, а числа всё равно измеряют гибрид — усреднённый
    # бэкбон с сырой головой и сырыми BN-буферами (долг 7 в §11 дизайна), так что
    # серия таких чисел по эпохам не добавит к ним смысла. `ema.module` — тот же
    # `CarClassifier`, голова у EMA общая с сырой: в `AveragedModel` она не заворачивалась.
    метрики_val_ema = evaluate(
        ema.module, голова, проверочные["val"], idx_to_model_id=обратный, device=устройство
    )
    метрики_test_ema = evaluate(
        ema.module, голова, проверочные["test"], idx_to_model_id=обратный, device=устройство
    )
    пик_vram = _пик_vram_мб(устройство)
    прошло = time.monotonic() - начало

    записать_метрики(
        val=asdict(метрики_val),
        test=asdict(метрики_test),
        val_ema=asdict(метрики_val_ema),
        test_ema=asdict(метрики_test_ema),
    )
    (прогон / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": ["stanford_cars"],
                "licence": "research-only",
                "catalog": аргументы.catalog.name,
                "seed": конфиг.seed,
                "device": str(устройство),
                "git_sha": хеш_коммита,
                "cli_overrides": переопределения,
                "packages": {
                    "torch": torch.__version__,
                    "config": str(аргументы.config),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (прогон / "report.md").write_text(
        render_report(
            конфиг,
            журнал,
            метрики_val,
            метрики_test,
            val_ema=метрики_val_ema,
            test_ema=метрики_test_ema,
            best_epoch=лучшая_эпоха,
            best_val_top1=лучший,
            device=str(устройство),
            seconds=прошло,
            peak_vram_mb=пик_vram,
            started_at=старт,
        ),
        encoding="utf-8",
    )
    print(f"Карточка: {прогон / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
