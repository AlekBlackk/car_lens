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
) -> tuple[float, float, float]:
    """Одна эпоха. Возвращает средний loss, последний применённый lr и обучающий top-1.

    Обучающий top-1 считается головой **без маржи** на уже полученном эмбеддинге:
    второго прохода бэкбона нет, есть повторный вызов головы — один матмул
    `embedding_dim × классы`, и он, как и первый, идёт в fp32 вне autocast.

    Маржа исключена намеренно, а не ради экономии. С `target` голова понижает колонку
    верного класса (`training/model.py:47-49`), а `m` едет 0.2 → 0.5 за первые
    `margin_warmup_epochs` эпох (`training/schedule.py:32-34`): по логитам с маржой
    линейка двигалась бы вместе с числом ровно там, где по §5 дизайна дообучения
    читается подпись «учится или сломано», и исправно обучающийся контур мог бы
    показать плоский или падающий top-1. Без проверочного сплита это единственное
    поэпохальное число, а не растущая обучающая точность означает сломанный контур,
    а не плохие данные.
    """
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
    попаданий = 0
    кадров = 0
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
            в_fp32 = эмбеддинг.float()
            логиты = head(в_fp32, метки)
            значение = потери(логиты, метки)
            # Логиты для обучающего top-1 — тем же вызовом головы, но без `target`, то
            # есть без маржи (см. докстринг). `no_grad`: число в градиенте не участвует,
            # и без него на каждом шаге оставался бы лишний граф.
            with torch.no_grad():
                без_маржи = head(в_fp32)
        значение.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), config.grad_clip
        )
        optimizer.step()
        ema.update_parameters(model)

        сумма += float(значение.detach())
        сколько += 1
        # Знаменатель — кадры, а не батчи: среднее батчевых точностей на батчах
        # разного размера даёт другое число, а не то же самое.
        попаданий += int((без_маржи.argmax(dim=1) == метки).sum())
        кадров += len(метки)
    if сколько == 0:
        raise RuntimeError("эпоха не сделала ни одного шага: пустой загрузчик")
    return сумма / сколько, последний_lr, попаданий / кадров


def _подпись_сплита(имя: str) -> str:
    """Подпись сплита в таблице карточки: `val` остаётся собой, `val_ema` → `val (EMA)`."""
    return f"{имя.removesuffix('_ema')} (EMA)" if имя.endswith("_ema") else имя


def _сплит_валидации(итоги: dict[str, Metrics]) -> str | None:
    """Сплит, по которому шли поэпохальный замер и отбор `best.pt`: первый не-EMA ключ.

    `None` — только когда итогов нет вовсе. Если все ключи почему-то EMA (из `main` так
    не приходит), подписи берут первый ключ как есть: объявить в карточке «проверочных
    сплитов не было» при непустых числах хуже.
    """
    for имя in итоги:
        if not имя.endswith("_ema"):
            return имя
    return next(iter(итоги), None)


def _секция_метрик(
    итоги: dict[str, Metrics], валидация: str, *, best_epoch: int, best_val_top1: float
) -> list[str]:
    """Секция «Метрики» карточки: таблица по сплитам, нарратив вехи, десять худших.

    Отдельной функцией, а не ветвью внутри `render_report`: нарратив — тридцать строк
    неизменного текста, и держать его на четыре пробела глубже ради `else` значит
    получать в каждом следующем диффе по карточке переотбивку вместо правки.
    """
    худшие = sorted(итоги[валидация].per_class_recall.items(), key=lambda пара: пара[1])[:10]
    строки = [
        "## Метрики",
        "",
        "| Сплит | top1_model | top5_model | top1_brand |",
        "|---|---|---|---|",
    ]
    строки += [
        f"| {_подпись_сплита(имя)} | {метрики.top1_model:.4f} | {метрики.top5_model:.4f} | "
        f"{метрики.top1_brand:.4f} |"
        for имя, метрики in итоги.items()
    ]
    строки += [
        "",
        f"Лучшая эпоха: {best_epoch}, {валидация} top1_model = {best_val_top1:.4f}. Именно её веса",
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
        f"## Десять худших классов по recall на {валидация}",
        "",
    ]
    строки += [f"- `{class_id}`: {значение:.3f}" for class_id, значение in худшие]
    return строки


def render_report(
    config: TrainConfig,
    эпохи: list[dict[str, Any]],
    итоги: dict[str, Metrics],
    *,
    card_title: str,
    best_epoch: int,
    best_val_top1: float,
    device: str,
    seconds: float,
    peak_vram_mb: float,
    started_at: datetime,
) -> str:
    """Карточка прогона на русском (§10 дизайна).

    `card_title` попадает в первую строку — ту, которую копируют в решения. Умолчание
    флага `--card-title` даёт прежнее «M-1.2»; прогон по русскому пулу называет себя
    сам, иначе он представлялся бы чужой вехой в самом заметном месте.

    `итоги` — метрики по сплитам в порядке замера: ключ `имя` для сырых весов и
    `f"{имя}_ema"` для усреднённых. Таблица строится обходом ключей, а число классов
    и десять худших по recall берутся из первого не-EMA ключа — того сплита, по
    которому шли поэпохальная валидация и отбор `best.pt`.

    Пустой словарь — законный случай, а не ошибка вызова: русский путь §3 дизайна
    дообучения идёт вовсе без проверочного сплита, а эталон меряется один раз
    отдельным шагом. Тогда обе секции про метрики заменяются одним объяснением:
    пустая таблица или строка нулей читалась бы как «померили и получили нуль».

    `started_at` — метка старта, из которой собрано имя каталога прогона.
    Рендеринг идёт в конце прогона, и `datetime.now()` здесь поставил бы дату
    финиша: для 33 минут это косметика, для прогона через полночь — уже враньё.
    """
    валидация = _сплит_валидации(итоги)
    строки = [
        f"# Карточка прогона {card_title}",
        "",
        f"Дата: {started_at.date().isoformat()}",
        "",
        f"- Бэкбон: `{config.backbone}`",
    ]
    if валидация is not None:
        строки.append(
            f"- Классов: {len(итоги[валидация].per_class_recall)} из встреченных в {валидация}"
        )
    строки += [
        f"- Эпох: {config.epochs}, batch {config.batch_size}, img_size {config.img_size}",
        f"- Устройство: {device}, время прогона: {seconds / 60:.1f} мин",
        f"- Пик VRAM: {peak_vram_mb:.0f} МБ",
        "",
    ]
    # Пустота берётся из самого словаря, а не из «не нашлось не-EMA ключа»: сплит с
    # именем на `_ema` — не повод написать в карточке, что замера не было.
    if not итоги:
        строки += [
            "Проверочных сплитов у прогона не было: `--eval-splits` пуст, поэтому в карточке",
            "нет ни таблицы метрик, ни худших классов, ни лучшей эпохи — поэпохального замера",
            "и отбора `best.pt` не было тоже. Эталон меряется один раз, отдельным шагом и",
            "весами последней эпохи (§3 дизайна дообучения): смотреть его в обучении значило",
            "бы выбирать веса по линейке, после чего сравнивать с ней уже нечего.",
        ]
    else:
        # Непустой словарь всегда даёт сплит валидации (`_сплит_валидации`), но mypy
        # выводит это только из проверки.
        assert валидация is not None
        строки += _секция_метрик(
            итоги, валидация, best_epoch=best_epoch, best_val_top1=best_val_top1
        )
    # Столбец точности зависит от того, была ли валидация: без неё в журнале нет
    # `val_top1_model` вовсе, зато обучающий top-1 есть всегда — и в таком прогоне он
    # единственное поэпохальное число.
    подпись, ключ = ("val top1", "val_top1_model") if итоги else ("train top1", "train_top1")
    строки += [
        "",
        "## По эпохам",
        "",
        f"| Эпоха | loss | lr | {подпись} | сек |",
        "|---|---|---|---|---|",
    ]
    строки += [
        f"| {запись['epoch']} | {запись['loss']:.4f} | {запись['lr']:.2e} | "
        f"{запись[ключ]:.4f} | {запись['seconds']:.0f} |"
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
    парсер.add_argument(
        "--crops-root",
        type=Path,
        default=None,
        help="корень, относительно которого раскрывается crop_path; по умолчанию кропы Stanford",
    )
    парсер.add_argument(
        "--train-split", default="train", help="имя parquet обучающего сплита, без расширения"
    )
    парсер.add_argument(
        "--eval-splits",
        nargs="*",
        default=("val", "test"),
        help="сплиты замера; первый — валидация и отбор best.pt, пустой список — без замера",
    )
    парсер.add_argument("--run-prefix", default="m1_2_train", help="префикс имени каталога прогона")
    парсер.add_argument(
        "--dataset", nargs="+", default=("stanford_cars",), help="датасеты прогона для манифеста"
    )
    парсер.add_argument("--licence", default="research-only", help="лицензия данных для манифеста")
    парсер.add_argument(
        "--card-title",
        default="M-1.2",
        help="заголовок карточки: «# Карточка прогона …»; умолчание — веха M-1.2",
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
    # Подмена данных и раскладки — по тому же правилу: пишется только отличие от
    # умолчания, поэтому у прогона M-1.2 объект остаётся прежним. Датасет и лицензия
    # лежат в манифесте отдельными полями, заголовок карточки виден в самой карточке, а
    # вот из какого корня читались кропы и как звался обучающий сплит, кроме манифеста,
    # узнать негде. Умолчания спрашиваются у парсера, а не переписываются литералом
    # второй раз.
    if аргументы.crops_root is not None:
        переопределения["crops_root"] = str(аргументы.crops_root)
    if аргументы.train_split != парсер.get_default("train_split"):
        переопределения["train_split"] = аргументы.train_split
    if tuple(аргументы.eval_splits) != tuple(парсер.get_default("eval_splits")):
        переопределения["eval_splits"] = list(аргументы.eval_splits)
    if аргументы.run_prefix != парсер.get_default("run_prefix"):
        переопределения["run_prefix"] = аргументы.run_prefix
    # Глобальные генераторы засеваются ради torch (инициализация головы, dropout)
    # и ради того, что дёргает `random`/`numpy` в самом проекте. Аугментации сюда
    # не входят: у трансформов albumentations 2.x свои `py_random` и
    # `random_generator` от энтропии ОС, они засеваются через
    # `build_transforms(seed=...)` и пересеваются в воркерах.
    random.seed(конфиг.seed)
    np.random.seed(конфиг.seed)
    torch.manual_seed(конфиг.seed)
    torch.backends.cudnn.benchmark = True  # Р-20: скорость важнее побитовой воспроизводимости

    каталог = load_catalog(аргументы.catalog)
    индекс = class_index(каталог)
    # Имя говорит, что лежит в значениях: `class_id`, а не `model_id`. У
    # платформенной группы это `grp__…`, и участников группы в карте нет
    # вовсе — она обратна `class_index`, то есть нумерует классы §13.2.
    класс_по_индексу = {значение: ключ for ключ, значение in индекс.items()}
    # Марка берётся из каталога, а не из префикса `class_id`: у разнобрендовой
    # группы её нет вовсе, и вывести её из идентификатора нечем.
    марка_по_индексу = {индекс[класс.class_id]: класс.brand for класс in каталог.classes}
    устройство = torch.device(аргументы.device)
    кропы = аргументы.crops_root or crops_dir() / "stanford"

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
            splits_dir() / f"{аргументы.train_split}.parquet",
            кропы,
            num_classes=len(индекс),
            skip_multi_car=True,
        ),
        build_transforms(
            конфиг.img_size, train=True, mean=модель.mean, std=модель.std, seed=конфиг.seed
        ),
        batch_size=конфиг.batch_size,
        shuffle=True,
        num_workers=конфиг.num_workers,
        seed=конфиг.seed,
    )
    # Пустой `--eval-splits` — законный режим, а не забытый аргумент: русский путь §3
    # дизайна дообучения идёт без валидационного среза (7.8 объявления на класс — срез
    # такого размера шумит сильнее, чем измеряет), а эталон в обучении не смотрится
    # вовсе, иначе отбор весов подгонялся бы под линейку. Тогда проверочных загрузчиков
    # нет ни одного, и ни один parquet, кроме обучающего, не читается.
    имена_замера: tuple[str, ...] = tuple(аргументы.eval_splits)
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
        for имя in имена_замера
    }
    # Поэпохальная валидация и отбор `best.pt` идут по первому сплиту списка; без списка
    # не идут вовсе.
    имя_валидации = имена_замера[0] if имена_замера else None

    def замерить(сеть: CarClassifier, имя: str) -> Metrics:
        """Метрики закрытого множества на проверочном сплите `имя`.

        Замыкание, а не пять почти одинаковых вызовов `evaluate`: карта классов,
        карта марок и устройство у всех замеров прогона одни и те же, и разъехаться
        между поэпохальным замером и финальным они не должны.

        Промежуточная переменная с аннотацией нужна mypy: импорты torch у него
        пропущены, из-за чего декоратор `@torch.no_grad()` делает `evaluate`
        нетипизированной, и прямой `return` был бы возвратом `Any`.
        """
        итог: Metrics = evaluate(
            сеть,
            голова,
            проверочные[имя],
            idx_to_class_id=класс_по_индексу,
            idx_to_brand=марка_по_индексу,
            device=устройство,
        )
        return итог

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
    прогон = runs_dir() / f"{аргументы.run_prefix}_{старт.strftime('%Y%m%d_%H%M%S')}"
    прогон.mkdir(parents=True, exist_ok=True)
    (прогон / "config.yaml").write_text(
        аргументы.config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    журнал: list[dict[str, Any]] = []
    лучший = -1.0
    лучшая_эпоха = -1

    def записать_метрики(**сплиты: Any) -> None:
        """Перезаписать `metrics.json` текущим журналом плюс переданные итоги замера.

        Вызывается в конце каждой эпохи, а не только после цикла: падение на
        55-й эпохе из 60 иначе теряет весь журнал, хотя чекпоинты уже на диске.
        Единственное место сборки файла — чтобы формат не разъехался.

        `best_epoch`, `best_val_top1` и `peak_vram_mb` пишутся на каждом вызове, а
        не один раз в конце: по ним сверяют, что в M-1.3 поедут веса из `best.pt`,
        и после падения на 55-й эпохе они нужны ровно так же, как журнал. Оставшиеся
        -1 означают, что отбора не было вовсе — прогон шёл без проверочного сплита,
        и `best.pt` на диске нет.
        """
        (прогон / "metrics.json").write_text(
            json.dumps(
                {
                    "epochs": журнал,
                    "best_epoch": лучшая_эпоха,
                    "best_val_top1": лучший,
                    "peak_vram_mb": _пик_vram_мб(устройство),
                    **сплиты,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    начало = time.monotonic()
    for эпоха in range(конфиг.epochs):
        отсечка = time.monotonic()
        loss, lr, обучающий_top1 = train_one_epoch(
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
        запись: dict[str, Any] = {
            "epoch": эпоха,
            "loss": loss,
            "lr": lr,
            "margin": голова.m,
            "train_top1": обучающий_top1,
        }
        метрики_val = замерить(модель, имя_валидации) if имя_валидации is not None else None
        if метрики_val is not None:
            запись["val_top1_model"] = метрики_val.top1_model
            запись["val_top5_model"] = метрики_val.top5_model
            запись["val_top1_brand"] = метрики_val.top1_brand
        запись["seconds"] = time.monotonic() - отсечка
        журнал.append(запись)
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
            # `None`, когда валидации не было. Ключ читателям чекпоинта не нужен —
            # ни `training/build_index.py`, ни `scripts/eval_index.py`, ни экспорт его
            # не смотрят, — а нулевые метрики соврали бы, что замер был.
            "metrics": asdict(метрики_val) if метрики_val is not None else None,
            "git_sha": хеш_коммита,
        }
        torch.save(чекпоинт, прогон / "last.pt")
        if метрики_val is not None and метрики_val.top1_model > лучший:
            лучший = метрики_val.top1_model
            лучшая_эпоха = эпоха
            torch.save(чекпоинт, прогон / "best.pt")
        записать_метрики()
        хвост = (
            f"val top1 {метрики_val.top1_model:.4f}"
            if метрики_val is not None
            else f"train top1 {обучающий_top1:.4f}"
        )
        print(f"эпоха {эпоха}: loss {loss:.4f}, lr {lr:.2e}, {хвост}")

    итоги: dict[str, Metrics] = {имя: замерить(модель, имя) for имя in имена_замера}
    # EMA оценивается один раз в конце прогона: на 171 классе поэпохальная оценка
    # удваивает время валидации, а числа всё равно измеряют гибрид — усреднённый
    # бэкбон с сырой головой и сырыми BN-буферами (долг 7 в §11 дизайна), так что
    # серия таких чисел по эпохам не добавит к ним смысла. `ema.module` — тот же
    # `CarClassifier`, голова у EMA общая с сырой: в `AveragedModel` она не заворачивалась.
    итоги.update({f"{имя}_ema": замерить(ema.module, имя) for имя in имена_замера})
    пик_vram = _пик_vram_мб(устройство)
    прошло = time.monotonic() - начало

    записать_метрики(**{имя: asdict(метрики) for имя, метрики in итоги.items()})
    (прогон / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": list(аргументы.dataset),
                "licence": аргументы.licence,
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
            итоги,
            card_title=аргументы.card_title,
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
