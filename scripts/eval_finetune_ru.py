"""Замер дообученной модели по русскому эталону holdout_ru двумя трассами (Р-11, §4, §5, §7).

Скрипт читает обученный чекпоинт MobileNetV4-M и собранный по обучающему пулу индекс центроидов
`catalog_index.bin`, прогоняет эталонный сплит `holdout_ru.parquet` и считает две трассы:

1. **Главная трасса — через индекс**: эмбеддинги эталона ранжируются по 4 центроидам каждого
   класса (`pipeline.knn.ranked_by_centroids`, k=3), собранным по обучающему пулу. Это то же
   число центроидов и та же опора, что у few-shot базиса M2, и ровно так же устроено мобильное
   приложение (§13.2 SPEC). Сравнивать индекс с индексом — единственный способ оценить эффект
   дообучения на реальном предмете сравнения без подмены предмета.
2. **Справочная трасса — через голову ArcFace**: top-3 из логитов классификатора
   (`training.evaluate.ranked_predictions`, k=3). Показывает, сколько точности теряется на
   индексной аппроксимации центроидами по сравнению с прямой классификационной головой.

Обе трассы сводятся одной и той же функцией `pipeline.fewshot.metrics.aggregate_fewshot`
с картой классов, полученной строго из `catalog_meta.json` бинарного индекса, а не из YAML:
замер обязан оценивать классы ровно в тех слотах и под теми именами, под которыми они собраны
в индекс.

Результат сопоставляется с предзарегистрированным базисом few-shot и записывается в
`bench/finetune-ru.md` через генератор отчёта `pipeline.fewshot.compare.render_comparison`.

Правило Р-15: torch, timm и модули `training.*` импортируются строго внутри функций.
Чистые функции модуля (разбор флагов, извлечение имён, оценка по центроидам) выполняются
и тестируются в базовом dev-окружении без группы `train`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.fewshot.compare import (
    RunProvenance,
    decision_rule,
    read_fewshot_metrics,
    render_comparison,
)
from pipeline.fewshot.gate import check_gate
from pipeline.fewshot.metrics import FEWSHOT_TOP_K, FewShotMetrics, aggregate_fewshot
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import ranked_by_centroids
from pipeline.paths import data_root, harvest_dir, splits_dir

DEFAULT_OUT = Path("bench/finetune-ru.md")
DEFAULT_PREREGISTRATION = Path("bench/finetune-ru-preregistration.md")


def sha256_of_file(path: Path) -> str:
    """Вычислить sha256 файла блоками по 1 МБ."""
    хеш = hashlib.sha256()
    with path.open("rb") as поток:
        for блок in iter(lambda: поток.read(1 << 20), b""):
            хеш.update(блок)
    return хеш.hexdigest()


def _config_sha256(checkpoint_path: Path, config: Mapping[str, Any]) -> str:
    """sha256 конфига: из файла config.yaml рядом с чекпоинтом либо дампа словаря."""
    рядом = checkpoint_path.parent / "config.yaml"
    if рядом.is_file():
        return sha256_of_file(рядом)
    дамп = json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(дамп).hexdigest()


def _git_sha(fallback: str | None = None) -> str:
    """Хеш текущего git-коммита либо запасной из чекпоинта."""
    try:
        вых = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        sha = вых.decode("utf-8").strip()
        if sha:
            return sha
    except Exception:
        pass
    return fallback or "unknown"


def idx_to_class_id_of(meta: Mapping[str, Any]) -> dict[int, str]:
    """Карта `индекс -> class_id`: позиция записи в `classes` — и есть индекс (§13.2)."""
    return {номер: класс["class_id"] for номер, класс in enumerate(meta["classes"])}


def load_index_classes(index_dir: Path) -> dict[int, str]:
    """Прочитать порядок классов из `catalog_meta.json` в каталоге индекса."""
    байты_меты = (index_dir / "catalog_meta.json").read_bytes()
    мета = json.loads(байты_меты)
    return idx_to_class_id_of(мета)


def evaluate_index_track(
    embeddings: np.ndarray,
    targets: Sequence[int],
    centroids: np.ndarray,
    idx_to_class_id: Mapping[int, str],
) -> tuple[list[list[int]], FewShotMetrics]:
    """Главная трасса замера: ранжирование по центроидам индекса и сведение метрик Р-11."""
    if centroids.shape[0] != len(idx_to_class_id):
        raise ValueError(
            f"размерность центроидов ({centroids.shape[0]}) не совпадает с числом классов в мете "
            f"({len(idx_to_class_id)})"
        )
    ранжирование = ranked_by_centroids(embeddings, centroids, k=FEWSHOT_TOP_K)
    метрики = aggregate_fewshot(ранжирование, targets, idx_to_class_id=idx_to_class_id)
    return ранжирование, метрики


def evaluate_head_track(
    model: Any,
    head: Any,
    loader: Any,
    *,
    device: Any,
    idx_to_class_id: Mapping[int, str],
) -> tuple[list[list[int]], list[int], FewShotMetrics]:
    """Справочная трасса замера: ранжирование по логитам головы ArcFace."""
    from training.evaluate import ranked_predictions

    ранжирование, цели = ranked_predictions(model, head, loader, device=device, k=FEWSHOT_TOP_K)
    метрики = aggregate_fewshot(ранжирование, цели, idx_to_class_id=idx_to_class_id)
    return ранжирование, цели, метрики


def _run_model_evaluation(
    *,
    checkpoint_path: Path,
    centroids: np.ndarray,
    idx_to_class_id: Mapping[int, str],
    holdout_path: Path,
    crops_root: Path,
    weights: str,
    device: str,
) -> tuple[FewShotMetrics, FewShotMetrics, dict[str, Any], int, str | None]:
    """Выполнить замер модели по обеим трассам (требует окружения с torch)."""
    import torch

    from training.build_index import collect_embeddings
    from training.dataset import build_transforms, make_loader
    from training.model import create_model
    from training.samples import read_samples

    чекпоинт: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ключ_модели = "model_state" if weights == "raw" else "ema_state"
    # `img_size`, `mean` и `std` требуются наравне с весами, а не подставляются умолчанием.
    # Чужая нормализация или чужой размер входа меняют эмбеддинг, то есть само замеряемое
    # число, и не роняют при этом ничего: в отчёт уехало бы правдоподобное неверное значение.
    # `training/train.py:599-614` кладёт все три в каждый чекпоинт, поэтому их отсутствие —
    # признак битого файла, а не старого формата, и молчать о нём нельзя.
    обязательные = ("config", "class_index", "head_state", ключ_модели, "img_size", "mean", "std")
    for обязательный in обязательные:
        if обязательный not in чекпоинт:
            raise ValueError(f"{checkpoint_path}: ключ {обязательный!r} отсутствует в чекпоинте")

    конфиг = dict(чекпоинт["config"])
    # Архитектура и размерность — из конфига прогона по той же причине. Зашитая константа
    # собрала бы другую сеть под теми же весами: при несовпадении форм отказ пришёл бы
    # стеной ключей `load_state_dict`, из которой причину не видно, а при совпадении замер
    # молча пошёл бы по чужой архитектуре.
    for поле in ("backbone", "embedding_dim"):
        if поле not in конфиг:
            raise ValueError(f"{checkpoint_path}: поле {поле!r} отсутствует в config чекпоинта")

    индекс = dict(чекпоинт["class_index"])
    чекпоинт_по_индексу = {номер: класс for класс, номер in индекс.items()}
    if чекпоинт_по_индексу != dict(idx_to_class_id):
        raise ValueError(
            f"{checkpoint_path}: нумерация классов чекпоинта разошлась с каталогом индекса "
            f"({len(индекс)} против {len(idx_to_class_id)} классов)"
        )
    чекпоинт_git_sha: str | None = чекпоинт.get("git_sha")

    модель, голова = create_model(
        str(конфиг["backbone"]),
        num_classes=len(индекс),
        embedding_dim=int(конфиг["embedding_dim"]),
        pretrained=False,
    )
    модель.load_state_dict(чекпоинт[ключ_модели])
    голова.load_state_dict(чекпоинт["head_state"])

    устройство = torch.device(device)
    модель.to(устройство)
    голова.to(устройство)

    образцы = read_samples(
        holdout_path,
        crops_root,
        num_classes=len(индекс),
        skip_multi_car=True,
    )
    среднее = (
        float(чекпоинт["mean"][0]),
        float(чекпоинт["mean"][1]),
        float(чекпоинт["mean"][2]),
    )
    отклонение = (
        float(чекпоинт["std"][0]),
        float(чекпоинт["std"][1]),
        float(чекпоинт["std"][2]),
    )
    трансформы = build_transforms(
        int(чекпоинт["img_size"]),
        train=False,
        mean=среднее,
        std=отклонение,
    )
    загрузчик = make_loader(
        образцы,
        трансформы,
        batch_size=int(конфиг.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(конфиг.get("num_workers", 0)),
    )

    # Главная трасса: эмбеддинги по центроидам индекса пула
    эмбеддинги, сырые_метки = collect_embeddings(модель, загрузчик, устройство)
    метки = [int(м) for м in сырые_метки]
    _, метрики_индекс = evaluate_index_track(эмбеддинги, метки, centroids, idx_to_class_id)

    # Справочная трасса: логиты головы ArcFace
    _, _, метрики_голова = evaluate_head_track(
        модель, голова, загрузчик, device=устройство, idx_to_class_id=idx_to_class_id
    )

    return метрики_индекс, метрики_голова, конфиг, len(метки), чекпоинт_git_sha


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов командной строки."""
    парсер = argparse.ArgumentParser(
        description="Замер дообученной модели на русском эталоне (Р-11, задача 6)"
    )
    парсер.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="путь к чекпоинту обучения (.pt)",
    )
    парсер.add_argument(
        "--index",
        type=Path,
        required=True,
        help="каталог с catalog_index.bin и catalog_meta.json",
    )
    парсер.add_argument(
        "--holdout",
        type=Path,
        default=splits_dir() / "holdout_ru.parquet",
        help="parquet-файл эталона holdout_ru",
    )
    парсер.add_argument(
        "--crops-root",
        type=Path,
        default=harvest_dir() / "crops",
        help="корень кропов",
    )
    парсер.add_argument(
        "--baseline",
        type=Path,
        default=data_root() / "index/fewshot_openclip_b32/fewshot_metrics.json",
        help="json-файл с метриками few-shot базиса",
    )
    парсер.add_argument(
        "--weights",
        choices=("raw", "ema"),
        default="raw",
        help="тип весов: сырые (raw) или усреднённые (ema)",
    )
    парсер.add_argument(
        "--device",
        default="cuda",
        help="устройство вычислений (cuda или cpu)",
    )
    парсер.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="путь для сохранения отчёта сравнения (.md)",
    )
    парсер.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
        help="путь к документу предрегистрации",
    )
    парсер.add_argument(
        "--train-minutes",
        type=float,
        default=0.0,
        help="время обучения модели в минутах (из карточки прогона)",
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Замерить модель на эталоне holdout_ru, сверить с базисом и записать отчёт."""
    парсер = build_parser()
    аргументы = парсер.parse_args(argv)

    if not аргументы.checkpoint.is_file():
        raise FileNotFoundError(f"чекпоинт не найден: {аргументы.checkpoint}")
    путь_бинарного_индекса = аргументы.index / "catalog_index.bin"
    if not путь_бинарного_индекса.is_file():
        raise FileNotFoundError(f"файл индекса не найден: {путь_бинарного_индекса}")
    путь_меты = аргументы.index / "catalog_meta.json"
    if not путь_меты.is_file():
        raise FileNotFoundError(f"файл меты индекса не найден: {путь_меты}")
    if not аргументы.baseline.is_file():
        raise FileNotFoundError(f"файл базиса не найден: {аргументы.baseline}")
    if аргументы.train_minutes < 0.0:
        raise ValueError(f"train_minutes не может быть отрицательным: {аргументы.train_minutes}")

    заголовок, центроиды = read_index(путь_бинарного_индекса)
    байты_меты = путь_меты.read_bytes()
    verify_meta(заголовок, байты_меты)
    мета = json.loads(байты_меты)
    имена = idx_to_class_id_of(мета)

    метрики_индекс, метрики_голова, конфиг, кадров_эталона, чекпоинт_sha = _run_model_evaluation(
        checkpoint_path=аргументы.checkpoint,
        centroids=центроиды,
        idx_to_class_id=имена,
        holdout_path=аргументы.holdout,
        crops_root=аргументы.crops_root,
        weights=аргументы.weights,
        device=аргументы.device,
    )

    гейт = check_gate(метрики_индекс, catalog_classes=заголовок.num_classes)
    базис = read_fewshot_metrics(аргументы.baseline)

    путь_пула = splits_dir() / "pool.parquet"
    if путь_пула.is_file():
        хеш_пула = sha256_of_file(путь_пула)
        from pipeline.schema import read_split

        кадров_пула = len(read_split(путь_пула))
    else:
        хеш_пула = "unknown"
        кадров_пула = 0

    хеш_эталона = sha256_of_file(аргументы.holdout) if аргументы.holdout.is_file() else "unknown"
    хеш_весов = (
        sha256_of_file(аргументы.checkpoint) if аргументы.checkpoint.is_file() else "unknown"
    )
    хеш_индекса = sha256_of_file(путь_бинарного_индекса)

    провенанс = RunProvenance(
        git_sha=_git_sha(чекпоинт_sha),
        run_date=date.today().isoformat(),
        backbone=str(конфиг.get("backbone", "unknown")),
        device=str(аргументы.device),
        epochs=int(конфиг.get("epochs", 0)),
        seed=int(конфиг.get("seed", 0)),
        train_minutes=float(аргументы.train_minutes),
        weights_sha256=хеш_весов,
        config_sha256=_config_sha256(аргументы.checkpoint, конфиг),
        index_sha256=хеш_индекса,
        pool_sha256=хеш_пула,
        holdout_sha256=хеш_эталона,
        pool_frames=кадров_пула,
        holdout_frames=кадров_эталона,
        catalog_classes=заголовок.num_classes,
        preregistration=str(аргументы.preregistration),
    )

    текст_отчёта = render_comparison(
        finetune=метрики_индекс,
        finetune_head=метрики_голова,
        baseline=базис,
        gate=гейт,
        provenance=провенанс,
    )

    аргументы.out.parent.mkdir(parents=True, exist_ok=True)
    аргументы.out.write_text(текст_отчёта, encoding="utf-8")

    print(f"Отчёт сохранён: {аргументы.out}")
    print(
        f"Главная трасса (индекс): top-1 {метрики_индекс.top1:.4f}, top-3 {метрики_индекс.top3:.4f}"
    )
    print(
        f"Справочная трасса (голова): top-1 {метрики_голова.top1:.4f}, "
        f"top-3 {метрики_голова.top3:.4f}"
    )
    print(f"Базис few-shot: top-1 {базис.top1:.4f}, top-3 {базис.top3:.4f}")
    print(decision_rule(baseline_top3=базис.top3, finetune_top3=метрики_индекс.top3))
    вердикт = "ПРОЙДЕН" if гейт.passed else "НЕ ПРОЙДЕН"
    print(f"Гейт по главной трассе: {вердикт}")
    for провал in гейт.failures:
        print(f"  ✗ {провал}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
