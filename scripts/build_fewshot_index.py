"""Индекс эталонов few-shot: центроиды классов по обучающему пулу Р-11 (§13.2).

Энкодер — `openclip_b32`, тот самый чекпоинт LAION, которым харвест фильтровал кандидатов и
считал `clip_rank`. Другой энкодер означал бы, что выборку отбирала одна модель, а мерят
другую.

Формат индекса не меняется: `catalog_index.bin` по §13.2, четыре центроида на класс, все
классы каталога. Класс без кадров в сплите — отказ, а не пустой слот: слот из нулей дал бы
косинус ноль ко всему и молча стал бы «классом, который никогда не угадывается».

`open_clip`, `torch` и `sklearn` импортируются внутри `main` (Р-15): раскладка кропов и
проверка полноты обязаны проверяться в CI, где стоит только группа `dev`.

Запуск:

    uv run --group train python -m scripts.build_fewshot_index
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, class_index, load_catalog
from pipeline.fewshot.embed import embed_crops
from pipeline.index_format import meta_json, write_index
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import read_split
from training.build_index import CENTROIDS_PER_CLASS, sha256_of_file, stack_centroids

ENCODER_NAME = "openclip_b32"
ENCODER_ARCH = "ViT-B-32"
ENCODER_WEIGHTS = "laion2b_s34b_b79k"
ENCODER_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
ENCODER_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
ENCODER_WEIGHTS_SHA256 = "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"

DEFAULT_SPLIT = "pool"


class BuildFewShotError(RuntimeError):
    """Индекс собрать нельзя."""


def index_dir() -> Path:
    """Каталог артефактов few-shot индекса."""
    return data_root() / "index" / f"fewshot_{ENCODER_NAME}"


def crops_by_class(df: pd.DataFrame, crops_root: Path) -> list[tuple[int, list[Path]]]:
    """Кропы сплита, сгруппированные по индексу класса, в порядке индекса.

    Путь собирается из `crop_path` (`{class_id}/{файл}`) и корня кропов: колонка хранит
    путь относительно каталога кропов, а не абсолютный, потому что данные лежат в WSL, а
    репозиторий на диске Windows (§16.1).

    Порядок внутри класса — порядок строк кадра, и он сохраняется: метки идут отдельным
    массивом, и перестановка молча перемешала бы разметку.
    """
    if df.empty:
        raise BuildFewShotError("сплит пуст: центроиды строить не из чего")
    без_класса = df["class_idx"].isna()
    if bool(без_класса.any()):
        примеры = df.loc[без_класса, "photo_id"].tolist()[:5]
        raise BuildFewShotError(f"строки без class_idx: {примеры}")
    по_классам: dict[int, list[Path]] = defaultdict(list)
    for номер, путь in zip(df["class_idx"], df["crop_path"], strict=True):
        по_классам[int(номер)].append(crops_root / str(путь))
    return [(номер, по_классам[номер]) for номер in sorted(по_классам)]


def check_full_coverage(
    groups: Sequence[tuple[int, list[Path]]],
    num_classes: int,
    idx_to_class_id: Mapping[int, str],
    *,
    allow_missing: bool = False,
) -> tuple[str, ...]:
    """Все классы каталога представлены в сплите.

    Отказ с именами, а не молчаливый пустой слот: `stack_centroids` упал бы на номере
    класса, по которому не понять, какой класс чинить и чем.
    """
    есть = {номер for номер, _ in groups}
    нет = tuple(idx_to_class_id[номер] for номер in range(num_classes) if номер not in есть)
    if нет and not allow_missing:
        raise BuildFewShotError(
            f"классов без кадров в сплите: {len(нет)} ({', '.join(нет[:5])}); "
            "эталон забрал всё — поднимать POOL_RESERVE в `pipeline/holdout.py`"
        )
    return нет


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов. Флага `--device` нет намеренно, см. докстринг модуля."""
    парсер = argparse.ArgumentParser(description="Индекс центроидов few-shot (Р-11)")
    парсер.add_argument("--split", default=DEFAULT_SPLIT, help="сплит-источник центроидов")
    парсер.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    парсер.add_argument("--out", type=Path, default=None)
    парсер.add_argument("--seed", type=int, default=1337, help="сид k-means")
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать `catalog_index.bin` по сплиту-пулу и записать отчёт сборки."""
    import open_clip  # type: ignore[import-untyped]

    from pipeline.harvest.backends import ClipImageEncoder
    from training.build_index import kmeans_factory

    аргументы = build_parser().parse_args(argv)
    каталог = load_catalog(аргументы.catalog)
    индекс = class_index(каталог)
    по_индексу = {значение: ключ for ключ, значение in индекс.items()}

    кадр = read_split(splits_dir() / f"{аргументы.split}.parquet")
    группы = crops_by_class(кадр, harvest_dir() / "crops")
    дефицитные = check_full_coverage(группы, len(индекс), по_индексу, allow_missing=True)
    if дефицитные:
        print(f"дефицит (уходят в синтетический слой): {', '.join(дефицитные)}", flush=True)

    модель, _, препроцесс = open_clip.create_model_and_transforms(
        ENCODER_ARCH, pretrained=ENCODER_WEIGHTS
    )
    энкодер = ClipImageEncoder(модель.eval(), препроцесс)

    куски: list[np.ndarray] = []
    метки: list[np.ndarray] = []
    for номер, пути in группы:
        свои = embed_crops(пути, энкодер)
        куски.append(свои)
        метки.append(np.full(len(свои), номер, dtype=np.int64))
        print(f"{по_индексу[номер]}: {len(свои)} кропов", flush=True)
    эмбеддинги = np.concatenate(куски)
    все_метки = np.concatenate(метки)

    центроиды = stack_centroids(
        эмбеддинги,
        все_метки,
        num_classes=len(индекс),
        centroids_per_class=CENTROIDS_PER_CLASS,
        kmeans=kmeans_factory(int(аргументы.seed)),
        allow_empty=True,
    )

    вывод = аргументы.out or index_dir()
    вывод.mkdir(parents=True, exist_ok=True)
    мета = meta_json(
        каталог.classes,
        embedding_dim=int(центроиды.shape[2]),
        centroids_per_class=CENTROIDS_PER_CLASS,
        model_weights_sha256=ENCODER_WEIGHTS_SHA256,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    путь_меты = вывод / "catalog_meta.json"
    путь_меты.write_bytes(мета)
    путь_индекса = вывод / "catalog_index.bin"
    заголовок = write_index(путь_индекса, центроиды, meta_bytes=путь_меты.read_bytes())

    размеры = np.bincount(все_метки, minlength=len(индекс))
    отчёт = {
        "split": аргументы.split,
        "encoder": ENCODER_NAME,
        "encoder_repo": ENCODER_REPO,
        "encoder_revision": ENCODER_REVISION,
        "encoder_weights_sha256": ENCODER_WEIGHTS_SHA256,
        "num_classes": заголовок.num_classes,
        "centroids_per_class": заголовок.centroids_per_class,
        "embedding_dim": заголовок.embedding_dim,
        "embeddings_total": len(все_метки),
        "embeddings_per_class": {
            "min": int(размеры.min()),
            "median": int(np.median(размеры)),
            "max": int(размеры.max()),
        },
        "kmeans_seed": int(аргументы.seed),
        "index_bytes": путь_индекса.stat().st_size,
        "index_sha256": sha256_of_file(путь_индекса),
        "meta_sha256": заголовок.meta_sha256.hex(),
    }
    (вывод / "report.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(отчёт, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
