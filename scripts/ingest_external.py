"""Приём одного внешнего источника в `splits/ext-<источник>.parquet`.

Эталон и пул читаются из сплитов, а не передаются флагами: отсечение утечки в `holdout_ru`
не опция, его нельзя забыть включить. Детектор — YOLO11n из группы `train`, как у Stanford.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog, member_index
from pipeline.ingest.external import (
    FOLDER_SOURCES,
    SOURCE_PRIORITY,
    build_external_frame,
    read_mad_cars,
    read_manual,
    read_mapped,
    read_mapping,
)
from pipeline.paths import harvest_dir, splits_dir
from pipeline.schema import read_split, write_split


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Принять внешний источник кадров")
    парсер.add_argument("--source", choices=SOURCE_PRIORITY, required=True)
    парсер.add_argument("--root", type=Path, required=True, help="корень папок источника")
    парсер.add_argument(
        "--mapping", type=Path, default=None, help="таблица соответствия dir_path,model_id"
    )
    парсер.add_argument("--weights", default="yolo11n.pt")
    аргументы = парсер.parse_args(argv)

    # Проверки аргументов — до загрузки каталога и детектора: ошибка в команде не должна
    # стоить минуты на импорт ultralytics.
    папочный = аргументы.source in FOLDER_SOURCES
    if папочный and аргументы.mapping is not None:
        raise SystemExit(f"у {аргументы.source} нет таблицы соответствия: папка и есть model_id")
    if not папочный and аргументы.mapping is None:
        raise SystemExit(f"{аргументы.source}: нужна --mapping catalog/external/<источник>.csv")

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    if аргументы.source == "mad_cars":
        фото = read_mad_cars(аргументы.root, каталог)
    elif аргументы.mapping is None:
        фото = read_manual(аргументы.root, каталог)
    else:
        фото = read_mapped(аргументы.root, read_mapping(аргументы.mapping, каталог))
    эталон = read_split(splits_dir() / "holdout_ru.parquet")
    пул = read_split(splits_dir() / "pool-clean.parquet")
    классы = member_index(каталог)

    # Импорт здесь: `ultralytics` живёт в группе `train`, которой в CI нет.
    from pipeline.detectors.yolo import YoloDetector

    итог = build_external_frame(
        фото,
        YoloDetector(weights=аргументы.weights),
        каталог,
        harvest_dir() / "crops",
        source=аргументы.source,
        holdout_hashes=[int(хеш) for хеш in эталон["phash"]],
        pool=[
            (классы[str(модель)], int(хеш))
            for модель, хеш in zip(пул["model_id"], пул["phash"], strict=True)
        ],
    )
    write_split(итог.frame, splits_dir() / f"ext-{аргументы.source}.parquet")
    счётчики = {"seen": итог.seen, "kept": len(итог.frame), "rejected": итог.rejected}
    (splits_dir() / f"ext-{аргументы.source}.json").write_text(
        json.dumps(счётчики, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{аргументы.source}: {счётчики}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
