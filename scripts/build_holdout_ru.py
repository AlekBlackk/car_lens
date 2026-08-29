"""Собрать `holdout_ru` и обучающий пул из манифеста харвеста (правило 2 Р-11).

Эталон отделяется от пула **до** построения центроидов и никогда в них не попадает — в
этом весь смысл правила 2. Проверяет это не человек, а инвариант 1 §13.1: ни один
`listing_id` не встречается в двух сплитах сразу.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, Catalog, class_index, load_catalog
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.holdout import HOLDOUT_MIN, HOLDOUT_TARGET, listing_id_of, select_holdout
from pipeline.paths import splits_dir
from pipeline.schema import empty_frame, write_split
from pipeline.split import check_invariants

# Инвариант 3 §13.1 требует 300 объявлений на класс. Р-11 сознательно меняет режим на
# few-shot: 20 кадров на центроиды — это другой способ обучения, а не «пока не набрали».
# Порог передаётся явно, чтобы умолчание 300 осталось нетронутым.
MIN_LISTINGS_ДЛЯ_FEW_SHOT = 5


def строки_в_кадр(rows: Sequence[HarvestRow], catalog: Catalog) -> pd.DataFrame:
    """Манифест харвеста в схему §13.1.

    `model_id` группы `grp__*` — её первый участник: какой именно ребейдж на фотографии,
    из открытого хранилища не восстановить, а правило 4 Р-11 на M0 считает группу одним
    классом. Инвариант 4 при этом держится: `model_id → class_idx` однозначно.
    """
    if not rows:
        return empty_frame()
    индекс = class_index(catalog)
    по_id = {к.class_id: к for к in catalog.classes}
    записи = []
    for строка in rows:
        model_id = по_id[строка.class_id].members[0]
        записи.append(
            {
                "photo_id": строка.sha256,
                "listing_id": listing_id_of(строка),
                "source": строка.source,
                "crop_path": f"{строка.class_id}/{строка.file}",
                "brand": catalog.by_id(model_id).brand,
                "model_id": model_id,
                "class_idx": индекс[строка.class_id],
                "body_style": None,
                "year": None,
                "bbox": [
                    float(строка.bbox_x),
                    float(строка.bbox_y),
                    float(строка.bbox_w),
                    float(строка.bbox_h),
                ],
                "bbox_area_frac": float(строка.bbox_area_frac),
                "det_conf": float(строка.det_conf),
                "phash": int(строка.phash),
                "raw_model_string": None,
                # Многомашинные кадры харвест отбраковывает на приёме (`run.py`, причина
                # `multi_car`), поэтому в манифесте их нет по построению.
                "is_multi_car": False,
            }
        )
    return pd.DataFrame(записи)


def main(argv: Sequence[str] | None = None) -> int:
    """Разделить манифест на эталон и пул, записать parquet, проверить инварианты."""
    парсер = argparse.ArgumentParser(description="Собрать holdout_ru (правило 2 Р-11)")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument("--target", type=int, default=HOLDOUT_TARGET)
    парсер.add_argument("--minimum", type=int, default=HOLDOUT_MIN)
    парсер.add_argument("--seed", type=int, default=1337)
    аргументы = парсер.parse_args(argv)

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки = load_manifest(аргументы.manifest)
    выбор = select_holdout(
        строки, target=аргументы.target, minimum=аргументы.minimum, seed=аргументы.seed
    )

    эталон = строки_в_кадр([с for с in строки if с.file in выбор.files], каталог)
    пул = строки_в_кадр([с for с in строки if с.file not in выбор.files], каталог)

    check_invariants(
        {"holdout_ru": эталон, "pool": пул},
        min_listings_per_class=MIN_LISTINGS_ДЛЯ_FEW_SHOT,
    )
    write_split(эталон, splits_dir() / "holdout_ru.parquet")
    write_split(пул, splits_dir() / "pool.parquet")

    print(f"holdout_ru: {len(эталон)} кадров, pool: {len(пул)} кадров")
    print(f"классов без эталона (меньше {аргументы.minimum}): {len(выбор.short_classes)}")
    for class_id in выбор.short_classes:
        print(f"  - {class_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
