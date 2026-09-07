"""Собрать `holdout_ru` и обучающий пул из манифеста харвеста (правило 2 Р-11).

Эталон отделяется от пула **до** построения центроидов и никогда в них не попадает — в
этом весь смысл правила 2. Проверяет это не человек, а инвариант 1 §13.1: ни один
`listing_id` не встречается в двух сплитах сразу.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, Catalog, class_index, load_catalog
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.holdout import HOLDOUT_MIN, HOLDOUT_TARGET, HoldoutSelection, select_holdout
from pipeline.paths import splits_dir
from pipeline.schema import empty_frame, write_split
from pipeline.split import CrossClassPhashPair, check_invariants, cross_class_close_phash

# Инвариант 3 §13.1 требует 300 объявлений на класс. Р-11 сознательно меняет режим на
# few-shot: 20 кадров на центроиды — это другой способ обучения, а не «пока не набрали».
# Порог передаётся явно, чтобы умолчание 300 осталось нетронутым.
MIN_LISTINGS_ДЛЯ_FEW_SHOT = 5


def строки_в_кадр(
    rows: Sequence[HarvestRow], catalog: Catalog, unit_of_file: Mapping[str, str]
) -> pd.DataFrame:
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
                "listing_id": unit_of_file[строка.file],
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


def сформировать_отчёт_сплита(
    holdout_frames: int,
    pool_frames: int,
    selection: HoldoutSelection,
    cross_class_pairs: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Данные для сайдкара провенанса сплита."""
    return {
        "holdout_frames": holdout_frames,
        "pool_frames": pool_frames,
        "refined_classes": list(selection.refined_classes),
        "short_classes": list(selection.short_classes),
        "cross_class_close_phash": list(cross_class_pairs),
    }


def пары_в_отчёт(
    пары: Sequence[CrossClassPhashPair], имя_класса: Mapping[int, str]
) -> list[dict[str, object]]:
    """Межклассовые пары для сайдкара провенанса: имена классов, расстояние, оба phash.

    В прокрутке терминала предупреждение живёт до первой очистки экрана, а сайдкар
    отвечает на вопрос «что нашла сборка», заданный через неделю.
    """
    return [
        {
            "class_a": имя_класса.get(п.class_a, str(п.class_a)),
            "class_b": имя_класса.get(п.class_b, str(п.class_b)),
            "distance": п.distance,
            "phash_a": п.phash_a,
            "phash_b": п.phash_b,
        }
        for п in пары
    ]


def строки_предупреждения(
    пары: Sequence[CrossClassPhashPair], имя_класса: Mapping[int, str]
) -> list[str]:
    """Предупреждение о межклассовой близости pHash — кандидатах на ошибку метки.

    Инвариант 5 §13.1 такие пары не блокирует, но человек о них знать обязан.
    Печатается `class_id`, а не `class_idx`: действовать человек будет по имени.
    Ноль печатается тоже — иначе молчание не отличить от невыполненной проверки.
    """
    заголовок = f"межклассовых близких пар phash: {len(пары)}"
    if not пары:
        return [заголовок]
    строки = [f"ВНИМАНИЕ: {заголовок} — утечкой это не считается, но метки стоит проверить"]
    строки.extend(
        f"  ~ {имя_класса.get(п.class_a, str(п.class_a))}"
        f" и {имя_класса.get(п.class_b, str(п.class_b))}:"
        f" расстояние {п.distance}, phash {п.phash_a} и {п.phash_b}"
        for п in пары
    )
    return строки


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

    единицы = dict(выбор.units)
    эталон = строки_в_кадр([с for с in строки if с.file in выбор.files], каталог, единицы)
    пул = строки_в_кадр([с for с in строки if с.file not in выбор.files], каталог, единицы)

    сплиты = {"holdout_ru": эталон, "pool": пул}
    check_invariants(сплиты, min_listings_per_class=MIN_LISTINGS_ДЛЯ_FEW_SHOT)
    write_split(эталон, splits_dir() / "holdout_ru.parquet")
    write_split(пул, splits_dir() / "pool.parquet")

    имя_класса = {индекс: class_id for class_id, индекс in class_index(каталог).items()}
    пары = cross_class_close_phash(сплиты)
    отчёт_данные = сформировать_отчёт_сплита(
        len(эталон), len(пул), выбор, cross_class_pairs=пары_в_отчёт(пары, имя_класса)
    )
    (splits_dir() / "holdout_ru_report.json").write_text(
        json.dumps(отчёт_данные, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"holdout_ru: {len(эталон)} кадров, pool: {len(пул)} кадров")
    print(f"дроблено по pHash: {len(выбор.refined_classes)} классов")
    for class_id in выбор.refined_classes:
        print(f"  ~ {class_id}")
    print(f"классов без эталона (меньше {аргументы.minimum}): {len(выбор.short_classes)}")
    for class_id in выбор.short_classes:
        print(f"  - {class_id}")
    for строка_вывода in строки_предупреждения(пары, имя_класса):
        print(строка_вывода)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
