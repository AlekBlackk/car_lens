"""Сборка `splits/reject-pool.parquet` — пула отказных прототипов (спека 2026-09-28 §2).

Чужие машины — кадры `scripts.mad_cars reject-pool-sample`, скачанные `fetch` в
`<mad-root>/<brand>__<model>/<car_id>_<view_id>.jpg`. Грузовики и автобусы — COCO по тем же
правилам бокса, что `truck_bus` замера, но только из картинок, не давших замеру ни одного
кропа. Пул обязан не пересекаться с `negatives-street.parquet`: прототип на модели или кадре
замера мерился бы на самом себе.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.ingest.mad_cars import off_catalog_reason, read_mad_disputed, read_mad_mapping
from pipeline.negatives import NegativesResult, build_negative_frame, read_exclusions
from pipeline.negatives_coco import coco_pools, read_coco_images
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import read_split, write_split
from scripts.build_negatives_street import (
    DEFAULT_EXCLUSIONS,
    DEFAULT_SEED,
    coco_candidates,
    street_candidates,
)
from scripts.mad_cars import DEFAULT_DISPUTED, DEFAULT_MAPPING, download_url

_MAD = "mad_cars/"
_COCO = "coco/"


def mad_models_of(ids: Iterable[str]) -> frozenset[str]:
    """Папки моделей MAD-Cars из `listing_id` или `origin`."""
    return frozenset(и.split(_MAD, 1)[1] for и in ids if _MAD in и)


def coco_files_of(ids: Iterable[str]) -> frozenset[str]:
    """Имена картинок COCO из `listing_id` или `origin`, любого вида."""
    return frozenset(и.split(_COCO, 1)[1] for и in ids if _COCO in и)


def check_disjoint(pool_ids: Iterable[str], measured_ids: Iterable[str]) -> None:
    """Пул и замер не делят ни модели MAD-Cars, ни картинки COCO."""
    пул, замер = list(pool_ids), list(measured_ids)
    общие = sorted(
        (mad_models_of(пул) & mad_models_of(замер)) | (coco_files_of(пул) & coco_files_of(замер))
    )
    if общие:
        raise ValueError(f"пул пересекается с замером ({len(общие)}): " + ", ".join(общие[:20]))


def check_off_catalog(
    models: Iterable[str],
    exact: Collection[tuple[str, str]],
    disputed: frozenset[tuple[str, str]],
    needles: Mapping[str, Sequence[str]],
) -> None:
    """Каждая модель пула всё ещё чужая по нынешним фильтрам выборки.

    Фильтры применялись при выборке; строка, добавленная в таблицу соответствий, спорных или
    игл после неё, иначе молча сделала бы модель каталога отказным прототипом своего класса.
    """
    причины: list[str] = []
    for папка in sorted(models):
        марка, разделитель, модель = папка.partition("__")
        причина = (
            off_catalog_reason((марка, модель), exact, disputed, needles)
            if разделитель
            else "bad_folder"
        )
        if причина is not None:
            причины.append(f"{папка} ({причина})")
    if причины:
        raise ValueError(f"в пуле модели каталога ({len(причины)}): " + ", ".join(причины[:20]))


def _счётчики(итог: NegativesResult) -> dict[str, dict[str, int]]:
    return {"kept": итог.kept, "rejected": итог.rejected}


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Собрать пул отказных прототипов")
    парсер.add_argument(
        "--mad-root", type=Path, default=data_root() / "external" / "mad_cars" / "reject_pool"
    )
    парсер.add_argument("--coco-annotations", type=Path, nargs="+", required=True)
    парсер.add_argument(
        "--coco-cache", type=Path, default=data_root() / "external" / "coco" / "images"
    )
    парсер.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    парсер.add_argument("--measured", type=Path, default=splits_dir() / "negatives-street.parquet")
    парсер.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    парсер.add_argument("--disputed", type=Path, default=DEFAULT_DISPUTED)
    парсер.add_argument("--truck-bus-cap", type=int, default=2000)
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--weights", default="yolo11n.pt")
    аргументы = парсер.parse_args(argv)

    замер = [str(л) for л in read_split(аргументы.measured)["listing_id"]]
    исключения = read_exclusions(аргументы.exclusions)
    улица = street_candidates(аргументы.mad_root, исключения.names)
    # До детектора: пересечение по модели ловится по папкам, не тратя час на нарезку.
    check_disjoint([к.origin for к in улица], замер)
    # Те же источники фильтров, что у `_чужие_пары` в `scripts/mad_cars.py`.
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    check_off_catalog(
        mad_models_of(к.origin for к in улица),
        read_mad_mapping(аргументы.mapping, каталог).keys(),
        read_mad_disputed(аргументы.disputed, каталог),
        load_alias_table(DEFAULT_ALIASES_PATH),
    )
    картинки = [к for путь in аргументы.coco_annotations for к in read_coco_images(путь)]
    грузовые, _ = coco_pools(картинки, аргументы.seed)
    замерные = coco_files_of(замер)
    свободные = [к for к in грузовые if к.file_name not in замерные]

    # Импорт здесь: `ultralytics` живёт в группе `train`, которой в CI нет.
    from pipeline.detectors.yolo import YoloDetector

    детектор = YoloDetector(weights=аргументы.weights)
    кропы = harvest_dir() / "crops"
    сбои: list[str] = []
    машины = build_negative_frame(
        улица,
        детектор,
        кропы,
        off_catalog_cap=10**9,  # объём задан выборкой reject-pool-sample
        excluded_photos=исключения.photos,
    )
    грузовики = build_negative_frame(
        coco_candidates(свободные, аргументы.coco_cache, download_url, сбои),
        детектор,
        кропы,
        off_catalog_cap=0,
        excluded_photos=исключения.photos,
        box_caps={"truck_bus": аргументы.truck_bus_cap},
    )
    пустые = [
        имя
        for имя, ч in (("off_catalog_street", машины), ("truck_bus", грузовики))
        if ч.frame.empty
    ]
    if пустые:
        raise SystemExit(f"пустой вид пула: {', '.join(пустые)}")
    кадр = pd.concat([машины.frame, грузовики.frame], ignore_index=True)
    check_disjoint([str(л) for л in кадр["listing_id"]], замер)
    write_split(кадр, splits_dir() / "reject-pool.parquet")
    счётчики = {
        "street": _счётчики(машины),
        "coco_truck_bus": _счётчики(грузовики),
        "coco_images": {
            "truck_bus_pool": len(грузовые),
            "measured_excluded": len(грузовые) - len(свободные),
        },
        "coco_failed": сбои,
        "seed": аргументы.seed,
    }
    (splits_dir() / "reject-pool.json").write_text(
        json.dumps(счётчики, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({к: в for к, в in счётчики.items() if к != "coco_failed"}, ensure_ascii=False))
    print(f"сбоев скачивания COCO: {len(сбои)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
