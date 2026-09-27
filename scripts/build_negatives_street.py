"""Сборка `splits/negatives-street.parquet` (спека 2026-09-27-open-set-street §2–3).

Два источника. Чужие машины — кадры MAD-Cars, скачанные `scripts.mad_cars fetch` по выборке
`negatives-sample` в `<mad-root>/<brand>__<model>/<car_id>_<view_id>.jpg`. Не-машины — COCO
2017: картинки качаются лениво по `coco_url` в кэш и режутся детектором; поток каждой стопки
останавливается на потолке своего вида. Старый `negatives.parquet` не трогается.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd

from pipeline.negatives import (
    NegativeCandidate,
    NegativesResult,
    build_negative_frame,
    read_exclusions,
)
from pipeline.negatives_coco import CocoImage, coco_pools, read_coco_images
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import empty_frame, write_split
from scripts.mad_cars import download_url

_КОРЕНЬ = Path(__file__).resolve().parent.parent
DEFAULT_EXCLUSIONS = _КОРЕНЬ / "catalog" / "external" / "negatives_exclude.csv"
DEFAULT_SEED = 20260927


def street_candidates(root: Path, excluded_names: frozenset[str]) -> list[NegativeCandidate]:
    """Кадры чужих моделей по папкам `<brand>__<model>`; папка из исключений — целиком мимо."""
    return [
        NegativeCandidate(
            путь, True, f"mad_cars/{папка.name}", папка.name, primary_kind="off_catalog_street"
        )
        for папка in sorted(п for п in root.iterdir() if п.is_dir())
        if папка.name not in excluded_names
        for путь in sorted(папка.glob("*.jpg"))
    ]


def coco_candidates(
    images: Iterable[CocoImage],
    cache: Path,
    download: Callable[[str, Path], None],
    failed: list[str],
) -> Iterator[NegativeCandidate]:
    """Лениво: картинка качается, только когда сборщик до неё дошёл; сбой — в `failed`."""
    cache.mkdir(parents=True, exist_ok=True)
    for картинка in images:
        путь = cache / картинка.file_name
        if not (путь.is_file() and путь.stat().st_size > 0):
            try:
                download(картинка.url, путь)
            except (OSError, http.client.HTTPException):
                путь.unlink(missing_ok=True)
                failed.append(картинка.url)
                continue
        yield NegativeCandidate(
            путь, False, f"coco/{картинка.file_name}", картинка.file_name, coco=картинка.truth
        )


def _счётчики(итог: NegativesResult) -> dict[str, dict[str, int]]:
    return {"kept": итог.kept, "rejected": итог.rejected}


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Собрать уличные негативы")
    парсер.add_argument(
        "--mad-root", type=Path, default=data_root() / "external" / "mad_cars" / "negatives"
    )
    парсер.add_argument("--coco-annotations", type=Path, nargs="+", required=True)
    парсер.add_argument(
        "--coco-cache", type=Path, default=data_root() / "external" / "coco" / "images"
    )
    парсер.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    парсер.add_argument("--truck-bus-cap", type=int, default=600)
    парсер.add_argument("--detector-fp-cap", type=int, default=600)
    парсер.add_argument("--detector-fp-budget", type=int, default=30000)
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--weights", default="yolo11n.pt")
    аргументы = парсер.parse_args(argv)

    исключения = read_exclusions(аргументы.exclusions)
    картинки = [к for путь in аргументы.coco_annotations for к in read_coco_images(путь)]
    грузовые, пустые = coco_pools(картинки, аргументы.seed)

    # Импорт здесь: `ultralytics` живёт в группе `train`, которой в CI нет.
    from pipeline.detectors.yolo import YoloDetector

    детектор = YoloDetector(weights=аргументы.weights)
    кропы = harvest_dir() / "crops"
    сбои: list[str] = []
    улица = build_negative_frame(
        street_candidates(аргументы.mad_root, исключения.names),
        детектор,
        кропы,
        off_catalog_cap=10**9,  # объём уже задан выборкой negatives-sample
        excluded_photos=исключения.photos,
    )
    грузовики = build_negative_frame(
        coco_candidates(грузовые, аргументы.coco_cache, download_url, сбои),
        детектор,
        кропы,
        off_catalog_cap=0,
        excluded_photos=исключения.photos,
        box_caps={"truck_bus": аргументы.truck_bus_cap},
    )
    пустота = build_negative_frame(
        coco_candidates(
            itertools.islice(пустые, аргументы.detector_fp_budget),
            аргументы.coco_cache,
            download_url,
            сбои,
        ),
        детектор,
        кропы,
        off_catalog_cap=0,
        excluded_photos=исключения.photos,
        box_caps={"detector_fp": аргументы.detector_fp_cap},
    )
    части = [ч.frame for ч in (улица, грузовики, пустота) if not ч.frame.empty]
    кадр = pd.concat(части, ignore_index=True) if части else empty_frame()
    write_split(кадр, splits_dir() / "negatives-street.parquet")
    счётчики = {
        "street": _счётчики(улица),
        "coco_truck_bus": _счётчики(грузовики),
        "coco_empty": _счётчики(пустота),
        "coco_images": {"truck_bus_pool": len(грузовые), "empty_pool": len(пустые)},
        "coco_failed": сбои,
        "seed": аргументы.seed,
    }
    (splits_dir() / "negatives-street.json").write_text(
        json.dumps(счётчики, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({к: в for к, в in счётчики.items() if к != "coco_failed"}, ensure_ascii=False))
    print(f"сбоев скачивания COCO: {len(сбои)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
