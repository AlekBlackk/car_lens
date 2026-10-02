"""Свежие не-машины для проверки порогов (`bench/catalog-192-thresholds-preregistration.md`).

Те же приём и детектор, что у `truck_bus` в `negatives-street`, но из картинок пула, которых
та сборка не трогала: всё, что она обработала, лежит в кэше COCO, поэтому свежая картинка —
та, которой в кэше нет. Порядок пула — по другому seed. Картинки качаются лениво, не больше
`--budget`; кадры пишутся в отдельный сплит, `negatives-street` не меняется.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Collection, Sequence
from pathlib import Path

from pipeline.negatives import build_negative_frame, read_exclusions
from pipeline.negatives_coco import CocoImage, coco_pools, read_coco_images
from pipeline.paths import data_root, harvest_dir
from pipeline.schema import write_split
from scripts.build_negatives_street import DEFAULT_EXCLUSIONS, coco_candidates
from scripts.mad_cars import download_url

DEFAULT_SEED = 20261003


def unseen_images(pool: Sequence[CocoImage], seen_names: Collection[str]) -> list[CocoImage]:
    """Картинки пула, чьих файлов ещё нет в кэше, в порядке пула."""
    return [картинка for картинка in pool if картинка.file_name not in seen_names]


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Свежие не-машины (truck_bus) из COCO")
    парсер.add_argument("--coco-annotations", type=Path, nargs="+", required=True)
    парсер.add_argument(
        "--coco-cache", type=Path, default=data_root() / "external" / "coco" / "images"
    )
    парсер.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--cap", type=int, default=600, help="сколько кадров принять")
    парсер.add_argument(
        "--budget", type=int, default=1800, help="сколько картинок скачать, не более"
    )
    парсер.add_argument("--weights", default="yolo11n.pt")
    парсер.add_argument("--out", type=Path, required=True, help="сплит свежих не-машин")
    аргументы = парсер.parse_args(argv)

    картинки = [к for путь in аргументы.coco_annotations for к in read_coco_images(путь)]
    грузовые, _ = coco_pools(картинки, аргументы.seed)
    аргументы.coco_cache.mkdir(parents=True, exist_ok=True)
    свежие = unseen_images(грузовые, {путь.name for путь in аргументы.coco_cache.iterdir()})
    print(
        f"пул truck_bus {len(грузовые)}, свежих {len(свежие)}, "
        f"к скачиванию не более {аргументы.budget}"
    )

    # Импорт здесь: `ultralytics` живёт в группе `train`, которой в CI нет.
    from pipeline.detectors.yolo import YoloDetector

    сбои: list[str] = []
    итог = build_negative_frame(
        coco_candidates(
            itertools.islice(свежие, аргументы.budget), аргументы.coco_cache, download_url, сбои
        ),
        YoloDetector(weights=аргументы.weights),
        harvest_dir() / "crops",
        off_catalog_cap=0,
        excluded_photos=read_exclusions(аргументы.exclusions).photos,
        box_caps={"truck_bus": аргументы.cap},
    )
    write_split(итог.frame, аргументы.out)
    счётчики = {
        "kept": итог.kept,
        "rejected": итог.rejected,
        "failed": сбои,
        "pool": len(грузовые),
        "fresh": len(свежие),
        "seed": аргументы.seed,
    }
    аргументы.out.with_suffix(".json").write_text(
        json.dumps(счётчики, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({к: в for к, в in счётчики.items() if к != "failed"}, ensure_ascii=False))
    print(f"сбоев скачивания COCO: {len(сбои)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
