"""Кампания MAD-Cars: выборка, скачивание, листы (`python -m scripts.mad_cars <команда>`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.ingest.mad_cars import read_mad_mapping, sample_frames
from pipeline.paths import data_root

DEFAULT_ROOT = data_root() / "external" / "mad_cars"
DEFAULT_MAPPING = Path("catalog/external/mad_cars.csv")


def _sample(аргументы: argparse.Namespace) -> int:
    таблица = read_mad_mapping(аргументы.mapping, load_catalog(DEFAULT_CATALOG_PATH))
    манифест = pd.read_csv(
        аргументы.manifest, dtype=str, usecols=["brand", "model", "car_id", "view_id", "url"]
    )
    выборка = sample_frames(
        манифест, таблица, cars_per_class=аргументы.cars_per_class, seed=аргументы.seed
    )
    выборка.to_csv(аргументы.out, index=False)
    print(
        f"выборка: {выборка['model_id'].nunique()} классов, {выборка['car_id'].nunique()} "
        f"машин, {len(выборка)} кадров → {аргументы.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Кампания MAD-Cars")
    команды = парсер.add_subparsers(dest="command", required=True)
    выборка = команды.add_parser("sample", help="выборка кадров из mad.csv")
    выборка.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "mad.csv")
    выборка.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    выборка.add_argument("--cars-per-class", type=int, default=100)
    выборка.add_argument("--seed", type=int, required=True)
    выборка.add_argument("--out", type=Path, required=True)
    выборка.set_defaults(run=_sample)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    аргументы = build_parser().parse_args(argv)
    return int(аргументы.run(аргументы))


if __name__ == "__main__":
    raise SystemExit(main())
