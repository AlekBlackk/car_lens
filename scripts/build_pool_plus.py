"""Склейка внешних источников с `pool-clean`: `pool-ext`, `pool-plus` и отчёт.

Читает `splits/ext-<источник>.parquet` и счётчики `ext-<источник>.json`, которые пишет
`scripts.ingest_external`. Отчёт `bench/external-ingest.md` генерируется целиком: рукописных
разделов в нём нет, перезапись безопасна.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, Catalog, load_catalog
from pipeline.ingest.external import (
    DEFAULT_MAX_PER_CLASS,
    GAP_CLASSES_PATH,
    SOURCE_PRIORITY,
    merge_external,
    read_gap_classes,
)
from pipeline.paths import splits_dir
from pipeline.schema import read_split, write_split

DEFAULT_REPORT = Path("bench/external-ingest.md")


def render_report(
    catalog: Catalog,
    pool: pd.DataFrame,
    ext: pd.DataFrame,
    ingest_stats: Mapping[str, Mapping[str, Any]],
    merge_rejected: Mapping[str, int],
    gap: frozenset[str],
) -> str:
    """Отчёт: кадры по классам и источникам, отказы приёма и склейки."""
    в_пуле = pool["class_idx"].value_counts()
    по_источникам = {
        источник: ext.loc[ext["source"] == источник, "class_idx"].value_counts()
        for источник in SOURCE_PRIORITY
    }
    строки = [
        "# Внешние данные: приём и склейка",
        "",
        "Генерируется `python -m scripts.build_pool_plus` целиком.",
        "",
        f"`pool-clean` {len(pool)} кадров + внешние {len(ext)} = `pool-plus` "
        f"{len(pool) + len(ext)}.",
        "",
        "## По классам",
        "",
        "| класс | дыра | pool-clean | " + " | ".join(SOURCE_PRIORITY) + " | итого |",
        "|---|---|---|" + "---|" * len(SOURCE_PRIORITY) + "---|",
    ]
    for номер, класс in enumerate(catalog.classes):
        пул = int(в_пуле.get(номер, 0))
        внешние = [int(по_источникам[и].get(номер, 0)) for и in SOURCE_PRIORITY]
        ячейки = " | ".join(str(n) for n in внешние)
        дыра = "да" if класс.class_id in gap else ""
        строки.append(f"| `{класс.class_id}` | {дыра} | {пул} | {ячейки} | {пул + sum(внешние)} |")
    строки += [
        "",
        "## Приём по источникам",
        "",
        "| источник | просмотрено | принято | отказы |",
        "|---|---|---|---|",
    ]
    for источник, счёт in ingest_stats.items():
        причины = ", ".join(f"{k} {v}" for k, v in sorted(счёт["rejected"].items()))
        строки.append(f"| {источник} | {счёт['seen']} | {счёт['kept']} | {причины or '—'} |")
    склейка = ", ".join(f"{k} {v}" for k, v in sorted(merge_rejected.items()))
    строки += ["", f"Склейка: {склейка or 'без отказов'}.", ""]
    return "\n".join(строки)


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Склеить внешние источники с pool-clean")
    парсер.add_argument("--pool", type=Path, default=splits_dir() / "pool-clean.parquet")
    парсер.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCE_PRIORITY,
        default=list(SOURCE_PRIORITY),
        help="источники; у каждого обязан быть splits/ext-<источник>.parquet",
    )
    парсер.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    парсер.add_argument("--seed", type=int, default=1337)
    парсер.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    аргументы = парсер.parse_args(argv)

    нет = [
        источник
        for источник in аргументы.sources
        if not (splits_dir() / f"ext-{источник}.parquet").is_file()
    ]
    if нет:
        raise SystemExit(
            f"нет сплитов источников: {', '.join(нет)} — сначала scripts.ingest_external "
            "или уберите их из --sources"
        )
    кадры = {и: read_split(splits_dir() / f"ext-{и}.parquet") for и in аргументы.sources}
    счётчики = {
        и: json.loads((splits_dir() / f"ext-{и}.json").read_text(encoding="utf-8"))
        for и in аргументы.sources
    }
    внешние, отказы = merge_external(
        кадры, max_per_class=аргументы.max_per_class, seed=аргументы.seed
    )
    пул = read_split(аргументы.pool)
    write_split(внешние, splits_dir() / "pool-ext.parquet")
    write_split(pd.concat([пул, внешние], ignore_index=True), splits_dir() / "pool-plus.parquet")

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    аргументы.report.write_text(
        render_report(
            каталог, пул, внешние, счётчики, отказы, read_gap_classes(GAP_CLASSES_PATH, каталог)
        ),
        encoding="utf-8",
    )
    print(f"pool-ext {len(внешние)}, pool-plus {len(пул) + len(внешние)}, склейка {отказы}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
