"""Собрать сплиты Stanford Cars: детектор → кропы → parquet → инварианты.

Единственное место, которое знает и про `YoloDetector`, и про каталог
скелета, и про запись на диск. Оркестратор `pipeline.ingest.build` о них
не знает: он принимает детектор параметром, и поэтому его тесты
обходятся подставным.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pandas as pd

from pipeline.catalog import STANFORD_CATALOG_PATH, load_catalog
from pipeline.ingest.build import BuildStats, build_frame
from pipeline.ingest.stanford import read_annotations
from pipeline.normalize import DEFAULT_ALIASES_PATH, load_aliases
from pipeline.paths import crops_dir, runs_dir, splits_dir
from pipeline.schema import write_split
from pipeline.split import check_invariants, split_by_listing

# Скелет: замер реального прогона (16185 фотографий на входе, отбраковка
# no_vehicle=213 и too_small=130, дедупом убрано 298, unmapped=0 — все 196
# имён и все 171 класс каталога сопоставились) даёт минимум 37 уникальных
# `listing_id` в train у самых бедных классов: Hyundai Accent (37), Fiat
# 500 Abarth (38), Rolls-Royce Phantom Drophead (38), Maybach Landaulet
# (38) — это модели, которых в Stanford Cars физически мало, а не следствие
# ошибок нормализатора. 30 выбрано заметно ниже наблюдаемого минимума —
# запас на пересчёт дедупа и порогов отбраковки, — но всё ещё ловит
# катастрофу сопоставления (класс с тремя объявлениями вместо семидесяти).
# Умолчание 300 из §3.3 — продакшн-требование, не трогается, за ним следит
# тест из M-1a.
MIN_LISTINGS_ДЛЯ_СКЕЛЕТА = 30


def render_report(stats: BuildStats, splits: Mapping[str, pd.DataFrame], classes: int) -> str:
    """Собрать текст отчёта о прогоне."""
    строки = [
        "# Отчёт ingest Stanford Cars (M-1b)",
        "",
        f"Дата: {datetime.now(UTC).date().isoformat()}",
        "",
        f"- Фотографий на входе: {stats.seen}",
        f"- Классов в каталоге скелета: {classes}",
        f"- Несопоставленных имён: {stats.unmapped}",
        f"- Убрано дедупом: {stats.deduped}",
        "",
        "## Отбраковка по причинам",
        "",
    ]
    if stats.rejected:
        строки += [
            f"- `{причина}`: {сколько}" for причина, сколько in sorted(stats.rejected.items())
        ]
    else:
        строки.append("- отбраковки не было")

    строки += ["", "## Сверка с ручной разметкой", ""]
    if stats.iou_values:
        медиана = statistics.median(stats.iou_values)
        строки.append(f"- Пар для сверки: {len(stats.iou_values)}")
        строки.append(f"- Медианный IoU YOLO против devkit: {медиана:.3f}")
    else:
        строки.append("- нет разметки в источнике, сверка не проводилась")

    строки += ["", "## Сплиты", ""]
    строки += [f"- `{имя}`: {len(d)} строк" for имя, d in sorted(splits.items())]
    return "\n".join(строки) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа. Возвращает код выхода процесса."""
    парсер = argparse.ArgumentParser(description="Собрать сплиты Stanford Cars (M-1b)")
    парсер.add_argument("--weights", default="yolo11n.pt")
    парсер.add_argument("--seed", type=int, default=1337)
    парсер.add_argument(
        "--min-listings",
        type=int,
        default=MIN_LISTINGS_ДЛЯ_СКЕЛЕТА,
        help="Порог инварианта 3. Умолчание §3.3 (300) на скелете невыполнимо",
    )
    аргументы = парсер.parse_args(argv)

    # Импорт здесь: `ultralytics` живёт в группе `train`, которой в CI нет.
    from pipeline.detectors.yolo import YoloDetector
    from scripts.fetch_stanford import dataset_dir

    каталог = load_catalog(STANFORD_CATALOG_PATH)
    фото = read_annotations(dataset_dir())
    итог = build_frame(
        фото,
        YoloDetector(weights=аргументы.weights),
        каталог,
        load_aliases(DEFAULT_ALIASES_PATH),
        crops_dir() / "stanford",
    )

    сплиты = split_by_listing(итог.frame, seed=аргументы.seed)
    check_invariants(сплиты, min_listings_per_class=аргументы.min_listings)

    for имя, кадр in сплиты.items():
        write_split(кадр, splits_dir() / f"{имя}.parquet")

    прогон = runs_dir() / f"m1b_ingest_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    прогон.mkdir(parents=True, exist_ok=True)
    (прогон / "report.md").write_text(
        render_report(итог.stats, сплиты, len(каталог)), encoding="utf-8"
    )
    # Механизм из решения Р-4: без списка датасетов через полгода нельзя
    # ответить, какие веса «чистые», и придётся переобучать всё на всякий случай.
    (прогон / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": ["stanford_cars"],
                "licence": "research-only",
                "catalog": str(STANFORD_CATALOG_PATH.name),
                "seed": аргументы.seed,
                "min_listings_per_class": аргументы.min_listings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Отчёт: {прогон / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
