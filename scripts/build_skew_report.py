"""Отчёт о перекосе источника (камера над пешеходным переходом).

Читает список кадров перекоса в формате DROP_COLUMNS (category=source_skew) и
считает по каждому затронутому классу: всего кадров, из них перекос, сколько
останется после гипотетического прореживания и ярлык `можно_проредить` или
`дыра_покрытия` по порогу `--min-street`.

Пул **не меняется** — это только замер. По умолчанию считается от
`pool-clean.parquet` (без tier 1), иначе «всего» завысится удалёнными кадрами.

Скрипт пишет **только таблицу** между маркерами в готовом файле отчёта: остальной
текст отчёта рукописный, и полная перезапись файла его бы уничтожила.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.class_purity import (
    LABEL_COVERAGE_HOLE,
    ClassPurityError,
    class_impact,
    read_drop_list,
    skew_label,
    unknown_ids,
)
from pipeline.paths import splits_dir
from pipeline.schema import read_split

DEFAULT_SKEW_LIST = Path("bench/source-skew-candidates.csv")
DEFAULT_REPORT = Path("bench/source-skew-report.md")

# Порог по умолчанию взят из --min-class сборки pool-clean (tier 1), не выведен из данных.
DEFAULT_MIN_STREET = 10

# Разметка перекоса визуальная и ошибается на кадр-другой в обе стороны, поэтому
# ярлык класса, чей остаток лежит в пределах этого числа кадров от порога, ненадёжен.
BORDERLINE_MARGIN = 2

START_MARKER = "<!-- ТАБЛИЦА:НАЧАЛО (генерируется build_skew_report) -->"
END_MARKER = "<!-- ТАБЛИЦА:КОНЕЦ -->"


@dataclass(frozen=True, slots=True)
class ClassRow:
    """Строка отчёта по одному классу."""

    model_id: str
    total: int
    skew: int
    remaining: int
    label: str
    borderline: bool


def build_rows(pool: pd.DataFrame, skew_ids: Collection[str], min_street: int) -> list[ClassRow]:
    """Собрать строки отчёта по классам, затронутым перекосом.

    Сортировка — по убыванию числа кадров перекоса, затем по имени класса.
    Кадры перекоса, которых нет в пуле, `class_impact` игнорирует (они уже
    удалены), в счёт они не идут.
    """
    impact = class_impact(pool, skew_ids)
    rows = [
        ClassRow(
            model_id=model_id,
            total=before,
            skew=before - after,
            remaining=after,
            label=skew_label(after, min_street),
            borderline=abs(after - min_street) <= BORDERLINE_MARGIN,
        )
        for model_id, (before, after) in impact.items()
    ]
    rows.sort(key=lambda r: (-r.skew, r.model_id))
    return rows


def render_table(rows: Sequence[ClassRow]) -> str:
    """Отрисовать markdown-таблицу отчёта."""
    lines = [
        "| класс | всего | перекос | останется | ярлык | у порога |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r.model_id} | {r.total} | {r.skew} | {r.remaining} | {r.label} "
        f"| {'да' if r.borderline else ''} |"
        for r in rows
    )
    return "\n".join(lines)


def fill_table(text: str, table: str) -> str:
    """Заменить содержимое между маркерами таблицы, не трогая остальной текст."""
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"В отчёте нет маркеров таблицы {START_MARKER!r}..{END_MARKER!r}; "
            "рукописный файл не трогаю"
        )
    head = text[: start + len(START_MARKER)]
    tail = text[end:]
    return f"{head}\n{table}\n{tail}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Отчёт о перекосе источника (пул не меняется)")
    parser.add_argument(
        "--split",
        type=Path,
        default=splits_dir() / "pool-clean.parquet",
        help="Пул для замера (по умолчанию splits_dir() / pool-clean.parquet)",
    )
    parser.add_argument(
        "--skew-list",
        type=Path,
        default=DEFAULT_SKEW_LIST,
        help=f"CSV кадров перекоса (по умолчанию {DEFAULT_SKEW_LIST})",
    )
    parser.add_argument(
        "--min-street",
        type=int,
        default=DEFAULT_MIN_STREET,
        help=f"Порог достаточности остатка (по умолчанию {DEFAULT_MIN_STREET})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Файл отчёта с маркерами таблицы (по умолчанию {DEFAULT_REPORT})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Посчитать перекос и вписать таблицу в отчёт."""
    args = parse_args(argv)

    rows_csv = read_drop_list(args.skew_list)
    skew_ids = [r.photo_id for r in rows_csv]
    pool = read_split(args.split)

    missing = unknown_ids(skew_ids, pool["photo_id"])
    if missing:
        print(f"нет в пуле (в счёт не идут): {len(missing)} шт., напр. {missing[:5]}")

    rows = build_rows(pool, skew_ids, args.min_street)
    can_thin = sum(1 for r in rows if r.label != LABEL_COVERAGE_HOLE)
    print(f"классов с перекосом: {len(rows)}, из них можно проредить: {can_thin}")
    print(f"пул: {args.split} ({len(pool)} кадров), порог остатка: {args.min_street}")

    try:
        text = args.out.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ClassPurityError(
            f"Нет файла отчёта {args.out}: сначала создайте его с маркерами таблицы"
        ) from None

    args.out.write_text(fill_table(text, render_table(rows)), encoding="utf-8")
    print(f"таблица вписана в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
