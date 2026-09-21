"""Сборка очищенного пула (pool-clean.parquet) без кандидатов на исключение.

Исключает кадры выбранных уровней tier из исходного сплита,
проверяет целостность идентификаторов, выводит статистику влияния на классы
и выполняет контрольное перечитывание записанного сплита.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pipeline.class_purity import (
    TIERS,
    ClassPurityError,
    class_impact,
    cleaned_pool,
    read_drop_list,
    select_tiers,
    unknown_ids,
)
from pipeline.paths import splits_dir
from pipeline.schema import read_split, write_split

DEFAULT_DROP_LIST = Path("bench/class-purity-drop-candidates.csv")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Собрать pool-clean.parquet без кандидатов на исключение"
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=splits_dir() / "pool.parquet",
        help="Путь к исходному пулу (по умолчанию splits_dir() / pool.parquet)",
    )
    parser.add_argument(
        "--drop-list",
        type=Path,
        default=DEFAULT_DROP_LIST,
        help=(
            "Путь к CSV со списком кандидатов на исключение "
            "(по умолчанию bench/class-purity-drop-candidates.csv)"
        ),
    )
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1],
        choices=TIERS,
        help="Уровни tier для исключения (по умолчанию [1])",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=splits_dir() / "pool-clean.parquet",
        help="Путь к результирующему файлу (по умолчанию splits_dir() / pool-clean.parquet)",
    )
    parser.add_argument(
        "--min-class",
        type=int,
        default=10,
        help="Порог предупреждения о малом числе кадров в классе (по умолчанию 10)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать очищенный сплит пула."""
    args = parse_args(argv)

    if args.out.resolve() == args.split.resolve():
        raise ValueError(
            f"Выходной файл не может указывать на исходный пул: {args.out} == {args.split}"
        )

    rows = read_drop_list(args.drop_list)
    pool = read_split(args.split)

    missing = unknown_ids([r.photo_id for r in rows], pool["photo_id"])
    if missing:
        raise ClassPurityError(
            f"В списке есть photo_id, отсутствующие в пуле ({len(missing)} шт.): {missing[:5]}"
        )

    drop_ids = select_tiers(rows, args.tiers)
    cleaned = cleaned_pool(pool, drop_ids)
    impact = class_impact(pool, drop_ids)

    print(f"удалено {len(drop_ids)}, осталось {len(cleaned)}")
    print(f"затронутых классов: {len(impact)}")
    for model_id, (before, after) in sorted(impact.items()):
        if after < args.min_class:
            print(f"{model_id}: {before} -> {after}")

    write_split(cleaned, args.out)

    re_read = read_split(args.out)
    if set(re_read["photo_id"]) != set(cleaned["photo_id"]):
        sys.stderr.write("Ошибка: photo_id перечитанного файла не сходятся с ожидаемыми\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
