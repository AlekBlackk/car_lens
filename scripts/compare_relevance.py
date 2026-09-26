"""Парное сравнение двух выгрузок `training.eval_relevance --predictions` по кадрам."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path

import pandas as pd

from pipeline.holdout_relevance import load_relevance


@dataclass(frozen=True, slots=True)
class PairedRow:
    name: str
    n: int
    base: float
    new: float
    wins: int
    losses: int
    p: float


def sign_test_p(wins: int, losses: int) -> float:
    """Точный двусторонний знаковый тест; без пар различий — p = 1."""
    n = wins + losses
    if n == 0:
        return 1.0
    # `int ** int` типизируется как Any (typeshed); аннотация возвращает арифметику к float.
    хвост: float = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * хвост)


def _строка(имя: str, база: pd.Series, новая: pd.Series) -> PairedRow:
    выигрыши = int((~база & новая).sum())
    проигрыши = int((база & ~новая).sum())
    return PairedRow(
        имя,
        len(база),
        float(база.mean()),
        float(новая.mean()),
        выигрыши,
        проигрыши,
        sign_test_p(выигрыши, проигрыши),
    )


def paired_rows(
    base: pd.DataFrame, new: pd.DataFrame, relevance: pd.DataFrame, *, weak_below: float
) -> list[PairedRow]:
    """Релевантный набор, слабые и сильные по базе классы; наборы кадров обязаны совпасть."""
    лишние = sorted(set(base["photo_id"]) ^ set(new["photo_id"]))
    if лишние:
        raise ValueError(f"выгрузки расходятся по photo_id: {len(лишние)} {лишние[:5]}")
    кадр = base.merge(new, on=["photo_id", "class_idx"], suffixes=("_base", "_new"))
    кадр = кадр.merge(relevance[["photo_id", "verdict"]], on="photo_id")
    кадр = кадр[кадр["verdict"] == "relevant"]
    база = кадр["top1_base"] == кадр["class_idx"]
    новая = кадр["top1_new"] == кадр["class_idx"]
    слабые = база.groupby(кадр["class_idx"]).transform("mean") < weak_below
    return [
        _строка("релевантный", база, новая),
        _строка("слабые классы", база[слабые], новая[слабые]),
        _строка("сильные классы", база[~слабые], новая[~слабые]),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Парное сравнение двух моделей на holdout_ru")
    парсер.add_argument("--base", type=Path, required=True)
    парсер.add_argument("--new", type=Path, required=True)
    парсер.add_argument("--relevance", type=Path, default=Path("catalog/holdout_relevance.csv"))
    парсер.add_argument("--weak-below", type=float, default=0.6)
    аргументы = парсер.parse_args(argv)
    читать = {"photo_id": str}
    строки = paired_rows(
        pd.read_csv(аргументы.base, dtype=читать),
        pd.read_csv(аргументы.new, dtype=читать),
        load_relevance(аргументы.relevance),
        weak_below=аргументы.weak_below,
    )
    print("| набор | кадров | база | новая | выигрыши / проигрыши | p |")
    print("|---|---|---|---|---|---|")
    for с in строки:
        доли = f"{с.base:.4f} | {с.new:.4f}"
        print(f"| {с.name} | {с.n} | {доли} | {с.wins} / {с.losses} | {с.p:.2g} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
