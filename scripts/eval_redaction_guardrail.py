"""Собрать и измерить frozen-unit few-shot guardrail выбранного профиля."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.manifest import HarvestRow, load_manifest
from pipeline.paths import data_root
from pipeline.redaction_calibration.baseline import FrozenUnit, Prediction
from pipeline.redaction_calibration.guardrail import (
    assert_no_frozen_units,
    build_guardrail_holdout,
    build_guardrail_pool,
    calculate_guardrail_report,
    render_guardrail_report,
    validate_guardrail_lineage,
)
from pipeline.schema import read_split, write_split
from scripts.eval_fewshot import write_predictions
from scripts.preregister_redaction_calibration import load_predictions

DEFAULT_CAMPAIGN_ROOT = data_root() / "redaction-calibration"


@dataclass(frozen=True, slots=True)
class GuardrailPaths:
    """Все изолированные артефакты одного guardrail-прогона."""

    pool: Path
    holdout: Path
    crops_root: Path
    index: Path
    predictions: Path
    report: Path

    @classmethod
    def for_fixture(cls, root: Path) -> GuardrailPaths:
        return cls(
            pool=root / "guardrail-pool.parquet",
            holdout=root / "guardrail-holdout.parquet",
            crops_root=root / "crops",
            index=root / "index",
            predictions=root / "candidate-predictions.csv",
            report=root / "guardrail.md",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_map(path: Path) -> tuple[FrozenUnit, ...]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected = ("photo_id", "listing_id", "split")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"колонки frozen map не совпали: {reader.fieldnames!r}")
        rows = list(reader)
    invalid = sorted({row["split"] for row in rows} - {"pool", "holdout_ru"})
    if invalid:
        raise ValueError(f"неизвестный split во frozen map: {invalid}")
    return tuple(
        FrozenUnit(
            row["photo_id"],
            row["listing_id"],
            cast(Literal["pool", "holdout_ru"], row["split"]),
        )
        for row in rows
    )


def load_baseline_predictions(path: Path) -> tuple[Prediction, ...]:
    """Прочитать замороженный exact baseline."""
    return load_predictions(path)


def _paths(arguments: argparse.Namespace) -> GuardrailPaths:
    root = arguments.out
    return GuardrailPaths(
        pool=root / "guardrail-pool.parquet",
        holdout=root / "guardrail-holdout.parquet",
        crops_root=arguments.crops_root,
        index=root / "index",
        predictions=root / "candidate-predictions.csv",
        report=arguments.report or root / "guardrail.md",
    )


def prepare_guardrail_splits(
    arguments: argparse.Namespace,
) -> tuple[tuple[FrozenUnit, ...], GuardrailPaths]:
    """Восстановить frozen units, записать новый pool и выживший holdout."""
    paths = _paths(arguments)
    frozen = _load_frozen_map(arguments.frozen_map)
    holdout_ids = {unit.photo_id for unit in frozen if unit.split == "holdout_ru"}
    manifest: list[HarvestRow] = load_manifest(arguments.manifest)
    catalog = load_catalog(arguments.catalog)
    pool = build_guardrail_pool(manifest, frozen, holdout_ids, catalog)
    holdout = build_guardrail_holdout(manifest, frozen, holdout_ids, catalog)
    write_split(pool, paths.pool)
    write_split(holdout, paths.holdout)
    return frozen, paths


def run_build_index(arguments: argparse.Namespace, paths: GuardrailPaths) -> None:
    """Построить индекс отдельным процессом в изолированный каталог."""
    command = [
        "python",
        "-m",
        "scripts.build_fewshot_index",
        "--split-path",
        str(paths.pool),
        "--crops-root",
        str(paths.crops_root),
        "--out",
        str(paths.index),
        "--catalog",
        str(arguments.catalog),
    ]
    if arguments.allow_missing_classes:
        command.append("--allow-missing-classes")
    subprocess.run(command, check=True)


def run_eval(arguments: argparse.Namespace, paths: GuardrailPaths) -> None:
    """Замерить выжившие кадры по новому индексу и сохранить per-frame predictions."""
    subprocess.run(
        [
            "python",
            "-m",
            "scripts.eval_fewshot",
            "--index",
            str(paths.index),
            "--split-path",
            str(paths.holdout),
            "--crops-root",
            str(paths.crops_root),
            "--predictions-out",
            str(paths.predictions),
            "--report",
            str(paths.report.with_name("fewshot-survivors.md")),
            "--diagnostic-only",
        ],
        check=True,
    )


def run_guardrail(arguments: argparse.Namespace) -> int:
    """Выполнить success path: lineage → leakage check → index → eval → report."""
    frozen, paths = prepare_guardrail_splits(arguments)
    baseline = load_baseline_predictions(arguments.baseline_predictions)
    validate_guardrail_lineage(baseline, frozen)
    final_pool = read_split(paths.pool)
    assert_no_frozen_units(final_pool, frozen)
    run_build_index(arguments, paths)
    final_holdout = read_split(paths.holdout)
    if final_holdout.empty:
        write_predictions(paths.predictions, ())
    else:
        run_eval(arguments, paths)
    candidate = load_predictions(paths.predictions)
    report = calculate_guardrail_report(
        baseline,
        candidate,
        pool_sha256=_sha256(paths.pool),
        split_sha256=_sha256(paths.holdout),
        index_sha256=_sha256(paths.index / "catalog_index.bin"),
    )
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(render_guardrail_report(report), encoding="utf-8", newline="\n")
    print(render_guardrail_report(report))
    return 0 if report.top1.passed and report.top3.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-unit quality guardrail профиля редакции")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crops-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--frozen-map",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT / "baseline" / "frozen-unit-map.csv",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT / "baseline" / "baseline-predictions.csv",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-missing-classes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_guardrail(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
