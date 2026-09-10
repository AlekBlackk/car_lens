"""CLI предрегистрации кампании рекалибровки редакции номерных знаков."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.harvest.detectors import DEFAULT_DETECTORS_MANIFEST, load_detectors_manifest
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH
from pipeline.harvest.redaction_profile import (
    CURRENT_REDACTION_PROFILE,
    DETECTOR_STAGE_SOURCES,
    detector_stage_sha256,
)
from pipeline.paths import data_root, splits_dir
from pipeline.redaction_calibration.baseline import (
    Prediction,
    create_baseline_passport,
    write_campaign_artifacts,
)
from pipeline.schema import read_split

DEFAULT_CAMPAIGN_ROOT = data_root() / "redaction-calibration"
DEFAULT_CALIBRATION = Path("bench") / "plate-leak-sample.csv"
DEFAULT_PREDICTIONS = DEFAULT_CAMPAIGN_ROOT / "baseline-predictions.csv"
DEFAULT_CREATED_AT = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_identity(*, allow_dirty: bool) -> tuple[str, bool]:
    sha = _git("rev-parse", "HEAD")
    clean = _git("status", "--porcelain") == ""
    if not clean and not allow_dirty:
        raise ValueError("Git-дерево не чистое; сначала зафиксируйте preregistration checkpoint")
    return sha, clean


def load_predictions(path: Path) -> tuple[Prediction, ...]:
    """Прочитать строгий CSV экспорта `eval_fewshot --predictions-out`."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected = ("photo_id", "target", "top_indices")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"колонки predictions не совпали: {reader.fieldnames!r} != {expected!r}"
            )
        return tuple(
            Prediction(
                photo_id=row["photo_id"],
                target=int(row["target"]),
                top_indices=tuple(int(value) for value in row["top_indices"].split()),
            )
            for row in reader
        )


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "opencv-runtime": cv2.__version__,
        "numpy-runtime": np.__version__,
    }
    for package in (
        "opencv-python-headless",
        "numpy",
        "pandas",
        "pyarrow",
        "pillow",
        "open-clip-torch",
        "torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "absent"
    return versions


def preregistration_document(created_at: str) -> dict[str, Any]:
    """Машиночитаемые решения до любого измеренного результата кандидата."""
    return {
        "schema_version": 1,
        "created_at": created_at,
        "candidate_results_present": False,
        "profile_space": {
            "geometry": {
                "ssd_tiles": [1280, 600, 300],
                "yolo_tiles": [1280, 640],
                "overlap": 0.25,
                "derivation": "real tile_origins plus observed canonical W_max/H_max",
            },
            "confidence": "observed canonical-plate confidence breakpoints",
            "independent_batches_may_expand_space": False,
        },
        "pareto_axes": [
            "ssd_confidence_margin",
            "yolo_confidence_margin",
            "median_positive_delta_black",
            "p95_positive_delta_black",
            "delta_black_above_0_25",
            "new_over_redacted_and_rejects",
            "changed_crops_for_h3",
        ],
        "pareto_rule": "not worse on every axis and strictly better on at least one",
        "visual_veto_fields": [
            "profile_id",
            "sha256_or_plate_id",
            "destroyed_model_feature",
            "disqualification_reason_not_cosmetic",
        ],
        "no_collapse_rule": "no weighted score or automatic tie-break; human selects full front",
        "quality_guardrail": {
            "lineage_frames": 1523,
            "axes": ["penalized_top1", "penalized_top3"],
            "boundary": "baseline exact k/n two-sided 95% Clopper-Pearson lower bound",
            "interpretation": (
                "guardrail did not detect collapse; not proof of absence of degradation"
            ),
        },
        "independent_taxonomy": {
            "pipeline_outcome": ["accepted", "rejected", "operational_error"],
            "review_status": ["pending", "resolved", "unreviewable"],
            "plate_verdict": [
                "leak",
                "leak_hist",
                "clean_partial",
                "clean_unreadable",
                "clean_filled",
                "clean_no_plate",
            ],
        },
        "upper_bound_reporting": (
            "two-sided 95% Clopper-Pearson and projection to actual post-profile pool"
        ),
        "party_budget": {
            "seed": 1337,
            "batches": 8,
            "batch_size": 300,
            "remainder_is_not_a_batch": True,
            "after_eight_failures": "budget_exhausted",
        },
        "training_authorized": False,
        "training_authorization_state": "full_pool_audited",
        "technical_states_do_not_authorize_training": True,
    }


def build_parser() -> argparse.ArgumentParser:
    """Явные пути позволяют сначала dry-run на фикстурах, затем реальный паспорт."""
    parser = argparse.ArgumentParser(description="Предрегистрация рекалибровки редакции")
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--pool", type=Path, default=splits_dir() / "pool.parquet")
    parser.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--created-at", required=True, help="фиксированное UTC ISO-время паспорта")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="только fixture dry-run до checkpoint; паспорт запишет clean=false",
    )
    return parser


def _calibration_ids(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if "sha256" not in (reader.fieldnames or ()):
            raise ValueError(f"в calibration нет колонки sha256: {path}")
        return tuple(row["sha256"] for row in reader)


def main(argv: Sequence[str] | None = None) -> int:
    """Проверить первичные данные и записать неизменяемую границу кампании."""
    arguments = build_parser().parse_args(argv)
    for path in (
        arguments.manifest,
        arguments.pool,
        arguments.holdout,
        arguments.calibration,
        arguments.predictions,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    pool = read_split(arguments.pool)
    holdout = read_split(arguments.holdout)
    predictions = load_predictions(arguments.predictions)
    git_sha, git_clean = _git_identity(allow_dirty=arguments.allow_dirty)
    pins = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    code_hashes = {
        str(path.relative_to(Path(__file__).resolve().parent.parent)).replace("\\", "/"): _sha256(
            path
        )
        for path in DETECTOR_STAGE_SOURCES
    }
    passport = create_baseline_passport(
        manifest_path=arguments.manifest,
        pool_path=arguments.pool,
        holdout_path=arguments.holdout,
        pool=pool,
        holdout=holdout,
        calibration_ids=_calibration_ids(arguments.calibration),
        predictions=predictions,
        profile=asdict(CURRENT_REDACTION_PROFILE),
        git_sha=git_sha,
        git_clean=git_clean,
        detector_stage_sha256=detector_stage_sha256(),
        code_sha256=code_hashes,
        weights_sha256={name: pin.sha256 for name, pin in pins.items()},
        jpeg={"codec": "cv2.imencode", "extension": ".jpg", "parameters": "defaults"},
        runtime=_runtime_versions(),
        created_at=arguments.created_at,
    )
    write_campaign_artifacts(
        arguments.campaign_root,
        passport,
        predictions,
        preregistration_document(arguments.created_at),
    )
    print(f"кампания: {arguments.campaign_root}")
    print(
        f"pool/holdout/calibration: {passport.pool_frames}/"
        f"{passport.holdout_frames}/{passport.calibration_frames}"
    )
    print(
        f"точный базис top-1 {passport.exact_metrics.top1_k}/{passport.exact_metrics.n}, "
        f"top-3 {passport.exact_metrics.top3_k}/{passport.exact_metrics.n}"
    )
    print(
        f"партии: {len(passport.independent_batches)}×{passport.independent_batch_size}, "
        f"остаток {len(passport.independent_remainder)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
