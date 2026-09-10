"""Предрегистрация: точный базис, замороженные единицы и запечатанные партии."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.redaction_calibration.baseline import (
    IndependentBatch,
    Prediction,
    campaign_budget_state,
    complete_batch,
    create_baseline_passport,
    exact_prediction_counts,
    freeze_units,
    open_batch,
    partition_independent_ids,
    start_batch_review,
    write_campaign_artifacts,
)
from scripts.preregister_redaction_calibration import preregistration_document


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["photo_id", "listing_id"])


def _passport_fixture(
    tmp_path: Path,
) -> tuple[Any, tuple[Prediction, ...], dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.csv"
    pool_path = tmp_path / "pool.parquet"
    holdout_path = tmp_path / "holdout_ru.parquet"
    manifest.write_bytes(b"sha256,class_id\npool-a,a\n")
    pool_path.write_bytes(b"fixture-pool")
    holdout_path.write_bytes(b"fixture-holdout")
    pool = _frame([("pool-a", "listing-a"), ("pool-b", "listing-b")])
    holdout = _frame([("hold-a", "listing-h1"), ("hold-b", "listing-h2")])
    predictions = (
        Prediction("hold-a", 4, (4, 2, 1)),
        Prediction("hold-b", 1, (3, 1, 0)),
    )
    arguments = {
        "manifest_path": manifest,
        "pool_path": pool_path,
        "holdout_path": holdout_path,
        "pool": pool,
        "holdout": holdout,
        "calibration_ids": ("pool-a",),
        "predictions": predictions,
        "profile": {
            "plate": {"confidence": 0.5},
            "plate_hires": {"confidence": 0.25},
        },
        "git_sha": "a" * 40,
        "git_clean": True,
        "detector_stage_sha256": "b" * 64,
        "code_sha256": {"pipeline/harvest/redact.py": "c" * 64},
        "weights_sha256": {"plate": "d" * 64, "plate_hires": "e" * 64},
        "jpeg": {"codec": "opencv.imencode", "parameters": "defaults"},
        "runtime": {"python": "3.11.9", "opencv-python-headless": "5.0.0.93"},
        "created_at": "2026-09-07T12:00:00Z",
        "seed": 1337,
        "batch_size": 1,
        "batch_count": 1,
    }
    passport = create_baseline_passport(**arguments)  # type: ignore[arg-type]
    return passport, predictions, arguments


# Следующая мутация обязана ломать тест: выбор партии начинает зависеть от порядка parquet.
def test_partition_is_stable_complete_and_disjoint() -> None:
    pool = [f"photo-{number:04d}" for number in range(2863)]
    calibration = pool[:300]

    batches, remainder = partition_independent_ids(pool, calibration)
    reversed_batches, reversed_remainder = partition_independent_ids(
        list(reversed(pool)), list(reversed(calibration))
    )

    assert len(batches) == 8
    assert [len(batch.photo_ids) for batch in batches] == [300] * 8
    assert len(remainder) == 163
    assert batches == reversed_batches
    assert remainder == reversed_remainder
    eligible = set(pool) - set(calibration)
    partitioned = [photo_id for batch in batches for photo_id in batch.photo_ids]
    assert set(partitioned) | set(remainder) == eligible
    assert len(partitioned) + len(remainder) == len(eligible)
    assert not set(partitioned) & set(remainder)
    assert all(batch.state == "sealed" for batch in batches)


def test_partition_rejects_duplicates_unknown_and_overlap() -> None:
    with pytest.raises(ValueError, match=r"дубликат.*pool_ids"):
        partition_independent_ids(["a", "a", "b"], ["a"], batch_size=1)
    with pytest.raises(ValueError, match=r"дубликат.*calibration_ids"):
        partition_independent_ids(["a", "b"], ["a", "a"], batch_size=1)
    with pytest.raises(ValueError, match=r"неизвестн.*calibration"):
        partition_independent_ids(["a", "b"], ["z"], batch_size=1)
    with pytest.raises(ValueError, match=r"полных партий"):
        partition_independent_ids(["a", "b"], [], batch_size=1, batch_count=3)


def test_each_batch_hash_binds_its_number_and_complete_list() -> None:
    batches, _ = partition_independent_ids(
        ["a", "b", "c", "d"], [], seed=7, batch_size=2, batch_count=2
    )
    changed, _ = partition_independent_ids(
        ["a", "b", "c", "e"], [], seed=7, batch_size=2, batch_count=2
    )

    assert len({batch.sha256 for batch in batches}) == 2
    assert [batch.sha256 for batch in batches] != [batch.sha256 for batch in changed]
    assert all(len(batch.sha256) == 64 for batch in batches)


def test_freeze_units_rejects_conflicting_or_duplicate_mappings() -> None:
    with pytest.raises(ValueError, match=r"конфликт.*photo_id"):
        freeze_units(_frame([("same", "one")]), _frame([("same", "two")]))
    with pytest.raises(ValueError, match=r"дубликат.*photo_id"):
        freeze_units(_frame([("same", "one"), ("same", "one")]), _frame([]))


def test_freeze_units_preserves_both_splits_without_holdout_overlap() -> None:
    units = freeze_units(
        _frame([("pool-a", "listing-a"), ("pool-b", "listing-b")]),
        _frame([("hold-a", "listing-h")]),
    )

    assert [(unit.photo_id, unit.listing_id, unit.split) for unit in units] == [
        ("hold-a", "listing-h", "holdout_ru"),
        ("pool-a", "listing-a", "pool"),
        ("pool-b", "listing-b", "pool"),
    ]


def test_exact_counts_are_derived_only_from_complete_predictions() -> None:
    predictions = [
        Prediction("a", 2, (2, 1, 0)),
        Prediction("b", 1, (3, 1, 4)),
        Prediction("c", 0, (3, 2, 1)),
    ]

    assert exact_prediction_counts(predictions, expected_ids={"a", "b", "c"}) == {
        "top1": {"k": 1, "n": 3},
        "top3": {"k": 2, "n": 3},
    }
    with pytest.raises((TypeError, ValueError)):
        create_baseline_passport(  # type: ignore[call-arg]
            predictions={"top1": 0.3333, "top3": 0.6667}  # type: ignore[arg-type]
        )


def test_exact_counts_reject_duplicate_missing_foreign_and_short_predictions() -> None:
    complete = [
        Prediction("a", 0, (0, 1, 2)),
        Prediction("b", 1, (1, 2, 0)),
    ]
    with pytest.raises(ValueError, match=r"дубликат.*prediction"):
        exact_prediction_counts([complete[0], complete[0]], expected_ids={"a"})
    with pytest.raises(ValueError, match=r"нет предсказаний"):
        exact_prediction_counts(complete[:1], expected_ids={"a", "b"})
    with pytest.raises(ValueError, match=r"неизвестные предсказания"):
        exact_prediction_counts(complete, expected_ids={"a"})
    with pytest.raises(ValueError, match=r"top_indices"):
        exact_prediction_counts([Prediction("a", 0, (0, 1))], expected_ids={"a"})


def test_passport_contains_full_frozen_provenance_and_exact_counts(tmp_path: Path) -> None:
    passport, _, _ = _passport_fixture(tmp_path)

    assert passport.pool_frames == 2
    assert passport.holdout_frames == 2
    assert passport.calibration_frames == 1
    assert passport.exact_metrics.top1_k == 1
    assert passport.exact_metrics.top3_k == 2
    assert passport.exact_metrics.n == 2
    assert len(passport.frozen_units) == 4
    assert len(passport.inputs["manifest"]["sha256"]) == 64
    assert passport.git == {"sha": "a" * 40, "clean": True}
    assert passport.created_at == "2026-09-07T12:00:00Z"


def test_passport_rejects_dirty_or_short_git_identity(tmp_path: Path) -> None:
    _, _, arguments = _passport_fixture(tmp_path)
    arguments = dict(arguments)
    arguments["git_clean"] = False
    with pytest.raises(ValueError, match=r"Git.*чист"):
        create_baseline_passport(**arguments)

    arguments["git_clean"] = True
    arguments["git_sha"] = "abc1234"
    with pytest.raises(ValueError, match=r"полный Git SHA"):
        create_baseline_passport(**arguments)


def test_open_and_review_batch_enforce_single_active_party() -> None:
    batches, _ = partition_independent_ids(["a", "b", "c"], [], seed=1, batch_size=1, batch_count=3)
    opened = open_batch(batches, 1)
    assert opened[0].state == "opened"
    with pytest.raises(ValueError, match=r"уже раскрыта"):
        open_batch(opened, 2)

    in_review = start_batch_review(opened, 1)
    assert in_review[0].state == "review_in_progress"
    with pytest.raises(ValueError, match=r"переход"):
        complete_batch(opened, 1, passed=True)
    completed = complete_batch(in_review, 1, passed=False)
    assert completed[0].state == "completed_fail"


def test_budget_exhausts_after_eight_failures_without_remainder() -> None:
    batches = tuple(
        IndependentBatch(
            number=number, state="completed_fail", photo_ids=(str(number),), sha256="x"
        )
        for number in range(1, 9)
    )
    assert campaign_budget_state(batches) == "budget_exhausted"

    with pytest.raises(ValueError, match=r"бюджет исчерпан"):
        open_batch(batches, 9)


def test_preregistration_freezes_every_step_five_decision_without_candidate_results() -> None:
    document = preregistration_document("2026-09-07T12:00:00Z")

    assert document["candidate_results_present"] is False
    assert set(document) >= {
        "profile_space",
        "pareto_axes",
        "pareto_rule",
        "visual_veto_fields",
        "no_collapse_rule",
        "quality_guardrail",
        "independent_taxonomy",
        "upper_bound_reporting",
        "party_budget",
        "technical_states_do_not_authorize_training",
    }
    assert document["training_authorized"] is False
    assert document["training_authorization_state"] == "full_pool_audited"


def test_fixture_dry_runs_are_byte_identical_and_keep_layout(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_passport, predictions, _ = _passport_fixture(tmp_path / "inputs-one")
    second_passport, _, _ = _passport_fixture(tmp_path / "inputs-two")
    preregistration = {
        "created_at": "2026-09-07T12:00:00Z",
        "candidate_results_present": False,
        "training_authorization_state": "full_pool_audited",
    }

    write_campaign_artifacts(first, first_passport, predictions, preregistration)
    write_campaign_artifacts(second, second_passport, predictions, preregistration)

    expected = {
        "campaign.json",
        "preregistration.json",
        "baseline/passport.json",
        "baseline/frozen-holdout.csv",
        "baseline/frozen-unit-map.csv",
        "baseline/baseline-predictions.csv",
        "baseline/independent-batches.json",
    }
    assert {
        str(path.relative_to(first)).replace("\\", "/")
        for path in first.rglob("*")
        if path.is_file()
    } == expected
    assert {
        str(path.relative_to(second)).replace("\\", "/")
        for path in second.rglob("*")
        if path.is_file()
    } == expected
    assert {
        str(path.relative_to(first)).replace("\\", "/"): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        str(path.relative_to(second)).replace("\\", "/"): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }

    campaign = json.loads((first / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["state"] == "preregistered"
    assert campaign["training_authorized"] is False
    assert campaign["training_authorization_state"] == "full_pool_audited"
    batch_document = json.loads(
        (first / "baseline" / "independent-batches.json").read_text(encoding="utf-8")
    )
    assert "remainder" in batch_document
    assert batch_document["remainder"]["photo_ids"] == []
