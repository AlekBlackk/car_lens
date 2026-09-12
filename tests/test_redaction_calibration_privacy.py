"""Тесты независимого privacy-ledger и точного отчёта."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pipeline.plate_leak import LEAK_DEFINITIONS, clopper_pearson
from pipeline.redaction_calibration.privacy import (
    IndependentRow,
    compute_independent_report,
    merge_independent_reviews,
    validate_independent_ledger,
)
from pipeline.redaction_calibration.state import CalibrationStateError
from scripts.measure_independent_plate_leak import STAGING_FIELDS, main
from scripts.stage_plate_leak_views import main as stage_party_views

VERDICTS = (
    "leak",
    "leak_hist",
    "clean_partial",
    "clean_unreadable",
    "clean_filled",
    "clean_no_plate",
)


def _accepted(photo_id: str, verdict: str = "clean_filled") -> IndependentRow:
    return IndependentRow(photo_id, "accepted", "", "resolved", verdict, "")


def test_ledger_accepts_all_six_plate_verdicts() -> None:
    rows = tuple(_accepted(f"id-{index}", verdict) for index, verdict in enumerate(VERDICTS))

    validate_independent_ledger(rows, tuple(row.photo_id for row in rows))


@pytest.mark.parametrize("verdict", ["unknown", ""])
def test_resolved_accepted_row_requires_known_verdict(verdict: str) -> None:
    row = _accepted("id-1", verdict)

    with pytest.raises(ValueError, match="plate_verdict"):
        validate_independent_ledger((row,), ("id-1",))


def test_rejected_row_requires_reason_and_no_plate_verdict() -> None:
    missing_reason = IndependentRow("id-1", "rejected", "", "resolved", "", "")
    has_verdict = replace(missing_reason, reject_reason="over_redacted", plate_verdict="leak")

    with pytest.raises(ValueError, match="reject_reason"):
        validate_independent_ledger((missing_reason,), ("id-1",))
    with pytest.raises(ValueError, match="plate_verdict"):
        validate_independent_ledger((has_verdict,), ("id-1",))


@pytest.mark.parametrize(
    "rows,sealed_ids,message",
    [
        ((_accepted("id-1"), _accepted("id-1")), ("id-1",), "дубликат"),
        ((_accepted("id-1"),), ("id-1", "id-2"), "не хватает"),
        ((_accepted("foreign"),), ("id-1",), "чужие"),
    ],
)
def test_ledger_rejects_duplicate_missing_and_foreign_ids(
    rows: tuple[IndependentRow, ...], sealed_ids: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_independent_ledger(rows, sealed_ids)


def test_report_arithmetic_and_blocking_categories_close() -> None:
    rows = (
        _accepted("resolved"),
        IndependentRow("unreviewable", "accepted", "", "unreviewable", "", "заливка"),
        IndependentRow("rejected", "rejected", "over_redacted", "resolved", "", ""),
        IndependentRow("operational", "operational_error", "missing_crop", "pending", "", ""),
    )
    validate_independent_ledger(rows, tuple(row.photo_id for row in rows))

    report = compute_independent_report(rows, post_profile_pool_size=2863)

    assert report.accepted_resolved == 1
    assert report.accepted_unreviewable == 1
    assert report.rejected == 1
    assert report.operational_error == 1
    assert report.accounted == 4
    assert not report.passed


def test_zero_of_300_uses_exact_interval_and_actual_population() -> None:
    rows = tuple(_accepted(f"id-{index:03d}", "clean_no_plate") for index in range(300))

    report = compute_independent_report(rows, post_profile_pool_size=2511)

    strict = next(bound for bound in report.bounds if bound.definition == "строго")
    assert strict.ci == clopper_pearson(0, 300)
    assert strict.projected_upper_count == pytest.approx(strict.ci[1] * 2511)
    assert tuple(bound.definition for bound in report.bounds) == tuple(
        definition for definition, _ in LEAK_DEFINITIONS
    )
    assert report.passed
    assert "observed zero does not prove population zero" in report.notes
    assert "does not satisfy rule 1 R-11" in report.notes


def test_report_requires_full_sealed_batch_not_arbitrary_subset() -> None:
    rows = tuple(_accepted(f"id-{index}") for index in range(299))

    report = compute_independent_report(rows, post_profile_pool_size=299)

    assert report.accounted == 299
    assert not report.passed


def test_pending_accepted_row_blocks_report() -> None:
    rows = (IndependentRow("id-1", "accepted", "", "pending", "", ""),)
    validate_independent_ledger(rows, ("id-1",))

    report = compute_independent_report(rows, post_profile_pool_size=1)

    assert report.accepted_pending == 1
    assert not report.passed
    assert report.bounds == ()


def test_human_review_is_merged_by_photo_id_not_row_position() -> None:
    pipeline_rows = (
        IndependentRow("id-a", "accepted", "", "pending", "", ""),
        IndependentRow("id-b", "accepted", "", "pending", "", ""),
    )
    review_rows = (
        {"photo_id": "id-b", "review_status": "resolved", "plate_verdict": "leak"},
        {
            "photo_id": "id-a",
            "review_status": "resolved",
            "plate_verdict": "clean_filled",
        },
    )

    merged = merge_independent_reviews(pipeline_rows, review_rows)

    assert [(row.photo_id, row.plate_verdict) for row in merged] == [
        ("id-a", "clean_filled"),
        ("id-b", "leak"),
    ]


def _sha(character: str) -> str:
    return character * 64


def _profile_document() -> dict[str, object]:
    return {
        "profile_id": "profile-a",
        "state": "quality_guardrail_passed",
        "profile_sha256": _sha("1"),
        "canonical_sha256": _sha("2"),
        "geometry_witness_sha256": _sha("3"),
        "w_max": 120,
        "h_max": 60,
    }


def _batch_document(state: str = "sealed") -> dict[str, Any]:
    return {
        "seed": 1337,
        "batch_size": 2,
        "budget_state": "available",
        "batches": [
            {
                "number": 1,
                "state": state,
                "photo_ids": ["id-a", "id-b"],
                "sha256": _sha("a"),
            },
            {
                "number": 2,
                "state": "sealed",
                "photo_ids": ["id-c", "id-d"],
                "sha256": _sha("b"),
            },
        ],
        "remainder": {"photo_ids": [], "sha256": _sha("c"), "usable_as_batch": False},
    }


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_pipeline(path: Path, outcomes: tuple[tuple[str, str, str], ...]) -> None:
    """Pipeline CSV несёт и колонки аудита: именно их ledger отдаёт party-видам."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("photo_id", "pipeline_outcome", "reject_reason", *STAGING_FIELDS),
        )
        writer.writeheader()
        for photo_id, outcome, reason in outcomes:
            writer.writerow(
                {
                    "photo_id": photo_id,
                    "pipeline_outcome": outcome,
                    "reject_reason": reason,
                    "crop_path": f"{photo_id}.jpg" if outcome == "accepted" else "",
                    "class_id": "class-a",
                    "plates": 1,
                    "faces": 0,
                }
            )


def test_cli_interruption_after_partial_review_keeps_review_in_progress(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    reviews = tmp_path / "reviews.csv"
    ledger = tmp_path / "ledger.csv"
    report = tmp_path / "report.json"
    _write_json(batches, _batch_document())
    _write_json(profile, _profile_document())
    _write_pipeline(pipeline, (("id-a", "accepted", ""), ("id-b", "accepted", "")))
    reviews.write_text(
        "photo_id,review_status,plate_verdict,note\nid-a,resolved,clean_filled,first only\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--batches",
            str(batches),
            "--profile",
            str(profile),
            "--pipeline",
            str(pipeline),
            "--reviews",
            str(reviews),
            "--ledger",
            str(ledger),
            "--report",
            str(report),
            "--post-profile-pool-size",
            "2",
        ]
    )

    document = json.loads(batches.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert document["batches"][0]["state"] == "review_in_progress"
    assert document["batches"][0]["review_result_sha256"]
    assert not report.exists()
    assert [row["photo_id"] for row in csv.DictReader(ledger.open(encoding="utf-8"))] == [
        "id-a",
        "id-b",
    ]


def test_cli_opened_batch_resumes_without_reopening(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    reviews = tmp_path / "reviews.csv"
    ledger = tmp_path / "ledger.csv"
    report = tmp_path / "report.json"
    document = _batch_document("opened")
    document["batches"][0].update(
        {
            "profile_id": "profile-a",
            "profile_sha256": _sha("1"),
            "geometry_witness_sha256": _sha("3"),
            "open_result_sha256": _sha("4"),
        }
    )
    _write_json(batches, document)
    _write_json(profile, _profile_document())
    _write_pipeline(pipeline, (("id-a", "accepted", ""), ("id-b", "accepted", "")))
    reviews.write_text(
        "photo_id,review_status,plate_verdict,note\n"
        "id-a,resolved,clean_filled,\n"
        "id-b,resolved,leak,\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--batches",
            str(batches),
            "--profile",
            str(profile),
            "--pipeline",
            str(pipeline),
            "--reviews",
            str(reviews),
            "--ledger",
            str(ledger),
            "--report",
            str(report),
            "--post-profile-pool-size",
            "2",
        ]
    )

    persisted = json.loads(batches.read_text(encoding="utf-8"))["batches"][0]
    assert exit_code == 1
    assert persisted["state"] == "completed_fail"
    assert persisted["open_result_sha256"] == _sha("4")


def test_cli_review_in_progress_batch_resumes_to_completion(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    reviews = tmp_path / "reviews.csv"
    document = _batch_document("review_in_progress")
    document["batches"][0].update(
        {
            "profile_id": "profile-a",
            "profile_sha256": _sha("1"),
            "geometry_witness_sha256": _sha("3"),
            "open_result_sha256": _sha("4"),
            "review_result_sha256": _sha("5"),
        }
    )
    _write_json(batches, document)
    _write_json(profile, _profile_document())
    _write_pipeline(pipeline, (("id-a", "accepted", ""), ("id-b", "accepted", "")))
    reviews.write_text(
        "photo_id,review_status,plate_verdict,note\n"
        "id-a,resolved,clean_filled,\n"
        "id-b,resolved,leak,\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--batches",
            str(batches),
            "--profile",
            str(profile),
            "--pipeline",
            str(pipeline),
            "--reviews",
            str(reviews),
            "--ledger",
            str(tmp_path / "ledger.csv"),
            "--report",
            str(tmp_path / "report.json"),
            "--post-profile-pool-size",
            "2",
        ]
    )

    assert exit_code == 1
    assert json.loads(batches.read_text(encoding="utf-8"))["batches"][0]["state"] == (
        "completed_fail"
    )


def test_cli_refuses_next_batch_while_prior_review_is_active(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    _write_json(batches, _batch_document("review_in_progress"))
    document = json.loads(batches.read_text(encoding="utf-8"))
    document["batches"][0].update(
        {
            "profile_id": "profile-a",
            "profile_sha256": _sha("1"),
            "geometry_witness_sha256": _sha("3"),
            "open_result_sha256": _sha("4"),
            "review_result_sha256": _sha("5"),
        }
    )
    _write_json(batches, document)
    _write_json(profile, _profile_document())
    _write_pipeline(pipeline, (("id-c", "accepted", ""), ("id-d", "accepted", "")))

    with pytest.raises(CalibrationStateError, match="активная партия"):
        main(
            [
                "--batches",
                str(batches),
                "--batch-number",
                "2",
                "--profile",
                str(profile),
                "--pipeline",
                str(pipeline),
                "--ledger",
                str(tmp_path / "ledger.csv"),
                "--report",
                str(tmp_path / "report.json"),
                "--post-profile-pool-size",
                "2",
            ]
        )


def test_cli_missing_accepted_crop_is_an_operational_failure(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    reviews = tmp_path / "reviews.csv"
    ledger = tmp_path / "ledger.csv"
    report_path = tmp_path / "report.json"
    _write_json(batches, _batch_document())
    _write_json(profile, _profile_document())
    _write_pipeline(
        pipeline,
        (("id-a", "accepted", ""), ("id-b", "operational_error", "missing_crop")),
    )
    reviews.write_text(
        "photo_id,review_status,plate_verdict,note\nid-a,resolved,clean_filled,\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--batches",
            str(batches),
            "--profile",
            str(profile),
            "--pipeline",
            str(pipeline),
            "--reviews",
            str(reviews),
            "--ledger",
            str(ledger),
            "--report",
            str(report_path),
            "--post-profile-pool-size",
            "2",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["accepted_resolved"] == 1
    assert report["operational_error"] == 1
    assert report["bounds"][0]["sample_size"] == 1
    assert json.loads(batches.read_text(encoding="utf-8"))["batches"][0]["state"] == (
        "completed_fail"
    )


def test_cli_defaults_to_next_sealed_batch_after_prior_failure(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    reviews = tmp_path / "reviews.csv"
    document = _batch_document("completed_fail")
    document["batches"][0].update(
        {
            "profile_id": "profile-a",
            "profile_sha256": _sha("1"),
            "geometry_witness_sha256": _sha("3"),
            "open_result_sha256": _sha("4"),
            "review_result_sha256": _sha("5"),
            "completion_result_sha256": _sha("6"),
        }
    )
    _write_json(batches, document)
    _write_json(profile, _profile_document())
    _write_pipeline(pipeline, (("id-c", "accepted", ""), ("id-d", "accepted", "")))
    reviews.write_text(
        "photo_id,review_status,plate_verdict,note\n"
        "id-c,resolved,clean_filled,\n"
        "id-d,resolved,leak,\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--batches",
            str(batches),
            "--profile",
            str(profile),
            "--pipeline",
            str(pipeline),
            "--reviews",
            str(reviews),
            "--ledger",
            str(tmp_path / "ledger.csv"),
            "--report",
            str(tmp_path / "report.json"),
            "--post-profile-pool-size",
            "2",
        ]
    )

    persisted = json.loads(batches.read_text(encoding="utf-8"))["batches"]
    assert exit_code == 1
    assert persisted[0]["state"] == "completed_fail"
    assert persisted[1]["state"] == "completed_fail"


def test_cli_completed_failure_is_terminal_and_keeps_exact_report(tmp_path: Path) -> None:
    batches = tmp_path / "batches.json"
    profile = tmp_path / "profile.json"
    pipeline = tmp_path / "pipeline.csv"
    reviews = tmp_path / "reviews.csv"
    ledger = tmp_path / "ledger.csv"
    report_path = tmp_path / "report.json"
    _write_json(batches, _batch_document())
    _write_json(profile, _profile_document())
    _write_pipeline(pipeline, (("id-a", "accepted", ""), ("id-b", "accepted", "")))
    reviews.write_text(
        "photo_id,review_status,plate_verdict,note\n"
        "id-b,resolved,clean_no_plate,\n"
        "id-a,resolved,leak,visible plate\n",
        encoding="utf-8",
    )
    arguments = [
        "--batches",
        str(batches),
        "--batch-number",
        "1",
        "--profile",
        str(profile),
        "--pipeline",
        str(pipeline),
        "--reviews",
        str(reviews),
        "--ledger",
        str(ledger),
        "--report",
        str(report_path),
        "--post-profile-pool-size",
        "2511",
    ]

    assert main(arguments) == 1
    document = json.loads(batches.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert document["batches"][0]["state"] == "completed_fail"
    assert document["batches"][0]["completion_result_sha256"]
    assert report["accepted_resolved"] == 2
    assert report["bounds"][0]["sample_size"] == 2
    assert report["bounds"][0]["projected_upper_count"] == pytest.approx(
        clopper_pearson(1, 2)[1] * 2511
    )
    with pytest.raises(CalibrationStateError, match="недопустимый переход"):
        main(arguments)


def _resolved_arguments(tmp_path: Path, **overrides: Path) -> list[str]:
    paths = {
        "--batches": tmp_path / "batches.json",
        "--profile": tmp_path / "profile.json",
        "--pipeline": tmp_path / "pipeline.csv",
        "--reviews": tmp_path / "reviews.csv",
        "--ledger": tmp_path / "ledger.csv",
        "--report": tmp_path / "report.json",
    }
    paths.update(overrides)
    arguments: list[str] = []
    for flag, path in paths.items():
        arguments.extend((flag, str(path)))
    arguments.extend(("--post-profile-pool-size", "2"))
    return arguments


def _resolved_batch_fixture(tmp_path: Path) -> None:
    _write_json(tmp_path / "batches.json", _batch_document())
    _write_json(tmp_path / "profile.json", _profile_document())
    _write_pipeline(tmp_path / "pipeline.csv", (("id-a", "accepted", ""), ("id-b", "accepted", "")))
    (tmp_path / "reviews.csv").write_text(
        "photo_id,review_status,plate_verdict,note\n"
        "id-a,resolved,clean_filled,\n"
        "id-b,resolved,clean_no_plate,\n",
        encoding="utf-8",
    )


def test_ledger_carries_staging_columns_for_party_views(tmp_path: Path) -> None:
    _resolved_batch_fixture(tmp_path)

    # Партия фикстуры меньше обязательных 300 кадров, поэтому отчёт закономерно не
    # проходит; проверяется состав записанного ledger, а не вердикт.
    assert main(_resolved_arguments(tmp_path)) == 1

    with (tmp_path / "ledger.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert set(STAGING_FIELDS) <= set(rows[0])
    assert [row["crop_path"] for row in rows] == ["id-a.jpg", "id-b.jpg"]
    assert rows[0]["class_id"] == "class-a"
    assert rows[0]["plates"] == "1"


def test_cli_requires_pipeline_columns_needed_by_party_views(tmp_path: Path) -> None:
    _resolved_batch_fixture(tmp_path)
    (tmp_path / "pipeline.csv").write_text(
        "photo_id,pipeline_outcome,reject_reason\nid-a,accepted,\nid-b,accepted,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="crop_path"):
        main(_resolved_arguments(tmp_path))

    persisted = json.loads((tmp_path / "batches.json").read_text(encoding="utf-8"))
    assert persisted["batches"][0]["state"] == "sealed"


@pytest.mark.parametrize("output_flag", ["--ledger", "--report"])
def test_cli_refuses_output_path_that_would_overwrite_an_input(
    tmp_path: Path, output_flag: str
) -> None:
    _resolved_batch_fixture(tmp_path)
    reviews_before = (tmp_path / "reviews.csv").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match=output_flag):
        main(_resolved_arguments(tmp_path, **{output_flag: tmp_path / "reviews.csv"}))

    assert (tmp_path / "reviews.csv").read_text(encoding="utf-8") == reviews_before
    persisted = json.loads((tmp_path / "batches.json").read_text(encoding="utf-8"))
    assert persisted["batches"][0]["state"] == "sealed"


def test_cli_validates_pipeline_before_opening_sealed_batch(tmp_path: Path) -> None:
    _resolved_batch_fixture(tmp_path)
    _write_pipeline(tmp_path / "pipeline.csv", (("id-a", "accepted", ""), ("id-x", "accepted", "")))

    with pytest.raises(ValueError, match="чужие"):
        main(_resolved_arguments(tmp_path))

    persisted = json.loads((tmp_path / "batches.json").read_text(encoding="utf-8"))["batches"][0]
    assert persisted["state"] == "sealed"
    assert persisted.get("open_result_sha256") is None
    assert not (tmp_path / "ledger.csv").exists()


def test_report_witnesses_ledger_and_final_reviews(tmp_path: Path) -> None:
    _resolved_batch_fixture(tmp_path)

    assert main(_resolved_arguments(tmp_path)) == 1

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert (
        report["ledger_sha256"]
        == hashlib.sha256((tmp_path / "ledger.csv").read_bytes()).hexdigest()
    )
    assert (
        report["reviews_sha256"]
        == hashlib.sha256((tmp_path / "reviews.csv").read_bytes()).hexdigest()
    )


def test_written_ledger_composes_with_party_staging(tmp_path: Path) -> None:
    _resolved_batch_fixture(tmp_path)
    crops = tmp_path / "crops"
    crops.mkdir()
    for photo_id in ("id-a", "id-b"):
        Image.new("RGB", (400, 300), "red").save(crops / f"{photo_id}.jpg")
    views = tmp_path / "views"

    assert main(_resolved_arguments(tmp_path)) == 1
    assert (
        stage_party_views(
            [
                "--party-ledger",
                str(tmp_path / "ledger.csv"),
                "--party-number",
                "1",
                "--crops-root",
                str(crops),
                "--out-dir",
                str(views),
                "--font",
                "default",
            ]
        )
        == 0
    )

    with (views / "views.csv").open(encoding="utf-8", newline="") as stream:
        staged = tuple(csv.DictReader(stream))
    assert [row["view_status"] for row in staged] == ["image", "image"]
    assert [row["class_id"] for row in staged] == ["class-a", "class-a"]
    assert [row["plates"] for row in staged] == ["1", "1"]
