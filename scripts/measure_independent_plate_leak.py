"""Открытие, свод и завершение запечатанной независимой privacy-партии."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from pipeline.redaction_calibration.privacy import (
    IndependentReport,
    IndependentRow,
    PipelineOutcome,
    ReviewStatus,
    compute_independent_report,
    merge_independent_reviews,
    validate_independent_ledger,
)
from pipeline.redaction_calibration.state import (
    BatchRecord,
    BatchState,
    ProfileRecord,
    ProfileState,
    transition_batch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def _write_ledger(path: Path, rows: Sequence[IndependentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(IndependentRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _batch_record(document: Mapping[str, Any]) -> BatchRecord:
    return BatchRecord(
        number=int(document["number"]),
        state=BatchState(document["state"]),
        batch_sha256=str(document["sha256"]),
        profile_id=document.get("profile_id"),
        profile_sha256=document.get("profile_sha256"),
        geometry_witness_sha256=document.get("geometry_witness_sha256"),
        open_result_sha256=document.get("open_result_sha256"),
        review_result_sha256=document.get("review_result_sha256"),
        completion_result_sha256=document.get("completion_result_sha256"),
    )


def _profile_record(document: Mapping[str, Any]) -> ProfileRecord:
    return ProfileRecord(
        profile_id=str(document["profile_id"]),
        state=ProfileState(document["state"]),
        profile_sha256=str(document["profile_sha256"]),
        canonical_sha256=cast(str | None, document.get("canonical_sha256")),
        geometry_witness_sha256=cast(str | None, document.get("geometry_witness_sha256")),
        w_max=cast(int | None, document.get("w_max")),
        h_max=cast(int | None, document.get("h_max")),
    )


def _persist_record(document: dict[str, Any], record: BatchRecord) -> None:
    document.update(
        {
            "state": record.state.value,
            "profile_id": record.profile_id,
            "profile_sha256": record.profile_sha256,
            "geometry_witness_sha256": record.geometry_witness_sha256,
            "open_result_sha256": record.open_result_sha256,
            "review_result_sha256": record.review_result_sha256,
            "completion_result_sha256": record.completion_result_sha256,
        }
    )


def _initial_rows(pipeline_rows: Sequence[Mapping[str, str]]) -> tuple[IndependentRow, ...]:
    rows = []
    for item in pipeline_rows:
        outcome = cast(PipelineOutcome, item.get("pipeline_outcome", ""))
        reason = item.get("reject_reason", "")
        status: ReviewStatus = "pending" if outcome != "rejected" else "resolved"
        rows.append(IndependentRow(item.get("photo_id", ""), outcome, reason, status, "", ""))
    return tuple(rows)


def _report_document(report: IndependentReport) -> dict[str, Any]:
    return asdict(report)


def _resolve_target_batch(
    document: Mapping[str, Any], number: int | None
) -> tuple[dict[str, Any], list[BatchRecord]]:
    batches = cast(list[dict[str, Any]], document["batches"])
    records = [_batch_record(item) for item in batches]
    if number is None:
        active = [
            record
            for record in records
            if record.state in {BatchState.OPENED, BatchState.REVIEW_IN_PROGRESS}
        ]
        sealed = [record for record in records if record.state is BatchState.SEALED]
        terminal = [
            record
            for record in records
            if record.state in {BatchState.COMPLETED_PASS, BatchState.COMPLETED_FAIL}
        ]
        if active:
            number = active[0].number
        elif sealed:
            number = min(record.number for record in sealed)
        elif terminal:
            number = terminal[-1].number
        else:
            raise ValueError("нет sealed-партии для открытия")
    matching = [item for item in batches if int(item["number"]) == number]
    if len(matching) != 1:
        raise ValueError(f"партия {number} должна встречаться ровно один раз")
    return matching[0], records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Точный замер запечатанной privacy-партии")
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--batch-number", type=int, default=None)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--post-profile-pool-size", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Продвинуть партию через единый state-автомат и записать независимые артефакты."""
    arguments = build_parser().parse_args(argv)
    batch_document = _read_json(arguments.batches)
    profile_document = _read_json(arguments.profile)
    persisted, records = _resolve_target_batch(batch_document, arguments.batch_number)
    record = _batch_record(persisted)
    if record.state in {BatchState.COMPLETED_PASS, BatchState.COMPLETED_FAIL}:
        # Терминальность и текст ошибки принадлежат state.transition_batch, а не CLI.
        transition_batch(
            record,
            BatchState.REVIEW_IN_PROGRESS,
            review_result_sha256=_sha256(arguments.pipeline),
        )
    profile = _profile_record(profile_document)
    sealed_ids = tuple(str(photo_id) for photo_id in persisted["photo_ids"])

    if record.state is BatchState.SEALED:
        record = transition_batch(
            record,
            BatchState.OPENED,
            batches=records,
            profile=profile,
            current_canonical_sha256=cast(str, profile_document["canonical_sha256"]),
            current_geometry_witness_sha256=cast(str, profile_document["geometry_witness_sha256"]),
            current_w_max=cast(int, profile_document["w_max"]),
            current_h_max=cast(int, profile_document["h_max"]),
            open_result_sha256=_sha256(arguments.pipeline),
        )
        _persist_record(persisted, record)
        _write_json(arguments.batches, batch_document)

    if record.state is BatchState.OPENED:
        witness = (
            _sha256(arguments.reviews)
            if arguments.reviews is not None
            else _sha256(arguments.pipeline)
        )
        record = transition_batch(
            record,
            BatchState.REVIEW_IN_PROGRESS,
            review_result_sha256=witness,
        )
        _persist_record(persisted, record)
        _write_json(arguments.batches, batch_document)

    pipeline_rows = _read_csv(arguments.pipeline)
    rows = _initial_rows(pipeline_rows)
    if arguments.reviews is not None:
        rows = merge_independent_reviews(rows, _read_csv(arguments.reviews))
    validate_independent_ledger(rows, sealed_ids)
    _write_ledger(arguments.ledger, rows)

    if any(row.pipeline_outcome == "accepted" and row.review_status == "pending" for row in rows):
        return 2

    report = compute_independent_report(rows, arguments.post_profile_pool_size)
    _write_json(arguments.report, _report_document(report))
    target = BatchState.COMPLETED_PASS if report.passed else BatchState.COMPLETED_FAIL
    record = transition_batch(
        record,
        target,
        completion_result_sha256=_sha256(arguments.report),
    )
    _persist_record(persisted, record)
    _write_json(arguments.batches, batch_document)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
