# tests/test_verify_redaction_control.py
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from pipeline.harvest.manifest import COLUMNS, HarvestRow, write_manifest
from pipeline.redaction_calibration.measure import write_frame_profile_results
from pipeline.redaction_calibration.model import FrameProfileResult
from scripts.verify_redaction_control import main, verify_control

REJECT_COLUMNS = ("file", "class_id", "source", "source_item_id", "sha256", "reason")


def _row(photo_id: str, *, class_id: str = "lada__granta") -> HarvestRow:
    return HarvestRow(
        file=f"{photo_id}.jpg",
        class_id=class_id,
        source="commons",
        source_item_id=photo_id[:8],
        page_url=f"https://example.test/page/{photo_id}",
        file_url=f"https://example.test/file/{photo_id}.jpg",
        author="Автор",
        license="CC0",
        license_tier=1,
        checked="2026-09-07",
        basis="category:Lada Granta",
        sha256=photo_id,
        phash=123,
        clip_rank=1,
        clip_margin=0.25,
        plates=1,
        faces=0,
        bbox_x=10.0,
        bbox_y=20.0,
        bbox_w=300.0,
        bbox_h=200.0,
        bbox_area_frac=0.5,
        det_conf=0.9,
        audit="ok",
    )


def _write_jpeg(path: Path, image: np.ndarray, *, quality: int = 91) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    path.write_bytes(encoded.tobytes())


def _write_rejects(path: Path, rows: list[tuple[HarvestRow, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(REJECT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row, reason in rows:
            writer.writerow(
                {
                    "file": row.file,
                    "class_id": row.class_id,
                    "source": row.source,
                    "source_item_id": row.source_item_id,
                    "sha256": row.sha256,
                    "reason": reason,
                }
            )


def _write_profile_result(
    path: Path,
    accepted: list[HarvestRow],
    rejected: list[tuple[HarvestRow, str]],
) -> None:
    results = [
        FrameProfileResult(
            photo_id=row.sha256,
            class_id=row.class_id,
            outcome="accepted",
            reject_reason=None,
            old_black_frac=0.0,
            new_black_frac=0.0,
            delta_black=0.0,
            pixels_equal=True,
            audit_preserved=True,
        )
        for row in accepted
    ]
    results.extend(
        FrameProfileResult(
            photo_id=row.sha256,
            class_id=row.class_id,
            outcome="rejected",
            reject_reason=reason,
            old_black_frac=0.0,
            new_black_frac=None,
            delta_black=None,
            pixels_equal=None,
            audit_preserved=False,
        )
        for row, reason in rejected
    )
    write_frame_profile_results(path, results)


def _write_profile(path: Path, *, traced: bool = False) -> None:
    profile = {
        "detector_pins": {"plate": {"sha256": "1" * 64}},
        "git_sha": "2" * 40,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "profile": {"plate_conf": 0.7},
        "profile_id": "3" * 64,
        "stage_sha256": "4" * 64,
        "timestamp": "2026-09-07T12:00:00Z" if not traced else "2026-09-07T13:00:00Z",
    }
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    accepted: list[HarvestRow],
    rejected: list[tuple[HarvestRow, str]],
    *,
    traced: bool,
) -> None:
    reasons: dict[str, int] = {}
    for _, reason in rejected:
        reasons[reason] = reasons.get(reason, 0) + 1
    report = {
        "counts": {
            "accepted": len(accepted),
            "input": len(accepted) + len(rejected),
            "rejected": len(rejected),
        },
        "output_hashes": {
            "manifest.csv": "a" * 64,
            "pixel-diff.csv": "b" * 64,
            "profile.json": "c" * 64,
            "quality-metrics.json": "d" * 64,
            "rejects.csv": "e" * 64,
            **({"observations.csv": "f" * 64} if traced else {}),
        },
        "reject_reasons": reasons,
    }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _workspace(
    root: Path,
    input_rows: list[HarvestRow],
    accepted: list[HarvestRow],
    rejected: list[tuple[HarvestRow, str]],
    *,
    traced: bool = False,
) -> Path:
    root.mkdir()
    write_manifest(root / "manifest.csv", accepted)
    _write_rejects(root / "rejects.csv", rejected)
    _write_profile_result(root / "pixel-diff.csv", accepted, rejected)
    _write_profile(root / "profile.json", traced=traced)
    (root / "quality-metrics.json").write_text("{}\n", encoding="utf-8")
    if traced:
        (root / "observations.csv").write_text("photo_id\n", encoding="utf-8")
    _write_report(root / "report.json", accepted, rejected, traced=traced)
    for row in accepted:
        source = next(item for item in input_rows if item.sha256 == row.sha256)
        old = root.parent / "input" / "crops" / source.class_id / source.file
        new = root / "crops" / row.class_id / row.file
        image = np.full((12, 16, 3), int(row.sha256[0], 16) * 8, dtype=np.uint8)
        if not old.exists():
            _write_jpeg(old, image)
        _write_jpeg(new, image)
    return root


def _fixture(tmp_path: Path, *, traced: bool = False) -> tuple[Path, Path, list[HarvestRow]]:
    accepted = _row("a" * 64)
    rejected = _row("b" * 64, class_id="gaz__gazelle_next")
    rows = [accepted, rejected]
    manifest = tmp_path / "input.csv"
    write_manifest(manifest, rows)
    workspace = _workspace(
        tmp_path / ("traced" if traced else "plain"),
        rows,
        [accepted],
        [(rejected, "redaction_failed")],
        traced=traced,
    )
    return manifest, workspace, rows


def _mismatches(report: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], report["mismatches"])


def _messages(report: dict[str, object]) -> list[str]:
    return [cast(str, mismatch["message"]) for mismatch in _mismatches(report)]


def _trace(report: dict[str, object]) -> dict[str, dict[str, object] | None]:
    return cast(dict[str, dict[str, object] | None], report["trace"])


def test_verify_control_exact_single_workspace_passes(tmp_path: Path) -> None:
    manifest, workspace, _ = _fixture(tmp_path)

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is True
    assert report["counts"] == {"accepted": 1, "input": 2, "rejected": 1}
    assert report["membership"] == {
        "accepted_ids": ["a" * 64],
        "input_ids": ["a" * 64, "b" * 64],
        "rejected_ids": ["b" * 64],
    }
    assert report["rejection_reasons"] == {"redaction_failed": 1}
    assert report["crops"] == [
        {
            "byte_equal": True,
            "new_sha256": hashlib.sha256(
                (workspace / "crops" / "lada__granta" / f"{'a' * 64}.jpg").read_bytes()
            ).hexdigest(),
            "old_sha256": hashlib.sha256(
                (tmp_path / "input" / "crops" / "lada__granta" / f"{'a' * 64}.jpg").read_bytes()
            ).hexdigest(),
            "photo_id": "a" * 64,
            "pixels_equal": True,
        }
    ]
    trace = _trace(report)
    plain_trace = trace["plain"]
    assert plain_trace is not None
    assert trace == {
        "plain": {"enabled": False, "hashes": plain_trace["hashes"]},
        "traced": None,
    }
    assert "observations.csv" not in cast(dict[str, str], plain_trace["hashes"])


def test_verify_control_reports_changed_pixel_and_extra_output(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    accepted = rows[0]
    changed = np.full((12, 16, 3), 240, dtype=np.uint8)
    _write_jpeg(workspace / "crops" / accepted.class_id / accepted.file, changed)
    (workspace / "crops" / accepted.class_id / "extra.jpg").write_bytes(b"extra")

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is False
    messages = _messages(report)
    assert any("пиксели кропа не совпали" in message for message in messages)
    assert any("лишний файл кропа" in message for message in messages)


def test_verify_control_reports_missing_required_output(tmp_path: Path) -> None:
    manifest, workspace, _ = _fixture(tmp_path)
    (workspace / "pixel-diff.csv").unlink()

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is False
    assert any(
        "отсутствует обязательный файл pixel-diff.csv" in message for message in _messages(report)
    )


def test_verify_control_reports_membership_and_reason_mismatches_together(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    accepted, rejected = rows
    foreign = _row("c" * 64)
    write_manifest(workspace / "manifest.csv", [accepted, foreign])
    _write_rejects(workspace / "rejects.csv", [(rejected, "duplicate"), (accepted, "conflict")])

    report = verify_control(manifest, tmp_path / "input", workspace)

    messages = _messages(report)
    assert report["passed"] is False
    assert any("посторонние photo_id" in message and "c" * 64 in message for message in messages)
    assert any(
        "пересечение accepted/rejected" in message and "a" * 64 in message for message in messages
    )
    assert any("причины workspace" in message for message in messages)
    assert any("арифметика workspace не сошлась" in message for message in messages)


def test_verify_control_accepts_regenerated_checked_date(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    regenerated = replace(rows[0], checked="2026-09-08")
    write_manifest(workspace / "manifest.csv", [regenerated])

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is True
    assert not any(mismatch["kind"] == "manifest_row" for mismatch in _mismatches(report))


def test_verify_control_reports_binding_manifest_semantic_change(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    changed = replace(rows[0], plates=2)
    write_manifest(workspace / "manifest.csv", [changed])

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is False
    assert any(
        mismatch["kind"] == "manifest_row" and mismatch["fields"] == ["plates"]
        for mismatch in _mismatches(report)
    )


def test_verify_control_rejects_duplicate_and_malformed_ids(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    duplicate = replace(rows[0], file="duplicate.jpg")
    write_manifest(workspace / "manifest.csv", [rows[0], duplicate])
    _write_rejects(workspace / "rejects.csv", [(replace(rows[1], sha256="not-a-sha"), "duplicate")])

    report = verify_control(manifest, tmp_path / "input", workspace)

    messages = _messages(report)
    assert report["passed"] is False
    assert any("повтор photo_id" in message and "a" * 64 in message for message in messages)
    assert any(
        "некорректный photo_id" in message and "not-a-sha" in message for message in messages
    )


def test_verify_control_keeps_checked_in_trace_manifest_parity(tmp_path: Path) -> None:
    manifest, plain, rows = _fixture(tmp_path)
    traced_row = replace(rows[0], checked="2026-09-08")
    traced = _workspace(
        tmp_path / "traced",
        rows,
        [traced_row],
        [(rows[1], "redaction_failed")],
        traced=True,
    )

    report = verify_control(manifest, tmp_path / "input", plain, traced)

    assert report["passed"] is False
    assert any(
        mismatch["kind"] == "trace_hash" and mismatch.get("artifact") == "manifest.csv"
        for mismatch in _mismatches(report)
    )


def test_verify_control_compares_plain_and_traced_artifacts_and_reasons(tmp_path: Path) -> None:
    manifest, plain, rows = _fixture(tmp_path)
    traced = _workspace(
        tmp_path / "traced",
        rows,
        [rows[0]],
        [(rows[1], "duplicate")],
        traced=True,
    )
    (traced / "quality-metrics.json").write_text('{"changed": true}\n', encoding="utf-8")

    report = verify_control(manifest, tmp_path / "input", plain, traced)

    assert report["passed"] is False
    messages = _messages(report)
    assert any("причины отказа plain/traced не совпали" in message for message in messages)
    assert any(
        "quality-metrics.json" in message and "trace off/on" in message for message in messages
    )
    assert not any("observations.csv" in message for message in messages)
    assert not any("report.json" in message for message in messages)
    trace = _trace(report)
    plain_trace = trace["plain"]
    traced_trace = trace["traced"]
    assert plain_trace is not None and plain_trace["enabled"] is False
    assert traced_trace is not None and traced_trace["enabled"] is True


def test_verify_control_compares_profile_provenance_semantically(tmp_path: Path) -> None:
    manifest, plain, rows = _fixture(tmp_path)
    traced = _workspace(
        tmp_path / "traced",
        rows,
        [rows[0]],
        [(rows[1], "redaction_failed")],
        traced=True,
    )

    report = verify_control(manifest, tmp_path / "input", plain, traced)

    assert report["passed"] is True
    assert not any(
        mismatch["kind"] == "trace_hash" and mismatch.get("artifact") == "profile.json"
        for mismatch in _mismatches(report)
    )


def test_verify_control_diagnosis_has_required_order(tmp_path: Path) -> None:
    manifest, workspace, _ = _fixture(tmp_path)
    (workspace / "quality-metrics.json").write_text('{"changed": true}\n', encoding="utf-8")

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["diagnosis_order"] == [
        "JPEG parameters/version",
        "detector pins",
        "code/profile hash",
        "device/runtime",
    ]
    assert "nondeterminism" not in json.dumps(report).lower()


def test_main_writes_default_report_and_returns_nonzero_on_mismatch(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    (workspace / "crops" / rows[0].class_id / rows[0].file).unlink()

    code = main(
        [
            "--manifest",
            str(manifest),
            "--data-root",
            str(tmp_path / "input"),
            "--plain-workspace",
            str(workspace),
        ]
    )

    report_path = workspace / "control-verification.json"
    assert code == 1
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False


def test_verify_control_reports_fatal_malformed_json(tmp_path: Path) -> None:
    manifest, workspace, _ = _fixture(tmp_path)
    (workspace / "report.json").write_text("{broken", encoding="utf-8")

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is False
    assert any(
        mismatch["kind"] == "unreadable" and "report.json" in cast(str, mismatch["message"])
        for mismatch in _mismatches(report)
    )


def test_verify_control_reports_malformed_manifest_schema(tmp_path: Path) -> None:
    manifest, workspace, _ = _fixture(tmp_path)
    (workspace / "manifest.csv").write_text(
        "file,class_id\nfoo.jpg,lada__granta\n", encoding="utf-8"
    )

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is False
    assert any(
        mismatch["kind"] == "unreadable"
        and "колонки manifest.csv не совпали" in cast(str, mismatch["message"])
        for mismatch in _mismatches(report)
    )


def test_verify_control_rejects_duplicate_input_ids(tmp_path: Path) -> None:
    manifest, workspace, rows = _fixture(tmp_path)
    duplicate = replace(rows[0], file="other.jpg")
    with manifest.open("a", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=list(COLUMNS)).writerow(asdict(duplicate))

    report = verify_control(manifest, tmp_path / "input", workspace)

    assert report["passed"] is False
    assert any("входной манифест: повтор photo_id" in message for message in _messages(report))
