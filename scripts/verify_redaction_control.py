"""Проверка Step-0: текущий профиль обязан точно воспроизвести принятые данные."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from pipeline.harvest.manifest import COLUMNS, HarvestRow, load_manifest
from pipeline.redaction_calibration.measure import load_frame_profile_results

PHOTO_ID = re.compile(r"[0-9a-f]{64}")
REJECT_FIELDS = ("file", "class_id", "source", "source_item_id", "sha256", "reason")
REQUIRED_ARTIFACTS = (
    "manifest.csv",
    "pixel-diff.csv",
    "profile.json",
    "quality-metrics.json",
    "rejects.csv",
    "report.json",
)
PARITY_ARTIFACTS = (
    "manifest.csv",
    "pixel-diff.csv",
    "quality-metrics.json",
    "rejects.csv",
)
PROFILE_VOLATILE_FIELDS = frozenset({"timestamp"})
DIAGNOSIS_ORDER = (
    "JPEG parameters/version",
    "detector pins",
    "code/profile hash",
    "device/runtime",
)


@dataclass(frozen=True, slots=True)
class RejectRow:
    """Строка rejects.csv после строгого разбора границы."""

    file: str
    class_id: str
    source: str
    source_item_id: str
    sha256: str
    reason: str


@dataclass(frozen=True, slots=True)
class WorkspaceData:
    """Прочитанные артефакты одного workspace с вычисленными хешами."""

    root: Path
    accepted: tuple[HarvestRow, ...]
    rejected: tuple[RejectRow, ...]
    profile_rows: tuple[Any, ...]
    profile: Mapping[str, Any]
    report: Mapping[str, Any]
    hashes: Mapping[str, str]
    trace_enabled: bool


def _mismatch(kind: str, message: str, **details: object) -> dict[str, object]:
    return {"kind": kind, "message": message, **details}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"не удалось прочитать {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"корень {path.name} должен быть объектом JSON")
    return cast(Mapping[str, Any], value)


def _manifest(path: Path) -> tuple[HarvestRow, ...]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        actual = tuple(next(reader, ()))
    if actual != COLUMNS:
        raise ValueError(f"колонки {path.name} не совпали: {actual!r} != {COLUMNS!r}")
    return tuple(load_manifest(path))


def _rejects(path: Path) -> tuple[RejectRow, ...]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != REJECT_FIELDS:
            raise ValueError(
                f"колонки rejects.csv не совпали: {reader.fieldnames!r} != {REJECT_FIELDS!r}"
            )
        rows = []
        for number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"rejects.csv, строка {number}: неполная CSV-строка")
            rows.append(RejectRow(**cast(dict[str, str], row)))
    return tuple(rows)


def _duplicates(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _validate_ids(
    label: str,
    values: Sequence[str],
    mismatches: list[dict[str, object]],
) -> None:
    for value in sorted(set(values)):
        if PHOTO_ID.fullmatch(value) is None:
            mismatches.append(
                _mismatch("malformed_photo_id", f"{label}: некорректный photo_id {value!r}")
            )
    for value in _duplicates(values):
        mismatches.append(_mismatch("duplicate_photo_id", f"{label}: повтор photo_id {value}"))


def _load_workspace(root: Path) -> WorkspaceData:
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "; ".join(f"отсутствует обязательный файл {name}" for name in missing)
        )
    accepted = _manifest(root / "manifest.csv")
    rejected = _rejects(root / "rejects.csv")
    profile_rows = tuple(load_frame_profile_results(root / "pixel-diff.csv"))
    profile = _json(root / "profile.json")
    report = _json(root / "report.json")
    trace_enabled = (root / "observations.csv").is_file()
    names = list(PARITY_ARTIFACTS)
    if trace_enabled:
        names.append("observations.csv")
    hashes = {name: _sha256(root / name) for name in sorted(names)}
    return WorkspaceData(
        root=root,
        accepted=accepted,
        rejected=rejected,
        profile_rows=profile_rows,
        profile=profile,
        report=report,
        hashes=hashes,
        trace_enabled=trace_enabled,
    )


def _read_workspace(
    root: Path,
    label: str,
    mismatches: list[dict[str, object]],
) -> WorkspaceData | None:
    try:
        return _load_workspace(root)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        mismatches.append(_mismatch("unreadable", f"{label}: {error}"))
        return None


def _reason_map(workspace: WorkspaceData) -> dict[str, str]:
    return {row.sha256: row.reason for row in workspace.rejected}


def _row_fields_changed(old: HarvestRow, new: HarvestRow) -> list[str]:
    return [
        field.name
        for field in fields(HarvestRow)
        if getattr(old, field.name) != getattr(new, field.name)
    ]


def _decode(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"пустой JPEG {path}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"JPEG не декодируется {path}")
    return cast(np.ndarray, image)


def _crop_record(
    input_root: Path,
    workspace: WorkspaceData,
    source: HarvestRow,
    rebuilt: HarvestRow,
    mismatches: list[dict[str, object]],
) -> dict[str, object] | None:
    old_path = input_root / "crops" / source.class_id / source.file
    new_path = workspace.root / "crops" / rebuilt.class_id / rebuilt.file
    missing = [str(path) for path in (old_path, new_path) if not path.is_file()]
    if missing:
        for path in missing:
            mismatches.append(_mismatch("missing_crop", f"отсутствует кроп {path}"))
        return None
    try:
        old_bytes = old_path.read_bytes()
        new_bytes = new_path.read_bytes()
        pixels_equal = bool(np.array_equal(_decode(old_path), _decode(new_path)))
    except (OSError, ValueError, cv2.error) as error:
        mismatches.append(_mismatch("unreadable_crop", str(error)))
        return None
    if not pixels_equal:
        mismatches.append(
            _mismatch(
                "crop_pixels",
                f"{source.sha256}: пиксели кропа не совпали",
                photo_id=source.sha256,
            )
        )
    return {
        "photo_id": source.sha256,
        "pixels_equal": pixels_equal,
        "byte_equal": old_bytes == new_bytes,
        "old_sha256": hashlib.sha256(old_bytes).hexdigest(),
        "new_sha256": hashlib.sha256(new_bytes).hexdigest(),
    }


def _extra_crops(workspace: WorkspaceData, expected: set[Path]) -> list[str]:
    crop_root = workspace.root / "crops"
    if not crop_root.exists():
        return []
    return sorted(
        path.relative_to(workspace.root).as_posix()
        for path in crop_root.rglob("*")
        if path.is_file() and path not in expected
    )


def _validate_report(
    workspace: WorkspaceData,
    input_count: int,
    mismatches: list[dict[str, object]],
) -> None:
    counts = workspace.report.get("counts")
    expected = {
        "accepted": len(workspace.accepted),
        "input": input_count,
        "rejected": len(workspace.rejected),
    }
    if counts != expected or expected["accepted"] + expected["rejected"] != input_count:
        mismatches.append(
            _mismatch(
                "arithmetic",
                f"арифметика workspace не сошлась: report={counts!r}, фактически={expected!r}",
            )
        )
    reasons = dict(sorted(Counter(row.reason for row in workspace.rejected).items()))
    if workspace.report.get("reject_reasons") != reasons:
        mismatches.append(
            _mismatch(
                "report_reasons",
                "причины workspace в report.json не совпали с rejects.csv",
            )
        )
    profile_ids = [row.photo_id for row in workspace.profile_rows]
    _validate_ids("pixel-diff.csv", profile_ids, mismatches)
    if sorted(profile_ids) != sorted(
        [row.sha256 for row in workspace.accepted] + [row.sha256 for row in workspace.rejected]
    ):
        mismatches.append(
            _mismatch("profile_membership", "pixel-diff.csv не покрывает все решения workspace")
        )


def _validate_workspace(
    input_rows: Sequence[HarvestRow],
    input_root: Path,
    workspace: WorkspaceData,
    label: str,
    mismatches: list[dict[str, object]],
) -> list[dict[str, object]]:
    input_ids = [row.sha256 for row in input_rows]
    accepted_ids = [row.sha256 for row in workspace.accepted]
    rejected_ids = [row.sha256 for row in workspace.rejected]
    _validate_ids(f"{label} manifest.csv", accepted_ids, mismatches)
    _validate_ids(f"{label} rejects.csv", rejected_ids, mismatches)
    input_set = set(input_ids)
    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)
    foreign = sorted((accepted_set | rejected_set) - input_set)
    missing = sorted(input_set - (accepted_set | rejected_set))
    overlap = sorted(accepted_set & rejected_set)
    if foreign:
        mismatches.append(_mismatch("foreign_ids", f"{label}: посторонние photo_id: {foreign}"))
    if missing:
        mismatches.append(_mismatch("missing_ids", f"{label}: пропавшие photo_id: {missing}"))
    if overlap:
        mismatches.append(
            _mismatch("overlap", f"{label}: пересечение accepted/rejected: {overlap}")
        )
    _validate_report(workspace, len(input_rows), mismatches)
    input_by_id = {row.sha256: row for row in input_rows}
    crop_records: list[dict[str, object]] = []
    expected_crops: set[Path] = set()
    for rebuilt in sorted(workspace.accepted, key=lambda row: row.sha256):
        source = input_by_id.get(rebuilt.sha256)
        if source is None:
            continue
        changed = _row_fields_changed(source, rebuilt)
        if changed:
            mismatches.append(
                _mismatch(
                    "manifest_row",
                    f"{label}: семантика строки {source.sha256} изменилась: {changed}",
                    photo_id=source.sha256,
                    fields=changed,
                )
            )
        expected_crops.add(workspace.root / "crops" / rebuilt.class_id / rebuilt.file)
        record = _crop_record(input_root, workspace, source, rebuilt, mismatches)
        if record is not None:
            crop_records.append(record)
    for path in _extra_crops(workspace, expected_crops):
        mismatches.append(_mismatch("extra_crop", f"{label}: лишний файл кропа {path}"))
    return crop_records


def _profile_witness(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in sorted(profile.items()) if key not in PROFILE_VOLATILE_FIELDS
    }


def _compare_workspaces(
    plain: WorkspaceData,
    traced: WorkspaceData,
    mismatches: list[dict[str, object]],
) -> None:
    if plain.trace_enabled:
        mismatches.append(_mismatch("trace_mode", "plain workspace содержит observations.csv"))
    if not traced.trace_enabled:
        mismatches.append(_mismatch("trace_mode", "traced workspace не содержит observations.csv"))
    plain_reasons = _reason_map(plain)
    traced_reasons = _reason_map(traced)
    if plain_reasons != traced_reasons:
        mismatches.append(
            _mismatch(
                "trace_reasons",
                "причины отказа plain/traced не совпали",
                plain=plain_reasons,
                traced=traced_reasons,
            )
        )
    if _profile_witness(plain.profile) != _profile_witness(traced.profile):
        mismatches.append(
            _mismatch(
                "trace_profile",
                "trace off/on: профиль и provenance не совпали",
                plain=_profile_witness(plain.profile),
                traced=_profile_witness(traced.profile),
            )
        )
    for name in PARITY_ARTIFACTS:
        plain_hash = plain.hashes.get(name)
        traced_hash = traced.hashes.get(name)
        if plain_hash != traced_hash:
            mismatches.append(
                _mismatch(
                    "trace_hash",
                    f"trace off/on: хеш {name} не совпал",
                    artifact=name,
                    plain=plain_hash,
                    traced=traced_hash,
                )
            )
    plain_crops = {
        path.relative_to(plain.root / "crops").as_posix(): _sha256(path)
        for path in (plain.root / "crops").rglob("*")
        if path.is_file()
    }
    traced_crops = {
        path.relative_to(traced.root / "crops").as_posix(): _sha256(path)
        for path in (traced.root / "crops").rglob("*")
        if path.is_file()
    }
    if plain_crops != traced_crops:
        mismatches.append(
            _mismatch(
                "trace_crops",
                "trace off/on: JPEG-буферы кропов не совпали",
                plain=plain_crops,
                traced=traced_crops,
            )
        )


def _trace_record(workspace: WorkspaceData | None) -> dict[str, object] | None:
    if workspace is None:
        return None
    return {"enabled": workspace.trace_enabled, "hashes": dict(workspace.hashes)}


def verify_control(
    manifest: Path,
    data_root: Path,
    plain_workspace: Path,
    traced_workspace: Path | None = None,
) -> dict[str, object]:
    """Проверить один control workspace и, при наличии, паритет tracing off/on."""
    mismatches: list[dict[str, object]] = []
    try:
        input_rows = load_manifest(manifest)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        input_rows = []
        mismatches.append(_mismatch("unreadable", f"входной манифест: {error}"))
    input_ids = [row.sha256 for row in input_rows]
    _validate_ids("входной манифест", input_ids, mismatches)
    plain = _read_workspace(plain_workspace, "plain workspace", mismatches)
    traced = (
        _read_workspace(traced_workspace, "traced workspace", mismatches)
        if traced_workspace is not None
        else None
    )
    crops: list[dict[str, object]] = []
    if plain is not None:
        crops = _validate_workspace(input_rows, data_root, plain, "plain", mismatches)
        if traced_workspace is None and plain.trace_enabled:
            mismatches.append(
                _mismatch("trace_mode", "одиночный plain workspace содержит observations.csv")
            )
    if traced is not None:
        _validate_workspace(input_rows, data_root, traced, "traced", mismatches)
    if plain is not None and traced is not None:
        _compare_workspaces(plain, traced, mismatches)
    accepted_ids = sorted(row.sha256 for row in plain.accepted) if plain is not None else []
    rejected_ids = sorted(row.sha256 for row in plain.rejected) if plain is not None else []
    reasons = (
        dict(sorted(Counter(row.reason for row in plain.rejected).items()))
        if plain is not None
        else {}
    )
    return {
        "passed": not mismatches,
        "counts": {
            "accepted": len(accepted_ids),
            "input": len(input_rows),
            "rejected": len(rejected_ids),
        },
        "membership": {
            "accepted_ids": accepted_ids,
            "input_ids": sorted(input_ids),
            "rejected_ids": rejected_ids,
        },
        "rejection_reasons": reasons,
        "crops": crops,
        "trace": {"plain": _trace_record(plain), "traced": _trace_record(traced)},
        "diagnosis_order": list(DIAGNOSIS_ORDER),
        "mismatches": mismatches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Проверка Step-0 control workspace")
    parser.add_argument("--manifest", type=Path, required=True, help="входной accepted-манифест")
    parser.add_argument("--data-root", type=Path, required=True, help="корень старых crops/")
    parser.add_argument("--plain-workspace", type=Path, required=True)
    parser.add_argument("--traced-workspace", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="путь отчёта; по умолчанию plain-workspace/control-verification.json",
    )
    return parser


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_control(
        args.manifest.resolve(),
        args.data_root.resolve(),
        args.plain_workspace.resolve(),
        args.traced_workspace.resolve() if args.traced_workspace is not None else None,
    )
    target = (
        args.report.resolve()
        if args.report is not None
        else args.plain_workspace.resolve() / "control-verification.json"
    )
    _write_json(target, report)
    if report["passed"]:
        print(f"Step-0 control подтверждён; отчёт: {target}")
        return 0
    mismatches = cast(list[object], report["mismatches"])
    print(f"Step-0 control не подтверждён; несоответствий: {len(mismatches)}")
    print(f"Отчёт: {target}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
