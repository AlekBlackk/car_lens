"""Артефакты Pareto-оценки профилей и трассируемого visual veto."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

from pipeline.redaction_calibration.geometry import DetectorLink, GeometryWitness
from pipeline.redaction_calibration.measure import (
    first_failed_gate,
    rank_profiles,
    stale_profile_ids,
)
from pipeline.redaction_calibration.model import (
    ConfidenceMargin,
    FrontierEntry,
    ProfileGateReport,
    ProfileScore,
    VisualVeto,
)
from scripts.stage_plate_leak_views import render_view

VETO_COLUMNS = tuple(field.name for field in fields(VisualVeto))
UNRESOLVED_COLUMNS = tuple(field.name for field in fields(DetectorLink))
VIEW_COLUMNS = ("profile_id", "frame_id", "sha256", "file")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COSMETIC_REASONS = frozenset(("cosmetic", "косметика", "только косметика"))


@dataclass(frozen=True, slots=True)
class ChangedCrop:
    """Изменившийся калибровочный кроп, который обязан попасть в visual review."""

    profile_id: str
    frame_id: str
    sha256: str
    source: Path


def _stable_component(value: str, name: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"{name} не является стабильным ID: {value!r}")
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stage_changed_crops(
    crops: Sequence[ChangedCrop],
    out_dir: Path,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    expected_changed_ids: Mapping[str, set[str]] | None = None,
) -> list[dict[str, str]]:
    """Записать каждый изменившийся кроп с profile/frame ID внутри пикселей."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(crops, key=lambda row: (row.profile_id, row.frame_id))
    identities = [(row.profile_id, row.frame_id) for row in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("дубликат profile_id/frame_id среди changed crops")
    if expected_changed_ids is not None:
        actual: dict[str, set[str]] = {}
        for profile_id, frame_id in identities:
            actual.setdefault(profile_id, set()).add(frame_id)
        if actual != dict(expected_changed_ids):
            raise ValueError("staging не покрывает все changed crops calibration 300")
    records: list[dict[str, str]] = []
    for crop in ordered:
        profile_id = _stable_component(crop.profile_id, "profile_id")
        frame_id = _stable_component(crop.frame_id, "frame_id")
        if _SHA256.fullmatch(crop.sha256) is None:
            raise ValueError(f"неверный sha256 changed crop: {crop.sha256!r}")
        if not crop.source.is_file():
            raise FileNotFoundError(crop.source)
        filename = f"{profile_id}__{frame_id}.png"
        image = Image.open(crop.source).convert("RGB")
        caption = f"profile={profile_id} frame={frame_id} sha={crop.sha256[:12]}"
        render_view(image, caption, long_side=None, font=font).save(out_dir / filename)
        records.append(
            {
                "profile_id": profile_id,
                "frame_id": frame_id,
                "sha256": crop.sha256,
                "file": filename,
            }
        )
    _write_csv(out_dir / "views.csv", VIEW_COLUMNS, records)
    return records


def _validate_veto(veto: VisualVeto) -> None:
    _stable_component(veto.profile_id, "profile_id")
    if veto.sha256 is None and veto.plate_id is None:
        raise ValueError("veto требует sha256 или plate_id")
    if veto.sha256 is not None and _SHA256.fullmatch(veto.sha256) is None:
        raise ValueError("veto содержит неверный sha256")
    if veto.plate_id is not None:
        _stable_component(veto.plate_id, "plate_id")
    if not veto.destroyed_feature.strip():
        raise ValueError("veto требует destroyed_feature")
    reason = veto.reason.strip().casefold()
    if len(reason) < 12 or reason in _COSMETIC_REASONS:
        raise ValueError("veto требует некосметическую причину дисквалификации")


def write_visual_vetoes(path: Path, vetoes: Sequence[VisualVeto]) -> None:
    """Проверить и записать stable-order veto ledger со строгой схемой."""
    for veto in vetoes:
        _validate_veto(veto)
    ordered = sorted(vetoes, key=lambda row: (row.profile_id, row.sha256 or "", row.plate_id or ""))
    _write_csv(path, VETO_COLUMNS, [asdict(veto) for veto in ordered])


def load_visual_vetoes(path: Path) -> list[VisualVeto]:
    """Прочитать veto ledger, отвергая неизвестные колонки и слабые причины."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != VETO_COLUMNS:
            raise ValueError(f"неверные колонки vetoes.csv: {reader.fieldnames}")
        vetoes = [
            VisualVeto(
                profile_id=row["profile_id"],
                sha256=row["sha256"] or None,
                plate_id=row["plate_id"] or None,
                destroyed_feature=row["destroyed_feature"],
                reason=row["reason"],
            )
            for row in reader
        ]
    for veto in vetoes:
        _validate_veto(veto)
    if vetoes != sorted(
        vetoes, key=lambda row: (row.profile_id, row.sha256 or "", row.plate_id or "")
    ):
        raise ValueError("vetoes.csv должен быть отсортирован")
    return vetoes


def apply_visual_vetoes(
    frontier: Sequence[FrontierEntry], vetoes: Sequence[VisualVeto]
) -> tuple[FrontierEntry, ...]:
    """Исключить vetoed profiles, не разрешая ledger добавить профиль вне frontier."""
    known = {entry.profile_id for entry in frontier}
    vetoed = {veto.profile_id for veto in vetoes}
    foreign = sorted(vetoed - known)
    if foreign:
        raise ValueError(f"veto ссылается на профиль вне frontier: {foreign}")
    return tuple(entry for entry in frontier if entry.profile_id not in vetoed)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _frontier_document(entry: FrontierEntry) -> dict[str, Any]:
    """Оси профиля вместе с явным правом на продвижение и причиной его отсутствия."""
    return {
        **asdict(entry.score),
        "promotable": entry.promotable,
        "failed_permitting_gates": list(entry.failed_permitting_gates),
    }


def write_campaign_artifacts(
    out_dir: Path,
    frontier: Sequence[FrontierEntry],
    *,
    unresolved_rows: Sequence[Mapping[str, str]],
    vetoes: Sequence[VisualVeto],
) -> tuple[FrontierEntry, ...]:
    """Записать минимальный строгий контракт кампании без автоматического выбора.

    Выбор идёт только среди продвигаемых выживших: непродвигаемый профиль остаётся в
    `pareto.json` со своей причиной, но кандидатом на продвижение не становится (§7.3).
    """
    if unresolved_rows:
        raise ValueError("annotations_resolved запрещён, пока unresolved.csv содержит строки")
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(frontier, key=lambda row: row.profile_id))
    _write_json(out_dir / "pareto.json", [_frontier_document(entry) for entry in ordered])
    _write_csv(out_dir / "unresolved.csv", UNRESOLVED_COLUMNS, [])
    write_visual_vetoes(out_dir / "vetoes.csv", vetoes)
    survivors = apply_visual_vetoes(ordered, vetoes)
    promotable = [entry for entry in survivors if entry.promotable]
    if len(promotable) > 1:
        _write_json(
            out_dir / "selection_required.json",
            {"frontier": [_frontier_document(entry) for entry in promotable]},
        )
    elif len(promotable) == 1:
        _write_json(
            out_dir / "selected_profile.json", {"profile": _frontier_document(promotable[0])}
        )
    return survivors


def _score(document: Mapping[str, Any]) -> ProfileScore:
    return ProfileScore(
        profile_id=str(document["profile_id"]),
        ssd_margin=ConfidenceMargin(**document["ssd_margin"]),
        yolo_margin=ConfidenceMargin(**document["yolo_margin"]),
        covered_plates=int(document["covered_plates"]),
        rejected_frames=int(document["rejected_frames"]),
        median_delta=float(document["median_delta"]),
        p95_delta=float(document["p95_delta"]),
        above_quarter=int(document["above_quarter"]),
        new_over_redacted=int(document["new_over_redacted"]),
        changed_crops=int(document["changed_crops"]),
    )


def _gate_report(document: Mapping[str, Any]) -> ProfileGateReport:
    return ProfileGateReport(
        profile_id=str(document["profile_id"]),
        provenance_fresh=bool(document["provenance_fresh"]),
        unresolved_count=int(document["unresolved_count"]),
        geometry_current=bool(document["geometry_current"]),
        geometry_preflight_passed=bool(document["geometry_preflight_passed"]),
        canonical_coverage_complete=bool(document["canonical_coverage_complete"]),
        strict_leaks=int(document["strict_leaks"]),
        unreviewable_accepted=int(document["unreviewable_accepted"]),
        report_arithmetic_closed=bool(document["report_arithmetic_closed"]),
        operational_errors=int(document.get("operational_errors", 0)),
        canonical_sha256=(
            str(document["canonical_sha256"])
            if document.get("canonical_sha256") is not None
            else None
        ),
        w_max=int(document["w_max"]) if document.get("w_max") is not None else None,
        h_max=int(document["h_max"]) if document.get("h_max") is not None else None,
        oversized_blind_zone_count=int(document.get("oversized_blind_zone_count", 0)),
        geometry_preflight_reason=(
            str(document["geometry_preflight_reason"])
            if document.get("geometry_preflight_reason") is not None
            else None
        ),
    )


def _load_candidates(
    path: Path, current: GeometryWitness
) -> tuple[list[tuple[ProfileGateReport, ProfileScore | None]], tuple[str, ...]]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError("candidates JSON должен быть массивом")
    reports = [_gate_report(document["report"]) for document in documents]
    stale = stale_profile_ids(reports, current)
    stale_set = set(stale)
    candidates: list[tuple[ProfileGateReport, ProfileScore | None]] = []
    for document, report in zip(documents, reports, strict=True):
        if report.profile_id in stale_set:
            report = replace(report, geometry_current=False)
        failed_gate = first_failed_gate(report)
        score_document = document.get("score")
        if failed_gate is None and score_document is None:
            raise ValueError(f"прошедший профиль {report.profile_id} требует score")
        score = _score(score_document) if score_document is not None else None
        if score is not None and report.profile_id != score.profile_id:
            raise ValueError("candidate report и score принадлежат разным профилям")
        candidates.append((report, score))
    return candidates, stale


def _load_changed_crops(path: Path) -> list[ChangedCrop]:
    columns = ("profile_id", "frame_id", "sha256", "source")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"неверные колонки changed crops: {reader.fieldnames}")
        return [
            ChangedCrop(
                profile_id=row["profile_id"],
                frame_id=row["frame_id"],
                sha256=row["sha256"],
                source=Path(row["source"]),
            )
            for row in reader
        ]


def _write_candidate_profiles(
    path: Path, candidates: Sequence[tuple[ProfileGateReport, ProfileScore | None]]
) -> None:
    rows = [
        {
            "profile_id": report.profile_id,
            "failed_gate": first_failed_gate(report),
            "report": asdict(report),
            "score": asdict(score) if score is not None else None,
        }
        for report, score in sorted(candidates, key=lambda item: item[0].profile_id)
    ]
    _write_json(path, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Запись Pareto и visual-veto артефактов")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--current-witness", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--vetoes", type=Path, required=True)
    parser.add_argument("--changed-crops", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Отсечь stale/failed профили и записать полный набор campaign-артефактов."""
    arguments = build_parser().parse_args(argv)
    witness_document = json.loads(arguments.current_witness.read_text(encoding="utf-8"))
    current = GeometryWitness(
        w_max=int(witness_document["w_max"]),
        h_max=int(witness_document["h_max"]),
        canonical_sha256=str(witness_document["canonical_sha256"]),
    )
    candidates, stale = _load_candidates(arguments.candidates, current)
    with arguments.unresolved.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != UNRESOLVED_COLUMNS:
            raise ValueError(f"неверные колонки unresolved.csv: {reader.fieldnames}")
        unresolved = list(reader)
    if unresolved:
        raise ValueError("annotations_resolved запрещён, пока unresolved.csv содержит строки")

    scored_candidates = [(report, score) for report, score in candidates if score is not None]
    frontier = rank_profiles(scored_candidates)
    vetoes = load_visual_vetoes(arguments.vetoes)
    changed_crops = _load_changed_crops(arguments.changed_crops)
    frontier_by_id = {entry.profile_id: entry.score for entry in frontier}
    review_crops = [crop for crop in changed_crops if crop.profile_id in frontier_by_id]
    changed_ids: dict[str, set[str]] = {}
    for crop in review_crops:
        changed_ids.setdefault(crop.profile_id, set()).add(crop.frame_id)
    for profile_id, score in frontier_by_id.items():
        actual = len(changed_ids.get(profile_id, set()))
        if actual != score.changed_crops:
            raise ValueError(
                f"changed_crops профиля {profile_id}: staged={actual}, report={score.changed_crops}"
            )
    font = (
        ImageFont.truetype(str(arguments.font), 30)
        if arguments.font is not None
        else ImageFont.load_default()
    )
    stage_changed_crops(
        review_crops,
        arguments.views,
        font=font,
        expected_changed_ids=changed_ids,
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    _write_candidate_profiles(arguments.out / "candidate_profiles.json", candidates)
    _write_json(arguments.out / "geometry_stale.json", {"profile_ids": list(stale)})
    write_campaign_artifacts(
        arguments.out,
        frontier,
        unresolved_rows=unresolved,
        vetoes=vetoes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
