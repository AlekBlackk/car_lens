"""Артефакты Pareto-оценки профилей и трассируемого visual veto."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

from pipeline.redaction_calibration.geometry import DetectorLink
from pipeline.redaction_calibration.model import ProfileScore, VisualVeto
from scripts.stage_plate_leak_views import render_view

VETO_COLUMNS = tuple(field.name for field in fields(VisualVeto))
UNRESOLVED_COLUMNS = tuple(field.name for field in fields(DetectorLink))
VIEW_COLUMNS = ("profile_id", "frame_id", "sha256", "file")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NONCOSMETIC_TOKENS = (
    "destroy",
    "unread",
    "recogn",
    "model",
    "feature",
    "уничтож",
    "нечита",
    "распозна",
    "модел",
    "признак",
)


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
    if (
        len(reason) < 12
        or reason == "косметика"
        or not any(token in reason for token in _NONCOSMETIC_TOKENS)
    ):
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
    frontier: Sequence[ProfileScore], vetoes: Sequence[VisualVeto]
) -> tuple[ProfileScore, ...]:
    """Исключить vetoed profiles, не разрешая ledger добавить профиль вне frontier."""
    known = {score.profile_id for score in frontier}
    vetoed = {veto.profile_id for veto in vetoes}
    foreign = sorted(vetoed - known)
    if foreign:
        raise ValueError(f"veto ссылается на профиль вне frontier: {foreign}")
    return tuple(score for score in frontier if score.profile_id not in vetoed)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_campaign_artifacts(
    out_dir: Path,
    frontier: Sequence[ProfileScore],
    *,
    unresolved_rows: Sequence[Mapping[str, str]],
    vetoes: Sequence[VisualVeto],
) -> tuple[ProfileScore, ...]:
    """Записать минимальный строгий контракт кампании без автоматического выбора."""
    if unresolved_rows:
        raise ValueError("annotations_resolved запрещён, пока unresolved.csv содержит строки")
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(frontier, key=lambda row: row.profile_id))
    _write_json(out_dir / "pareto.json", [asdict(score) for score in ordered])
    _write_csv(out_dir / "unresolved.csv", UNRESOLVED_COLUMNS, [])
    write_visual_vetoes(out_dir / "vetoes.csv", vetoes)
    survivors = apply_visual_vetoes(ordered, vetoes)
    if len(survivors) > 1:
        _write_json(
            out_dir / "selection_required.json",
            {"frontier": [asdict(score) for score in survivors]},
        )
    elif len(survivors) == 1:
        _write_json(out_dir / "selected_profile.json", {"profile": asdict(survivors[0])})
    return survivors


def _score(document: Mapping[str, Any]) -> ProfileScore:
    from pipeline.redaction_calibration.model import ConfidenceMargin

    return ProfileScore(
        profile_id=str(document["profile_id"]),
        ssd_margin=ConfidenceMargin(**document["ssd_margin"]),
        yolo_margin=ConfidenceMargin(**document["yolo_margin"]),
        median_delta=float(document["median_delta"]),
        p95_delta=float(document["p95_delta"]),
        above_quarter=int(document["above_quarter"]),
        new_over_redacted=int(document["new_over_redacted"]),
        changed_crops=int(document["changed_crops"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Запись Pareto и visual-veto артефактов")
    parser.add_argument("--frontier", type=Path, required=True, help="JSON-массив ProfileScore")
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--vetoes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Перенести проверенные pure-результаты в строгие campaign-артефакты."""
    arguments = build_parser().parse_args(argv)
    frontier = [_score(row) for row in json.loads(arguments.frontier.read_text(encoding="utf-8"))]
    with arguments.unresolved.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != UNRESOLVED_COLUMNS:
            raise ValueError(f"неверные колонки unresolved.csv: {reader.fieldnames}")
        unresolved = list(reader)
    vetoes = load_visual_vetoes(arguments.vetoes)
    write_campaign_artifacts(arguments.out, frontier, unresolved_rows=unresolved, vetoes=vetoes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
