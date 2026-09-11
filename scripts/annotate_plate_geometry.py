"""Full-resolution виды и строгая сборка канонической геометрии номеров."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Literal, TypeAlias, cast

from PIL import Image, ImageFont

from pipeline.redaction_calibration.geometry import (
    CanonicalPlate,
    DetectorLink,
    DetectorObservation,
    load_detector_links,
    load_detector_observations,
    write_canonical_plates,
    write_detector_links,
)
from scripts.stage_plate_leak_views import render_view

ReviewAction: TypeAlias = Literal["create", "correct", "false_plate", "ambiguous", "link"]
REVIEW_ACTIONS = frozenset(("create", "correct", "false_plate", "ambiguous", "link"))
REVIEW_COLUMNS = (
    "action",
    "observation_id",
    "plate_id",
    "sha256",
    "x0",
    "y0",
    "x1",
    "y1",
    "readable",
    "note",
)
LINK_COLUMNS = tuple(field.name for field in fields(DetectorLink))
STAGE_COLUMNS = (
    *tuple(field.name for field in fields(CanonicalPlate)),
    "source_number",
    "source_path",
)


class GeometryReviewError(ValueError):
    """Нарушение строгого контракта человеческой геометрической разметки."""


def review_caption(plate: CanonicalPlate, source_number: int) -> str:
    """Собрать подпись, однозначно связывающую пиксели с canonical-строкой."""
    return (
        f"{plate.plate_id} sha={plate.sha256[:12]} #{source_number:03d} "
        f"({plate.x0},{plate.y0},{plate.x1},{plate.y1})"
    )


def stage_geometry_views(
    rows: Sequence[tuple[CanonicalPlate, int, Path]],
    out_dir: Path,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> list[dict[str, object]]:
    """Записать stable-order виды без downscale и индекс с явными идентификаторами."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row[0].plate_id)
    plate_ids = [plate.plate_id for plate, _, _ in ordered]
    if len(plate_ids) != len(set(plate_ids)):
        raise GeometryReviewError("дубликат plate_id в staging views")
    records: list[dict[str, object]] = []
    for plate, source_number, source in ordered:
        image = Image.open(source).convert("RGB")
        view = render_view(image, review_caption(plate, source_number), long_side=None, font=font)
        filename = f"{plate.plate_id}.png"
        view.save(out_dir / filename)
        records.append(
            {
                "plate_id": plate.plate_id,
                "sha256": plate.sha256,
                "source_number": source_number,
                "coordinates": f"{plate.x0},{plate.y0},{plate.x1},{plate.y1}",
                "file": filename,
            }
        )
    _write_dict_rows(
        out_dir / "views.csv",
        records,
        ("plate_id", "sha256", "source_number", "coordinates", "file"),
    )
    return records


def merge_review(
    review_path: Path,
    canonical: Sequence[CanonicalPlate],
    observations: Sequence[DetectorObservation],
    links: Sequence[DetectorLink],
) -> tuple[list[CanonicalPlate], list[DetectorLink]]:
    """Применить явные human actions к проверенным canonical и link записям."""
    plates_by_id = _unique_by_id(canonical, "plate_id")
    observations_by_id = _unique_observations(observations)
    links_by_observation = _unique_links(links)
    reviewed_observations: set[str] = set()
    for line, row in _read_review(review_path):
        observation_id = row["observation_id"]
        if observation_id in reviewed_observations:
            raise GeometryReviewError(
                f"строка {line}: observation_id повторно использован: {observation_id}"
            )
        reviewed_observations.add(observation_id)
        action = _review_action(row["action"], line)
        observation = _require_observation(observation_id, observations_by_id, line)
        plate_id = row["plate_id"]
        if action == "link":
            if plate_id not in plates_by_id:
                raise GeometryReviewError(f"строка {line}: неизвестный plate_id {plate_id}")
            linked_plate = plates_by_id[plate_id]
            if linked_plate.sha256 != observation.sha256:
                raise GeometryReviewError(
                    f"строка {line}: номер и наблюдение принадлежат разным кадрам"
                )
            _upsert_link(links_by_observation, observation.observation_id, plate_id, None)
            continue
        if action in {"false_plate", "ambiguous"} and not plate_id:
            _upsert_link(
                links_by_observation,
                observation.observation_id,
                None,
                action,
            )
            continue
        plate = _plate_from_review(row, action, line)
        if plate.sha256 != observation.sha256:
            raise GeometryReviewError(f"строка {line}: sha256 номера не совпадает с наблюдением")
        if action == "correct":
            if plate_id not in plates_by_id:
                raise GeometryReviewError(f"строка {line}: нечего correct: {plate_id}")
            existing_plate = plates_by_id[plate_id]
            if existing_plate.sha256 != plate.sha256:
                raise GeometryReviewError(
                    f"строка {line}: sha256 не совпадает с существующим номером"
                )
            if existing_plate.verdict != "true_plate":
                raise GeometryReviewError(f"строка {line}: correct применим только к true_plate")
        elif plate_id in plates_by_id:
            raise GeometryReviewError(f"строка {line}: дубликат plate_id {plate_id}")
        plates_by_id[plate_id] = plate
        _upsert_link(links_by_observation, observation.observation_id, plate_id, None)
    plates = sorted(plates_by_id.values(), key=lambda plate: plate.plate_id)
    _reject_contradictory_overlaps(plates)
    return plates, sorted(links_by_observation.values(), key=lambda link: link.observation_id)


def write_outputs(
    out_dir: Path,
    canonical: Sequence[CanonicalPlate],
    links: Sequence[DetectorLink],
) -> None:
    """Записать stable-order canonical, все связи и отдельные unresolved строки."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_plates(out_dir / "canonical-plates.csv", canonical)
    ordered_links = _validated_links(links)
    write_detector_links(out_dir / "detector-links.csv", ordered_links)
    unresolved = [link for link in ordered_links if link.plate_id is None]
    write_detector_links(out_dir / "unresolved.csv", unresolved)


def _read_review(path: Path) -> list[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise GeometryReviewError(f"неверные колонки review CSV: {reader.fieldnames}")
        return [(line, row) for line, row in enumerate(reader, start=2)]


def _review_action(value: str, line: int) -> ReviewAction:
    if value not in REVIEW_ACTIONS:
        raise GeometryReviewError(f"строка {line}: неизвестный action {value!r}")
    return cast(ReviewAction, value)


def _require_observation(
    observation_id: str,
    observations: dict[str, DetectorObservation],
    line: int,
) -> DetectorObservation:
    try:
        return observations[observation_id]
    except KeyError as error:
        raise GeometryReviewError(
            f"строка {line}: неизвестный observation_id {observation_id!r}"
        ) from error


def _plate_from_review(row: dict[str, str], action: ReviewAction, line: int) -> CanonicalPlate:
    note = row["note"].strip()
    if action in {"create", "correct"} and not note:
        raise GeometryReviewError(f"строка {line}: для {action} обязателен непустой note")
    verdict = {
        "create": "true_plate",
        "correct": "true_plate",
        "false_plate": "false_plate",
        "ambiguous": "ambiguous",
    }[action]
    try:
        coordinates = tuple(int(row[name]) for name in ("x0", "y0", "x1", "y1"))
        readable = _parse_bool(row["readable"])
    except ValueError as error:
        raise GeometryReviewError(f"строка {line}: неверные координаты/readable") from error
    return CanonicalPlate(
        plate_id=row["plate_id"],
        sha256=row["sha256"],
        x0=coordinates[0],
        y0=coordinates[1],
        x1=coordinates[2],
        y1=coordinates[3],
        readable=readable,
        verdict=cast(Literal["true_plate", "false_plate", "ambiguous"], verdict),
        note=note,
    )


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"неверное логическое значение: {value!r}")


def _unique_by_id(
    records: Sequence[CanonicalPlate], attribute: Literal["plate_id"]
) -> dict[str, CanonicalPlate]:
    indexed: dict[str, CanonicalPlate] = {}
    for record in records:
        key = cast(str, getattr(record, attribute))
        if key in indexed:
            raise GeometryReviewError(f"дубликат {attribute}: {key}")
        indexed[key] = record
    return indexed


def _unique_observations(
    observations: Sequence[DetectorObservation],
) -> dict[str, DetectorObservation]:
    indexed: dict[str, DetectorObservation] = {}
    for observation in observations:
        if observation.observation_id in indexed:
            raise GeometryReviewError(f"дубликат observation_id: {observation.observation_id}")
        indexed[observation.observation_id] = observation
    return indexed


def _unique_links(links: Sequence[DetectorLink]) -> dict[str, DetectorLink]:
    indexed: dict[str, DetectorLink] = {}
    for link in links:
        if link.observation_id in indexed:
            raise GeometryReviewError(f"дубликат observation_id: {link.observation_id}")
        indexed[link.observation_id] = link
    return indexed


def _upsert_link(
    links: dict[str, DetectorLink],
    observation_id: str,
    plate_id: str | None,
    reason: str | None,
) -> None:
    existing = links.get(observation_id)
    link_id = existing.link_id if existing is not None else _link_id(observation_id)
    links[observation_id] = DetectorLink(link_id, observation_id, plate_id, reason)


def _link_id(observation_id: str) -> str:
    digest = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()[:16]
    return f"link-{digest}"


def _reject_contradictory_overlaps(plates: Sequence[CanonicalPlate]) -> None:
    for index, left in enumerate(plates):
        for right in plates[index + 1 :]:
            if left.sha256 != right.sha256 or left.verdict == right.verdict:
                continue
            left_box = left.x0, left.y0, left.x1, left.y1
            right_box = right.x0, right.y0, right.x1, right.y1
            if _intersects(left_box, right_box):
                raise GeometryReviewError(
                    f"противоречивые перекрывающиеся истины: {left.plate_id}, {right.plate_id}"
                )


def _intersects(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(
        left[3], right[3]
    )


def _validated_links(links: Sequence[DetectorLink]) -> list[DetectorLink]:
    seen_ids: set[str] = set()
    seen_observations: set[str] = set()
    for link in links:
        if not link.link_id or not link.observation_id:
            raise GeometryReviewError("link_id и observation_id обязательны")
        if link.link_id in seen_ids or link.observation_id in seen_observations:
            raise GeometryReviewError("дубликат link_id или observation_id")
        if link.plate_id is None and not (link.reason or "").strip():
            raise GeometryReviewError("у unresolved link обязателен reason")
        if link.plate_id is not None and link.reason is not None:
            raise GeometryReviewError("resolved link не должен содержать reason")
        seen_ids.add(link.link_id)
        seen_observations.add(link.observation_id)
    return sorted(links, key=lambda link: link.link_id)


def _write_dict_rows(
    path: Path, rows: Sequence[dict[str, object]], columns: tuple[str, ...]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stage_rows(path: Path) -> list[tuple[CanonicalPlate, int, Path]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != STAGE_COLUMNS:
            raise GeometryReviewError(f"неверные колонки staging CSV: {reader.fieldnames}")
        rows: list[tuple[CanonicalPlate, int, Path]] = []
        for line, row in enumerate(reader, start=2):
            try:
                plate = CanonicalPlate(
                    plate_id=row["plate_id"],
                    sha256=row["sha256"],
                    x0=int(row["x0"]),
                    y0=int(row["y0"]),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                    readable=_parse_bool(row["readable"]),
                    verdict=cast(
                        Literal["true_plate", "false_plate", "ambiguous"],
                        row["verdict"],
                    ),
                    note=row["note"],
                )
                rows.append((plate, int(row["source_number"]), Path(row["source_path"])))
            except (KeyError, TypeError, ValueError) as error:
                raise GeometryReviewError(f"строка {line}: неверные данные staging CSV") from error
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Построить CLI staging и merge человеческой review CSV."""
    parser = argparse.ArgumentParser(description="Каноническая геометрия номеров")
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="собрать full-resolution просмотровые виды")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--out-dir", type=Path, required=True)
    stage.add_argument("--font", type=Path, default=None)
    merge = commands.add_parser("merge", help="слить человеческую review CSV")
    merge.add_argument("--review", type=Path, required=True)
    merge.add_argument("--canonical", type=Path, required=True)
    merge.add_argument("--observations", type=Path, required=True)
    merge.add_argument("--links", type=Path, required=True)
    merge.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить staging либо merge строгих геометрических артефактов."""
    from pipeline.redaction_calibration.geometry import load_canonical_plates

    args = build_parser().parse_args(argv)
    if args.command == "stage":
        font = (
            ImageFont.load_default()
            if args.font is None
            else ImageFont.truetype(str(args.font), 30)
        )
        stage_geometry_views(_stage_rows(args.manifest), args.out_dir, font=font)
        return 0
    canonical = load_canonical_plates(args.canonical)
    observations = load_detector_observations(args.observations)
    links = load_detector_links(args.links)
    merged, merged_links = merge_review(args.review, canonical, observations, links)
    write_outputs(args.out_dir, merged, merged_links)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
