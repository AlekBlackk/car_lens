"""Строгие артефакты канонических номеров и целочисленная геометрия."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from pipeline.harvest.tiling import tile_origins

Box: TypeAlias = tuple[int, int, int, int]
PlateVerdict: TypeAlias = Literal["true_plate", "false_plate", "ambiguous"]
StableId = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
Sha256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CanonicalPlate:
    """Проверенный человеком полный бокс номера в координатах исходного кадра."""

    plate_id: str
    sha256: str
    x0: int
    y0: int
    x1: int
    y1: int
    readable: bool
    verdict: PlateVerdict
    note: str


@dataclass(frozen=True, slots=True)
class DetectorObservation:
    """Стабильное raw-наблюдение со всей геометрической provenance детектора."""

    observation_id: str
    sha256: str
    detector: str
    profile_id: str
    pass_name: str
    zone: Box
    tile: Box
    box: Box
    merged_box: Box | None = None
    confidence: float = 0.0
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class DetectorLink:
    """Разрешённая либо явно неразрешённая связь наблюдения с номером."""

    link_id: str
    observation_id: str
    plate_id: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class GeometryWitness:
    """Граница полных истинных номеров, привязанная к точным CSV-байтам."""

    w_max: int
    h_max: int
    canonical_sha256: str


CANONICAL_COLUMNS = tuple(field.name for field in fields(CanonicalPlate))
PLATE_VERDICTS = frozenset(("true_plate", "false_plate", "ambiguous"))


def tile_guarantee(tile: int, overlap: float) -> int:
    """Вернуть гарантированный размер объекта вдоль одной оси тайла."""
    if tile <= 0 or not 0.0 <= overlap < 1.0:
        raise ValueError("tile должен быть положительным, overlap — в диапазоне [0, 1)")
    return tile - max(1, int(tile * (1.0 - overlap)))


def box_fits_actual_tile(
    frame_size: tuple[int, int],
    canonical: CanonicalPlate,
    tile: int,
    overlap: float,
) -> bool:
    """Проверить, что один реальный full-frame тайл целиком содержит номер."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame_size должен содержать положительные width и height")
    _validate_plate(canonical)
    if canonical.x1 > width or canonical.y1 > height:
        return False
    x_origins = tile_origins(width, tile, overlap)
    y_origins = tile_origins(height, tile, overlap)
    return any(
        x0 <= canonical.x0
        and canonical.x1 <= min(width, x0 + tile)
        and y0 <= canonical.y0
        and canonical.y1 <= min(height, y0 + tile)
        for y0 in y_origins
        for x0 in x_origins
    )


def fill_contains(fill: Box, box: Box | CanonicalPlate) -> bool:
    """Проверить точное half-open вложение бокса в один срез заливки."""
    target = _plate_box(box) if isinstance(box, CanonicalPlate) else box
    return (
        fill[0] <= target[0]
        and fill[1] <= target[1]
        and target[2] <= fill[2]
        and target[3] <= fill[3]
    )


def fills_cover(fills: Sequence[Box], canonical: CanonicalPlate) -> bool:
    """Проверить покрытие номера объединением заливок, растеризуя только его бокс."""
    _validate_plate(canonical)
    width = canonical.x1 - canonical.x0
    height = canonical.y1 - canonical.y0
    mask = np.zeros((height, width), dtype=bool)
    for fill in fills:
        x0 = max(canonical.x0, fill[0])
        y0 = max(canonical.y0, fill[1])
        x1 = min(canonical.x1, fill[2])
        y1 = min(canonical.y1, fill[3])
        if x1 > x0 and y1 > y0:
            mask[y0 - canonical.y0 : y1 - canonical.y0, x0 - canonical.x0 : x1 - canonical.x0] = (
                True
            )
    return bool(mask.all())


def is_subtile_plate(plate: CanonicalPlate, tiles: int | Sequence[int]) -> bool:
    """Проверить, что бокс номера не превосходит хотя бы один из тайлов."""
    _validate_plate(plate)
    tile_list = (tiles,) if isinstance(tiles, int) else tuple(tiles)
    if not tile_list or any(t <= 0 for t in tile_list):
        raise ValueError("tiles должен содержать положительные размеры тайлов")
    w = plate.x1 - plate.x0
    h = plate.y1 - plate.y0
    return any(w <= t and h <= t for t in tile_list)


def partition_canonical_plates(
    canonical: Sequence[CanonicalPlate],
    tiles: int | Sequence[int],
) -> tuple[list[CanonicalPlate], list[CanonicalPlate]]:
    """Разделить истинные канонические номера на подтайловые и надтайловые."""
    subtile: list[CanonicalPlate] = []
    oversized: list[CanonicalPlate] = []
    for plate in canonical:
        if plate.verdict != "true_plate":
            continue
        if is_subtile_plate(plate, tiles):
            subtile.append(plate)
        else:
            oversized.append(plate)
    return subtile, oversized


def match_observation(
    observation: DetectorObservation,
    canonical: Sequence[CanonicalPlate],
    fills: Sequence[Box],
    w_max: int,
    h_max: int,
) -> str | None:
    """Консервативно связать raw-бокс только с одним полностью залитым номером."""
    _validate_observation(observation)
    if w_max <= 0 or h_max <= 0:
        raise ValueError("w_max и h_max должны быть положительными")
    width = observation.box[2] - observation.box[0]
    height = observation.box[3] - observation.box[1]
    if width > w_max or height > h_max:
        return None
    fill = _observation_fill(observation, fills)
    if fill is None:
        return None
    candidates = [
        plate
        for plate in canonical
        if plate.sha256 == observation.sha256
        and plate.verdict == "true_plate"
        and fill_contains(fill, plate)
    ]
    if len(candidates) != 1:
        return None
    competing = [
        plate
        for plate in canonical
        if plate.sha256 == observation.sha256
        and plate.plate_id != candidates[0].plate_id
        and _intersects(fill, _plate_box(plate))
    ]
    return None if competing else candidates[0].plate_id


def write_detector_observations(path: Path, observations: Sequence[DetectorObservation]) -> None:
    """Проверить и записать raw-наблюдения в порядке стабильных ID."""
    ordered = _validated_observations(observations)
    _write_records(path, ordered, tuple(field.name for field in fields(DetectorObservation)))


def load_detector_observations(path: Path) -> list[DetectorObservation]:
    """Загрузить raw-наблюдения со строгой схемой и stable-order проверкой."""
    columns = tuple(field.name for field in fields(DetectorObservation))
    rows = _read_rows(path, columns)
    parsed: list[DetectorObservation] = []
    for line, row in rows:
        try:
            parsed.append(
                DetectorObservation(
                    observation_id=row["observation_id"],
                    sha256=row["sha256"],
                    detector=row["detector"],
                    profile_id=row["profile_id"],
                    pass_name=row["pass_name"],
                    zone=_parse_box(row["zone"]),
                    tile=_parse_box(row["tile"]),
                    box=_parse_box(row["box"]),
                    merged_box=_parse_optional_box(row["merged_box"]),
                    confidence=float(row["confidence"]),
                    class_id=int(row["class_id"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"строка {line}: неверные типы observations CSV") from error
    ordered = _validated_observations(parsed)
    if parsed != ordered:
        raise ValueError("observations CSV должен быть отсортирован по observation_id")
    return parsed


def write_detector_links(path: Path, links: Sequence[DetectorLink]) -> None:
    """Проверить и записать resolved/unresolved связи в порядке стабильных ID."""
    ordered = _validated_links(links)
    _write_records(path, ordered, tuple(field.name for field in fields(DetectorLink)))


def load_detector_links(path: Path) -> list[DetectorLink]:
    """Загрузить связи со строгой схемой и stable-order проверкой."""
    columns = tuple(field.name for field in fields(DetectorLink))
    rows = _read_rows(path, columns)
    parsed = [
        DetectorLink(
            link_id=row["link_id"],
            observation_id=row["observation_id"],
            plate_id=row["plate_id"] or None,
            reason=row["reason"] or None,
        )
        for _, row in rows
    ]
    ordered = _validated_links(parsed)
    if parsed != ordered:
        raise ValueError("links CSV должен быть отсортирован по link_id")
    return parsed


def write_canonical_plates(path: Path, plates: Sequence[CanonicalPlate]) -> None:
    """Проверить, отсортировать по `plate_id` и записать канонический CSV."""
    ordered = _validated_plates(plates)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CANONICAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for plate in ordered:
            writer.writerow(asdict(plate))


def load_canonical_plates(path: Path) -> list[CanonicalPlate]:
    """Загрузить канонический CSV со строгой схемой, типами и порядком."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CANONICAL_COLUMNS:
            raise ValueError(f"неверные колонки canonical CSV: {reader.fieldnames}")
        parsed = [_parse_plate(row, index) for index, row in enumerate(reader, start=2)]
    ordered = _validated_plates(parsed)
    if parsed != ordered:
        raise ValueError("canonical CSV должен быть отсортирован по plate_id")
    return parsed


def geometry_witness(path: Path) -> GeometryWitness:
    """Вычислить свидетель из проверенных байтов stable-order canonical CSV."""
    plates = load_canonical_plates(path)
    true_plates = [plate for plate in plates if plate.verdict == "true_plate"]
    if not true_plates:
        raise ValueError("canonical CSV не содержит истинных номеров")
    return GeometryWitness(
        w_max=max(plate.x1 - plate.x0 for plate in true_plates),
        h_max=max(plate.y1 - plate.y0 for plate in true_plates),
        canonical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _validated_observations(
    observations: Sequence[DetectorObservation],
) -> list[DetectorObservation]:
    seen: set[str] = set()
    for observation in observations:
        _validate_observation(observation)
        if observation.observation_id in seen:
            raise ValueError(f"дубликат observation_id: {observation.observation_id}")
        seen.add(observation.observation_id)
    return sorted(observations, key=lambda row: row.observation_id)


def _validated_links(links: Sequence[DetectorLink]) -> list[DetectorLink]:
    seen_ids: set[str] = set()
    seen_observations: set[str] = set()
    for link in links:
        _validate_id("link_id", link.link_id)
        _validate_id("observation_id", link.observation_id)
        if link.plate_id is not None:
            _validate_id("plate_id", link.plate_id)
        if link.link_id in seen_ids or link.observation_id in seen_observations:
            raise ValueError("дубликат link_id или observation_id")
        if link.plate_id is None and not (link.reason or "").strip():
            raise ValueError("у unresolved link обязателен reason")
        if link.plate_id is not None and link.reason is not None:
            raise ValueError("resolved link не должен содержать reason")
        seen_ids.add(link.link_id)
        seen_observations.add(link.observation_id)
    return sorted(links, key=lambda row: row.link_id)


def _write_records(
    path: Path,
    records: Sequence[DetectorObservation] | Sequence[DetectorLink],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for key in ("zone", "tile", "box", "merged_box"):
                if row.get(key) is not None:
                    row[key] = ",".join(str(value) for value in row[key])
            writer.writerow(row)


def _read_rows(path: Path, columns: tuple[str, ...]) -> list[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"неверные колонки {path}: {reader.fieldnames}")
        return [(line, row) for line, row in enumerate(reader, start=2)]


def _parse_optional_box(value: str) -> Box | None:
    return None if not value else _parse_box(value)


def _parse_box(value: str) -> Box:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("бокс должен содержать четыре координаты")
    return parts


def _validated_plates(plates: Sequence[CanonicalPlate]) -> list[CanonicalPlate]:
    seen: set[str] = set()
    for plate in plates:
        _validate_plate(plate)
        if plate.plate_id in seen:
            raise ValueError(f"дубликат plate_id: {plate.plate_id}")
        seen.add(plate.plate_id)
    return sorted(plates, key=lambda plate: plate.plate_id)


def _validate_plate(plate: CanonicalPlate) -> None:
    _validate_id("plate_id", plate.plate_id)
    _validate_sha(plate.sha256)
    _validate_box(_plate_box(plate), "канонические координаты")
    if plate.verdict not in PLATE_VERDICTS:
        raise ValueError(f"неизвестный verdict: {plate.verdict}")
    if not isinstance(plate.readable, bool):
        raise ValueError("readable должен быть логическим")
    if plate.verdict == "true_plate" and not plate.readable:
        raise ValueError("истинный номер должен иметь readable=True")
    if plate.verdict == "false_plate" and plate.readable:
        raise ValueError("ложный номер должен иметь readable=False")
    if not isinstance(plate.note, str):
        raise ValueError("note должен быть строкой")
    if plate.verdict == "true_plate" and not plate.note.strip():
        raise ValueError("у истинного номера обязателен непустой note")


def _validate_observation(observation: DetectorObservation) -> None:
    _validate_id("observation_id", observation.observation_id)
    _validate_sha(observation.sha256)
    _validate_id("profile_id", observation.profile_id)
    if not observation.detector or not observation.pass_name:
        raise ValueError("detector и pass_name обязательны")
    _validate_box(observation.box, "бокс наблюдения")
    if observation.merged_box is not None:
        _validate_box(observation.merged_box, "merged-бокс")
    for name, box in (("zone", observation.zone), ("tile", observation.tile)):
        if box != (0, 0, 0, 0):
            _validate_box(box, name)
    if isinstance(observation.class_id, bool) or not isinstance(observation.class_id, int):
        raise ValueError("class_id должен быть целым")
    if isinstance(observation.confidence, bool) or not isinstance(
        observation.confidence, (int, float)
    ):
        raise ValueError("confidence должен быть числом")


def _validate_id(name: str, value: str) -> None:
    if not isinstance(value, str) or StableId.fullmatch(value) is None:
        raise ValueError(f"{name} не является стабильным ID: {value!r}")


def _validate_sha(value: str) -> None:
    if not isinstance(value, str) or Sha256.fullmatch(value) is None:
        raise ValueError(f"неверный sha256: {value!r}")


def _validate_box(box: Box, name: str) -> None:
    if (
        len(box) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
        or box[0] < 0
        or box[1] < 0
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"{name} должны задавать положительный integer half-open бокс: {box}")


def _parse_plate(row: dict[str, str], line: int) -> CanonicalPlate:
    try:
        readable = _parse_bool(row["readable"])
        coordinates = tuple(int(row[name]) for name in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"строка {line}: неверные типы canonical CSV") from error
    return CanonicalPlate(
        plate_id=row["plate_id"],
        sha256=row["sha256"],
        x0=coordinates[0],
        y0=coordinates[1],
        x1=coordinates[2],
        y1=coordinates[3],
        readable=readable,
        verdict=row["verdict"],  # type: ignore[arg-type]
        note=row["note"],
    )


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"неверное логическое значение: {value!r}")


def _plate_box(plate: CanonicalPlate) -> Box:
    return plate.x0, plate.y0, plate.x1, plate.y1


def _observation_fill(observation: DetectorObservation, fills: Sequence[Box]) -> Box | None:
    containing = [fill for fill in fills if fill_contains(fill, observation.box)]
    if len(containing) != 1:
        return None
    return containing[0]


def _intersects(left: Box, right: Box) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(
        left[3], right[3]
    )
