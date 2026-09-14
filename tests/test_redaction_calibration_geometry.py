"""Тесты канонической геометрии номеров и conservative matcher."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pipeline.redaction_calibration.geometry import (
    CanonicalPlate,
    DetectorLink,
    DetectorObservation,
    GeometryWitness,
    PlateVerdict,
    box_fits_actual_tile,
    fill_contains,
    fills_cover,
    geometry_witness,
    is_subtile_plate,
    load_canonical_plates,
    load_detector_links,
    load_detector_observations,
    match_observation,
    partition_canonical_plates,
    tile_guarantee,
    write_canonical_plates,
    write_detector_links,
    write_detector_observations,
)


def _plate(
    plate_id: str = "plate-001",
    *,
    sha256: str = "a" * 64,
    box: tuple[int, int, int, int] = (100, 100, 260, 150),
    verdict: PlateVerdict = "true_plate",
) -> CanonicalPlate:
    return CanonicalPlate(
        plate_id,
        sha256,
        *box,
        readable=verdict == "true_plate",
        verdict=verdict,
        note="проверено",
    )


def _observation(
    observation_id: str = "obs-001",
    *,
    sha256: str = "a" * 64,
    box: tuple[int, int, int, int] = (120, 110, 240, 145),
) -> DetectorObservation:
    return DetectorObservation(
        observation_id=observation_id,
        sha256=sha256,
        detector="ssd",
        profile_id="profile-001",
        pass_name="first",
        zone=(0, 0, 800, 600),
        tile=(0, 0, 640, 600),
        box=box,
        confidence=0.91,
        class_id=0,
    )


def test_observation_and_link_csv_roundtrip_is_strict_and_sorted(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.csv"
    links_path = tmp_path / "links.csv"
    write_detector_observations(
        observations_path,
        [_observation("obs-002"), _observation("obs-001")],
    )
    write_detector_links(
        links_path,
        [
            DetectorLink("link-002", "obs-002", None, "ambiguous_match"),
            DetectorLink("link-001", "obs-001", "plate-001", None),
        ],
    )

    assert [row.observation_id for row in load_detector_observations(observations_path)] == [
        "obs-001",
        "obs-002",
    ]
    assert [row.link_id for row in load_detector_links(links_path)] == ["link-001", "link-002"]

    links_path.write_text(
        links_path.read_text(encoding="utf-8").replace("ambiguous_match", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reason"):
        load_detector_links(links_path)


def test_detector_csv_rejects_unsorted_and_duplicate_stable_ids(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.csv"
    write_detector_observations(
        observations_path,
        [_observation("obs-002"), _observation("obs-001")],
    )
    lines = observations_path.read_text(encoding="utf-8").splitlines()
    observations_path.write_text(
        "\n".join((lines[0], lines[2], lines[1])) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="отсортирован"):
        load_detector_observations(observations_path)

    with pytest.raises(ValueError, match="дубликат"):
        write_detector_observations(
            tmp_path / "duplicates.csv",
            [_observation(), _observation()],
        )


def test_detector_observation_keeps_raw_and_merged_boxes_separate() -> None:
    observation = DetectorObservation(
        observation_id="obs-merged",
        sha256="a" * 64,
        detector="ssd",
        profile_id="profile-001",
        pass_name="first",
        zone=(0, 0, 800, 600),
        tile=(0, 0, 640, 600),
        box=(120, 110, 240, 145),
        merged_box=(100, 100, 260, 150),
        confidence=0.91,
        class_id=0,
    )

    assert observation.box == (120, 110, 240, 145)
    assert observation.merged_box == (100, 100, 260, 150)


def test_records_immutable_and_keep_stable_ids_and_provenance() -> None:
    observation = _observation()
    link = DetectorLink("link-001", observation.observation_id, "plate-001", None)

    assert observation.zone == (0, 0, 800, 600)
    assert observation.tile == (0, 0, 640, 600)
    assert link.observation_id == "obs-001"
    with pytest.raises(FrozenInstanceError):
        observation.detector = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tile", "expected"),
    [(300, 75), (640, 160), (1280, 320)],
)
def test_tile_guarantee_uses_integer_production_step(tile: int, expected: int) -> None:
    assert tile_guarantee(tile, 0.25) == expected


def test_box_fits_actual_tile_rejects_non_positive_frame_size() -> None:
    with pytest.raises(ValueError, match="frame_size"):
        box_fits_actual_tile((0, 600), _plate(), 640, 0.25)


def test_actual_tile_uses_final_origin_with_increased_overlap() -> None:
    plate = _plate(box=(650, 10, 850, 60))

    assert box_fits_actual_tile((1000, 200), plate, 640, 0.25)


def test_actual_tile_accepts_one_tile_axes_and_requires_same_cartesian_tile() -> None:
    plate = _plate(box=(900, 40, 990, 160))

    assert box_fits_actual_tile((1000, 200), plate, 1280, 0.25)
    assert not box_fits_actual_tile((1000, 100), plate, 640, 0.25)


def test_actual_eight_box_preflight_fixture() -> None:
    boxes = (
        (40, 30, 200, 80),
        (450, 70, 610, 120),
        (650, 90, 810, 140),
        (840, 130, 1000, 180),
        (20, 390, 180, 440),
        (470, 410, 630, 460),
        (670, 450, 830, 500),
        (840, 500, 1000, 550),
    )

    assert all(box_fits_actual_tile((1000, 600), _plate(box=box), 640, 0.25) for box in boxes)


def test_fill_contains_accepts_exact_half_open_boundary() -> None:
    assert fill_contains((10, 20, 30, 40), (10, 20, 30, 40))
    assert not fill_contains((10, 20, 29, 40), (10, 20, 30, 40))


def test_fills_cover_uses_union_of_two_integer_slices() -> None:
    plate = _plate(box=(10, 20, 30, 40))

    assert fills_cover(((10, 20, 20, 40), (20, 20, 30, 40)), plate)
    assert not fills_cover(((10, 20, 20, 40), (21, 20, 30, 40)), plate)


def test_matcher_auto_links_single_containing_fill() -> None:
    plate = _plate()
    observation = _observation()

    assert match_observation(observation, [plate], [(90, 90, 270, 160)], 160, 50) == "plate-001"


def test_matcher_rejects_two_canonical_intersections() -> None:
    observation = _observation(box=(120, 110, 250, 145))
    plates = [
        _plate(box=(100, 100, 180, 150)),
        _plate("plate-002", box=(200, 100, 270, 150)),
    ]

    assert match_observation(observation, plates, [(90, 90, 280, 160)], 160, 50) is None


def test_matcher_rejects_fill_intersecting_competing_canonical() -> None:
    observation = _observation()
    plates = [
        _plate(box=(100, 100, 180, 150)),
        _plate("plate-002", box=(200, 100, 270, 150)),
    ]

    assert match_observation(observation, plates, [(90, 90, 280, 160)], 160, 50) is None


def test_matcher_uses_exact_observation_fill_not_unrelated_covering_union() -> None:
    observation = _observation(box=(100, 100, 180, 150))
    fills = [(100, 100, 180, 150), (180, 100, 260, 150)]

    assert match_observation(observation, [_plate()], fills, 160, 50) is None


def test_matcher_rejects_partial_fill() -> None:
    plate = _plate()

    assert (
        match_observation(
            _observation(box=(100, 100, 250, 150)),
            [plate],
            [(100, 100, 250, 150)],
            160,
            50,
        )
        is None
    )


def test_matcher_rejects_invalid_witness_bounds() -> None:
    with pytest.raises(ValueError, match=r"w_max.*h_max"):
        match_observation(_observation(), [_plate()], [(90, 90, 270, 160)], 0, 50)


def test_matcher_rejects_raw_box_above_witness_even_when_contained() -> None:
    plate = _plate(box=(50, 50, 300, 200))
    observation = _observation(box=(60, 60, 250, 130))

    assert match_observation(observation, [plate], [(50, 50, 300, 200)], 160, 50) is None


def test_matcher_rejects_different_sha() -> None:
    assert (
        match_observation(
            _observation(sha256="b" * 64),
            [_plate()],
            [(90, 90, 270, 160)],
            160,
            50,
        )
        is None
    )


def test_canonical_csv_is_strict_sorted_and_hashes_validated_bytes(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    plates = [_plate("plate-002", box=(0, 0, 80, 30)), _plate("plate-001")]

    write_canonical_plates(path, plates)
    loaded = load_canonical_plates(path)
    witness = geometry_witness(path)

    assert [plate.plate_id for plate in loaded] == ["plate-001", "plate-002"]
    assert witness == GeometryWitness(
        w_max=160,
        h_max=50,
        canonical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_witness_excludes_false_and_ambiguous_rows(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    write_canonical_plates(
        path,
        [
            _plate(box=(0, 0, 40, 20)),
            _plate("plate-002", box=(0, 0, 400, 200), verdict="false_plate"),
            _plate("plate-003", box=(0, 0, 300, 100), verdict="ambiguous"),
        ],
    )

    assert geometry_witness(path).w_max == 40
    assert geometry_witness(path).h_max == 20


def test_witness_rejects_false_readable_and_true_unreadable_rows(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    with pytest.raises(ValueError, match="readable"):
        write_canonical_plates(
            path,
            [
                CanonicalPlate(
                    "plate-001",
                    "a" * 64,
                    0,
                    0,
                    40,
                    20,
                    True,
                    "false_plate",
                    "не номер",
                )
            ],
        )
    with pytest.raises(ValueError, match="readable"):
        write_canonical_plates(
            path,
            [
                CanonicalPlate(
                    "plate-001",
                    "a" * 64,
                    0,
                    0,
                    40,
                    20,
                    False,
                    "true_plate",
                    "номер",
                )
            ],
        )


def test_witness_rejects_true_plate_with_unreviewed_note(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="note"):
        write_canonical_plates(
            tmp_path / "canonical.csv",
            [
                CanonicalPlate(
                    "plate-001",
                    "a" * 64,
                    0,
                    0,
                    40,
                    20,
                    True,
                    "true_plate",
                    " ",
                )
            ],
        )


@pytest.mark.parametrize(
    "plate",
    [
        _plate(box=(0, 0, 0, 10)),
        _plate(box=(0, 0, 10, 0)),
        _plate(box=(-1, 0, 10, 10)),
    ],
)
def test_canonical_csv_rejects_non_positive_or_negative_boxes(
    tmp_path: Path, plate: CanonicalPlate
) -> None:
    with pytest.raises(ValueError, match=r"координат|положитель"):
        write_canonical_plates(tmp_path / "canonical.csv", [plate])


def test_canonical_csv_rejects_schema_ids_order_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    write_canonical_plates(path, [_plate()])
    text = path.read_text(encoding="utf-8")

    path.write_text(text.replace("plate_id", "wrong"), encoding="utf-8")
    with pytest.raises(ValueError, match="колон"):
        load_canonical_plates(path)

    with pytest.raises(ValueError, match="plate_id"):
        write_canonical_plates(tmp_path / "bad-id.csv", [_plate("bad id")])

    with pytest.raises(ValueError, match="дубликат"):
        write_canonical_plates(tmp_path / "duplicate.csv", [_plate(), _plate()])


def test_is_subtile_plate_классифицирует_по_тайлам() -> None:
    plate = _plate("plate-001", box=(100, 100, 260, 150))  # 160 x 50
    assert is_subtile_plate(plate, 640)
    assert is_subtile_plate(plate, [320, 640])
    assert not is_subtile_plate(plate, 100)  # width 160 > 100
    assert not is_subtile_plate(plate, 40)  # height 50 > 40

    large = _plate("plate-002", box=(100, 100, 900, 400))  # 800 x 300
    assert not is_subtile_plate(large, 640)
    assert is_subtile_plate(large, [640, 1280])

    with pytest.raises(ValueError, match="положительные"):
        is_subtile_plate(plate, 0)
    with pytest.raises(ValueError, match="положительные"):
        is_subtile_plate(plate, [])


def test_partition_canonical_plates_разделяет_только_истинные_номера() -> None:
    sub = _plate("plate-001", box=(100, 100, 260, 150))  # 160 x 50
    over = _plate("plate-002", box=(100, 100, 900, 400))  # 800 x 300
    false_plate = _plate(
        "plate-003",
        box=(100, 100, 200, 150),
        verdict="false_plate",
    )
    subtiles, oversized = partition_canonical_plates([sub, over, false_plate], 640)
    assert subtiles == [sub]
    assert oversized == [over]
