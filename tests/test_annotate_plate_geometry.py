"""Тесты full-resolution аннотации и слияния человеческой геометрии."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageFont

from pipeline.redaction_calibration.geometry import (
    CanonicalPlate,
    DetectorLink,
    DetectorObservation,
    PlateVerdict,
    load_canonical_plates,
)
from scripts.annotate_plate_geometry import (
    merge_review,
    review_caption,
    stage_geometry_views,
)


def _observation(
    observation_id: str = "obs-001",
    *,
    sha256: str = "a" * 64,
    box: tuple[int, int, int, int] = (10, 20, 90, 50),
) -> DetectorObservation:
    return DetectorObservation(
        observation_id=observation_id,
        sha256=sha256,
        detector="ssd",
        profile_id="profile-001",
        pass_name="first",
        zone=(0, 0, 200, 120),
        tile=(0, 0, 200, 120),
        box=box,
        confidence=0.9,
        class_id=0,
    )


def _plate(
    plate_id: str = "plate-001",
    *,
    sha256: str = "a" * 64,
    box: tuple[int, int, int, int] = (10, 20, 90, 50),
    verdict: PlateVerdict = "true_plate",
    note: str = "проверено",
) -> CanonicalPlate:
    return CanonicalPlate(
        plate_id,
        sha256,
        *box,
        readable=True,
        verdict=verdict,
        note=note,
    )


def _write_review(path: Path, rows: list[dict[str, object]]) -> None:
    columns = (
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
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_review_caption_contains_every_required_burned_in_identifier() -> None:
    text = review_caption(_plate(), source_number=7)

    for value in ("plate-001", "aaaaaaaaaaaa", "#007", "(10,20,90,50)"):
        assert value in text


def test_stage_geometry_views_keeps_full_resolution_and_maps_rows_by_id(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (200, 120), "white").save(source)
    out = tmp_path / "views"

    records = stage_geometry_views([(_plate(), 7, source)], out, font=ImageFont.load_default())

    assert records == [
        {
            "plate_id": "plate-001",
            "sha256": "a" * 64,
            "source_number": 7,
            "coordinates": "10,20,90,50",
            "file": "plate-001.png",
        }
    ]
    view = Image.open(out / "plate-001.png").convert("RGB")
    assert view.size == (200, 120 + 46)
    reference = Image.new("RGB", view.size, "white")
    assert ImageChops.difference(view, reference).getbbox() is not None


def test_stage_geometry_views_rejects_duplicate_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (200, 120), "white").save(source)

    with pytest.raises(ValueError, match="дубликат plate_id"):
        stage_geometry_views(
            [(_plate(), 7, source), (_plate(), 8, source)],
            tmp_path / "views",
            font=ImageFont.load_default(),
        )


def test_stage_cli_reads_manifest_and_writes_every_fixture_view(tmp_path: Path) -> None:
    source_1 = tmp_path / "one.jpg"
    source_2 = tmp_path / "two.jpg"
    Image.new("RGB", (200, 120), "red").save(source_1)
    Image.new("RGB", (180, 100), "blue").save(source_2)
    manifest = tmp_path / "views-input.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "plate_id",
                "sha256",
                "x0",
                "y0",
                "x1",
                "y1",
                "readable",
                "verdict",
                "note",
                "source_number",
                "source_path",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "plate_id": "plate-002",
                    "sha256": "b" * 64,
                    "x0": 20,
                    "y0": 30,
                    "x1": 100,
                    "y1": 60,
                    "readable": "False",
                    "verdict": "ambiguous",
                    "note": "проверить",
                    "source_number": 8,
                    "source_path": source_2,
                },
                {
                    "plate_id": "plate-001",
                    "sha256": "a" * 64,
                    "x0": 10,
                    "y0": 20,
                    "x1": 90,
                    "y1": 50,
                    "readable": "True",
                    "verdict": "true_plate",
                    "note": "проверено",
                    "source_number": 7,
                    "source_path": source_1,
                },
            ]
        )

    from scripts.annotate_plate_geometry import main

    out = tmp_path / "views"
    assert main(["stage", "--manifest", str(manifest), "--out-dir", str(out)]) == 0
    rows = list(csv.DictReader((out / "views.csv").open(encoding="utf-8", newline="")))
    assert [row["plate_id"] for row in rows] == ["plate-001", "plate-002"]
    assert {path.name for path in out.glob("*.png")} == {"plate-001.png", "plate-002.png"}
    for row in rows:
        view = Image.open(out / row["file"]).convert("RGB")
        image_height = 120 if row["plate_id"] == "plate-001" else 100
        assert view.height == image_height + 46
        caption_strip = view.crop((0, 0, view.width, 46))
        assert (
            ImageChops.difference(
                caption_strip,
                Image.new("RGB", caption_strip.size, "white"),
            ).getbbox()
            is not None
        )


def test_non_geometry_actions_require_no_manual_coordinates(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "false_plate",
                "observation_id": "obs-001",
                "plate_id": "",
                "sha256": "",
                "x0": "",
                "y0": "",
                "x1": "",
                "y1": "",
                "readable": "",
                "note": "не номер",
            }
        ],
    )

    canonical, links = merge_review(review, [], [_observation()], [])

    assert canonical == []
    assert links == [
        DetectorLink(
            "link-8f4a991a5ce8e125",
            "obs-001",
            None,
            "false_plate",
        )
    ]


def test_merge_supports_create_correct_false_ambiguous_and_link(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "correct",
                "observation_id": "obs-001",
                "plate_id": "plate-001",
                "sha256": "a" * 64,
                "x0": 8,
                "y0": 18,
                "x1": 92,
                "y1": 52,
                "readable": "True",
                "note": "исправлен полный бокс",
            },
            {
                "action": "create",
                "observation_id": "obs-002",
                "plate_id": "plate-002",
                "sha256": "b" * 64,
                "x0": 20,
                "y0": 30,
                "x1": 100,
                "y1": 60,
                "readable": "False",
                "note": "добавлен вручную",
            },
            {
                "action": "false_plate",
                "observation_id": "obs-003",
                "plate_id": "plate-003",
                "sha256": "c" * 64,
                "x0": 30,
                "y0": 40,
                "x1": 110,
                "y1": 70,
                "readable": "False",
                "note": "не номер",
            },
            {
                "action": "ambiguous",
                "observation_id": "obs-004",
                "plate_id": "plate-004",
                "sha256": "d" * 64,
                "x0": 40,
                "y0": 50,
                "x1": 120,
                "y1": 80,
                "readable": "False",
                "note": "неоднозначно",
            },
            {
                "action": "link",
                "observation_id": "obs-005",
                "plate_id": "plate-001",
                "sha256": "",
                "x0": "",
                "y0": "",
                "x1": "",
                "y1": "",
                "readable": "",
                "note": "",
            },
        ],
    )

    canonical, links = merge_review(
        review,
        [_plate()],
        [
            _observation(
                f"obs-00{index}",
                sha256="a" * 64 if index == 5 else chr(96 + index) * 64,
            )
            for index in range(1, 6)
        ],
        [DetectorLink("link-auto", "obs-009", None, "partial_fill")],
    )

    assert [(plate.plate_id, plate.verdict) for plate in canonical] == [
        ("plate-001", "true_plate"),
        ("plate-002", "true_plate"),
        ("plate-003", "false_plate"),
        ("plate-004", "ambiguous"),
    ]
    assert canonical[0].x0 == 8
    assert [link.observation_id for link in links] == [
        "obs-001",
        "obs-002",
        "obs-003",
        "obs-004",
        "obs-005",
        "obs-009",
    ]
    assert next(link for link in links if link.observation_id == "obs-005").plate_id == "plate-001"


@pytest.mark.parametrize("action", ["create", "correct"])
def test_manual_create_and_correct_require_non_empty_note(tmp_path: Path, action: str) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": action,
                "observation_id": "obs-001",
                "plate_id": "plate-001" if action == "correct" else "plate-002",
                "sha256": "a" * 64,
                "x0": 10,
                "y0": 20,
                "x1": 90,
                "y1": 50,
                "readable": "True",
                "note": " ",
            }
        ],
    )

    with pytest.raises(ValueError, match="note"):
        merge_review(review, [_plate()], [_observation()], [])


def test_link_requires_plate_from_same_frame(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "link",
                "observation_id": "obs-001",
                "plate_id": "plate-001",
                "sha256": "",
                "x0": "",
                "y0": "",
                "x1": "",
                "y1": "",
                "readable": "",
                "note": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="разным кадрам"):
        merge_review(review, [_plate(sha256="b" * 64)], [_observation()], [])


def test_review_row_cannot_reuse_an_observation(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": action,
                "observation_id": "obs-001",
                "plate_id": plate_id,
                "sha256": "a" * 64,
                "x0": x0,
                "y0": 20,
                "x1": x1,
                "y1": 50,
                "readable": "True",
                "note": "проверено",
            }
            for action, plate_id, x0, x1 in (
                ("create", "plate-001", 10, 40),
                ("create", "plate-002", 50, 90),
            )
        ],
    )

    with pytest.raises(ValueError, match=r"observation_id.*повтор"):
        merge_review(review, [], [_observation()], [])


def test_merge_rejects_duplicate_ids_unknown_actions_and_links(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "create",
                "observation_id": "obs-001",
                "plate_id": "plate-001",
                "sha256": "a" * 64,
                "x0": 10,
                "y0": 20,
                "x1": 90,
                "y1": 50,
                "readable": "True",
                "note": "дубликат",
            }
        ],
    )
    with pytest.raises(ValueError, match="дубликат"):
        merge_review(review, [_plate()], [_observation()], [])

    text = review.read_text(encoding="utf-8").replace("create", "guess")
    review.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="action"):
        merge_review(review, [], [_observation()], [])


def test_correct_only_applies_to_true_plate(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "correct",
                "observation_id": "obs-001",
                "plate_id": "plate-001",
                "sha256": "a" * 64,
                "x0": 8,
                "y0": 18,
                "x1": 92,
                "y1": 52,
                "readable": "False",
                "note": "уточнён полный бокс",
            }
        ],
    )

    with pytest.raises(ValueError, match=r"correct.*true_plate"):
        merge_review(
            review,
            [_plate(verdict="ambiguous", note="неоднозначно")],
            [_observation()],
            [],
        )


def test_correct_cannot_move_existing_plate_to_another_frame(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "correct",
                "observation_id": "obs-001",
                "plate_id": "plate-001",
                "sha256": "a" * 64,
                "x0": 10,
                "y0": 20,
                "x1": 90,
                "y1": 50,
                "readable": "True",
                "note": "исправлена геометрия",
            }
        ],
    )

    with pytest.raises(ValueError, match=r"sha256.*существующ"):
        merge_review(review, [_plate(sha256="b" * 64)], [_observation()], [])


def test_merge_rejects_plate_sha_different_from_observation(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "create",
                "observation_id": "obs-001",
                "plate_id": "plate-002",
                "sha256": "b" * 64,
                "x0": 10,
                "y0": 20,
                "x1": 90,
                "y1": 50,
                "readable": "True",
                "note": "ручная разметка",
            }
        ],
    )

    with pytest.raises(ValueError, match=r"sha256.*наблюден"):
        merge_review(review, [], [_observation()], [])


def test_merge_rejects_overlapping_contradictory_truths(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    _write_review(
        review,
        [
            {
                "action": "false_plate",
                "observation_id": "obs-002",
                "plate_id": "plate-002",
                "sha256": "a" * 64,
                "x0": 20,
                "y0": 20,
                "x1": 80,
                "y1": 50,
                "readable": "False",
                "note": "ложный",
            }
        ],
    )

    with pytest.raises(ValueError, match="противореч"):
        merge_review(review, [_plate()], [_observation("obs-002")], [])


def test_cli_outputs_are_sorted_and_unresolved_have_reason(tmp_path: Path) -> None:
    from scripts.annotate_plate_geometry import write_outputs

    out = tmp_path / "out"
    write_outputs(
        out,
        [_plate("plate-002"), _plate("plate-001")],
        [
            DetectorLink("link-002", "obs-002", None, "partial_fill"),
            DetectorLink("link-001", "obs-001", "plate-001", None),
        ],
    )

    assert [plate.plate_id for plate in load_canonical_plates(out / "canonical-plates.csv")] == [
        "plate-001",
        "plate-002",
    ]
    unresolved = list(csv.DictReader((out / "unresolved.csv").open(encoding="utf-8", newline="")))
    assert unresolved == [
        {
            "link_id": "link-002",
            "observation_id": "obs-002",
            "plate_id": "",
            "reason": "partial_fill",
        }
    ]
