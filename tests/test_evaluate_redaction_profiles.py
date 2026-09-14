"""Поведенческие тесты campaign CLI выбора профилей редакции."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageFont

from pipeline.redaction_calibration.geometry import GeometryWitness
from pipeline.redaction_calibration.model import (
    ConfidenceMargin,
    ProfileGateReport,
    ProfileScore,
    VisualVeto,
)
from scripts.evaluate_redaction_profiles import (
    ChangedCrop,
    apply_visual_vetoes,
    load_visual_vetoes,
    main,
    stage_changed_crops,
    write_campaign_artifacts,
    write_visual_vetoes,
)


def _score(profile_id: str, *, changed: int = 1) -> ProfileScore:
    return ProfileScore(
        profile_id=profile_id,
        ssd_margin=ConfidenceMargin(0.10, 0.20),
        yolo_margin=ConfidenceMargin(0.15, 0.25),
        median_delta=0.01,
        p95_delta=0.05,
        above_quarter=0,
        new_over_redacted=0,
        changed_crops=changed,
    )


def _gate_report(
    profile_id: str,
    *,
    canonical_sha256: str = "c" * 64,
    w_max: int = 160,
    h_max: int = 50,
) -> ProfileGateReport:
    return ProfileGateReport(
        profile_id=profile_id,
        provenance_fresh=True,
        unresolved_count=0,
        geometry_current=True,
        geometry_preflight_passed=True,
        canonical_coverage_complete=True,
        strict_leaks=0,
        unreviewable_accepted=0,
        report_arithmetic_closed=True,
        canonical_sha256=canonical_sha256,
        w_max=w_max,
        h_max=h_max,
    )


def test_stage_changed_crops_пишет_все_изменения_с_burned_in_id(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (180, 100), "red").save(first)
    Image.new("RGB", (200, 120), "blue").save(second)
    out = tmp_path / "views"

    rows = stage_changed_crops(
        [
            ChangedCrop("profile-b", "frame-002", "b" * 64, second),
            ChangedCrop("profile-a", "frame-001", "a" * 64, first),
        ],
        out,
        font=ImageFont.load_default(),
    )

    assert rows == [
        {
            "profile_id": "profile-a",
            "frame_id": "frame-001",
            "sha256": "a" * 64,
            "file": "profile-a__frame-001.png",
        },
        {
            "profile_id": "profile-b",
            "frame_id": "frame-002",
            "sha256": "b" * 64,
            "file": "profile-b__frame-002.png",
        },
    ]
    for row, source in zip(rows, (first, second), strict=True):
        view = Image.open(out / row["file"]).convert("RGB")
        original = Image.open(source).convert("RGB")
        assert view.size == (original.width, original.height + 46)
        blank = Image.new("RGB", view.size, "white")
        assert ImageChops.difference(view, blank).getbbox() is not None
        pixels = np.asarray(view)
        assert np.unique(pixels[-46:], axis=0).size > 1

    with (out / "views.csv").open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == rows

    with pytest.raises(ValueError, match="calibration 300"):
        stage_changed_crops(
            [ChangedCrop("profile-a", "frame-000", "a" * 64, first)],
            tmp_path / "incomplete",
            font=ImageFont.load_default(),
            expected_changed_ids={"profile-a": {"frame-000", "frame-001"}},
        )


def test_visual_veto_требует_уничтоженный_признак_и_некосметическую_причину(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vetoes.csv"

    with pytest.raises(ValueError, match="destroyed_feature"):
        write_visual_vetoes(
            path,
            [VisualVeto("profile-a", "a" * 64, None, "", "номер стал нечитаем")],
        )
    with pytest.raises(ValueError, match="некосмет"):
        write_visual_vetoes(
            path,
            [VisualVeto("profile-a", "a" * 64, None, "badge", "косметика")],
        )
    write_visual_vetoes(
        path,
        [VisualVeto("profile-a", "a" * 64, None, "badge", "заливка скрыла эмблему")],
    )


def test_veto_только_удаляет_профиль_и_ledger_roundtrip_строгий(tmp_path: Path) -> None:
    path = tmp_path / "vetoes.csv"
    vetoes = [
        VisualVeto(
            "profile-b",
            "b" * 64,
            "plate-002",
            "manufacturer_badge",
            "заливка уничтожила обязательный признак модели",
        )
    ]
    write_visual_vetoes(path, vetoes)

    assert load_visual_vetoes(path) == vetoes
    assert apply_visual_vetoes([_score("profile-a"), _score("profile-b")], vetoes) == (
        _score("profile-a"),
    )
    with pytest.raises(ValueError, match="frontier"):
        apply_visual_vetoes([_score("profile-a")], vetoes)


def test_campaign_artifacts_не_автовыбирают_из_полного_frontier(tmp_path: Path) -> None:
    frontier = (_score("profile-a"), _score("profile-b"))

    write_campaign_artifacts(tmp_path, frontier, unresolved_rows=[], vetoes=[])

    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    selection = json.loads((tmp_path / "selection_required.json").read_text(encoding="utf-8"))
    assert pareto == [asdict(score) for score in frontier]
    assert selection == {"frontier": [asdict(score) for score in frontier]}
    assert not (tmp_path / "selected_profile.json").exists()
    unresolved_text = (tmp_path / "unresolved.csv").read_text(encoding="utf-8")
    assert unresolved_text == "link_id,observation_id,plate_id,reason\n"
    assert (tmp_path / "vetoes.csv").is_file()


def test_single_survivor_пишет_явный_результат_без_selection_required(tmp_path: Path) -> None:
    write_campaign_artifacts(tmp_path, [_score("profile-a")], unresolved_rows=[], vetoes=[])

    assert json.loads((tmp_path / "selected_profile.json").read_text(encoding="utf-8")) == {
        "profile": asdict(_score("profile-a"))
    }
    assert not (tmp_path / "selection_required.json").exists()


def test_cli_пишет_candidates_frontier_stale_и_changed_views(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "report": asdict(_gate_report("profile-fresh")),
                    "score": asdict(_score("profile-fresh")),
                },
                {
                    "report": asdict(_gate_report("profile-stale", canonical_sha256="d" * 64)),
                },
            ]
        ),
        encoding="utf-8",
    )
    witness = tmp_path / "witness.json"
    witness.write_text(
        json.dumps(asdict(GeometryWitness(160, 50, "c" * 64))),
        encoding="utf-8",
    )
    unresolved = tmp_path / "unresolved.csv"
    unresolved.write_text("link_id,observation_id,plate_id,reason\n", encoding="utf-8")
    vetoes = tmp_path / "vetoes.csv"
    write_visual_vetoes(vetoes, [])
    source = tmp_path / "changed.jpg"
    Image.new("RGB", (180, 100), "red").save(source)
    changed = tmp_path / "changed.csv"
    with changed.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("profile_id", "frame_id", "sha256", "source"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "profile_id": "profile-fresh",
                "frame_id": "frame-001",
                "sha256": "a" * 64,
                "source": source,
            }
        )
    out = tmp_path / "campaign"
    views = tmp_path / "views"

    assert (
        main(
            [
                "--candidates",
                str(candidates),
                "--current-witness",
                str(witness),
                "--unresolved",
                str(unresolved),
                "--vetoes",
                str(vetoes),
                "--changed-crops",
                str(changed),
                "--views",
                str(views),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    candidate_rows = json.loads((out / "candidate_profiles.json").read_text(encoding="utf-8"))
    assert [row["profile_id"] for row in candidate_rows] == ["profile-fresh", "profile-stale"]
    assert candidate_rows[0]["failed_gate"] is None
    assert candidate_rows[1]["failed_gate"] == "current_geometry_witness"
    assert json.loads((out / "geometry_stale.json").read_text(encoding="utf-8")) == {
        "profile_ids": ["profile-stale"]
    }
    assert json.loads((out / "pareto.json").read_text(encoding="utf-8")) == [
        asdict(_score("profile-fresh"))
    ]
    assert (views / "profile-fresh__frame-001.png").is_file()


def test_cli_отвергает_неполный_staging_changed_crops(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "report": asdict(_gate_report("profile-a")),
                    "score": asdict(_score("profile-a", changed=2)),
                }
            ]
        ),
        encoding="utf-8",
    )
    witness = tmp_path / "witness.json"
    witness.write_text(
        json.dumps(asdict(GeometryWitness(160, 50, "c" * 64))),
        encoding="utf-8",
    )
    unresolved = tmp_path / "unresolved.csv"
    unresolved.write_text("link_id,observation_id,plate_id,reason\n", encoding="utf-8")
    vetoes = tmp_path / "vetoes.csv"
    write_visual_vetoes(vetoes, [])
    source = tmp_path / "changed.jpg"
    Image.new("RGB", (180, 100), "red").save(source)
    changed = tmp_path / "changed.csv"
    changed.write_text(
        f"profile_id,frame_id,sha256,source\nprofile-a,frame-001,{'a' * 64},{source}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed_crops"):
        main(
            [
                "--candidates",
                str(candidates),
                "--current-witness",
                str(witness),
                "--unresolved",
                str(unresolved),
                "--vetoes",
                str(vetoes),
                "--changed-crops",
                str(changed),
                "--views",
                str(tmp_path / "views"),
                "--out",
                str(tmp_path / "campaign"),
            ]
        )


def test_campaign_не_объявляет_annotations_resolved_с_unresolved(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unresolved"):
        write_campaign_artifacts(
            tmp_path,
            [_score("profile-a")],
            unresolved_rows=[
                {
                    "link_id": "link-001",
                    "observation_id": "obs-001",
                    "plate_id": "",
                    "reason": "ambiguous_match",
                }
            ],
            vetoes=[],
        )


def test_gate_report_parses_oversized_blind_zone_and_reason() -> None:
    from scripts.evaluate_redaction_profiles import _gate_report

    doc = {
        "profile_id": "prof-1",
        "provenance_fresh": True,
        "unresolved_count": 0,
        "geometry_current": True,
        "geometry_preflight_passed": True,
        "canonical_coverage_complete": True,
        "strict_leaks": 0,
        "unreviewable_accepted": 0,
        "report_arithmetic_closed": True,
        "oversized_blind_zone_count": 5,
        "geometry_preflight_reason": "oversized_uncovered: p-1",
    }
    report = _gate_report(doc)
    assert report.oversized_blind_zone_count == 5
    assert report.geometry_preflight_reason == "oversized_uncovered: p-1"

    doc_legacy = {
        k: v
        for k, v in doc.items()
        if k not in ("oversized_blind_zone_count", "geometry_preflight_reason")
    }
    legacy_report = _gate_report(doc_legacy)
    assert legacy_report.oversized_blind_zone_count == 0
    assert legacy_report.geometry_preflight_reason is None
