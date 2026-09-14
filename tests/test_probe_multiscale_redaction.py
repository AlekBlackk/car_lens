"""Тесты модуля раннего зонда мультимасштабной редакции номеров."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pipeline.harvest.cache import cache_path
from pipeline.harvest.manifest import HarvestRow, write_manifest
from scripts.probe_multiscale_redaction import (
    CandidateMetrics,
    create_diagnostics_for_candidate,
    make_probe_candidates,
    parse_candidate_report,
    render_diagnostic_frame,
    select_probe_sample,
)


def test_make_probe_candidates_creates_four_valid_profiles() -> None:
    candidates = make_probe_candidates()
    assert set(candidates.keys()) == {
        "baseline",
        "yolo_1280_640",
        "yolo_1280_640_320",
        "yolo_1280_640_ssd_1280",
    }

    # baseline: SSD 1280 (0.50) + YOLO 1280 (0.25)
    baseline = candidates["baseline"]
    assert len(baseline.detectors) == 2
    assert baseline.detectors[0].instance == "plate"
    assert baseline.detectors[0].weights == "plate"
    assert baseline.detectors[0].params.confidence == 0.50
    assert baseline.detectors[0].params.tile == 1280

    assert baseline.detectors[1].instance == "plate_hires"
    assert baseline.detectors[1].weights == "plate_hires"
    assert baseline.detectors[1].params.confidence == 0.25
    assert baseline.detectors[1].params.tile == 1280

    # yolo_1280_640
    c2 = candidates["yolo_1280_640"]
    assert len(c2.detectors) == 2
    assert [d.instance for d in c2.detectors] == ["yolo_1280", "yolo_640"]
    assert all(d.weights == "plate_hires" for d in c2.detectors)
    assert all(d.params.confidence == 0.25 for d in c2.detectors)
    assert c2["yolo_1280"].params.tile == 1280
    assert c2["yolo_640"].params.tile == 640

    # yolo_1280_640_320
    c3 = candidates["yolo_1280_640_320"]
    assert len(c3.detectors) == 3
    assert [d.instance for d in c3.detectors] == ["yolo_1280", "yolo_320", "yolo_640"]
    assert all(d.weights == "plate_hires" for d in c3.detectors)
    assert all(d.params.confidence == 0.25 for d in c3.detectors)
    assert c3["yolo_1280"].params.tile == 1280
    assert c3["yolo_640"].params.tile == 640
    assert c3["yolo_320"].params.tile == 320

    # yolo_1280_640_ssd_1280
    c4 = candidates["yolo_1280_640_ssd_1280"]
    assert len(c4.detectors) == 3
    assert [d.instance for d in c4.detectors] == ["ssd_1280", "yolo_1280", "yolo_640"]
    assert c4["ssd_1280"].weights == "plate"
    assert c4["ssd_1280"].params.confidence == 0.50
    assert c4["ssd_1280"].params.tile == 1280
    assert c4["yolo_1280"].weights == "plate_hires"
    assert c4["yolo_1280"].params.confidence == 0.25
    assert c4["yolo_1280"].params.tile == 1280
    assert c4["yolo_640"].weights == "plate_hires"
    assert c4["yolo_640"].params.confidence == 0.25
    assert c4["yolo_640"].params.tile == 640


def _make_row(sha: str) -> HarvestRow:
    return HarvestRow(
        file=f"{sha}.jpg",
        class_id="gaz__gazel",
        source="commons",
        source_item_id="123",
        page_url="https://example.invalid/page",
        file_url="https://example.invalid/file",
        author="test",
        license="cc0",
        license_tier=1,
        checked="2026-09-14",
        basis="category:test",
        sha256=sha,
        phash=12345,
        clip_rank=1,
        clip_margin=0.0,
        plates=1,
        faces=0,
        bbox_x=10.0,
        bbox_y=10.0,
        bbox_w=100.0,
        bbox_h=100.0,
        bbox_area_frac=0.5,
        det_conf=0.9,
        audit="ok",
    )


def test_select_probe_sample_excludes_calibration_and_canonical(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    calibration_path = tmp_path / "calibration.csv"
    canonical_path = tmp_path / "canonical.csv"

    # Создаём 20 строк манифеста с hex-хешами
    rows = [_make_row(f"{i:02x}" * 32) for i in range(20)]
    write_manifest(manifest_path, rows)

    # Исключим хеш 0 в калибровке и хеш 1 в канонике
    hash0 = f"{0:02x}" * 32
    hash1 = f"{1:02x}" * 32
    calibration_path.write_text(f"номер,sha256\n1,{hash0}\n", encoding="utf-8")
    canonical_path.write_text(f"plate_id,sha256\nL1,{hash1}\n", encoding="utf-8")

    sampled, excluded = select_probe_sample(
        manifest_path=manifest_path,
        calibration_sample_path=calibration_path,
        canonical_path=canonical_path,
        seed=42,
        sample_size=10,
    )

    assert len(sampled) == 10
    assert excluded == {hash0, hash1}
    sampled_hashes = {r.sha256 for r in sampled}
    assert hash0 not in sampled_hashes
    assert hash1 not in sampled_hashes

    # Проверка детерминизма сида
    sampled2, _ = select_probe_sample(
        manifest_path=manifest_path,
        calibration_sample_path=calibration_path,
        canonical_path=canonical_path,
        seed=42,
        sample_size=10,
    )
    assert [r.sha256 for r in sampled] == [r.sha256 for r in sampled2]


def test_parse_candidate_report(tmp_path: Path) -> None:
    report_file = tmp_path / "report.json"
    report_data = {
        "counts": {"input": 500, "accepted": 480, "rejected": 20},
        "reject_reasons": {"redaction_failed": 15, "multi_car": 5},
    }
    report_file.write_text(json.dumps(report_data), encoding="utf-8")

    metrics: CandidateMetrics = parse_candidate_report("test_cand", report_file, budget=0.038)
    assert metrics.name == "test_cand"
    assert metrics.input_count == 500
    assert metrics.accepted_count == 480
    assert metrics.rejected_count == 20
    assert metrics.redaction_failed_count == 15
    assert pytest.approx(metrics.redaction_failed_share, 1e-4) == 15 / 500  # 0.03
    # 95% CI для 15/500: [0.0169, 0.0490], нижняя граница 0.0169 < 0.038 => exceeds_budget = False
    assert metrics.ci_low < 0.038
    assert not metrics.exceeds_budget

    # Теперь случай превышения бюджета
    report_data2 = {
        "counts": {"input": 500, "accepted": 400, "rejected": 100},
        "reject_reasons": {"redaction_failed": 90, "multi_car": 10},
    }
    report_file.write_text(json.dumps(report_data2), encoding="utf-8")
    metrics2 = parse_candidate_report("failing_cand", report_file, budget=0.038)
    assert metrics2.redaction_failed_count == 90
    assert metrics2.ci_low > 0.038
    assert metrics2.exceeds_budget


def test_render_diagnostic_frame(tmp_path: Path) -> None:
    raw_image = np.zeros((100, 120, 3), dtype=np.uint8)
    out_file = tmp_path / "diag.jpg"
    obs = [
        {"pass": "first", "merged": True, "x": 10.0, "y": 10.0, "w": 30.0, "h": 15.0},
        {
            "pass": "second",
            "merged": True,
            "x": 50.0,
            "y": 50.0,
            "w": 20.0,
            "h": 10.0,
            "confidence": 0.45,
            "detector": "yolo_640",
        },
    ]
    render_diagnostic_frame(raw_image, "photo123", "cand_test", obs, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_create_diagnostics_for_candidate(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    harvest_data_dir = tmp_path / "harvest"
    harvest_data_dir.mkdir()
    diagnostics_dir = tmp_path / "diagnostics"

    photo_id = "a" * 64
    source = "commons"

    # Создаём dummy rejects.csv
    rejects_file = candidate_dir / "rejects.csv"
    rejects_df = pd.DataFrame(
        [
            {
                "file": f"{photo_id}.jpg",
                "class_id": "test_car",
                "source": source,
                "source_item_id": "123",
                "page_url": "http://example.invalid",
                "file_url": "http://example.invalid",
                "author": "author",
                "license": "cc0",
                "license_tier": 1,
                "checked": "2026-09-14",
                "basis": "cat",
                "sha256": photo_id,
                "phash": 0,
                "clip_rank": 1,
                "clip_margin": 0.0,
                "plates": 1,
                "faces": 0,
                "bbox_x": 0.0,
                "bbox_y": 0.0,
                "bbox_w": 10.0,
                "bbox_h": 10.0,
                "bbox_area_frac": 0.1,
                "det_conf": 0.9,
                "audit": "",
                "reason": "redaction_failed",
            }
        ]
    )
    rejects_df.to_csv(rejects_file, index=False)

    # Создаём dummy observations.csv
    obs_file = candidate_dir / "observations.csv"
    obs_df = pd.DataFrame(
        [
            {
                "photo_id": photo_id,
                "detector": "plate",
                "profile": "prof",
                "zone_x0": 0,
                "zone_y0": 0,
                "zone_x1": 100,
                "zone_y1": 100,
                "tile_x0": 0,
                "tile_y0": 0,
                "tile_x1": 100,
                "tile_y1": 100,
                "x": 10.0,
                "y": 10.0,
                "w": 30.0,
                "h": 15.0,
                "confidence": 0.85,
                "class_id": 0,
                "raw": False,
                "merged": True,
                "pass": "first",
            },
            {
                "photo_id": photo_id,
                "detector": "yolo_640",
                "profile": "prof",
                "zone_x0": 0,
                "zone_y0": 0,
                "zone_x1": 100,
                "zone_y1": 100,
                "tile_x0": 0,
                "tile_y0": 0,
                "tile_x1": 100,
                "tile_y1": 100,
                "x": 50.0,
                "y": 50.0,
                "w": 20.0,
                "h": 10.0,
                "confidence": 0.45,
                "class_id": 0,
                "raw": False,
                "merged": True,
                "pass": "second",
            },
        ]
    )
    obs_df.to_csv(obs_file, index=False)

    # Создаём синтетическое изображение в кэше харвеста
    img_cache_path = cache_path(harvest_data_dir, source, photo_id, "jpg")
    img_cache_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", synthetic_img)
    assert ok
    img_cache_path.write_bytes(buf.tobytes())

    inspected_ids = create_diagnostics_for_candidate(
        candidate_name="cand_test",
        candidate_dir=candidate_dir,
        harvest_data_dir=harvest_data_dir,
        out_diagnostics_dir=diagnostics_dir,
        max_frames=10,
    )

    assert inspected_ids == [photo_id]
    expected_file = diagnostics_dir / f"cand_test_{photo_id[:12]}.jpg"
    assert expected_file.exists()
    assert expected_file.stat().st_size > 0
