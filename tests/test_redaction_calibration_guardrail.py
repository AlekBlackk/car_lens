"""Quality guardrail на замороженной few-shot линейке после выбора профиля."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.manifest import HarvestRow
from pipeline.redaction_calibration.baseline import FrozenUnit, Prediction
from pipeline.redaction_calibration.guardrail import (
    GUARDRAIL_SUCCESS_PHRASE,
    assert_no_frozen_units,
    build_guardrail_holdout,
    build_guardrail_pool,
    calculate_guardrail_report,
    render_guardrail_report,
)
from pipeline.schema import read_split, write_split
from scripts.eval_redaction_guardrail import GuardrailPaths, run_guardrail


def _manifest_row(
    photo_id: str,
    *,
    file: str,
    class_id: str = "lada__2107",
    source_item_id: str = "newly-computed-unit",
    phash: int = 99,
) -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id=source_item_id,
        page_url=f"https://example.test/{photo_id}",
        file_url=f"https://example.test/{file}",
        author="Автор",
        license="CC BY 4.0",
        license_tier=1,
        checked="2026-09-12",
        basis="fixture",
        sha256=photo_id,
        phash=phash,
        clip_rank=1,
        clip_margin=0.5,
        plates=0,
        faces=0,
        bbox_x=1.0,
        bbox_y=2.0,
        bbox_w=30.0,
        bbox_h=20.0,
        bbox_area_frac=0.4,
        det_conf=0.9,
        audit="ok",
    )


def _frozen_units() -> tuple[FrozenUnit, ...]:
    return (
        FrozenUnit("holdout", "original-unit", "holdout_ru"),
        FrozenUnit("sibling", "original-unit", "pool"),
        FrozenUnit("survivor", "surviving-unit", "pool"),
        FrozenUnit("surviving-holdout", "holdout-unit", "holdout_ru"),
    )


def test_guardrail_pool_excludes_entire_frozen_holdout_unit_and_keeps_authority() -> None:
    """Пересчёт pHash/unit не должен вернуть sibling holdout-кадра в центроиды."""
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    rows = [
        _manifest_row("holdout", file="holdout-new.jpg", phash=111),
        _manifest_row("sibling", file="sibling-new.jpg", phash=222),
        _manifest_row("survivor", file="survivor-new.jpg", phash=333),
    ]

    pool = build_guardrail_pool(rows, _frozen_units(), {"holdout", "surviving-holdout"}, catalog)

    assert pool["photo_id"].tolist() == ["survivor"]
    assert pool["listing_id"].tolist() == ["surviving-unit"]
    assert pool["crop_path"].tolist() == ["lada__2107/survivor-new.jpg"]
    assert pool["phash"].tolist() == [333]
    assert pool["class_idx"].tolist() == [
        load_catalog(DEFAULT_CATALOG_PATH).classes.index(
            next(row for row in catalog.classes if row.class_id == "lada__2107")
        )
    ]


def test_guardrail_pool_rejects_photo_absent_from_frozen_map() -> None:
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    unknown = _manifest_row("unknown", file="unknown.jpg")

    with pytest.raises(ValueError, match=r"неизвестные photo_id.*unknown"):
        build_guardrail_pool([unknown], _frozen_units(), {"holdout"}, catalog)


def test_guardrail_holdout_uses_surviving_new_crops_and_original_listing_ids() -> None:
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    row = _manifest_row("surviving-holdout", file="holdout-rebuilt.jpg", phash=444)

    holdout = build_guardrail_holdout(
        [row], _frozen_units(), {"holdout", "surviving-holdout"}, catalog
    )

    assert holdout["photo_id"].tolist() == ["surviving-holdout"]
    assert holdout["listing_id"].tolist() == ["holdout-unit"]
    assert holdout["crop_path"].tolist() == ["lada__2107/holdout-rebuilt.jpg"]


def test_final_parquet_check_rejects_deliberate_frozen_unit_contamination(tmp_path: Path) -> None:
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    clean = build_guardrail_pool(
        [_manifest_row("survivor", file="survivor.jpg")],
        _frozen_units(),
        {"holdout"},
        catalog,
    )
    contaminated = pd.concat(
        [clean, clean.assign(photo_id="intruder", listing_id="original-unit")],
        ignore_index=True,
    )
    path = tmp_path / "guardrail-pool.parquet"
    write_split(contaminated, path)

    final_pool = read_split(path)
    with pytest.raises(ValueError, match=r"замороженными holdout-единицами.*original-unit"):
        assert_no_frozen_units(final_pool, _frozen_units())


def test_guardrail_metrics_use_exact_baseline_and_penalize_destroyed_frames() -> None:
    baseline = (
        Prediction("a", 1, (1, 2, 3)),
        Prediction("b", 2, (4, 2, 0)),
        Prediction("c", 3, (0, 1, 2)),
        Prediction("d", 4, (4, 5, 6)),
    )
    candidate = (
        Prediction("a", 1, (1, 2, 3)),
        Prediction("b", 2, (2, 4, 0)),
        Prediction("d", 4, (0, 4, 6)),
    )

    report = calculate_guardrail_report(
        baseline,
        candidate,
        pool_sha256="a" * 64,
        split_sha256="b" * 64,
        index_sha256="c" * 64,
    )

    assert (report.top1.baseline_k, report.top1.candidate_k_all) == (2, 2)
    assert (report.top3.baseline_k, report.top3.candidate_k_all) == (3, 3)
    assert report.top1.survivors == report.top3.survivors == 3
    assert report.top1.destroyed == report.top3.destroyed == 1
    assert report.top1.candidate_k_survivors == 2
    assert report.top3.candidate_k_survivors == 3
    assert report.top1.candidate_k_survivors / report.top1.survivors == pytest.approx(2 / 3)
    assert report.top3.candidate_k_survivors / report.top3.survivors == pytest.approx(1.0)
    assert report.top1.lower_bound == pytest.approx(0.0675859865)
    assert report.top3.lower_bound == pytest.approx(0.1941204497)
    assert report.top1.passed is True
    assert report.top3.passed is True


def test_guardrail_fails_when_one_penalized_axis_is_below_exact_bound() -> None:
    baseline = tuple(Prediction(str(index), 1, (1, 2, 3)) for index in range(20))
    candidate = tuple(Prediction(str(index), 1, (1, 2, 3)) for index in range(15))
    candidate += tuple(Prediction(str(index), 1, (2, 3, 1)) for index in range(15, 20))

    report = calculate_guardrail_report(
        baseline,
        candidate,
        pool_sha256="a" * 64,
        split_sha256="b" * 64,
        index_sha256="c" * 64,
    )

    assert report.top1.candidate_k_all == 15
    assert report.top1.lower_bound == pytest.approx(0.831566529)
    assert report.top1.passed is False
    assert report.top3.passed is True


def test_passing_report_claims_not_collapsed_but_never_not_worse() -> None:
    predictions = (Prediction("a", 1, (1, 2, 3)),)
    report = calculate_guardrail_report(
        predictions,
        predictions,
        pool_sha256="a" * 64,
        split_sha256="b" * 64,
        index_sha256="c" * 64,
    )

    rendered = render_guardrail_report(report)

    assert GUARDRAIL_SUCCESS_PHRASE in rendered
    assert "guardrail не обнаружил обвала; это не доказательство отсутствия ухудшения" in rendered
    assert "not_worse" not in rendered
    assert "не хуже" not in rendered
    assert "эффект профиля при постоянной линейке из 1523 кадров" in rendered


def test_candidate_prediction_target_must_match_frozen_baseline() -> None:
    baseline = (Prediction("a", 1, (1, 2, 3)),)
    candidate = (Prediction("a", 2, (1, 2, 3)),)

    with pytest.raises(ValueError, match=r"target.*a"):
        calculate_guardrail_report(
            baseline,
            candidate,
            pool_sha256="a" * 64,
            split_sha256="b" * 64,
            index_sha256="c" * 64,
        )


def test_guardrail_cli_checks_final_parquet_before_building_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen_units()
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    clean = build_guardrail_pool(
        [_manifest_row("survivor", file="survivor.jpg")], frozen, {"holdout"}, catalog
    )
    contaminated = pd.concat(
        [clean, clean.assign(photo_id="intruder", listing_id="original-unit")],
        ignore_index=True,
    )
    paths = GuardrailPaths.for_fixture(tmp_path)
    write_split(contaminated, paths.pool)
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.prepare_guardrail_splits",
        lambda arguments: (frozen, paths),
    )
    encoder_started = False

    def forbidden_build(*arguments: object) -> None:
        nonlocal encoder_started
        encoder_started = True

    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_build_index", forbidden_build)

    with pytest.raises(ValueError, match="замороженными holdout-единицами"):
        run_guardrail(argparse.Namespace())
    assert encoder_started is False


def test_guardrail_cli_writes_report_predictions_and_input_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = GuardrailPaths.for_fixture(tmp_path)
    write_split(
        build_guardrail_pool(
            [_manifest_row("survivor", file="survivor.jpg")],
            _frozen_units(),
            {"holdout"},
            load_catalog(DEFAULT_CATALOG_PATH),
        ),
        paths.pool,
    )
    write_split(
        build_guardrail_holdout(
            [_manifest_row("surviving-holdout", file="holdout.jpg")],
            _frozen_units(),
            {"holdout", "surviving-holdout"},
            load_catalog(DEFAULT_CATALOG_PATH),
        ),
        paths.holdout,
    )
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    paths.predictions.write_text("photo_id,target,top_indices\na,1,1 2 3\n", encoding="utf-8")
    paths.index.mkdir(parents=True)
    (paths.index / "catalog_index.bin").write_bytes(b"index")
    baseline = (Prediction("a", 1, (1, 2, 3)),)
    candidate = (Prediction("a", 1, (1, 2, 3)),)
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.prepare_guardrail_splits",
        lambda arguments: (_frozen_units(), paths),
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.assert_no_frozen_units", lambda *args: None
    )
    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_build_index", lambda *args: None)
    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_eval", lambda *args: None)
    monkeypatch.setattr("scripts.eval_redaction_guardrail.load_predictions", lambda path: candidate)
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.load_baseline_predictions", lambda _: baseline
    )

    result = run_guardrail(argparse.Namespace(baseline_predictions=tmp_path / "base.csv"))

    assert result == 0
    assert paths.predictions.is_file()
    report = paths.report.read_text(encoding="utf-8")
    assert GUARDRAIL_SUCCESS_PHRASE in report
    assert "sha256 guardrail pool" in report
    assert "sha256 survivor split" in report
    assert "sha256 index" in report
