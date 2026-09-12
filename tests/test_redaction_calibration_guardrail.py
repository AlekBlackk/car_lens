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
    validate_guardrail_lineage,
)
from pipeline.schema import empty_frame, read_split, write_split
from scripts.eval_redaction_guardrail import (
    SURVIVOR_REPORT_NAME,
    GuardrailPaths,
    build_parser,
    run_eval,
    run_guardrail,
)


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


@pytest.mark.parametrize(
    "unsafe_file",
    [
        "../../../other.jpg",
        "/tmp/other.jpg",
        "..\\..\\other.jpg",
        "\\rooted.jpg",
        "\\\\server\\share\\other.jpg",
        "C:\\tmp\\other.jpg",
        "C:other.jpg",
        "other.jpg:stream",
        "nested/other.jpg",
        "nested\\other.jpg",
        "",
        ".",
        "..",
    ],
)
def test_guardrail_split_rejects_unsafe_manifest_filename(unsafe_file: str) -> None:
    row = _manifest_row("survivor", file=unsafe_file)

    with pytest.raises(ValueError, match=r"небезопасное имя кропа"):
        build_guardrail_pool(
            [row],
            _frozen_units(),
            {"holdout", "surviving-holdout"},
            load_catalog(DEFAULT_CATALOG_PATH),
        )


def test_guardrail_split_accepts_single_filename_segment() -> None:
    pool = build_guardrail_pool(
        [_manifest_row("survivor", file="valid crop-01.jpg")],
        _frozen_units(),
        {"holdout", "surviving-holdout"},
        load_catalog(DEFAULT_CATALOG_PATH),
    )

    assert pool["crop_path"].tolist() == ["lada__2107/valid crop-01.jpg"]


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


def test_guardrail_prediction_boundaries_reject_duplicates_and_foreign_candidate() -> None:
    baseline = (
        Prediction("a", 1, (1, 2, 3)),
        Prediction("a", 1, (1, 2, 3)),
    )
    with pytest.raises(ValueError, match=r"дубликат photo_id в baseline.*a"):
        calculate_guardrail_report(
            baseline,
            (),
            pool_sha256="a" * 64,
            split_sha256="b" * 64,
            index_sha256="c" * 64,
        )

    clean = (Prediction("a", 1, (1, 2, 3)),)
    duplicate_candidate = (clean[0], clean[0])
    with pytest.raises(ValueError, match=r"дубликат photo_id в candidate.*a"):
        calculate_guardrail_report(
            clean,
            duplicate_candidate,
            pool_sha256="a" * 64,
            split_sha256="b" * 64,
            index_sha256="c" * 64,
        )

    with pytest.raises(ValueError, match=r"неизвестные candidate photo_id.*foreign"):
        calculate_guardrail_report(
            clean,
            (Prediction("foreign", 1, (1, 2, 3)),),
            pool_sha256="a" * 64,
            split_sha256="b" * 64,
            index_sha256="c" * 64,
        )


def test_lineage_requires_exact_frozen_holdout_ids_and_production_count() -> None:
    frozen = (
        FrozenUnit("a", "unit-a", "holdout_ru"),
        FrozenUnit("b", "unit-b", "holdout_ru"),
        FrozenUnit("pool", "unit-p", "pool"),
    )
    baseline = (
        Prediction("a", 1, (1, 2, 3)),
        Prediction("b", 2, (2, 1, 3)),
    )

    validate_guardrail_lineage(baseline, frozen, expected_frames=2)
    with pytest.raises(ValueError, match=r"baseline.*missing.*b"):
        validate_guardrail_lineage(baseline[:1], frozen, expected_frames=2)
    with pytest.raises(ValueError, match=r"baseline.*extra.*foreign"):
        validate_guardrail_lineage(
            (*baseline, Prediction("foreign", 1, (1, 2, 3))),
            frozen,
            expected_frames=2,
        )
    with pytest.raises(ValueError, match=r"дубликат photo_id в baseline.*a"):
        validate_guardrail_lineage((baseline[0], baseline[0]), frozen, expected_frames=2)
    with pytest.raises(ValueError, match=r"ровно 1523.*2"):
        validate_guardrail_lineage(baseline, frozen)


def test_guardrail_cli_rejects_truncated_baseline_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = GuardrailPaths.for_fixture(tmp_path)
    write_split(empty_frame(), paths.pool)
    write_split(empty_frame(), paths.holdout)
    frozen = tuple(FrozenUnit(str(index), f"unit-{index}", "holdout_ru") for index in range(1523))
    truncated = tuple(Prediction(str(index), 1, (1, 2, 3)) for index in range(1000))
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.prepare_guardrail_splits",
        lambda arguments: (frozen, paths),
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.validate_guardrail_paths", lambda _: paths
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.load_baseline_predictions", lambda path: truncated
    )
    evaluation_started = False

    def forbidden_eval(*arguments: object) -> None:
        nonlocal evaluation_started
        evaluation_started = True

    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_eval", forbidden_eval)

    with pytest.raises(ValueError, match=r"baseline.*missing"):
        run_guardrail(argparse.Namespace(baseline_predictions=tmp_path / "baseline.csv"))
    assert evaluation_started is False


def test_guardrail_all_destroyed_writes_header_only_predictions_and_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = GuardrailPaths.for_fixture(tmp_path)
    write_split(empty_frame(), paths.pool)
    write_split(empty_frame(), paths.holdout)
    paths.index.mkdir(parents=True)
    (paths.index / "catalog_index.bin").write_bytes(b"index")
    frozen = tuple(FrozenUnit(str(index), f"unit-{index}", "holdout_ru") for index in range(1523))
    baseline = tuple(Prediction(str(index), 1, (1, 2, 3)) for index in range(1523))
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.prepare_guardrail_splits",
        lambda arguments: (frozen, paths),
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.validate_guardrail_paths", lambda _: paths
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.load_baseline_predictions", lambda path: baseline
    )
    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_build_index", lambda *args: None)
    encoder_started = False

    def forbidden_eval(*arguments: object) -> None:
        nonlocal encoder_started
        encoder_started = True

    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_eval", forbidden_eval)

    result = run_guardrail(argparse.Namespace(baseline_predictions=tmp_path / "baseline.csv"))

    assert result == 1
    assert encoder_started is False
    assert paths.predictions.read_text(encoding="utf-8") == "photo_id,target,top_indices\n"
    report = paths.report.read_text(encoding="utf-8")
    assert "уничтожено/не принято кадров линейки: 1523" in report
    assert "| top-1 | 1523/1523" in report
    assert "| 0/1523 | 0/0 (0.0000) | НЕ ПРОЙДЕН |" in report


def test_guardrail_eval_uses_nonblocking_diagnostic_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = GuardrailPaths.for_fixture(tmp_path)
    captured: list[str] = []

    def capture(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.extend(command)

    monkeypatch.setattr("scripts.eval_redaction_guardrail.subprocess.run", capture)

    run_eval(argparse.Namespace(), paths)

    assert "--diagnostic-only" in captured


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
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.validate_guardrail_paths", lambda _: paths
    )
    baseline = tuple(
        Prediction(unit.photo_id, 1, (1, 2, 3)) for unit in frozen if unit.split == "holdout_ru"
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.load_baseline_predictions", lambda path: baseline
    )
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.validate_guardrail_lineage", lambda *args: None
    )
    encoder_started = False

    def forbidden_build(*arguments: object) -> None:
        nonlocal encoder_started
        encoder_started = True

    monkeypatch.setattr("scripts.eval_redaction_guardrail.run_build_index", forbidden_build)

    with pytest.raises(ValueError, match="замороженными holdout-единицами"):
        run_guardrail(argparse.Namespace(baseline_predictions=tmp_path / "base.csv"))
    assert encoder_started is False


def _guardrail_arguments(tmp_path: Path, *, report: Path | None = None) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--crops-root",
            str(tmp_path / "crops"),
            "--frozen-map",
            str(tmp_path / "frozen.csv"),
            "--baseline-predictions",
            str(tmp_path / "baseline.csv"),
            "--out",
            str(tmp_path / "out"),
            *(["--report", str(report)] if report is not None else []),
        ]
    )


@pytest.mark.parametrize(
    "collision",
    [
        "manifest",
        "frozen_map",
        "baseline_predictions",
        "pool",
        "holdout",
        "predictions",
        "survivor_report",
        "index_file",
        "index_subtree",
        "crops_subtree",
    ],
)
def test_guardrail_rejects_report_collisions_before_writing(
    tmp_path: Path, collision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _guardrail_arguments(tmp_path)
    paths = GuardrailPaths.for_fixture(tmp_path / "out")
    reports = {
        "manifest": arguments.manifest,
        "frozen_map": arguments.frozen_map,
        "baseline_predictions": arguments.baseline_predictions,
        "pool": paths.pool,
        "holdout": paths.holdout,
        "predictions": paths.predictions,
        "survivor_report": paths.report.with_name(SURVIVOR_REPORT_NAME),
        "index_file": paths.index / "catalog_index.bin",
        "index_subtree": paths.index / "nested" / "report.md",
        "crops_subtree": arguments.crops_root / "report.md",
    }
    arguments.report = reports[collision]
    prepared = False

    def forbidden_prepare(arguments: argparse.Namespace) -> None:
        nonlocal prepared
        prepared = True

    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.prepare_guardrail_splits", forbidden_prepare
    )

    with pytest.raises(ValueError, match=r"--report.*пересекается"):
        run_guardrail(arguments)
    assert prepared is False


def test_guardrail_rejects_normalized_report_alias(tmp_path: Path) -> None:
    arguments = _guardrail_arguments(
        tmp_path,
        report=tmp_path / "out" / "x" / ".." / "candidate-predictions.csv",
    )

    with pytest.raises(ValueError, match=r"--report.*пересекается"):
        run_guardrail(arguments)


def test_guardrail_default_report_path_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _guardrail_arguments(tmp_path)
    validated = False

    def prepared_after_validation(arguments: argparse.Namespace) -> None:
        nonlocal validated
        validated = True
        raise RuntimeError("prepared")

    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.prepare_guardrail_splits", prepared_after_validation
    )

    with pytest.raises(RuntimeError, match="prepared"):
        run_guardrail(arguments)
    assert validated is True


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
        "scripts.eval_redaction_guardrail.validate_guardrail_paths", lambda _: paths
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
    monkeypatch.setattr(
        "scripts.eval_redaction_guardrail.validate_guardrail_lineage", lambda *args: None
    )

    result = run_guardrail(argparse.Namespace(baseline_predictions=tmp_path / "base.csv"))

    assert result == 0
    assert paths.predictions.is_file()
    report = paths.report.read_text(encoding="utf-8")
    assert GUARDRAIL_SUCCESS_PHRASE in report
    assert "sha256 guardrail pool" in report
    assert "sha256 survivor split" in report
    assert "sha256 index" in report
