from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from pipeline.redaction_calibration.geometry import (
    CanonicalPlate,
    DetectorObservation,
    GeometryWitness,
)
from pipeline.redaction_calibration.measure import (
    black_fraction,
    confidence_breakpoints,
    confidence_margin,
    dominates,
    first_failed_gate,
    geometry_preflight,
    load_frame_profile_results,
    pareto_frontier,
    rank_profiles,
    stale_profile_ids,
    summarize_profile_results,
    write_frame_profile_results,
)
from pipeline.redaction_calibration.model import (
    ConfidenceMargin,
    FrameProfileResult,
    ProfileGateReport,
    ProfileMetrics,
    ProfileScore,
)


def _result(
    *,
    photo_id: str,
    old: float | None,
    new: float | None,
    equal: bool | None,
    reason: str | None = None,
) -> FrameProfileResult:
    delta = None if old is None or new is None else max(0.0, new - old)
    return FrameProfileResult(
        photo_id=photo_id,
        class_id="lada__granta",
        outcome="rejected" if reason else "accepted",
        reject_reason=reason,
        old_black_frac=old,
        new_black_frac=new,
        delta_black=delta,
        pixels_equal=equal,
        audit_preserved=False,
    )


def test_black_fraction_использует_grayscale_и_строгий_порог_8() -> None:
    image = np.array([[[0, 0, 20], [0, 0, 30], [8, 8, 8]]], dtype=np.uint8)

    assert black_fraction(image) == pytest.approx(1 / 3)


def test_profile_metrics_считает_положительный_delta_и_строгие_границы() -> None:
    results = [
        _result(photo_id="a", old=0.40, new=0.30, equal=True),
        _result(photo_id="b", old=0.00, new=0.25, equal=False),
        _result(photo_id="c", old=0.10, new=0.36, equal=False, reason="duplicate"),
        _result(photo_id="d", old=0.00, new=0.60, equal=False, reason="over_redacted"),
        _result(photo_id="e", old=None, new=None, equal=None, reason="decode_failed"),
    ]

    assert summarize_profile_results(results) == ProfileMetrics(
        median_delta_black=0.255,
        p95_delta_black=0.60,
        max_delta_black=0.60,
        above_quarter=2,
        new_over_redacted=1,
        changed_crops=3,
        rejects=(("decode_failed", 1), ("duplicate", 1), ("over_redacted", 1)),
    )


def test_profile_metrics_p95_использует_nearest_rank() -> None:
    results = [
        _result(photo_id=str(index), old=0.0, new=index / 100, equal=False)
        for index in range(1, 21)
    ]

    metrics = summarize_profile_results(results)

    assert metrics.p95_delta_black == pytest.approx(0.19)
    assert metrics.median_delta_black == pytest.approx(0.105)


def test_profile_metrics_отвергает_пустой_ввод() -> None:
    with pytest.raises(ValueError, match="пуст"):
        summarize_profile_results([])


def test_profile_result_csv_roundtrip_имеет_строгую_схему(tmp_path: Path) -> None:
    path = tmp_path / "pixel-diff.csv"
    expected = [
        _result(photo_id="a" * 64, old=0.1, new=0.2, equal=False),
        _result(
            photo_id="b" * 64,
            old=None,
            new=None,
            equal=None,
            reason="decode_failed",
        ),
    ]

    write_frame_profile_results(path, expected)

    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == [
        field.name for field in fields(FrameProfileResult)
    ]
    assert load_frame_profile_results(path) == expected

    path.write_text(
        path.read_text(encoding="utf-8").replace("audit_preserved", "unexpected"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="колон"):
        load_frame_profile_results(path)


def test_profile_result_csv_rejects_blank_required_audit_preserved(tmp_path: Path) -> None:
    path = tmp_path / "pixel-diff.csv"
    write_frame_profile_results(
        path,
        [_result(photo_id="a" * 64, old=0.1, new=0.2, equal=False)],
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(",False\n", ",\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="логическ"):
        load_frame_profile_results(path)


def _observation(detector: str, confidence: float, index: int) -> DetectorObservation:
    return DetectorObservation(
        observation_id=f"obs-{index:03d}",
        sha256="a" * 64,
        detector=detector,
        profile_id="trace-profile",
        pass_name="first",
        zone=(0, 0, 1000, 600),
        tile=(0, 0, 640, 600),
        box=(10, 20, 90, 50),
        confidence=confidence,
    )


def _score(
    profile_id: str,
    *,
    ssd: tuple[float, float] = (0.10, 0.20),
    yolo: tuple[float, float] = (0.15, 0.25),
    costs: tuple[float, float, int, int, int] = (0.10, 0.20, 2, 1, 5),
) -> ProfileScore:
    return ProfileScore(
        profile_id=profile_id,
        ssd_margin=ConfidenceMargin(*ssd),
        yolo_margin=ConfidenceMargin(*yolo),
        median_delta=costs[0],
        p95_delta=costs[1],
        above_quarter=costs[2],
        new_over_redacted=costs[3],
        changed_crops=costs[4],
    )


def _gate_report(profile_id: str, **overrides: object) -> ProfileGateReport:
    values: dict[str, object] = {
        "profile_id": profile_id,
        "provenance_fresh": True,
        "unresolved_count": 0,
        "geometry_current": True,
        "geometry_preflight_passed": True,
        "canonical_coverage_complete": True,
        "strict_leaks": 0,
        "unreviewable_accepted": 0,
        "report_arithmetic_closed": True,
    }
    values.update(overrides)
    return ProfileGateReport(**values)  # type: ignore[arg-type]


def test_confidence_breakpoints_оставляет_уникальные_наблюдавшиеся_границы() -> None:
    observations = [
        _observation("ssd", 0.50, 1),
        _observation("ssd", 0.35, 2),
        _observation("ssd", 0.50, 3),
        _observation("ssd", 0.20, 4),
        _observation("yolo", 0.95, 5),
    ]

    assert confidence_breakpoints(observations, "ssd") == (0.20, 0.35, 0.50)
    assert confidence_breakpoints(observations, "yolo") == (0.95,)


def test_confidence_margin_использует_соседний_breakpoint_а_не_скрытые_005() -> None:
    breakpoints = (0.20, 0.35, 0.50)

    margin = confidence_margin(0.35, breakpoints, min_true_confidence=0.50)

    assert margin.to_next_lower == pytest.approx(0.15)
    assert margin.to_min_true == pytest.approx(0.15)
    with pytest.raises(ValueError, match="breakpoint"):
        confidence_margin(0.30, breakpoints, min_true_confidence=0.50)
    with pytest.raises(ValueError, match="слабейшего истинного"):
        confidence_margin(0.50, breakpoints, min_true_confidence=0.35)


def test_pareto_сохраняет_раздельные_margin_и_равные_профили() -> None:
    a = _score("a")
    b = _score("b", costs=(0.20, 0.30, 3, 1, 7))
    c = _score("c", ssd=(0.12, 0.22), costs=(0.10, 0.20, 2, 1, 9))
    equal = _score("z")

    assert dominates(a, b)
    assert not dominates(a, c)
    assert not dominates(a, equal)
    assert [row.profile_id for row in pareto_frontier([equal, c, b, a])] == ["a", "c", "z"]


def test_hard_gates_идут_в_заданном_порядке_до_pareto() -> None:
    failures = _gate_report(
        "failed",
        provenance_fresh=False,
        unresolved_count=2,
        geometry_current=False,
        geometry_preflight_passed=False,
        canonical_coverage_complete=False,
        strict_leaks=1,
        unreviewable_accepted=1,
        report_arithmetic_closed=False,
    )

    assert first_failed_gate(failures) == "provenance_workspace_freshness"
    assert rank_profiles(
        [(failures, _score("failed")), (_gate_report("survivor"), _score("survivor"))]
    ) == (_score("survivor"),)

    ordered_failures: list[tuple[str, dict[str, object]]] = [
        ("unresolved_empty", {"unresolved_count": 1}),
        ("current_geometry_witness", {"geometry_current": False}),
        ("structural_actual_tile_preflight", {"geometry_preflight_passed": False}),
        ("full_canonical_coverage", {"canonical_coverage_complete": False}),
        ("strict_calibration_review", {"strict_leaks": 1}),
        ("report_arithmetic", {"report_arithmetic_closed": False}),
    ]
    for expected, overrides in ordered_failures:
        assert first_failed_gate(_gate_report(expected, **overrides)) == expected


def test_geometry_preflight_использует_структурную_границу_и_реальные_origins() -> None:
    witness = GeometryWitness(w_max=160, h_max=50, canonical_sha256="c" * 64)
    plates = [
        CanonicalPlate(
            plate_id="plate-001",
            sha256="a" * 64,
            x0=650,
            y0=10,
            x1=810,
            y1=60,
            readable=True,
            verdict="true_plate",
            note="проверено",
        )
    ]

    assert geometry_preflight(640, 0.25, witness, plates, {"a" * 64: (1000, 200)})
    assert not geometry_preflight(600, 0.25, witness, plates, {"a" * 64: (1000, 200)})
    assert not geometry_preflight(640, 0.30, witness, plates, {"a" * 64: (1000, 200)})


def test_stale_profiles_сравнивает_canonical_и_оба_максимума() -> None:
    current = GeometryWitness(w_max=160, h_max=50, canonical_sha256="c" * 64)
    reports = [
        _gate_report(
            "fresh",
            canonical_sha256="c" * 64,
            w_max=160,
            h_max=50,
        ),
        _gate_report(
            "hash-changed",
            canonical_sha256="d" * 64,
            w_max=160,
            h_max=50,
        ),
        _gate_report(
            "height-changed",
            canonical_sha256="c" * 64,
            w_max=160,
            h_max=49,
        ),
    ]

    assert stale_profile_ids(reports, current) == ("hash-changed", "height-changed")
