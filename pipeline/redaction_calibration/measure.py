"""Чистые пиксельные меры и строгий CSV-контракт профильных результатов."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from pipeline.harvest.run import BLACK_PIXEL_THRESHOLD, MAX_BLACK_PIXEL_FRAC
from pipeline.redaction_calibration.geometry import (
    CanonicalPlate,
    DetectorObservation,
    GeometryWitness,
    box_fits_actual_tile,
    tile_guarantee,
)
from pipeline.redaction_calibration.model import (
    ConfidenceMargin,
    FrameOutcome,
    FrameProfileResult,
    HardGate,
    ProfileGateReport,
    ProfileMetrics,
    ProfileScore,
)


def black_fraction(image: np.ndarray, threshold: int = BLACK_PIXEL_THRESHOLD) -> float:
    """Доля пикселей с яркостью строго ниже порога боевого правила редакции."""
    if image.size == 0:
        raise ValueError("нельзя измерить пустое изображение")
    серый = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float((серый < threshold).mean())


def summarize_profile_results(results: Sequence[FrameProfileResult]) -> ProfileMetrics:
    """Свести один результат на входную строку в детерминированные метрики профиля.

    P95 — nearest-rank: для N отсортированных значений берётся элемент с индексом
    `ceil(0.95 * N) - 1`. Дельта учитывает только прирост чёрной доли.
    """
    if not results:
        raise ValueError("нельзя свести пустой набор профильных результатов")
    дельты = sorted(result.delta_black for result in results if result.delta_black is not None)
    if not дельты:
        дельты = [0.0]
    середина = len(дельты) // 2
    медиана = дельты[середина] if len(дельты) % 2 else (дельты[середина - 1] + дельты[середина]) / 2
    p95 = дельты[math.ceil(0.95 * len(дельты)) - 1]
    отказы = Counter(
        result.reject_reason
        for result in results
        if result.outcome == "rejected" and result.reject_reason is not None
    )
    return ProfileMetrics(
        median_delta_black=медиана,
        p95_delta_black=p95,
        max_delta_black=дельты[-1],
        above_quarter=sum(delta > 0.25 for delta in дельты),
        new_over_redacted=sum(
            result.new_black_frac is not None and result.new_black_frac > MAX_BLACK_PIXEL_FRAC
            for result in results
        ),
        changed_crops=sum(result.pixels_equal is False for result in results),
        rejects=tuple(sorted(отказы.items())),
    )


_PROFILE_FIELDS = tuple(field.name for field in fields(FrameProfileResult))


def write_frame_profile_results(path: Path, results: Sequence[FrameProfileResult]) -> None:
    """Записать профильные результаты с колонками из полей доменной записи."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_PROFILE_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _optional_bool(value: str) -> bool | None:
    if value == "":
        return None
    if value not in {"True", "False"}:
        raise ValueError(f"недопустимое логическое значение: {value!r}")
    return value == "True"


def _required_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"недопустимое обязательное логическое значение: {value!r}")
    return value == "True"


def load_frame_profile_results(path: Path) -> list[FrameProfileResult]:
    """Прочитать CSV, отвергая отсутствующие или неизвестные колонки контракта."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        actual = tuple(reader.fieldnames or ())
        if actual != _PROFILE_FIELDS:
            raise ValueError(
                f"колонки профильных результатов не совпали: {actual!r} != {_PROFILE_FIELDS!r}"
            )
        loaded: list[FrameProfileResult] = []
        for row in reader:
            outcome = row["outcome"]
            if outcome not in {"accepted", "rejected"}:
                raise ValueError(f"неизвестный исход профильной строки: {outcome!r}")
            loaded.append(
                FrameProfileResult(
                    photo_id=row["photo_id"],
                    class_id=row["class_id"],
                    outcome=cast(FrameOutcome, outcome),
                    reject_reason=row["reject_reason"] or None,
                    old_black_frac=_optional_float(row["old_black_frac"]),
                    new_black_frac=_optional_float(row["new_black_frac"]),
                    delta_black=_optional_float(row["delta_black"]),
                    pixels_equal=_optional_bool(row["pixels_equal"]),
                    audit_preserved=_required_bool(row["audit_preserved"]),
                )
            )
    return loaded


_HARD_GATES: tuple[tuple[HardGate, Callable[[ProfileGateReport], bool]], ...] = (
    ("provenance_workspace_freshness", lambda row: row.provenance_fresh),
    ("unresolved_empty", lambda row: row.unresolved_count == 0),
    ("current_geometry_witness", lambda row: row.geometry_current),
    ("structural_actual_tile_preflight", lambda row: row.geometry_preflight_passed),
    ("full_canonical_coverage", lambda row: row.canonical_coverage_complete),
    (
        "strict_calibration_review",
        lambda row: row.strict_leaks == 0 and row.unreviewable_accepted == 0,
    ),
    (
        "report_arithmetic",
        lambda row: row.operational_errors == 0 and row.report_arithmetic_closed,
    ),
)


def confidence_breakpoints(
    observations: Sequence[DetectorObservation], detector: str
) -> tuple[float, ...]:
    """Вернуть только наблюдавшиеся уникальные confidence выбранного детектора."""
    if not detector.strip():
        raise ValueError("detector должен быть непустым")
    return tuple(
        sorted({float(row.confidence) for row in observations if row.detector == detector})
    )


def confidence_margin(
    selected: float,
    breakpoints: Sequence[float],
    *,
    min_true_confidence: float,
) -> ConfidenceMargin:
    """Вычислить запас выбранного breakpoint без произвольного шага между наблюдениями."""
    ordered = tuple(sorted(set(float(value) for value in breakpoints)))
    if selected not in ordered:
        raise ValueError("выбранный confidence должен быть наблюдавшимся breakpoint")
    index = ordered.index(selected)
    if selected > min_true_confidence:
        raise ValueError("порог выше confidence слабейшего истинного номера")
    lower = ordered[index - 1] if index else 0.0
    return ConfidenceMargin(
        to_next_lower=selected - lower,
        to_min_true=min_true_confidence - selected,
    )


def first_failed_gate(report: ProfileGateReport) -> HardGate | None:
    """Вернуть первый провал строго в предзарегистрированном порядке семи гейтов."""
    for name, predicate in _HARD_GATES:
        if not predicate(report):
            return name
    return None


def geometry_fixpoint_converged(
    previous_plate_ids: set[str],
    current_plate_ids: set[str],
    previous_witness: GeometryWitness,
    current_witness: GeometryWitness,
    *,
    unresolved_count: int,
) -> bool:
    """Подтвердить отсутствие новых canonical-объектов и расширения observed boundary."""
    boundary_unchanged = (
        previous_witness.w_max == current_witness.w_max
        and previous_witness.h_max == current_witness.h_max
    )
    return unresolved_count == 0 and current_plate_ids == previous_plate_ids and boundary_unchanged


def geometry_preflight(
    tile: int,
    overlap: float,
    witness: GeometryWitness,
    canonical: Sequence[CanonicalPlate],
    frame_sizes: Mapping[str, tuple[int, int]],
) -> bool:
    """Проверить структурную границу и фактические origins каждого истинного номера."""
    if overlap != 0.25:
        return False
    if tile_guarantee(tile, overlap) < max(witness.w_max, witness.h_max):
        return False
    true_plates = [plate for plate in canonical if plate.verdict == "true_plate"]
    try:
        return bool(true_plates) and all(
            box_fits_actual_tile(frame_sizes[plate.sha256], plate, tile, overlap)
            for plate in true_plates
        )
    except KeyError:
        return False


def dominates(left: ProfileScore, right: ProfileScore) -> bool:
    """Проверить строгое Pareto-доминирование по отдельным margin и cost осям."""
    left_margins = (
        left.ssd_margin.to_next_lower,
        left.ssd_margin.to_min_true,
        left.yolo_margin.to_next_lower,
        left.yolo_margin.to_min_true,
    )
    right_margins = (
        right.ssd_margin.to_next_lower,
        right.ssd_margin.to_min_true,
        right.yolo_margin.to_next_lower,
        right.yolo_margin.to_min_true,
    )
    left_costs = (
        left.median_delta,
        left.p95_delta,
        left.above_quarter,
        left.new_over_redacted,
        left.changed_crops,
    )
    right_costs = (
        right.median_delta,
        right.p95_delta,
        right.above_quarter,
        right.new_over_redacted,
        right.changed_crops,
    )
    no_worse = all(a >= b for a, b in zip(left_margins, right_margins, strict=True)) and all(
        a <= b for a, b in zip(left_costs, right_costs, strict=True)
    )
    strictly_better = any(a > b for a, b in zip(left_margins, right_margins, strict=True)) or any(
        a < b for a, b in zip(left_costs, right_costs, strict=True)
    )
    return no_worse and strictly_better


def pareto_frontier(scores: Sequence[ProfileScore]) -> tuple[ProfileScore, ...]:
    """Оставить недоминируемые профили; равные строки не дедуплицировать."""
    return tuple(
        sorted(
            (
                score
                for index, score in enumerate(scores)
                if not any(
                    other_index != index and dominates(other, score)
                    for other_index, other in enumerate(scores)
                )
            ),
            key=lambda row: row.profile_id,
        )
    )


def rank_profiles(
    candidates: Sequence[tuple[ProfileGateReport, ProfileScore]],
) -> tuple[ProfileScore, ...]:
    """Пустить в Pareto только профили, прошедшие семь гейтов."""
    survivors: list[ProfileScore] = []
    for report, score in candidates:
        if report.profile_id != score.profile_id:
            raise ValueError("gate report и score принадлежат разным профилям")
        if first_failed_gate(report) is None:
            survivors.append(score)
    return pareto_frontier(survivors)


def stale_profile_ids(
    reports: Sequence[ProfileGateReport], current: GeometryWitness
) -> tuple[str, ...]:
    """Назвать только отчёты, не совпадающие с текущими canonical SHA и W_max/H_max."""
    return tuple(
        sorted(
            report.profile_id
            for report in reports
            if (
                report.canonical_sha256 != current.canonical_sha256
                or report.w_max != current.w_max
                or report.h_max != current.h_max
            )
        )
    )
