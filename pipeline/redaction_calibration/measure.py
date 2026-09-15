"""Чистые пиксельные меры и строгий CSV-контракт профильных результатов."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, fields
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from pipeline.harvest.redaction_profile import RedactionProfile
from pipeline.harvest.run import BLACK_PIXEL_THRESHOLD, MAX_BLACK_PIXEL_FRAC
from pipeline.redaction_calibration.geometry import (
    Box,
    CanonicalPlate,
    DetectorObservation,
    GeometryWitness,
    box_fits_actual_tile,
    fills_cover,
    partition_canonical_plates,
    tile_guarantee,
)
from pipeline.redaction_calibration.model import (
    ConfidenceMargin,
    FrameOutcome,
    FrameProfileResult,
    GeometryGateVerdict,
    HardGate,
    PlateCoverage,
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
    tile_or_profile: int | Sequence[int] | RedactionProfile,
    overlap: float = 0.25,
    witness: GeometryWitness | None = None,
    canonical: Sequence[CanonicalPlate] = (),
    frame_sizes: Mapping[str, tuple[int, int]] | None = None,
    fills: Mapping[str, Sequence[Box]] | Sequence[Box] | None = None,
) -> GeometryGateVerdict:
    """Проверить вмещение подтайловых номеров и покрытие надтайловых номеров заливками.

    Для многокадровых выборок заливки `fills` передаются как словарь `sha256 -> Sequence[Box]`.
    Плоский `Sequence[Box]` допускается для однокадровых выборок или фикстур.
    """
    scales: list[tuple[int, float]]
    if isinstance(tile_or_profile, RedactionProfile):
        scales = [(d.params.tile, d.params.overlap) for d in tile_or_profile.detectors]
    elif isinstance(tile_or_profile, int):
        scales = [(tile_or_profile, overlap)]
    elif isinstance(tile_or_profile, (Sequence, tuple, list)) and not isinstance(
        tile_or_profile, (str, bytes)
    ):
        scales = [(t, overlap) for t in tile_or_profile]
    else:
        raise TypeError(f"недопустимый tile_or_profile: {type(tile_or_profile).__name__}")

    if any(o != 0.25 for _, o in scales):
        return GeometryGateVerdict(
            passed=False,
            subtile_passed=False,
            oversized_passed=True,
            oversized_count=0,
            reason="invalid_overlap: overlap must be 0.25",
        )

    tiles = [t for t, _ in scales]
    subtiles, oversized = partition_canonical_plates(canonical, tiles)
    total_true = len(subtiles) + len(oversized)
    if total_true == 0:
        return GeometryGateVerdict(
            passed=False,
            subtile_passed=False,
            oversized_passed=False,
            oversized_count=0,
            reason="no_true_plates",
        )

    # При отсутствии надтайловых номеров свидетель проверяет структурную гарантию
    if witness is not None and not oversized:
        max_tile = max(tiles)
        if tile_guarantee(max_tile, scales[0][1]) < max(witness.w_max, witness.h_max):
            return GeometryGateVerdict(
                passed=False,
                subtile_passed=False,
                oversized_passed=True,
                oversized_count=0,
                reason="tile_guarantee_exceeded",
            )

    frame_sizes_map = frame_sizes if frame_sizes is not None else {}
    subtile_passed = True
    subtile_reason: str | None = None

    for plate in subtiles:
        candidate_scales = [
            (t, o) for t, o in scales if (plate.x1 - plate.x0) <= t and (plate.y1 - plate.y0) <= t
        ]
        if plate.sha256 not in frame_sizes_map:
            subtile_passed = False
            subtile_reason = f"missing_frame_size: {plate.sha256}"
            break
        frame_size = frame_sizes_map[plate.sha256]
        contained = any(box_fits_actual_tile(frame_size, plate, t, o) for t, o in candidate_scales)
        if not contained:
            subtile_passed = False
            subtile_reason = f"subtile_uncontained: {plate.plate_id}"
            break

    oversized_passed = True
    oversized_reason: str | None = None
    oversized_count = len(oversized)

    for plate in oversized:
        if isinstance(fills, Mapping):
            plate_fills = fills.get(plate.sha256, ())
        elif isinstance(fills, Sequence) and not isinstance(fills, (str, bytes)):
            plate_fills = fills
        else:
            plate_fills = ()
        if not fills_cover(plate_fills, plate):
            oversized_passed = False
            oversized_reason = f"oversized_uncovered: {plate.plate_id}"
            break

    passed = subtile_passed and oversized_passed
    reason = (
        subtile_reason
        if not subtile_passed
        else (oversized_reason if not oversized_passed else None)
    )

    return GeometryGateVerdict(
        passed=passed,
        subtile_passed=subtile_passed,
        oversized_passed=oversized_passed,
        oversized_count=oversized_count,
        reason=reason,
    )


def canonical_coverage(
    canonical: Sequence[CanonicalPlate],
    fills: Mapping[str, Sequence[Box]],
    *,
    rejected_sha256: AbstractSet[str],
) -> PlateCoverage:
    """Счесть накрытые, непокрытые и отсеянные истинные канонические номера.

    Заливки заданы по `sha256` кадра, как в `geometry_preflight`; отсутствие ключа значит
    «у кадра нет заливок», то есть номер не накрыт, но не отсеян. Отсев приходит отдельным
    обязательным множеством `rejected_sha256` и не смешивается с непокрытием: у номера
    ровно один из трёх исходов. Накрытие считает `fills_cover` по объединению заливок
    кадра. Набор без единого истинного номера отвергается: ось приватности там не
    определена.
    """
    true_plates = [plate for plate in canonical if plate.verdict == "true_plate"]
    if not true_plates:
        raise ValueError("канонический набор не содержит истинных номеров")
    covered = 0
    uncovered = 0
    rejected = 0
    for plate in true_plates:
        if plate.sha256 in rejected_sha256:
            rejected += 1
        elif fills_cover(fills.get(plate.sha256, ()), plate):
            covered += 1
        else:
            uncovered += 1
    return PlateCoverage(covered=covered, uncovered=uncovered, rejected=rejected)


def dominates(left: ProfileScore, right: ProfileScore) -> bool:
    """Проверить строгое Pareto-доминирование по осям запаса, приватности и издержек.

    Правило буквальное: не хуже по всем осям и строго лучше хотя бы по одной. Взвешенного
    score, скрытых весов и автоматического tie-break нет.
    """
    left_higher = (
        left.ssd_margin.to_next_lower,
        left.ssd_margin.to_min_true,
        left.yolo_margin.to_next_lower,
        left.yolo_margin.to_min_true,
        left.covered_plates,
    )
    right_higher = (
        right.ssd_margin.to_next_lower,
        right.ssd_margin.to_min_true,
        right.yolo_margin.to_next_lower,
        right.yolo_margin.to_min_true,
        right.covered_plates,
    )
    left_lower = (
        left.median_delta,
        left.p95_delta,
        left.above_quarter,
        left.new_over_redacted,
        left.changed_crops,
    )
    right_lower = (
        right.median_delta,
        right.p95_delta,
        right.above_quarter,
        right.new_over_redacted,
        right.changed_crops,
    )
    no_worse = all(a >= b for a, b in zip(left_higher, right_higher, strict=True)) and all(
        a <= b for a, b in zip(left_lower, right_lower, strict=True)
    )
    strictly_better = any(a > b for a, b in zip(left_higher, right_higher, strict=True)) or any(
        a < b for a, b in zip(left_lower, right_lower, strict=True)
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
