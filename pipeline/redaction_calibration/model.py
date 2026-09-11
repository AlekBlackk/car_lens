"""Типизированные результаты одного профильного прогона и его сводка."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

FrameOutcome: TypeAlias = Literal["accepted", "rejected"]
HardGate: TypeAlias = Literal[
    "provenance_workspace_freshness",
    "unresolved_empty",
    "current_geometry_witness",
    "structural_actual_tile_preflight",
    "full_canonical_coverage",
    "strict_calibration_review",
    "report_arithmetic",
]


@dataclass(frozen=True, slots=True)
class FrameProfileResult:
    """Пиксельный результат одной входной строки изолированной пересборки."""

    photo_id: str
    class_id: str
    outcome: FrameOutcome
    reject_reason: str | None
    old_black_frac: float | None
    new_black_frac: float | None
    delta_black: float | None
    pixels_equal: bool | None
    audit_preserved: bool


@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    """Стоимость профиля по новым чёрным пикселям, изменениям и отказам."""

    median_delta_black: float
    p95_delta_black: float
    max_delta_black: float
    above_quarter: int
    new_over_redacted: int
    changed_crops: int
    rejects: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ConfidenceMargin:
    """Запас порога детектора до соседней границы и слабейшего истинного номера."""

    to_next_lower: float
    to_min_true: float


@dataclass(frozen=True, slots=True)
class ProfileScore:
    """Раздельные Pareto-оси профиля без скалярной свёртки."""

    profile_id: str
    ssd_margin: ConfidenceMargin
    yolo_margin: ConfidenceMargin
    median_delta: float
    p95_delta: float
    above_quarter: int
    new_over_redacted: int
    changed_crops: int


@dataclass(frozen=True, slots=True)
class ProfileGateReport:
    """Факты семи упорядоченных калибровочных гейтов одного профиля."""

    profile_id: str
    provenance_fresh: bool
    unresolved_count: int
    geometry_current: bool
    geometry_preflight_passed: bool
    canonical_coverage_complete: bool
    strict_leaks: int
    unreviewable_accepted: int
    report_arithmetic_closed: bool
    canonical_sha256: str | None = None
    w_max: int | None = None
    h_max: int | None = None


@dataclass(frozen=True, slots=True)
class VisualVeto:
    """Трассируемая дисквалификация профиля по уничтоженному признаку модели."""

    profile_id: str
    sha256: str | None
    plate_id: str | None
    destroyed_feature: str
    reason: str
