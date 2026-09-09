"""Типизированные результаты одного профильного прогона и его сводка."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

FrameOutcome: TypeAlias = Literal["accepted", "rejected"]


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
