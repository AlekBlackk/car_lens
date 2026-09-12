"""Независимый ledger проверки приватности и точные границы утечки номера."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, TypeAlias, cast

from pipeline.plate_leak import (
    LEAK_DEFINITIONS,
    LEAK_STRICT,
    LEAK_WITH_HISTORICAL,
    LEAK_WITH_PARTIAL,
    clopper_pearson,
)

PipelineOutcome: TypeAlias = Literal["accepted", "rejected", "operational_error"]
ReviewStatus: TypeAlias = Literal["pending", "resolved", "unreviewable"]

PLATE_VERDICTS: Final = frozenset(
    {
        "leak",
        "leak_hist",
        "clean_partial",
        "clean_unreadable",
        "clean_filled",
        "clean_no_plate",
    }
)
_PIPELINE_OUTCOMES: Final = frozenset({"accepted", "rejected", "operational_error"})
_REVIEW_STATUSES: Final = frozenset({"pending", "resolved", "unreviewable"})


@dataclass(frozen=True, slots=True)
class IndependentRow:
    """Одна запечатанная единица с независимыми исходом pipeline и ручным вердиктом."""

    photo_id: str
    pipeline_outcome: PipelineOutcome
    reject_reason: str
    review_status: ReviewStatus
    plate_verdict: str
    note: str


@dataclass(frozen=True, slots=True)
class IndependentBound:
    """Точная граница для одного импортированного определения утечки."""

    definition: str
    leaks: int
    sample_size: int
    share: float
    ci: tuple[float, float]
    projected_upper_count: float


@dataclass(frozen=True, slots=True)
class IndependentReport:
    """Полная арифметика партии и статистика только по принятым решённым кадрам."""

    accepted_resolved: int
    accepted_pending: int
    accepted_unreviewable: int
    rejected: int
    operational_error: int
    accounted: int
    post_profile_pool_size: int
    bounds: tuple[IndependentBound, ...]
    passed: bool
    notes: str


# Ссылки удерживают импортированные множества явной частью контракта модуля. Нельзя
# воспроизводить их литералами: определения утечки принадлежат pipeline.plate_leak.
_IMPORTED_LEAK_SETS = (LEAK_STRICT, LEAK_WITH_HISTORICAL, LEAK_WITH_PARTIAL)


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _validate_row(row: IndependentRow) -> None:
    if not row.photo_id.strip():
        raise ValueError("photo_id должен быть непустым")
    if row.pipeline_outcome not in _PIPELINE_OUTCOMES:
        raise ValueError(f"{row.photo_id}: неизвестный pipeline_outcome")
    if row.review_status not in _REVIEW_STATUSES:
        raise ValueError(f"{row.photo_id}: неизвестный review_status")
    if row.pipeline_outcome == "accepted":
        if row.reject_reason:
            raise ValueError(f"{row.photo_id}: accepted не должен иметь reject_reason")
        if row.review_status == "resolved" and row.plate_verdict not in PLATE_VERDICTS:
            raise ValueError(f"{row.photo_id}: resolved accepted требует plate_verdict")
        if row.review_status != "resolved" and row.plate_verdict:
            raise ValueError(f"{row.photo_id}: нерешённый accepted не должен иметь plate_verdict")
        return
    if not row.reject_reason:
        raise ValueError(f"{row.photo_id}: {row.pipeline_outcome} требует reject_reason")
    if row.plate_verdict:
        raise ValueError(f"{row.photo_id}: {row.pipeline_outcome} не должен иметь plate_verdict")
    if row.pipeline_outcome == "rejected" and row.review_status != "resolved":
        raise ValueError(f"{row.photo_id}: rejected должен быть resolved")
    if row.pipeline_outcome == "operational_error" and row.review_status != "pending":
        raise ValueError(f"{row.photo_id}: operational_error должен быть pending")


def validate_independent_ledger(rows: Sequence[IndependentRow], sealed_ids: Sequence[str]) -> None:
    """Проверить таксономию и взаимно-однозначное покрытие запечатанных photo_id."""
    row_ids = [row.photo_id for row in rows]
    duplicate_rows = _duplicates(row_ids)
    if duplicate_rows:
        raise ValueError(f"дубликат photo_id в ledger: {duplicate_rows[:5]}")
    duplicate_sealed = _duplicates(tuple(sealed_ids))
    if duplicate_sealed:
        raise ValueError(f"дубликат photo_id в sealed IDs: {duplicate_sealed[:5]}")
    foreign = sorted(set(row_ids) - set(sealed_ids))
    if foreign:
        raise ValueError(f"ledger содержит чужие photo_id: {foreign[:5]}")
    missing = sorted(set(sealed_ids) - set(row_ids))
    if missing:
        raise ValueError(f"ledger не хватает sealed photo_id: {missing[:5]}")
    for row in rows:
        _validate_row(row)


def merge_independent_reviews(
    rows: Sequence[IndependentRow], reviews: Sequence[Mapping[str, str]]
) -> tuple[IndependentRow, ...]:
    """Присоединить ручные решения строго по неизменяемому photo_id."""
    review_ids = [review.get("photo_id", "") for review in reviews]
    duplicates = _duplicates(review_ids)
    if duplicates:
        raise ValueError(f"дубликат photo_id в review: {duplicates[:5]}")
    row_ids = {row.photo_id for row in rows}
    foreign = sorted(set(review_ids) - row_ids)
    if foreign:
        raise ValueError(f"review содержит чужие photo_id: {foreign[:5]}")
    by_id = {review["photo_id"]: review for review in reviews}
    merged = []
    for row in rows:
        review = by_id.get(row.photo_id)
        if review is None:
            merged.append(row)
            continue
        if row.pipeline_outcome != "accepted":
            raise ValueError(f"{row.photo_id}: review допустим только для accepted")
        merged.append(
            replace(
                row,
                review_status=cast(ReviewStatus, review.get("review_status", "")),
                plate_verdict=review.get("plate_verdict", ""),
                note=review.get("note", ""),
            )
        )
    return tuple(merged)


def compute_independent_report(
    rows: Sequence[IndependentRow], post_profile_pool_size: int
) -> IndependentReport:
    """Посчитать точные интервалы, не смешивая rejects и ошибки со знаменателем."""
    if isinstance(post_profile_pool_size, bool) or post_profile_pool_size < 0:
        raise ValueError("post_profile_pool_size должен быть неотрицательным int")
    for row in rows:
        _validate_row(row)
    accepted_resolved = tuple(
        row
        for row in rows
        if row.pipeline_outcome == "accepted" and row.review_status == "resolved"
    )
    accepted_pending = sum(
        row.pipeline_outcome == "accepted" and row.review_status == "pending" for row in rows
    )
    accepted_unreviewable = sum(
        row.pipeline_outcome == "accepted" and row.review_status == "unreviewable" for row in rows
    )
    rejected = sum(row.pipeline_outcome == "rejected" for row in rows)
    operational_error = sum(row.pipeline_outcome == "operational_error" for row in rows)
    bounds = _compute_bounds(accepted_resolved, post_profile_pool_size)
    strict_leaks = sum(row.plate_verdict in LEAK_STRICT for row in accepted_resolved)
    passed = (
        len(rows) == 300
        and accepted_pending == 0
        and accepted_unreviewable == 0
        and operational_error == 0
        and strict_leaks == 0
    )
    return IndependentReport(
        accepted_resolved=len(accepted_resolved),
        accepted_pending=accepted_pending,
        accepted_unreviewable=accepted_unreviewable,
        rejected=rejected,
        operational_error=operational_error,
        accounted=len(rows),
        post_profile_pool_size=post_profile_pool_size,
        bounds=bounds,
        passed=passed,
        notes=("observed zero does not prove population zero; does not satisfy rule 1 R-11"),
    )


def _compute_bounds(
    accepted_resolved: Sequence[IndependentRow], post_profile_pool_size: int
) -> tuple[IndependentBound, ...]:
    n = len(accepted_resolved)
    if n == 0:
        return ()
    bounds = []
    for definition, verdicts in LEAK_DEFINITIONS:
        leaks = sum(row.plate_verdict in verdicts for row in accepted_resolved)
        ci = clopper_pearson(leaks, n)
        bounds.append(
            IndependentBound(
                definition=definition,
                leaks=leaks,
                sample_size=n,
                share=leaks / n,
                ci=ci,
                projected_upper_count=ci[1] * post_profile_pool_size,
            )
        )
    return tuple(bounds)
