"""Арифметика расхождений между рантаймами (§4.4 дизайна M-1.4).

Два слоя сравнения разделены намеренно. Сырое — по ненормированным эмбеддингам —
детектор ошибки конвертации: если fp32-TFLite расходится с PyTorch на 1e-2,
сломан граф, и мерить после этого потери квантизации бессмысленно. Косинусное —
прикладной эффект: индекс ищет именно косинусом.

Модуль принимает готовые массивы и готовые метрики. Ни путей, ни интерпретаторов:
кто и чем считал эмбеддинги — забота `export/verify_parity.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from training.metrics import Metrics

# Знаменатель относительной ошибки прижимается снизу: эмбеддинги проходят через
# BatchNorm и содержат значения около нуля, деление на которые даёт бесконечность
# вместо диагноза.
_MIN_DENOMINATOR = 1e-6


class ParityError(ValueError):
    """Расхождение посчитать нельзя."""


@dataclass(frozen=True, slots=True)
class RawDelta:
    """Расхождение по сырым, ненормированным эмбеддингам."""

    max_abs: float
    median_abs: float
    p99_abs: float
    max_rel: float


@dataclass(frozen=True, slots=True)
class CosineDelta:
    """Косинус между эмбеддингами одного и того же кропа в двух рантаймах."""

    min: float
    median: float
    p1: float


def _проверить(reference: np.ndarray, candidate: np.ndarray) -> None:
    if reference.shape != candidate.shape:
        raise ParityError(f"несовпадение форм: {reference.shape} против {candidate.shape}")
    if reference.ndim != 2:
        raise ParityError(f"ожидалась матрица (n, dim), получено {reference.shape}")
    if len(reference) == 0:
        raise ParityError("выборка пуста: сравнивать нечего")


def raw_deltas(reference: np.ndarray, candidate: np.ndarray) -> RawDelta:
    """Поэлементное расхождение: детектор ошибки конвертации."""
    _проверить(reference, candidate)
    разница = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    знаменатель = np.maximum(np.abs(reference.astype(np.float64)), _MIN_DENOMINATOR)
    return RawDelta(
        max_abs=float(разница.max()),
        median_abs=float(np.median(разница)),
        p99_abs=float(np.percentile(разница, 99)),
        max_rel=float((разница / знаменатель).max()),
    )


def cosine_deltas(reference: np.ndarray, candidate: np.ndarray) -> CosineDelta:
    """Косинус попарно. Нормализация делается здесь: на вход идут сырые эмбеддинги."""
    _проверить(reference, candidate)
    эталон = reference.astype(np.float64)
    кандидат = candidate.astype(np.float64)
    норма_эталона = np.linalg.norm(эталон, axis=1)
    норма_кандидата = np.linalg.norm(кандидат, axis=1)
    if bool(np.any(норма_эталона == 0)) or bool(np.any(норма_кандидата == 0)):
        raise ParityError("нулевой эмбеддинг: косинус не определён")
    косинусы = np.sum(эталон * кандидат, axis=1) / (норма_эталона * норма_кандидата)
    return CosineDelta(
        min=float(косинусы.min()),
        median=float(np.median(косинусы)),
        p1=float(np.percentile(косинусы, 1)),
    )


def top1_churn(
    reference_top5: Sequence[Sequence[int]], candidate_top5: Sequence[Sequence[int]]
) -> float:
    """Доля примеров, у которых сменился top-1.

    Метрики двух рантаймов могут совпасть при том, что половина ответов разъехалась
    попарно: одинаковое число правильных не значит те же самые правильные.
    """
    if len(reference_top5) != len(candidate_top5):
        raise ParityError(f"разные длины: {len(reference_top5)} против {len(candidate_top5)}")
    if not reference_top5:
        raise ParityError("выборка пуста: сравнивать нечего")
    сменилось = sum(
        1
        for эталон, кандидат in zip(reference_top5, candidate_top5, strict=True)
        if эталон[0] != кандидат[0]
    )
    return сменилось / len(reference_top5)


def metric_delta(reference: Metrics, candidate: Metrics) -> dict[str, float]:
    """Разница метрик в пунктах процента. Минус — просадка кандидата."""
    return {
        "top1_model": (candidate.top1_model - reference.top1_model) * 100.0,
        "top5_model": (candidate.top5_model - reference.top5_model) * 100.0,
        "top1_brand": (candidate.top1_brand - reference.top1_brand) * 100.0,
    }
