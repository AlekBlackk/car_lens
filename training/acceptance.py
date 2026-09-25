"""Метрики приёмки §5.2 на подмножестве эталона (спека holdout-relevance §7).

top-1, top-5 и строгая марка — через `training.metrics.aggregate`, тот же код, что у
паритета: число полного набора обязано совпасть с `parity.json` байт в байт. Открытое
множество и покрытие — через `training.calibration`, тот же код, что у подбора порогов.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from training.calibration import Scores, auroc, fpr_at_tpr, operating_point
from training.metrics import aggregate


@dataclass(frozen=True, slots=True)
class SetMetrics:
    """Приёмка одного набора: точность, открытое множество и что видит пользователь."""

    n: int
    classes: int
    top1_model: float
    top5_model: float
    top1_brand_strict: float
    top1_brand_grouped: float
    auroc: float
    fpr_at_tpr95: float
    coverage: float
    error: float


def member_brands(members: Sequence[Sequence[str]]) -> list[frozenset[str]]:
    """Марки участников класса по префиксу `model_id` («geely__coolray» → «geely»)."""
    return [frozenset(м.split("__", 1)[0] for м in участники) for участники in members]


def grouped_brand_hits(
    top1: np.ndarray, truth: np.ndarray, brands: Sequence[frozenset[str]]
) -> np.ndarray:
    """Марка угадана, если угадан класс или у классов есть общая марка участников.

    Строгая марка `training.metrics` у разнобрендовой группы не совпадает ни с чем, и верно
    узнанный Coolray там — промах по марке. Для пользователя это верный ответ.
    """
    return np.array(
        [p == t or bool(brands[p] & brands[t]) for p, t in zip(top1, truth, strict=True)],
        dtype=bool,
    )


def set_metrics(
    *,
    top5: np.ndarray,
    truth: np.ndarray,
    known: Scores,
    negatives: Scores,
    mask: np.ndarray,
    idx_to_class_id: Mapping[int, str],
    idx_to_brand: Mapping[int, str | None],
    brands: Sequence[frozenset[str]],
    sim_min: float,
    margin_min: float,
) -> SetMetrics:
    """Метрики набора `mask`; негативы общие для всех наборов."""
    if not mask.any():
        raise ValueError("пустой набор: метрики не определены")
    т5 = top5[mask]
    истина = truth[mask]
    строгие = aggregate(
        т5.tolist(),
        истина.tolist(),
        idx_to_class_id=idx_to_class_id,
        idx_to_brand=idx_to_brand,
    )
    свои = known.subset(mask)
    точка = operating_point(свои, negatives, sim_min, margin_min)
    return SetMetrics(
        n=int(mask.sum()),
        classes=len(set(истина.tolist())),
        top1_model=строгие.top1_model,
        top5_model=строгие.top5_model,
        top1_brand_strict=строгие.top1_brand,
        top1_brand_grouped=float(grouped_brand_hits(т5[:, 0], истина, brands).mean()),
        auroc=auroc(свои.sim1, negatives.sim1),
        fpr_at_tpr95=fpr_at_tpr(свои.sim1, negatives.sim1),
        coverage=точка.coverage,
        error=точка.error,
    )
