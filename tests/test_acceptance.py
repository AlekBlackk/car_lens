"""Метрики приёмки §5.2 на подмножестве эталона (спека holdout-relevance §7)."""

from __future__ import annotations

import numpy as np
import pytest

from training.acceptance import SetMetrics, grouped_brand_hits, member_brands, set_metrics
from training.calibration import Scores
from training.tune_thresholds import scores_from

КЛАССЫ = {0: "lada__vesta", 1: "lada__granta", 2: "grp__coolray"}
МАРКИ = {0: "Lada", 1: "Lada", 2: None}
УЧАСТНИКИ = [["lada__vesta"], ["lada__granta"], ["geely__coolray", "belgee__x50"]]

# Четыре кадра: верная Веста, Гранта вместо Весты, верный Coolray, Гранта вместо Coolray.
ТОП5 = np.array([[0, 1, 2, 1, 2], [1, 0, 2, 0, 2], [2, 0, 1, 0, 1], [1, 2, 0, 2, 0]])
ИСТИНА = np.array([0, 0, 2, 2])
SIM1 = np.array([0.9, 0.6, 0.8, 0.4])
SIM2 = np.array([0.1, 0.5, 0.3, 0.3])


def _знакомые() -> Scores:
    return scores_from(ТОП5[:, 0], SIM1, ТОП5[:, 1], SIM2, ИСТИНА, [МАРКИ[i] for i in range(3)])


def _чужие() -> Scores:
    return scores_from(
        np.array([0, 1]),
        np.array([0.7, 0.2]),
        np.array([1, 0]),
        np.array([0.1, 0.1]),
        None,
        [МАРКИ[i] for i in range(3)],
    )


def _метрики(маска: np.ndarray) -> SetMetrics:
    return set_metrics(
        top5=ТОП5,
        truth=ИСТИНА,
        known=_знакомые(),
        negatives=_чужие(),
        mask=маска,
        idx_to_class_id=КЛАССЫ,
        idx_to_brand=МАРКИ,
        brands=member_brands(УЧАСТНИКИ),
        sim_min=0.5,
        margin_min=0.0,
    )


def test_марки_участников_по_префиксу() -> None:
    assert member_brands(УЧАСТНИКИ) == [
        frozenset({"lada"}),
        frozenset({"lada"}),
        frozenset({"geely", "belgee"}),
    ]


def test_групповая_марка_засчитывает_верный_класс_группы() -> None:
    марки = member_brands([*УЧАСТНИКИ, ["geely__monjaro"]])
    попадания = grouped_brand_hits(np.array([2, 1, 3, 0]), np.array([2, 0, 2, 2]), марки)
    # Coolray верно; Гранта вместо Весты — Lada; Monjaro вместо Coolray — Geely; Веста — нет.
    assert попадания.tolist() == [True, True, True, False]


def test_полный_набор() -> None:
    м = _метрики(np.ones(4, dtype=bool))
    assert (м.n, м.classes) == (4, 2)
    assert м.top1_model == pytest.approx(0.5)
    assert м.top5_model == pytest.approx(1.0)
    # Строгая марка: у группы марки нет, верный Coolray — промах.
    assert м.top1_brand_strict == pytest.approx(0.5)
    assert м.top1_brand_grouped == pytest.approx(0.75)
    # Пары знакомый > чужой: 0.9 — 2, 0.6 — 1, 0.8 — 2, 0.4 — 1 из 8.
    assert м.auroc == pytest.approx(0.75)
    # sim_min 0.5: показаны три кадра, неверен один (Гранта вместо Весты).
    assert м.coverage == pytest.approx(0.75)
    assert м.error == pytest.approx(1 / 3)


def test_подмножество_считается_только_по_своим_кадрам() -> None:
    м = _метрики(np.array([True, True, False, False]))
    assert (м.n, м.classes) == (2, 1)
    assert м.top1_model == pytest.approx(0.5)
    assert м.top1_brand_grouped == pytest.approx(1.0)


def test_пустой_набор_отвергается() -> None:
    with pytest.raises(ValueError, match="пустой набор"):
        _метрики(np.zeros(4, dtype=bool))
