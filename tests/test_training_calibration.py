"""Подбор порогов отказа и шкалы уверенности (спека calibration-ru §5)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from scripts.stage_assets import ПЛЕЙСХОЛДЕР_КАЛИБРОВКИ, ПЛЕЙСХОЛДЕР_ПОРОГОВ
from training.calibration import (
    BRAND,
    MODEL,
    REJECT,
    CalibrationError,
    Operating,
    Scores,
    auroc,
    calibration_doc,
    calibration_knots,
    confidence_from_sim,
    decide,
    ece,
    fpr_at_tpr,
    operating_point,
    select_pair,
    split_halves,
    thresholds_doc,
)


def _оценки(sim1, sim2, same_brand, class_hit, brand_hit) -> Scores:  # type: ignore[no-untyped-def]
    return Scores(
        sim1=np.asarray(sim1, dtype=np.float64),
        sim2=np.asarray(sim2, dtype=np.float64),
        same_brand=np.asarray(same_brand, dtype=bool),
        class_hit=np.asarray(class_hit, dtype=bool),
        brand_hit=np.asarray(brand_hit, dtype=bool),
    )


def test_решение_зеркалит_decide_ts() -> None:
    оценки = _оценки(
        sim1=[0.30, 0.60, 0.60, 0.60, 0.50],
        sim2=[0.10, 0.58, 0.58, 0.40, 0.30],
        same_brand=[True, True, False, False, False],
        class_hit=[False] * 5,
        brand_hit=[False] * 5,
    )

    решения = decide(оценки, sim_min=0.42, margin_min=0.05)

    # sim ниже порога; разрыв мал, марка общая; разрыв мал, марки разные; чисто; чисто
    assert решения.tolist() == [REJECT, BRAND, REJECT, MODEL, MODEL]


def test_порог_включительный_как_в_приложении() -> None:
    оценки = _оценки([0.42], [0.0], [False], [True], [True])
    assert decide(оценки, sim_min=0.42, margin_min=0.0).tolist() == [MODEL]


def test_ответ_маркой_верен_по_марке() -> None:
    знакомые = _оценки(
        sim1=[0.9, 0.9],
        sim2=[0.89, 0.1],
        same_brand=[True, False],
        class_hit=[False, True],
        brand_hit=[True, True],
    )
    негативы = _оценки([0.1], [0.0], [False], [False], [False])

    точка = operating_point(знакомые, негативы, sim_min=0.5, margin_min=0.05)

    assert точка.coverage == 1.0
    assert точка.error == 0.0
    assert точка.fpr == 0.0


def test_выбор_пары_максимизирует_покрытие_при_ограничениях() -> None:
    # Три верных с высоким sim, два неверных с низким, негатив посередине.
    знакомые = _оценки(
        sim1=[0.9, 0.8, 0.7, 0.5, 0.45],
        sim2=[0.1] * 5,
        same_brand=[False] * 5,
        class_hit=[True, True, True, False, False],
        brand_hit=[True, True, True, False, False],
    )
    негативы = _оценки([0.6, 0.3], [0.1, 0.1], [False, False], [False] * 2, [False] * 2)

    точка = select_pair(
        знакомые,
        негативы,
        max_error=0.0,
        max_fpr=0.5,
        sim_grid=np.array([0.4, 0.5, 0.55, 0.7]),
        margin_grid=np.array([0.0, 0.05]),
    )

    assert точка is not None
    assert (точка.sim_min, точка.margin_min) == (0.55, 0.0)
    assert точка.coverage == pytest.approx(0.6)


def test_нет_допустимой_пары_значит_none() -> None:
    знакомые = _оценки([0.9], [0.1], [False], [False], [False])
    негативы = _оценки([0.9], [0.1], [False], [False], [False])

    assert (
        select_pair(
            знакомые,
            негативы,
            max_error=0.1,
            max_fpr=0.1,
            sim_grid=np.array([0.5]),
            margin_grid=np.array([0.0]),
        )
        is None
    )


def test_узлы_монотонны_и_первый_стоит_на_пороге() -> None:
    генератор = np.random.default_rng(1)
    sims = генератор.uniform(0.5, 1.0, 500)
    hits = генератор.uniform(size=500) < (sims - 0.4)

    узлы = calibration_knots(sims, hits, sim_min=0.47, max_knots=32)

    assert 2 <= len(узлы) <= 32
    assert узлы[0][0] == 0.47
    for (x0, y0), (x1, y1) in pairwise(узлы):
        assert x1 > x0 and y1 > y0
    assert all(0.0 <= y <= 1.0 for _, y in узлы)


def test_одинаковые_sim_с_разным_исходом_не_дают_двойного_x() -> None:
    sims = np.array([0.5, 0.5, 0.6, 0.6, 0.7, 0.8])
    hits = np.array([False, True, False, True, True, True])

    узлы = calibration_knots(sims, hits, sim_min=0.45)

    xs = [x for x, _ in узлы]
    assert len(xs) == len(set(xs))


def test_одна_ступень_роняет() -> None:
    with pytest.raises(CalibrationError, match="ступен"):
        calibration_knots(np.array([0.5, 0.6, 0.7]), np.array([True] * 3), sim_min=0.4)


def test_уверенность_зеркалит_confidence_from_sim() -> None:
    узлы = [(0.5, 0.2), (0.7, 0.6), (0.9, 0.99)]

    значения = confidence_from_sim(np.array([0.4, 0.5, 0.6, 0.95]), узлы)

    # ниже и на первом узле — clipLo; между — линейно; правее — последний, зажатый clipHi
    assert значения.tolist() == pytest.approx([0.01, 0.01, 0.4, 0.98])


def test_ece_ноль_на_идеальной_шкале() -> None:
    уверенность = np.array([0.25] * 4 + [0.75] * 4)
    исходы = np.array([True, False, False, False, True, True, True, False])
    assert ece(уверенность, исходы) == pytest.approx(0.0)


def test_половины_стратифицированы_и_детерминированы() -> None:
    группы = np.array([0] * 10 + [1] * 4 + [2])

    первая = split_halves(группы, seed=5)

    assert (первая == split_halves(группы, seed=5)).all()
    assert int(первая[группы == 0].sum()) == 5
    assert int(первая[группы == 1].sum()) == 2


def test_auroc_и_fpr() -> None:
    свои = np.array([0.9, 0.8, 0.7, 0.6])
    чужие = np.array([0.5, 0.65])
    assert auroc(свои, чужие) == pytest.approx(7 / 8)
    assert fpr_at_tpr(свои, чужие, tpr=0.75) == pytest.approx(0.0)


def test_документы_в_форме_ассетов() -> None:
    точка = Operating(0.55, 0.03, coverage=0.61, error=0.14, fpr=0.08, shown=100)

    пороги = thresholds_doc(точка, val_n=1500, neg_n=2400)
    калибровка = calibration_doc([(0.55, 0.3), (0.9, 0.95)])

    assert set(пороги) == set(ПЛЕЙСХОЛДЕР_ПОРОГОВ)
    assert пороги["tuned"] is True
    assert пороги["min_observations"] == ПЛЕЙСХОЛДЕР_ПОРОГОВ["min_observations"]
    assert пороги["min_short_side_px"] == ПЛЕЙСХОЛДЕР_ПОРОГОВ["min_short_side_px"]
    assert пороги["tuned_on"] == {
        "val_n": 1500,
        "neg_n": 2400,
        "achieved_tpr": 0.61,
        "achieved_fpr": 0.08,
    }
    assert set(калибровка) == set(ПЛЕЙСХОЛДЕР_КАЛИБРОВКИ)
    assert калибровка["tuned"] is True
    assert калибровка["clip"] == [0.01, 0.98]
