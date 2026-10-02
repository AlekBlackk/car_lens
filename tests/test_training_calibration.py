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
    REJECT_GRID,
    REJECT_OFF,
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


def _с_прототипом(sim1, sim2, same_brand, sim_reject) -> Scores:  # type: ignore[no-untyped-def]
    n = len(sim1)
    return Scores(
        sim1=np.asarray(sim1, dtype=np.float64),
        sim2=np.asarray(sim2, dtype=np.float64),
        same_brand=np.asarray(same_brand, dtype=bool),
        class_hit=np.zeros(n, dtype=bool),
        brand_hit=np.zeros(n, dtype=bool),
        sim_reject=np.asarray(sim_reject, dtype=np.float64),
    )


ОБЩИЕ_С_TS = _с_прототипом(
    sim1=[0.30, 0.75, 0.75, 0.75, 0.75],
    sim2=[0.10, 0.25, 0.25, 0.74, 0.74],
    same_brand=[False, False, False, True, True],
    sim_reject=[0.90, 0.6875, 0.625, 0.0, 0.6875],
)


def test_ветка_прототипа_зеркалит_decide_ts() -> None:
    решения = decide(ОБЩИЕ_С_TS, sim_min=0.42, margin_min=0.05, reject_margin=0.125)
    assert решения.tolist() == [REJECT, REJECT, MODEL, BRAND, REJECT]


def test_выключенная_ветка_решает_как_без_прототипов() -> None:
    без = Scores(
        ОБЩИЕ_С_TS.sim1,
        ОБЩИЕ_С_TS.sim2,
        ОБЩИЕ_С_TS.same_brand,
        ОБЩИЕ_С_TS.class_hit,
        ОБЩИЕ_С_TS.brand_hit,
    )
    выключена = decide(ОБЩИЕ_С_TS, 0.42, 0.05, REJECT_OFF)
    assert выключена.tolist() == decide(без, 0.42, 0.05).tolist()
    assert decide(без, 0.42, 0.05, reject_margin=0.125).tolist() == выключена.tolist()


def test_подмножество_несёт_прототип() -> None:
    часть = ОБЩИЕ_С_TS.subset(np.array([False, True, False, False, True]))
    assert часть.sim_reject is not None
    assert часть.sim_reject.tolist() == [0.6875, 0.6875]


def test_сетка_прототипа_начинается_с_выключенной_ветки() -> None:
    assert REJECT_GRID[0] == REJECT_OFF
    assert REJECT_GRID[1:].tolist() == [round(-0.05 + 0.01 * i, 2) for i in range(16)]


def test_рабочая_точка_несёт_порог_прототипа() -> None:
    свои = _с_прототипом([0.75, 0.75], [0.25, 0.25], [False, False], [0.0, 0.7])
    свои = Scores(
        свои.sim1, свои.sim2, свои.same_brand, np.ones(2, bool), np.ones(2, bool), свои.sim_reject
    )
    чужие = _с_прототипом([0.75], [0.25], [False], [0.74])
    точка = operating_point(свои, чужие, 0.42, 0.05, reject_margin=0.125)
    assert точка.reject_margin == 0.125
    assert точка.coverage == 0.5
    assert точка.fpr == 0.0


def test_документ_порогов_пишет_прототип_только_при_конечном_пороге() -> None:
    с_веткой = Operating(0.8, 0.01, 0.8, 0.05, 0.1, shown=10, reject_margin=0.03)
    без_ветки = Operating(0.8, 0.01, 0.8, 0.05, 0.1, shown=10)
    документ = thresholds_doc(с_веткой, val_n=10, neg_n=10)
    assert документ["reject_margin"] == 0.03
    assert list(документ)[:5] == ["version", "tuned", "sim_min", "margin_min", "reject_margin"]
    assert "reject_margin" not in thresholds_doc(без_ветки, val_n=10, neg_n=10)


def test_лишняя_группа_негативов_сужает_допустимые_пары() -> None:
    знакомые = _оценки(
        sim1=[0.9, 0.8, 0.7, 0.5, 0.45],
        sim2=[0.1] * 5,
        same_brand=[False] * 5,
        class_hit=[True, True, True, False, False],
        brand_hit=[True, True, True, False, False],
    )
    негативы = _оценки([0.6, 0.3], [0.1, 0.1], [False, False], [False] * 2, [False] * 2)
    лишние = {"не-машины": _оценки([0.6, 0.6], [0.1, 0.1], [False] * 2, [False] * 2, [False] * 2)}

    точка = select_pair(
        знакомые,
        негативы,
        max_error=0.0,
        max_fpr=0.5,
        extra_negatives=лишние,
        sim_grid=np.array([0.4, 0.5, 0.55, 0.7]),
        margin_grid=np.array([0.0, 0.05]),
    )

    assert точка is not None
    assert (точка.sim_min, точка.margin_min) == (0.7, 0.0)
