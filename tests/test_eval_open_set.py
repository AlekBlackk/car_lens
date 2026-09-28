"""Метрика открытого множества: пара держит все ограничения сразу (спека §4)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from training.calibration import REJECT_OFF, Scores, select_pair
from training.eval_open_set import (
    NO_REJECT,
    NONCAR,
    STREET,
    ModelResult,
    OpenSetFold,
    OpenSetPoint,
    binding_constraint,
    cross_fit_open_set,
    open_set_point,
    render_open_set_report,
    select_open_set_pair,
    split_by_group,
)

СЕТКА = np.array([0.70, 0.76, 0.85])
НОЛЬ = np.array([0.0])


def знакомые(*sims: float) -> Scores:
    n = len(sims)
    да = np.ones(n, dtype=bool)
    return Scores(np.array(sims), np.zeros(n), np.zeros(n, dtype=bool), да, да)


def чужие(*sims: float) -> Scores:
    n = len(sims)
    нет = np.zeros(n, dtype=bool)
    return Scores(np.array(sims), np.zeros(n), нет, нет, нет)


def test_пара_держит_ограничения_всех_групп() -> None:
    свои = знакомые(0.6, 0.7, 0.8, 0.9)
    группы = {STREET: чужие(0.55, 0.75), NONCAR: чужие(0.5)}

    точка = select_open_set_pair(свои, группы, sim_grid=СЕТКА, margin_grid=НОЛЬ)

    assert точка is not None
    assert (точка.sim_min, точка.margin_min, точка.coverage) == (0.76, 0.0, 0.5)
    assert точка.fpr == {STREET: 0.0, NONCAR: 0.0}


def test_пара_с_двумя_ограничениями_из_трёх_недопустима() -> None:
    свои = знакомые(0.9, 0.9)
    группы = {STREET: чужие(0.5), NONCAR: чужие(0.95)}

    assert select_pair(свои, группы[STREET], max_error=0.15, max_fpr=0.15) is not None
    assert select_open_set_pair(свои, группы) is None


def test_пустая_группа_роняет_замер() -> None:
    with pytest.raises(ValueError, match="не-машины"):
        select_open_set_pair(знакомые(0.9), {STREET: чужие(0.5), NONCAR: чужие()})


def test_узкое_ограничение_то_у_которого_меньше_запас() -> None:
    точка = OpenSetPoint(0.8, 0.0, 0.7, 0.05, 10, {STREET: 0.14, NONCAR: 0.02})
    assert binding_constraint(точка) == f"FPR {STREET}"


def test_деление_по_группам_не_кладёт_группу_в_обе_половины() -> None:
    группы = np.array(["a", "b", "a", "c", "b", "d", "a", "c"])
    маска = split_by_group(группы, seed=3)

    for группа in np.unique(группы):
        assert len(set(маска[группы == группа].tolist())) == 1
    assert маска.any() and not маска.all()
    assert np.array_equal(маска, split_by_group(группы, seed=3))


def test_отложенная_сторона_меряется_на_другой_половине() -> None:
    свои = знакомые(0.8, 0.9, 0.6, 0.7)
    группы = {STREET: чужие(0.5, 0.65), NONCAR: чужие(0.4, 0.45)}
    своя_a = np.array([True, True, False, False])
    половины = {STREET: np.array([True, False]), NONCAR: np.array([True, False])}

    стороны = cross_fit_open_set(свои, группы, своя_a, половины)

    a = стороны[0]
    assert a.tuned_on == "A" and a.chosen is not None and a.held_out is not None
    ожидание = open_set_point(
        свои.subset(~своя_a),
        {г: s.subset(~половины[г]) for г, s in группы.items()},
        a.chosen.sim_min,
        a.chosen.margin_min,
    )
    assert a.held_out == ожидание


def test_отчёт_несёт_стороны_итог_телефон_и_разделимость() -> None:
    точка = OpenSetPoint(0.84, 0.015, 0.72, 0.05, 1000, {STREET: 0.14, NONCAR: 0.03})
    результат = ModelResult(
        label="0.5",
        classifier_sha256="31e3f13c",
        folds=[OpenSetFold("A", точка, точка), OpenSetFold("B", None, None)],
        final=точка,
        phone=точка,
        reference_fpr_at_phone=0.149,
        separability={STREET: (0.81, 0.77), NONCAR: (0.95, 0.30)},
    )

    текст = render_open_set_report([результат], {"off_catalog_street": 1900}, {"seed": "20260927"})

    assert "## Модель 0.5" in текст
    assert "| A | подбор | 0.84 | 0.015 | 0.720 | 0.050 | 0.140 | 0.030 |" in текст
    assert "| B | подбор | — | — | — | — | — | — |" in текст
    assert f"узкое ограничение: FPR {STREET}" in текст
    assert "справочный off_catalog в точке телефона: FPR 0.149" in текст
    assert f"| {STREET} | 0.810 | 0.770 |" in текст
    assert "- off_catalog_street: 1900" in текст


С_ВЕТКОЙ = np.array([REJECT_OFF, 0.05])


def с_прототипом(оценки: Scores, *r: float) -> Scores:
    return replace(оценки, sim_reject=np.array(r))


def test_прототип_открывает_пару_которой_без_него_нет() -> None:
    свои = с_прототипом(знакомые(0.6, 0.7, 0.8, 0.9), 0.1, 0.1, 0.1, 0.1)
    группы = {
        STREET: с_прототипом(чужие(0.85, 0.95), 0.84, 0.94),
        NONCAR: с_прототипом(чужие(0.5), 0.1),
    }

    assert select_open_set_pair(свои, группы, sim_grid=СЕТКА, margin_grid=НОЛЬ) is None
    точка = select_open_set_pair(
        свои, группы, sim_grid=СЕТКА, margin_grid=НОЛЬ, reject_grid=С_ВЕТКОЙ
    )

    assert точка is not None
    assert (точка.sim_min, точка.reject_margin, точка.coverage) == (0.70, 0.05, 0.75)
    assert точка.fpr == {STREET: 0.0, NONCAR: 0.0}


def test_при_равном_покрытии_выбирается_выключенная_ветка() -> None:
    свои = с_прототипом(знакомые(0.6, 0.7, 0.8, 0.9), 0.1, 0.1, 0.1, 0.1)
    группы = {STREET: с_прототипом(чужие(0.55), 0.1), NONCAR: с_прототипом(чужие(0.5), 0.1)}

    точка = select_open_set_pair(
        свои, группы, sim_grid=СЕТКА, margin_grid=НОЛЬ, reject_grid=С_ВЕТКОЙ
    )

    assert точка is not None and точка.reject_margin == REJECT_OFF


def test_пределы_задаются_по_группам() -> None:
    свои = знакомые(0.9, 0.9)
    группы = {STREET: чужие(0.5), NONCAR: чужие(0.95)}

    assert select_open_set_pair(свои, группы) is None
    точка = select_open_set_pair(свои, группы, max_fpr={STREET: 0.15, NONCAR: 1.0})
    assert точка is not None and точка.fpr[NONCAR] == 1.0


def test_узкое_ограничение_по_пределам_групп() -> None:
    точка = OpenSetPoint(0.8, 0.0, 0.7, 0.05, 10, {STREET: 0.10, NONCAR: 0.08})
    assert binding_constraint(точка, max_fpr={STREET: 0.113, NONCAR: 0.086}) == f"FPR {NONCAR}"


def test_отложенная_сторона_меряется_с_порогом_прототипа() -> None:
    свои = с_прототипом(знакомые(0.8, 0.9, 0.8, 0.9), 0.1, 0.1, 0.1, 0.1)
    группы = {
        STREET: с_прототипом(чужие(0.95, 0.95), 0.94, 0.94),
        NONCAR: с_прототипом(чужие(0.4, 0.45), 0.1, 0.1),
    }
    своя_a = np.array([True, True, False, False])
    половины = {STREET: np.array([True, False]), NONCAR: np.array([True, False])}

    стороны = cross_fit_open_set(свои, группы, своя_a, половины, reject_grid=С_ВЕТКОЙ)

    a = стороны[0]
    assert a.chosen is not None and a.held_out is not None
    assert a.chosen.reject_margin == 0.05
    assert a.held_out.reject_margin == 0.05
    assert a.held_out.fpr[STREET] == 0.0


def test_без_сетки_прототипа_поведение_прежнее() -> None:
    assert NO_REJECT.tolist() == [REJECT_OFF]
    свои = знакомые(0.6, 0.7, 0.8, 0.9)
    группы = {STREET: чужие(0.55, 0.75), NONCAR: чужие(0.5)}
    точка = select_open_set_pair(свои, группы, sim_grid=СЕТКА, margin_grid=НОЛЬ)
    assert точка is not None and точка.reject_margin == REJECT_OFF
