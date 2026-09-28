"""Гейт доставки прототипов и парный счёт показанных знакомых (спека 2026-09-28 §4)."""

from __future__ import annotations

import numpy as np

from training.calibration import REJECT_OFF, Scores
from training.eval_open_set import NONCAR, STREET, OpenSetFold, OpenSetPoint
from training.eval_reject import (
    GateResult,
    Paired,
    RejectResult,
    gate,
    paired_shown,
    render_reject_report,
)


def точка(покрытие: float, reject: float = REJECT_OFF) -> OpenSetPoint:
    return OpenSetPoint(0.8, 0.01, покрытие, 0.05, 100, {STREET: 0.1, NONCAR: 0.08}, reject)


def стороны(a: float | None, b: float | None) -> list[OpenSetFold]:
    return [
        OpenSetFold("A", точка(0.8), None if a is None else точка(a)),
        OpenSetFold("B", точка(0.8), None if b is None else точка(b)),
    ]


БАЗА = стороны(0.772, 0.806)
ТЕЛЕФОН = точка(0.757)
ХОРОШИЙ_ПАРНЫЙ = Paired(wins=40, losses=10, p=0.00001)


def test_гейт_проходит_при_всех_трёх_условиях() -> None:
    итог = gate(БАЗА, стороны(0.80, 0.83), ТЕЛЕФОН, точка(0.78, 0.03), ХОРОШИЙ_ПАРНЫЙ)
    assert итог.passed
    assert [проверка.passed for проверка in итог.checks] == [True, True, True]


def test_прирост_ровно_002_проходит_несмотря_на_округление() -> None:
    # 0.826 − 0.806 во float64 даёт 0.019999999999999907.
    итог = gate(БАЗА, стороны(0.792, 0.826), ТЕЛЕФОН, точка(0.78, 0.03), ХОРОШИЙ_ПАРНЫЙ)
    assert итог.checks[0].passed


def test_мал_прирост_одной_половины_роняет_первое_условие() -> None:
    итог = gate(БАЗА, стороны(0.80, 0.81), ТЕЛЕФОН, точка(0.78, 0.03), ХОРОШИЙ_ПАРНЫЙ)
    assert not итог.passed and not итог.checks[0].passed


def test_нет_отложенной_точки_роняет_первое_условие() -> None:
    итог = gate(БАЗА, стороны(0.80, None), ТЕЛЕФОН, точка(0.78, 0.03), ХОРОШИЙ_ПАРНЫЙ)
    assert not итог.checks[0].passed


def test_не_хуже_с_выключенной_веткой_не_проходит() -> None:
    итог = gate(БАЗА, стороны(0.80, 0.83), ТЕЛЕФОН, точка(0.78), ХОРОШИЙ_ПАРНЫЙ)
    assert not итог.passed and not итог.checks[1].passed


def test_не_хуже_без_прироста_покрытия_не_проходит() -> None:
    итог = gate(БАЗА, стороны(0.80, 0.83), ТЕЛЕФОН, точка(0.757, 0.03), ХОРОШИЙ_ПАРНЫЙ)
    assert not итог.checks[1].passed


def test_нет_точки_не_хуже_роняет_второе_и_третье() -> None:
    итог = gate(БАЗА, стороны(0.80, 0.83), ТЕЛЕФОН, None, None)
    assert [проверка.passed for проверка in итог.checks] == [True, False, False]


def test_незначимый_парный_счёт_роняет_третье_условие() -> None:
    итог = gate(БАЗА, стороны(0.80, 0.83), ТЕЛЕФОН, точка(0.78, 0.03), Paired(12, 8, 0.5))
    assert not итог.passed and not итог.checks[2].passed


def test_парный_счёт_по_показанным_кадрам() -> None:
    n = 4
    знакомые = Scores(
        np.array([0.9, 0.9, 0.82, 0.5]),
        np.zeros(n),
        np.zeros(n, dtype=bool),
        np.ones(n, dtype=bool),
        np.ones(n, dtype=bool),
        np.array([0.1, 0.89, 0.1, 0.1]),
    )
    база = OpenSetPoint(0.84, 0.0, 0.5, 0.0, 2, {STREET: 0.0, NONCAR: 0.0})
    новая = OpenSetPoint(0.80, 0.0, 0.5, 0.0, 2, {STREET: 0.0, NONCAR: 0.0}, 0.05)

    # База показывает кадры 0 и 1; новая — 0 и 2 (кадр 1 срезан прототипом, 2 открыт порогом).
    assert paired_shown(знакомые, база, новая) == Paired(wins=1, losses=1, p=1.0)


def test_отчёт_несёт_обе_модели_точки_и_вердикт() -> None:
    результат = RejectResult(
        base_folds=БАЗА,
        new_folds=стороны(0.80, 0.83),
        base_final=точка(0.795),
        new_final=точка(0.82, 0.04),
        base_phone=ТЕЛЕФОН,
        no_worse=точка(0.78, 0.03),
        paired=ХОРОШИЙ_ПАРНЫЙ,
        separability={STREET: (0.871, 0.905)},
        gate=GateResult(True, []),
    )

    текст = render_reject_report(результат, {"off_catalog_street": 1445}, {"seed": "20260927"})

    assert "| 0.5 | итог | — | 0.80 | 0.010 | выкл | 0.795 |" in текст
    assert "| с прототипами | итог | — | 0.80 | 0.010 | 0.04 | 0.820 |" in текст
    assert "| «не хуже 0.5» | 0.80 | 0.010 | 0.03 | 0.780 |" in текст
    assert "выигрышей 40, проигрышей 10" in текст
    assert f"| {STREET} | 0.871 | 0.905 |" in текст
    assert "**Вердикт:** доставка" in текст
