from dataclasses import replace

import pytest

from spike.clip_zeroshot.gate import (
    G1_TOP1,
    G2_CONTROL,
    G6_MS,
    Outcome,
    PhaseOneResult,
    check_phase_one,
    decide,
)

ПРОХОДНОЙ = PhaseOneResult(
    stanford_top1=0.80,
    stanford_top3=0.90,
    control_top3=0.85,
    ru_cn_top3=0.55,
    core_top3={
        "lada__2107": 0.5,
        "lada__granta": 0.5,
        "lada__niva_2121": 0.5,
        "uaz__patriot": 0.5,
    },
)


def test_пороги_дословно_из_спеки() -> None:
    assert G1_TOP1 == 0.75
    assert G2_CONTROL == 0.80
    assert pytest.approx(48.7) == G6_MS


def test_проходной_кандидат_проходит() -> None:
    прошёл, причины = check_phase_one(ПРОХОДНОЙ)
    assert прошёл is True
    assert причины == []


def test_один_ядровой_класс_на_нуле_роняет_всё() -> None:
    """G4 поклассовый: провал одного из четырёх — приговор, а не среднее по больнице."""
    итог = replace(ПРОХОДНОЙ, core_top3={**ПРОХОДНОЙ.core_top3, "uaz__patriot": 0.1})
    прошёл, причины = check_phase_one(итог)
    assert прошёл is False
    assert any("uaz__patriot" in причина for причина in причины)


def test_провал_контроля_даёт_особую_причину() -> None:
    """G2 — валидность замера: выборку переделывают, метод не хоронят."""
    итог = replace(ПРОХОДНОЙ, control_top3=0.5)
    прошёл, причины = check_phase_one(итог)
    assert прошёл is False
    assert any("замер недействителен" in причина for причина in причины)


def test_граница_включительно() -> None:
    итог = PhaseOneResult(
        stanford_top1=0.75,
        stanford_top3=0.85,
        control_top3=0.80,
        ru_cn_top3=0.50,
        core_top3=dict.fromkeys(ПРОХОДНОЙ.core_top3, 0.40),
    )
    прошёл, _ = check_phase_one(итог)
    assert прошёл is True


def test_исход_go_требует_всех_ворот() -> None:
    паритет = {"g7a": 0.99995, "g7b": 0.995, "g7c": 0.95}
    assert decide(True, 30.0, 40.0, паритет) is Outcome.GO


def test_точность_есть_размер_не_влез_это_частично() -> None:
    паритет = {"g7a": 0.99995, "g7b": 0.995, "g7c": 0.95}
    assert decide(True, 86.0, 40.0, паритет) is Outcome.PARTIAL


def test_провал_точности_это_no_go_независимо_от_остального() -> None:
    assert decide(False, 8.0, 10.0, {"g7a": 1.0, "g7b": 1.0, "g7c": 1.0}) is Outcome.NO_GO


def test_провал_паритета_это_no_go_а_не_частично() -> None:
    """Модель, считающая на телефоне не то же самое, не годится ни в каком смысле."""
    паритет = {"g7a": 0.99995, "g7b": 0.995, "g7c": 0.7}
    assert decide(True, 8.0, 10.0, паритет) is Outcome.NO_GO


def test_фаза_два_не_запускалась_это_no_go() -> None:
    assert decide(True, None, None, None) is Outcome.NO_GO
