"""Сопоставление боксов fp16 с fp32 и гейт предрегистрации (M4)."""

from __future__ import annotations

from export.detector_parity import (
    GATE_EQUAL_COUNT,
    SetTotals,
    match,
    primary_agrees,
    render_report,
    verdict,
)
from pipeline.crop import Detection


def бокс(
    x: float, y: float, w: float = 100, h: float = 80, conf: float = 0.9, cls: int = 2
) -> Detection:
    return Detection(x=x, y=y, w=w, h=h, conf=conf, cls=cls)


def test_совпавшие_боксы_образуют_пару() -> None:
    assert match([бокс(10, 10)], [бокс(11, 10)]) == [(0, 0)]


def test_другой_класс_пары_не_даёт() -> None:
    assert match([бокс(10, 10, cls=2)], [бокс(10, 10, cls=7)]) == []


def test_бокс_ниже_порога_iou_пары_не_даёт() -> None:
    assert match([бокс(0, 0)], [бокс(90, 0)]) == []


def test_один_бокс_проверяемого_не_достаётся_двум() -> None:
    пары = match([бокс(10, 10, conf=0.9), бокс(12, 10, conf=0.8)], [бокс(11, 10)])
    assert пары == [(0, 0)]


def test_главный_бокс_с_другим_классом_расходится() -> None:
    assert not primary_agrees([бокс(10, 10, cls=2)], [бокс(10, 10, cls=7)])


def test_пустой_кадр_с_обеих_сторон_согласен_а_с_одной_нет() -> None:
    assert primary_agrees([], [])
    assert not primary_agrees([бокс(10, 10)], [])


def _набор(имя: str, кадры: int, расхождений: int = 0) -> SetTotals:
    набор = SetTotals(имя)
    for номер in range(кадры):
        эталон = [бокс(10, 10)]
        проверяемый = [бокс(10, 10)] if номер >= расхождений else []
        набор.add(f"{имя}/{номер}.jpg", эталон, проверяемый)
    return набор


def test_вердикт_проходит_на_идентичных_боксах() -> None:
    итог = verdict([_набор("mad_cars", 200), _набор("coco", 200)])
    assert итог["passed"]
    assert итог["conditions"]["c_iou_min"]["value"] == 1.0


def test_вердикт_падает_когда_кадров_с_равным_числом_меньше_предела() -> None:
    # 2 из 100 = 0.98 < 0.995: пропавший бокс на двух кадрах не проходит.
    итог = verdict([_набор("mad_cars", 100, расхождений=2)])
    условие = итог["conditions"]["a_equal_count[mad_cars]"]
    assert условие["value"] < GATE_EQUAL_COUNT
    assert not итог["passed"]


def test_вердикт_без_пар_не_проходит() -> None:
    # Минимум IoU по пустому списку — не «идеально», а отсутствие измерения.
    итог = verdict([SetTotals("coco")])
    assert not итог["passed"]


def test_отчёт_называет_вердикт_и_худшие_кадры() -> None:
    наборы = [_набор("mad_cars", 10)]
    текст = render_report(наборы, verdict(наборы), {"fp32": "a.tflite", "fp16": "b.tflite"})
    assert "гейт" in текст
    assert "mad_cars/0.jpg" in текст
