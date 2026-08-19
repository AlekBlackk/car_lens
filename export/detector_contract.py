"""Контракт экспортированного детектора: что именно ждёт граф (§7.1 дизайна M-1.5).

Формы тензоров контрактом не являются. Два графа с одинаковыми `1×416×416×3` и
`1×84×N` расходятся диапазоном входа, порядком каналов, форматом бокса и единицами
координат, и каждое расхождение даёт рамку, уехавшую на четверть кадра, при полностью
зелёной сверке форм. Поэтому контракт выясняется целиком и записывается в отчёт.

Модуль намеренно не импортирует ultralytics и не грузит веса: он разбирает уже
снятые с интерпретатора подробности, поэтому проверяется в CI, где нет ни того ни
другого.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# §4.1: COCO 2 — car, 5 — bus, 7 — truck. То же множество, что `pipeline/crop.py:17`.
VEHICLE_CLASSES = (2, 5, 7)
COCO_CLASSES = 80
# Ultralytics заполняет поля letterbox серым 114 и делит вход на 255 без вычитания
# среднего. Оба числа — часть контракта, а не деталь реализации: применить к этому
# входу нормализацию классификатора значит сдвинуть все активации.
LETTERBOX_PAD_VALUE = 114
MAX_DET = 100


class ContractError(ValueError):
    """Граф не соответствует ожидаемому контракту детектора."""


@dataclass(frozen=True, slots=True)
class DetectorContract:
    imgsz: int
    input_dtype: str
    input_layout: str
    channel_order: str
    input_range: str
    letterbox_pad_value: int
    output_layout: str
    box_format: str
    coord_units: str
    num_classes: int
    vehicle_classes: tuple[int, ...]
    score_layout: str
    max_det: int

    def to_json(self) -> dict[str, Any]:
        return {
            "imgsz": self.imgsz,
            "input_dtype": self.input_dtype,
            "input_layout": self.input_layout,
            "channel_order": self.channel_order,
            "input_range": self.input_range,
            "letterbox_pad_value": self.letterbox_pad_value,
            "output_layout": self.output_layout,
            "box_format": self.box_format,
            "coord_units": self.coord_units,
            "num_classes": self.num_classes,
            "vehicle_classes": list(self.vehicle_classes),
            "score_layout": self.score_layout,
            "max_det": self.max_det,
        }


def _раскладка_входа(форма: list[int], imgsz: int) -> str:
    if форма == [1, imgsz, imgsz, 3]:
        return "NHWC"
    if форма == [1, 3, imgsz, imgsz]:
        return "NCHW"
    несовпавшие = [значение for значение in форма if значение not in (1, 3, imgsz)]
    raise ContractError(
        f"вход {форма} не сводится к 1×{imgsz}×{imgsz}×3 ни в одной раскладке"
        + (f"; лишние размеры {несовпавшие}" if несовпавшие else "")
    )


def _раскладка_выхода(форма: list[int]) -> tuple[str, int]:
    """Вернуть раскладку и число каналов признаков (4 + классы)."""
    if len(форма) != 3 or форма[0] != 1:
        raise ContractError(f"выход {форма}, а ожидается три оси с ведущей единицей")
    _, вторая, третья = форма
    # Каналов всегда меньше, чем предсказаний: у v11 на 416 их 3549 против 84.
    if вторая < третья:
        return "CN", вторая
    return "NC", третья


def _единицы_координат(probe: Mapping[str, Any], imgsz: int) -> str:
    максимум = float(probe["max_coord"])
    if 0.0 <= максимум <= 1.5:
        return "normalized"
    # Пиксели начинаются не сразу за 1.5, а с десятой доли кадра. Полоса между
    # ними — (1.5, imgsz*0.1] — правдоподобна в обеих трактовках: три пикселя на
    # кадре 416 это и крошечная рамка в углу, и нормализованная координата, слегка
    # уехавшая за единицу. Обе догадки молча дают рамку не на машине, поэтому полоса
    # не выбирается ни в одну сторону, а роняет прогон.
    if float(imgsz) * 0.1 < максимум <= float(imgsz) * 1.05:
        return "pixels"
    raise ContractError(
        f"максимальная координата {максимум} не похожа ни на нормализованную (0…1), "
        f"ни на пиксельную относительно {imgsz}"
    )


def contract_from_graph(
    input_details: Mapping[str, Any],
    output_details: Mapping[str, Any],
    *,
    imgsz: int,
    probe: Mapping[str, Any],
) -> DetectorContract:
    """Собрать контракт из подробностей графа и зонда одного прогона."""
    тип_входа = str(input_details["dtype"])
    if тип_входа != "float32":
        raise ContractError(
            f"граница входа {тип_входа}, а веха берёт fp32-детектор: квантованный "
            "требует своих шкал и своей приёмки"
        )
    раскладка = _раскладка_входа([int(значение) for значение in input_details["shape"]], imgsz)
    раскладка_выхода, каналов = _раскладка_выхода(
        [int(значение) for значение in output_details["shape"]]
    )
    классов = каналов - 4
    if классов != COCO_CLASSES:
        raise ContractError(
            f"классов в выходе {классов}, а веса COCO дают {COCO_CLASSES}: "
            "перепутана модель или в выходе есть objectness"
        )
    return DetectorContract(
        imgsz=imgsz,
        input_dtype=тип_входа,
        input_layout=раскладка,
        channel_order="RGB",
        input_range="unit",
        letterbox_pad_value=LETTERBOX_PAD_VALUE,
        output_layout=раскладка_выхода,
        box_format="xywh",
        coord_units=_единицы_координат(probe, imgsz),
        num_classes=классов,
        vehicle_classes=VEHICLE_CLASSES,
        score_layout="classes_only",
        max_det=MAX_DET,
    )
