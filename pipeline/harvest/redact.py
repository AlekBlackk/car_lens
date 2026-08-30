"""Редакция номеров и лиц — правило 1 решения Р-11.

Кроп машины не удаляет её собственный номер, поэтому стадия обязательная и обхода у неё
нет: кадр, который не удалось замыть, отбраковывается целиком.

Редакция полнокадровая и идёт до кропа (§6): номер фоновой машины, попавшей краем в
сохраняемый кроп, обязан уйти так же, как собственный номер главной. Кроп вырезается из
уже очищенного кадра — всё, что в него попадёт, по построению отредактировано.

Номера заливаются сплошным цветом сразу: полумеры с размытием оставляли символы
читаемыми. Лица сохраняют лестницу размыв→заливка — размытого лица достаточно для «не
идентифицируется», а чёрный прямоугольник вредит данным.

Вход и выход — BGR: детекторы OpenCV работают в нём же, конверсий здесь нет.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.crop import Detection, Detector
from pipeline.harvest.tiling import nms_merge

MIN_PLATE_W_PX = 16
MIN_FACE_SIDE_PX = 20
PLATE_FILL_PAD = 0.15
VEHICLE_SEARCH_PAD = 0.15
MIN_KERNEL = 15


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Итог редакции одного кадра."""

    image: np.ndarray
    plates: int
    faces: int
    ok: bool


def blur_kernel(bbox_w: int) -> int:
    """Нечётный радиус размытия, пропорциональный ширине бокса."""
    return max(MIN_KERNEL, (bbox_w // 3) | 1)


def vehicle_search_box(
    box: Detection, frame_w: int, frame_h: int, pad: float = VEHICLE_SEARCH_PAD
) -> tuple[int, int, int, int]:
    """Бокс ТС, расширенный на `pad`: номер часто ниже бампера, вне тесного бокса."""
    dx = box.w * pad
    dy = box.h * pad
    x0 = max(0, round(box.x - dx))
    y0 = max(0, round(box.y - dy))
    x1 = min(frame_w, round(box.x + box.w + dx))
    y1 = min(frame_h, round(box.y + box.h + dy))
    return x0, y0, x1, y1


def _крупные(боксы: Sequence[Detection], min_w: int, min_h: int) -> list[Detection]:
    """Отсечь детекции ниже порога различимости."""
    return [б for б in боксы if б.w >= min_w and б.h >= min_h]


def _детект_номера(
    image: np.ndarray, vehicle_boxes: Sequence[Detection], detectors: Sequence[Detector]
) -> list[Detection]:
    """Ансамбль детекторов номеров: по всему кадру и по каждому боксу ТС с запасом."""
    высота, ширина = image.shape[:2]
    зоны: list[tuple[int, int, int, int]] = [(0, 0, ширина, высота)]
    зоны += [vehicle_search_box(б, ширина, высота) for б in vehicle_boxes]

    найдено: list[Detection] = []
    for детектор in detectors:
        for x0, y0, x1, y1 in зоны:
            вырез = image[y0:y1, x0:x1]
            if вырез.size == 0:
                continue
            for б in детектор.detect(вырез):
                найдено.append(Detection(б.x + x0, б.y + y0, б.w, б.h, б.conf, б.cls))
    return nms_merge(_крупные(найдено, MIN_PLATE_W_PX, 1))


def _залить_номер(image: np.ndarray, box: Detection) -> None:
    """Сплошная заливка нулём, расширенная на PLATE_FILL_PAD."""
    высота, ширина = image.shape[:2]
    dx = box.w * PLATE_FILL_PAD
    dy = box.h * PLATE_FILL_PAD
    x0 = max(0, int(box.x - dx))
    y0 = max(0, int(box.y - dy))
    x1 = min(ширина, int(box.x + box.w + dx))
    y1 = min(высота, int(box.y + box.h + dy))
    if x1 > x0 and y1 > y0:
        image[y0:y1, x0:x1] = 0


def _замыть_лицо(image: np.ndarray, box: Detection, заливкой: bool) -> None:
    """Одна ступень замыва лица на месте."""
    высота, ширина = image.shape[:2]
    x0 = max(0, int(box.x))
    y0 = max(0, int(box.y))
    x1 = min(ширина, int(box.x + box.w))
    y1 = min(высота, int(box.y + box.h))
    if x1 <= x0 or y1 <= y0:
        return
    if заливкой:
        image[y0:y1, x0:x1] = 0
        return
    ядро = blur_kernel(x1 - x0)
    image[y0:y1, x0:x1] = cv2.GaussianBlur(image[y0:y1, x0:x1], (ядро, ядро), 0)


def redact_frame(
    image: np.ndarray,
    vehicle_boxes: Sequence[Detection],
    plate_detectors: Sequence[Detector],
    face_detector: Detector,
) -> RedactionResult:
    """Очистить весь кадр: номера — заливкой, лица — лестницей, с повторной проверкой."""
    рабочая = image.copy()

    номера = _детект_номера(рабочая, vehicle_boxes, plate_detectors)
    for бокс in номера:
        _залить_номер(рабочая, бокс)
    if _детект_номера(рабочая, vehicle_boxes, plate_detectors):
        return RedactionResult(рабочая, len(номера), 0, False)

    лица = _крупные(face_detector.detect(рабочая), MIN_FACE_SIDE_PX, MIN_FACE_SIDE_PX)
    лица_число = len(лица)
    for заливкой in (False, True):
        if not лица:
            break
        for бокс in лица:
            _замыть_лицо(рабочая, бокс, заливкой)
        лица = _крупные(face_detector.detect(рабочая), MIN_FACE_SIDE_PX, MIN_FACE_SIDE_PX)
    ok = not лица
    return RedactionResult(рабочая, len(номера), лица_число, ok)
