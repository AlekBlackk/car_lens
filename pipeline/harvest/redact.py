"""Редакция номеров и лиц — правило 1 решения Р-11.

Кроп машины не удаляет её собственный номер, поэтому стадия обязательная и обхода у
неё нет: кадр, который не удалось замыть, отбраковывается целиком.

Замыв проверяется замывом. Повторная детекция обязательна на каждой ступени — без неё
утверждение «номер замыт» ничем не подтверждено.

Ступеней две, и это не перестраховка. Фиксированное ядро на снимке высокого разрешения
оставляет номер читаемым, поэтому радиус пропорционален боксу; но и терять кадр после
первой неудачи рано — сплошная заливка снимает вопрос там, где размытие на границе
порога ещё пробивается.

Вход и выход — BGR: детекторы OpenCV работают в нём же, конверсий здесь нет вовсе.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.crop import Detection, Detector

REDACTION_MARGIN = 0.10
MIN_KERNEL = 15


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Итог редакции одного изображения."""

    image: np.ndarray
    plates: int
    faces: int
    ok: bool


def redaction_zone(
    box: Detection, frame_w: int, frame_h: int, margin: float = REDACTION_MARGIN
) -> tuple[int, int, int, int]:
    """Зона поиска номеров и лиц: бокс с запасом, обрезанный кадром.

    Запас 10% против 8% у сохраняемого кропа (`crop.pad_box`) намеренно: зона детекции
    обязана быть шире кропа, иначе номер, наполовину срезанный границей, останется
    читаемым внутри.
    """
    dx = box.w * margin
    dy = box.h * margin
    x0 = max(0, round(box.x - dx))
    y0 = max(0, round(box.y - dy))
    x1 = min(frame_w, round(box.x + box.w + dx))
    y1 = min(frame_h, round(box.y + box.h + dy))
    return x0, y0, x1, y1


def blur_kernel(bbox_w: int) -> int:
    """Нечётный радиус размытия, пропорциональный ширине бокса."""
    return max(MIN_KERNEL, (bbox_w // 3) | 1)


def _замазать(image: np.ndarray, боксы: list[Detection], заливкой: bool) -> None:
    """Применить ступень замыва ко всем боксам на месте."""
    высота, ширина = image.shape[:2]
    for бокс in боксы:
        x0 = max(0, int(бокс.x))
        y0 = max(0, int(бокс.y))
        x1 = min(ширина, int(бокс.x + бокс.w))
        y1 = min(высота, int(бокс.y + бокс.h))
        if x1 <= x0 or y1 <= y0:
            continue
        if заливкой:
            image[y0:y1, x0:x1] = 0
            continue
        ядро = blur_kernel(x1 - x0)
        image[y0:y1, x0:x1] = cv2.GaussianBlur(image[y0:y1, x0:x1], (ядро, ядро), 0)


def redact(
    image: np.ndarray, plate_detector: Detector, face_detector: Detector
) -> RedactionResult:
    """Замыть номера и лица, подтвердив результат повторной детекцией."""
    рабочая = image.copy()
    номера = plate_detector.detect(рабочая)
    лица = face_detector.detect(рабочая)
    if not номера and not лица:
        return RedactionResult(image, 0, 0, True)

    найдено_номеров, найдено_лиц = len(номера), len(лица)
    for заливкой in (False, True):
        _замазать(рабочая, номера, заливкой)
        _замазать(рабочая, лица, заливкой)
        номера = plate_detector.detect(рабочая)
        лица = face_detector.detect(рабочая)
        if not номера and not лица:
            return RedactionResult(рабочая, найдено_номеров, найдено_лиц, True)

    return RedactionResult(рабочая, найдено_номеров, найдено_лиц, False)
