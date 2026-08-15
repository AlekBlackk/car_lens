"""Выбор главного автомобиля на фотографии и нарезка кропа (§3.4, §4.2).

Детектор передаётся снаружи через протокол `Detector`. Без этого ни один
тест не запустится без скачивания весов YOLO, а тесты, требующие сети,
перестают запускаться примерно через неделю.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

# COCO: 2 — car, 5 — bus, 7 — truck (§4.1).
VEHICLE_CLASSES = frozenset({2, 5, 7})


class Detection(NamedTuple):
    """Один бокс в координатах исходного фото: левый верхний угол и размеры."""

    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int

    @property
    def area(self) -> float:
        return self.w * self.h


class Detector(Protocol):
    """Что угодно, умеющее находить боксы на изображении."""

    def detect(self, image: np.ndarray) -> list[Detection]:  # pragma: no cover
        ...


@dataclass(frozen=True, slots=True)
class CropDecision:
    """Решение по одной фотографии."""

    box: Detection | None
    is_multi_car: bool
    area_frac: float
    reject_reason: str | None


def select_primary(
    dets: Sequence[Detection],
    frame_w: int,
    frame_h: int,
    min_conf: float = 0.5,
    min_area_frac: float = 0.15,
    rival_ratio: float = 0.40,
) -> CropDecision:
    """Выбрать главный автомобиль кадра по правилам §3.4.

    Кадры с двумя сопоставимыми машинами не отбрасываются, а помечаются
    `is_multi_car`. Из `train` их исключает `split.py`; в `val` и `test`
    они остаются, потому что на улице такой кадр — норма.
    """
    транспорт = [d for d in dets if d.cls in VEHICLE_CLASSES and d.conf > min_conf]
    if not транспорт:
        return CropDecision(None, False, 0.0, "no_vehicle")

    по_убыванию = sorted(транспорт, key=lambda d: d.area, reverse=True)
    главный = по_убыванию[0]
    площадь_кадра = float(frame_w * frame_h)
    доля = главный.area / площадь_кадра

    if доля < min_area_frac:
        return CropDecision(None, False, доля, "too_small")

    соперник = по_убыванию[1] if len(по_убыванию) > 1 else None
    много_машин = bool(соперник and соперник.area >= rival_ratio * главный.area)

    return CropDecision(главный, много_машин, доля, None)


def pad_box(
    box: Detection, frame_w: int, frame_h: int, pad: float = 0.08
) -> tuple[int, int, int, int]:
    """Расширить бокс на `pad` с каждой стороны, не выходя за кадр.

    Восемь процентов из §4.2: контекст крыши и колёс помогает различать
    кузова.
    """
    dx = box.w * pad
    dy = box.h * pad
    # `round` от float уже возвращает int, поэтому внешний `int(...)` из плана
    # убран: он был лишним и ruff (RUF046) на него ругается.
    x0 = max(0, round(box.x - dx))
    y0 = max(0, round(box.y - dy))
    x1 = min(frame_w, round(box.x + box.w + dx))
    y1 = min(frame_h, round(box.y + box.h + dy))
    return x0, y0, x1, y1


def crop_image(image: np.ndarray, box: Detection, pad: float = 0.08) -> np.ndarray:
    """Вырезать расширенный кроп из изображения."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = pad_box(box, w, h, pad)
    return image[y0:y1, x0:x1]
