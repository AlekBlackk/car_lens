"""Два фильтра разной природы: жёсткий отсев мусора и мягкий ранг для аудита.

YOLO-гейт переиспользует `crop.select_primary` целиком — правила §3.4 уже написаны и
протестированы. Отличие одно: многомашинный кадр здесь отбраковывается, а не помечается.
На M1 такой кадр остаётся в `val` намеренно (на улице это норма), но здесь метка класса
пришла из категории Commons и относится к галерее, а не к конкретному боксу — какая из
двух машин та самая, неизвестно.

CLIP делает ровно две вещи. Жёстко режет то, в чём он силён: категорию изображения —
салон, двигатель, игрушка, рисунок. И мягко ранжирует по классам каталога, ничего не
отбраковывая: у сильнейшего кандидата Р-10 top-3 на RU+CN равен 0.511, и гейт по рангу
выбросил бы половину правильных фотографий в самых дефицитных классах.

**Единственная конверсия BGR → RGB конвейера живёт здесь**: энкодеры CLIP обучены на
RGB, всё остальное вокруг работает в BGR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from pipeline.crop import Detection, Detector, select_primary

NEGATIVE_PROMPTS = (
    "a photo of a car interior",
    "a photo of a car dashboard",
    "a photo of a car engine",
    "a photo of a car part close-up",
    "a photo of a toy car or scale model",
    "a photo of a drawing or blueprint of a car",
    "a photo of a truck or a bus",
)


class ImageEncoder(Protocol):
    """Энкодер изображений CLIP. Вход — RGB, выход — нормированный вектор."""

    def encode_rgb(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


@dataclass(frozen=True, slots=True)
class VehicleVerdict:
    """Решение YOLO-гейта: бокс главной машины или причина отказа."""

    box: Detection | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ClipVerdict:
    """Решение CLIP: жёсткий отсев мусора и мягкие числа для аудита."""

    is_car_photo: bool
    rank: int
    margin: float


def vehicle_gate(image: np.ndarray, detector: Detector) -> VehicleVerdict:
    """Проверить кадр правилами §3.4 и вернуть бокс главной машины."""
    высота, ширина = image.shape[:2]
    решение = select_primary(detector.detect(image), ширина, высота)
    if решение.reject_reason is not None:
        return VehicleVerdict(None, решение.reject_reason)
    if решение.is_multi_car:
        return VehicleVerdict(None, "multi_car")
    return VehicleVerdict(решение.box, None)


def clip_verdict(
    image: np.ndarray,
    encoder: ImageEncoder,
    class_matrix: np.ndarray,
    negative_matrix: np.ndarray,
    target_index: int,
) -> ClipVerdict:
    """Отсеять мусор и посчитать ранг целевого класса среди всех классов каталога."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    вектор = encoder.encode_rgb(rgb)
    по_классам = class_matrix @ вектор
    по_негативам = negative_matrix @ вектор

    лучший_класс = float(по_классам.max())
    целевой = float(по_классам[target_index])
    # Ранг честный по отношению к продукту: сравнение идёт со всеми классами каталога,
    # а не только с теми, что сейчас собираются.
    ранг = int((по_классам > целевой).sum()) + 1
    return ClipVerdict(
        is_car_photo=лучший_класс > float(по_негативам.max()),
        rank=ранг,
        margin=лучший_класс - целевой,
    )
