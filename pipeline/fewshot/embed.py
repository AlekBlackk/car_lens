"""Проход энкодера CLIP по кропам: пути → матрица нормализованных эмбеддингов.

Модуль тонкий намеренно. Всё, что зависит от весов, живёт в `ClipImageEncoder`
(`pipeline/harvest/backends.py`) и приходит сюда параметром, поэтому проход проверяется в
CI на подставном энкодере — без torch, без весов и без сети.

Протокол энкодера берётся из `pipeline/harvest/filters.py`, а не объявляется заново: это
тот же контракт «RGB на входе, вектор на выходе», и вторая копия разъехалась бы с первой
при первой же правке.

Конвертация BGR → RGB — по тому же правилу, что в харвесте: `cv2.imread` отдаёт BGR, а
энкодеры CLIP обучены на RGB.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from pipeline.harvest.filters import ImageEncoder


class EmbedError(RuntimeError):
    """Кропы нельзя превратить в эмбеддинги."""


def embed_crops(paths: Sequence[Path], encoder: ImageEncoder) -> np.ndarray:
    """Матрица `(len(paths), dim)` нормализованных эмбеддингов в порядке `paths`.

    Порядок сохраняется, потому что метки классов вызывающий держит отдельным массивом:
    перестановка строк молча перемешала бы разметку.

    Нормализация делается здесь, даже если энкодер уже нормализовал. Косинус в
    `pipeline.knn` — это скалярное произведение, и его правильность не должна зависеть от
    того, какую реализацию энкодера подставили.
    """
    if not paths:
        raise EmbedError("пустой список путей: эмбеддинги считать нечему")
    векторы: list[np.ndarray] = []
    for путь in paths:
        кадр = cv2.imread(str(путь))
        if кадр is None:
            raise EmbedError(f"{путь}: файл не читается как изображение")
        векторы.append(encoder.encode_rgb(cv2.cvtColor(кадр, cv2.COLOR_BGR2RGB)))
    сложенные = np.stack(векторы).astype(np.float32)
    if сложенные.ndim != 2:
        raise EmbedError(f"энкодер вернул форму {сложенные.shape}, ожидалась (n, dim)")
    нормы = np.linalg.norm(сложенные, axis=1, keepdims=True)
    if bool(np.any(нормы == 0)):
        raise EmbedError("нулевой эмбеддинг: нормировать нечем")
    итог: np.ndarray = (сложенные / нормы).astype(np.float32)
    return итог
