"""Сведение эмбеддингов класса к фиксированному числу центроидов (§13.2).

Это не формат: `CentroidError` означает, что виноваты данные класса, а не файл.
k-means передаётся параметром — sklearn стоит только в группе `train`, а раскладка
слотов обязана проверяться в CI.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class CentroidError(ValueError):
    """Класс нельзя свести к центроидам."""


def centroids_for_class(
    embeddings: np.ndarray,
    *,
    centroids_per_class: int,
    kmeans: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Центроиды одного класса: `(centroids_per_class, dim)`, L2-нормализованные.

    `embeddings` — `(n, dim)`, уже нормализованные. При `n < centroids_per_class`
    кластеров берётся `n`, недостающие слоты заполняются последним центроидом:
    формат требует фиксированное число слотов, а выдумывать эталоны неоткуда.
    """
    if embeddings.ndim != 2:
        raise CentroidError(f"ожидалась матрица (n, dim), получено {embeddings.shape}")
    if len(embeddings) == 0:
        raise CentroidError("класс без эталонов: центроиды не определены")
    кластеров = min(centroids_per_class, len(embeddings))
    центры = np.asarray(kmeans(embeddings, кластеров), dtype=np.float32)
    ожидаемая_форма = (кластеров, embeddings.shape[1])
    if центры.shape != ожидаемая_форма:
        raise CentroidError(f"k-means вернул форму {центры.shape}, ожидалась {ожидаемая_форма}")
    нормы = np.linalg.norm(центры, axis=1, keepdims=True)
    if bool(np.any(нормы == 0)):
        raise CentroidError("нулевой центр кластера: нормировать нечем")
    центры = (центры / нормы).astype(np.float32)
    if кластеров < centroids_per_class:
        добивка = np.repeat(центры[-1:], centroids_per_class - кластеров, axis=0)
        центры = np.concatenate([центры, добивка], axis=0)
    return центры
