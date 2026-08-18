"""Поиск ближайших классов по центроидам индекса (§13.2).

Модуль не импортирует torch: одной и той же функцией меряется индекс на M-1.3
(`scripts/eval_index.py`) и паритет квантованной модели на M-1.4
(`export/verify_parity.py`). Вторая копия сравнивала бы заодно две реализации
сравнения, а джоб `python` в CI обходится группой `dev`, где torch нет.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TOP_K


def top5_by_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> list[list[int]]:
    """Индексы top-5 классов по максимуму косинуса к их центроидам.

    `centroids` — `(классы, центроиды, измерения)`, всё нормализовано, поэтому
    косинус — скалярное произведение. Максимум по центроидам берётся до сортировки:
    top-5 — это пять разных классов, а не пять ближайших центроидов, которые могут
    принадлежать одному.
    """
    классов, _, размерность = centroids.shape
    плоские = centroids.reshape(-1, размерность)
    сходства = embeddings @ плоские.T
    по_классам = сходства.reshape(len(embeddings), классов, -1).max(axis=2)
    порядок = np.argsort(-по_классам, axis=1)[:, :TOP_K]
    return [[int(класс) for класс in строка] for строка in порядок]
