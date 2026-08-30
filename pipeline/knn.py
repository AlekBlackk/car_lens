"""Поиск ближайших классов по центроидам индекса (§13.2).

Модуль не импортирует torch: одной и той же функцией меряется индекс на M-1.3
(`scripts/eval_index.py`) и паритет квантованной модели на M-1.4
(`export/verify_parity.py`). Вторая копия сравнивала бы заодно две реализации
сравнения, а джоб `python` в CI обходится группой `dev`, где torch нет.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TOP_K


def ranked_by_centroids(
    embeddings: np.ndarray, centroids: np.ndarray, k: int = TOP_K
) -> list[list[int]]:
    """Индексы `k` ближайших классов по максимуму косинуса к их центроидам.

    `centroids` — `(классы, центроиды, измерения)`, всё нормализовано, поэтому косинус —
    скалярное произведение. Максимум по центроидам берётся до сортировки: результат — это
    `k` разных классов, а не `k` ближайших центроидов, которые могут принадлежать одному.

    `k` проверяется явно. `argsort`-срез длиннее числа классов молча вернул бы короткие
    строки, а вызывающий (метрика) считал бы по ним попадание в top-k, которого нет.
    """
    классов, _, размерность = centroids.shape
    if not 1 <= k <= классов:
        raise ValueError(f"k={k} вне диапазона 1..{классов}")
    плоские = centroids.reshape(-1, размерность)
    сходства = embeddings @ плоские.T
    по_классам = сходства.reshape(len(embeddings), классов, -1).max(axis=2)
    порядок = np.argsort(-по_классам, axis=1)[:, :k]
    return [[int(класс) for класс in строка] for строка in порядок]


def top5_by_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> list[list[int]]:
    """top-5 классов: имя, на которое опираются M-1.3 и сверка паритета M-1.4 (§13.2)."""
    return ranked_by_centroids(embeddings, centroids, TOP_K)
