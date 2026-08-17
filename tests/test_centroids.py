"""Тесты раскладки центроидов."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from pipeline.centroids import CentroidError, centroids_for_class


def _нормировать(матрица: np.ndarray) -> np.ndarray:
    нормы = np.linalg.norm(матрица, axis=1, keepdims=True)
    return np.asarray(матрица / нормы, dtype=np.float32)


def _заглушка(вызовы: list[tuple[int, int]]) -> Callable[[np.ndarray, int], np.ndarray]:
    """k-means, возвращающий первые k строк. Записывает свои вызовы."""

    def kmeans(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
        вызовы.append((len(эмбеддинги), k))
        return эмбеддинги[:k].copy()

    return kmeans


def test_обычный_класс_зовёт_kmeans_один_раз() -> None:
    вызовы: list[tuple[int, int]] = []
    эмбеддинги = _нормировать(np.random.default_rng(0).normal(size=(20, 8)))
    результат = centroids_for_class(эмбеддинги, centroids_per_class=4, kmeans=_заглушка(вызовы))
    assert вызовы == [(20, 4)]
    assert результат.shape == (4, 8)
    assert результат.dtype == np.float32


def test_результат_нормализован() -> None:
    """Среднее единичных векторов не единично, заголовок объявляет normalized = 1."""
    вызовы: list[tuple[int, int]] = []

    def kmeans(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
        вызовы.append((len(эмбеддинги), k))
        # Центр кластера длиной заметно меньше единицы.
        return np.repeat(эмбеддинги[:1] * 0.3, k, axis=0)

    эмбеддинги = _нормировать(np.random.default_rng(1).normal(size=(10, 8)))
    результат = centroids_for_class(эмбеддинги, centroids_per_class=4, kmeans=kmeans)
    np.testing.assert_allclose(np.linalg.norm(результат, axis=1), 1.0, rtol=1e-6)


def test_недобор_дублирует_последний_центроид() -> None:
    вызовы: list[tuple[int, int]] = []
    эмбеддинги = _нормировать(np.random.default_rng(2).normal(size=(2, 8)))
    результат = centroids_for_class(эмбеддинги, centroids_per_class=4, kmeans=_заглушка(вызовы))
    assert вызовы == [(2, 2)]
    assert результат.shape == (4, 8)
    np.testing.assert_array_equal(результат[2], результат[1])
    np.testing.assert_array_equal(результат[3], результат[1])


def test_один_эмбеддинг() -> None:
    вызовы: list[tuple[int, int]] = []
    эмбеддинги = _нормировать(np.random.default_rng(3).normal(size=(1, 8)))
    результат = centroids_for_class(эмбеддинги, centroids_per_class=4, kmeans=_заглушка(вызовы))
    assert результат.shape == (4, 8)
    for слот in range(4):
        np.testing.assert_allclose(результат[слот], результат[0], rtol=1e-6)


def test_класс_без_эмбеддингов_падает() -> None:
    """Нулевой вектор дал бы неопределённый косинус, то есть случайное сравнение."""
    вызовы: list[tuple[int, int]] = []
    with pytest.raises(CentroidError, match="без эталонов"):
        centroids_for_class(
            np.zeros((0, 8), dtype=np.float32), centroids_per_class=4, kmeans=_заглушка(вызовы)
        )


def test_нулевой_центр_падает() -> None:
    """Вырожденный центр нормировать нечем — молча делить на ноль нельзя."""

    def kmeans(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
        return np.zeros((k, эмбеддинги.shape[1]), dtype=np.float32)

    эмбеддинги = _нормировать(np.random.default_rng(4).normal(size=(10, 8)))
    with pytest.raises(CentroidError, match="нулевой"):
        centroids_for_class(эмбеддинги, centroids_per_class=4, kmeans=kmeans)
