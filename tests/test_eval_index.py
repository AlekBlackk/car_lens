"""Тесты оценки индекса. Поиск по центроидам проверяется без torch."""

from __future__ import annotations

import numpy as np

from scripts.eval_index import top5_by_centroids


def test_top5_ставит_свой_класс_первым() -> None:
    центроиды = np.zeros((6, 4, 3), dtype=np.float32)
    for класс in range(6):
        база = np.zeros(3, dtype=np.float32)
        база[класс % 3] = 1.0
        # Слегка разные центроиды одного класса, все около своей оси.
        for k in range(4):
            вектор = база + 0.01 * k
            центроиды[класс, k] = вектор / np.linalg.norm(вектор)
    эмбеддинги = центроиды[2, 0][None, :]
    результат = top5_by_centroids(эмбеддинги, центроиды)
    assert len(результат) == 1
    assert результат[0][0] == 2
    assert len(результат[0]) == 5
    assert len(set(результат[0])) == 5


def test_каждая_строка_даёт_разные_классы() -> None:
    """Пять слотов top-5 — это пять классов, а не пять центроидов одного класса."""
    генератор = np.random.default_rng(0)
    сырые = генератор.normal(size=(8, 4, 16))
    центроиды = (сырые / np.linalg.norm(сырые, axis=2, keepdims=True)).astype(np.float32)
    эмбеддинги = центроиды[:, 0, :]
    результат = top5_by_centroids(эмбеддинги, центроиды)
    for строка in результат:
        assert len(set(строка)) == 5
