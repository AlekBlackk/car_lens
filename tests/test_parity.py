"""Тесты арифметики расхождений (§4.4 дизайна M-1.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from export.parity import (
    ParityError,
    cosine_deltas,
    metric_delta,
    raw_deltas,
    top1_churn,
)
from training.metrics import Metrics


def _метрики(top1: float, top5: float, brand: float) -> Metrics:
    return Metrics(top1_model=top1, top5_model=top5, top1_brand=brand, per_class_recall={})


def test_совпадающие_матрицы_дают_нули() -> None:
    матрица = np.random.default_rng(0).normal(size=(10, 256)).astype(np.float32)
    дельта = raw_deltas(матрица, матрица.copy())
    assert дельта.max_abs == 0.0
    assert дельта.median_abs == 0.0
    assert дельта.p99_abs == 0.0
    assert дельта.max_rel == 0.0


def test_сырое_расхождение_считается_поэлементно() -> None:
    """Все четыре поля различают статистики: разница [0, 0.5, 0] даёт им разные числа."""
    эталон = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    кандидат = np.array([[1.0, 2.5, 4.0]], dtype=np.float32)
    дельта = raw_deltas(эталон, кандидат)
    assert дельта.max_abs == pytest.approx(0.5)
    assert дельта.median_abs == pytest.approx(0.0)
    assert дельта.p99_abs == pytest.approx(0.49)
    assert дельта.max_rel == pytest.approx(0.25)


def test_max_rel_прижимает_знаменатель_у_нуля() -> None:
    """Ноль в эталоне не имеет права превратить относительную ошибку в бесконечность."""
    эталон = np.array([[0.0, 1.0]], dtype=np.float32)
    кандидат = np.array([[1e-3, 1.0]], dtype=np.float32)
    дельта = raw_deltas(эталон, кандидат)
    assert math.isfinite(дельта.max_rel)
    assert дельта.max_rel == pytest.approx(1000.0)


def test_разные_формы_падают() -> None:
    with pytest.raises(ParityError, match="форм"):
        raw_deltas(np.zeros((4, 8), dtype=np.float32), np.zeros((4, 16), dtype=np.float32))


def test_пустой_вход_падает() -> None:
    with pytest.raises(ParityError, match="пуст"):
        raw_deltas(np.zeros((0, 8), dtype=np.float32), np.zeros((0, 8), dtype=np.float32))


def test_нулевая_размерность_падает() -> None:
    """Строки есть, а столбцов нет: numpy упал бы на `max()` сырым ValueError."""
    with pytest.raises(ParityError, match="пуст"):
        raw_deltas(np.zeros((5, 0), dtype=np.float32), np.zeros((5, 0), dtype=np.float32))


def test_nan_в_кандидате_падает() -> None:
    """nan прошёл бы порог приёмки как «расхождения нет»: `nan > 1e-3` — это False."""
    эталон = np.ones((3, 4), dtype=np.float32)
    кандидат = эталон.copy()
    кандидат[1, 2] = np.nan
    with pytest.raises(ParityError, match="неконечн"):
        raw_deltas(эталон, кандидат)


def test_inf_в_эталоне_падает() -> None:
    """inf даёт из одного вызова несогласованные поля: `max_abs=inf` при `p99_abs=nan`."""
    кандидат = np.ones((3, 4), dtype=np.float32)
    эталон = кандидат.copy()
    эталон[0, 0] = np.inf
    with pytest.raises(ParityError, match="неконечн"):
        cosine_deltas(эталон, кандидат)


def test_косинус_не_зависит_от_длины_вектора() -> None:
    """Эмбеддинги приходят ненормированными, косинус обязан нормировать сам."""
    эталон = np.random.default_rng(1).normal(size=(20, 256)).astype(np.float32)
    дельта = cosine_deltas(эталон, эталон * 5.0)
    assert дельта.min == pytest.approx(1.0, abs=1e-5)
    assert дельта.median == pytest.approx(1.0, abs=1e-5)
    assert дельта.p1 == pytest.approx(1.0, abs=1e-5)


def test_косинус_видит_хвост() -> None:
    """Медиана может быть единицей при испорченном одном примере — минимум это ловит."""
    эталон = np.tile(np.eye(1, 4, dtype=np.float32), (100, 1))
    кандидат = эталон.copy()
    кандидат[7] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    дельта = cosine_deltas(эталон, кандидат)
    assert дельта.median == pytest.approx(1.0, abs=1e-6)
    assert дельта.min == pytest.approx(0.0, abs=1e-6)
    assert дельта.p1 == pytest.approx(0.99, abs=1e-6)


def test_нулевой_вектор_падает() -> None:
    эталон = np.ones((3, 4), dtype=np.float32)
    кандидат = эталон.copy()
    кандидат[1] = 0.0
    with pytest.raises(ParityError, match="нулев"):
        cosine_deltas(эталон, кандидат)


def test_churn_считает_долю_смен_top1() -> None:
    эталон = [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5], [3, 1, 2, 4, 5], [4, 1, 2, 3, 5]]
    кандидат = [[1, 2, 3, 4, 5], [9, 1, 3, 4, 5], [3, 9, 2, 4, 5], [7, 1, 2, 3, 5]]
    assert top1_churn(эталон, кандидат) == pytest.approx(0.5)


def test_churn_на_разной_длине_падает() -> None:
    with pytest.raises(ParityError, match="длин"):
        top1_churn([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5]])


def test_дельта_метрик_в_пунктах_со_знаком() -> None:
    """Минус означает просадку кандидата, и это единственное читаемое соглашение."""
    дельта = metric_delta(_метрики(0.9432, 0.9916, 0.9774), _метрики(0.9350, 0.9916, 0.9800))
    assert дельта["top1_model"] == pytest.approx(-0.82, abs=1e-6)
    assert дельта["top5_model"] == pytest.approx(0.0, abs=1e-6)
    assert дельта["top1_brand"] == pytest.approx(0.26, abs=1e-6)
