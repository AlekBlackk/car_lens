"""Чистая часть замера few-shot: имена классов, недостача эталона, связка метрик."""

from __future__ import annotations

import numpy as np

from pipeline.fewshot.metrics import FewShotMetrics
from scripts.eval_fewshot import evaluate, idx_to_class_id_of, missing_holdout_classes

МЕТА = {
    "classes": [
        {"class_id": "lada__2107", "members": ["lada__2107"]},
        {"class_id": "kia__rio", "members": ["kia__rio"]},
        {"class_id": "uaz__patriot", "members": ["uaz__patriot"]},
    ]
}


def test_карты_имён_собираются_по_позиции_в_мете() -> None:
    assert idx_to_class_id_of(МЕТА) == {0: "lada__2107", 1: "kia__rio", 2: "uaz__patriot"}


def test_недостающие_классы_названы_в_порядке_индекса() -> None:
    имена = {0: "lada__2107", 1: "kia__rio", 2: "uaz__patriot"}
    метрики = FewShotMetrics(
        top1=1.0,
        top3=1.0,
        per_class_top3={"lada__2107": 1.0, "uaz__patriot": 1.0},
        per_class_support={"lada__2107": 5, "uaz__patriot": 3},
    )
    assert missing_holdout_classes(имена, метрики) == ("kia__rio",)


def test_evaluate_связывает_ранжирование_и_агрегацию() -> None:
    """Точный косинус 1.0 на правильном центроиде — попадание top-1 и top-3."""
    генератор = np.random.default_rng(3)
    сырые = генератор.normal(size=(3, 2, 4))
    центроиды = (сырые / np.linalg.norm(сырые, axis=2, keepdims=True)).astype(np.float32)
    эмбеддинги = np.stack([центроиды[0, 0], центроиды[1, 1], центроиды[2, 0]])
    метки = [0, 1, 2]
    имена = {0: "a", 1: "b", 2: "c"}
    итог = evaluate(эмбеддинги, метки, центроиды, имена)
    assert итог.top1 == 1.0
    assert итог.top3 == 1.0
    assert set(итог.per_class_top3) == {"a", "b", "c"}


def test_evaluate_ловит_ошибки_ранжирования() -> None:
    """Размерности эмбеддингов и центроидов обязаны совпадать."""
    import pytest

    центроиды = np.zeros((3, 2, 8), dtype=np.float32)
    эмбеддинги = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        evaluate(эмбеддинги, [0, 1], центроиды, {0: "a", 1: "b", 2: "c"})
