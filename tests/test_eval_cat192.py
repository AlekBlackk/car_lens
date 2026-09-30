"""Проверки каталога 192: слоты истины, группировка, вердикт условия (в)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.acceptance import SetMetrics
from training.eval_cat192 import confusions, per_group, truth_slots, verdict


def метрики(top1: float, error: float) -> SetMetrics:
    return SetMetrics(
        n=100,
        classes=10,
        top1_model=top1,
        top5_model=0.9,
        top1_brand_strict=0.9,
        top1_brand_grouped=0.9,
        auroc=0.8,
        fpr_at_tpr95=0.5,
        coverage=0.7,
        error=error,
    )


def test_слот_истины_по_классу_мета() -> None:
    кадры = pd.DataFrame({"model_id": ["belgee__x50", "audi__a3"]})
    члены = {"belgee__x50": "grp__coolray", "audi__a3": "audi__a3"}
    слоты = truth_slots(кадры, члены, ["audi__a3", "grp__coolray"])
    assert слоты is not None
    assert list(слоты) == [1, 0]


def test_класса_нет_в_индексе_набор_не_определён() -> None:
    кадры = pd.DataFrame({"model_id": ["audi__a3"]})
    assert truth_slots(кадры, {"audi__a3": "audi__a3"}, ["audi__a4"]) is None


def test_группы_сортируются_от_худшей() -> None:
    итог = per_group(np.array([True, False, True, True]), ["a", "a", "b", "b"])
    assert list(итог.index) == ["a", "b"]
    assert list(итог["n"]) == [2, 2]
    assert list(итог["top1"]) == [0.5, 1.0]


def test_частые_ошибки_без_верных_ответов() -> None:
    итог = confusions(["a", "a", "a", "b"], ["b", "b", "a", "b"])
    assert итог == [("a", "b", 2)]


def test_вердикт_проходит_когда_новые_не_слабее_старых() -> None:
    годно, _ = verdict(метрики(0.80, 0.10), метрики(0.85, 0.05))
    assert годно


def test_вердикт_называет_каждую_причину() -> None:
    годно, причины = verdict(метрики(0.60, 0.20), метрики(0.85, 0.05))
    assert not годно
    assert причины.count(";") == 2
