"""Замер приёмки на трёх наборах: сверка с паритетом и отчёт (спека holdout-relevance §7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.holdout_relevance import SET_FULL, SET_RELEVANT, SET_RELEVANT_LEVEL
from training.acceptance import SetMetrics
from training.eval_relevance import check_reference, read_parity, render_report


def _м(top1: float = 0.69, n: int = 1523) -> SetMetrics:
    return SetMetrics(
        n=n,
        classes=149,
        top1_model=top1,
        top5_model=0.828,
        top1_brand_strict=0.731,
        top1_brand_grouped=0.765,
        auroc=0.766,
        fpr_at_tpr95=0.826,
        coverage=0.533,
        error=0.068,
    )


def test_паритет_читается_из_tflite_fp32(tmp_path: Path) -> None:
    путь = tmp_path / "parity.json"
    fp32 = {"top1_model": 0.1, "top5_model": 0.2, "top1_brand": 0.3, "per_class_recall": {}}
    путь.write_text(
        json.dumps({"runtimes": {"holdout_ru": {"tflite_fp32": fp32, "pytorch": {}}}}),
        encoding="utf-8",
    )
    assert read_parity(путь) == {"top1_model": 0.1, "top5_model": 0.2, "top1_brand": 0.3}


def test_сверка_пропускает_совпадение_и_называет_расхождение() -> None:
    check_reference(_м(), {"top1_model": (0.69, 0.0), "auroc": (0.7662, 5e-4)})
    with pytest.raises(ValueError, match="top1_model"):
        check_reference(_м(top1=0.70), {"top1_model": (0.69, 1e-9)})


def test_отчёт_три_набора_цели_и_разметка() -> None:
    отчёт = render_report(
        results={SET_FULL: _м(), SET_RELEVANT: _м(0.8, 900), SET_RELEVANT_LEVEL: _м(0.85, 700)},
        verdict_counts={"relevant": 900, "off_street": 500, "wrong_label": 40, "unsure": 83},
        top_down=260,
        provenance={"classifier_sha256": "b105"},
    )
    for фраза in ("полный", "релевантный", "релевантный без «сверху»", "≥ 0.75", "≤ 0.15"):
        assert фраза in отчёт
    assert "| top-1 по модели | ≥ 0.75 | 0.6900 | 0.8000 | 0.8500 |" in отчёт
    assert "unsure больше 5%" in отчёт


def test_без_предупреждения_если_unsure_мало() -> None:
    отчёт = render_report(
        results={SET_FULL: _м(), SET_RELEVANT: _м(), SET_RELEVANT_LEVEL: _м()},
        verdict_counts={"relevant": 1500, "off_street": 20, "wrong_label": 0, "unsure": 3},
        top_down=0,
        provenance={},
    )
    assert "unsure больше 5%" not in отчёт


def test_ответы_по_кадрам_пишутся_csv(tmp_path: Path) -> None:
    from training.eval_relevance import PREDICTION_COLUMNS, write_predictions

    путь = tmp_path / "pred.csv"
    write_predictions(путь, ["a", "b"], np.array([1, 2]), np.array([1, 0]), np.array([0.9, 0.4]))
    кадр = pd.read_csv(путь, dtype={"photo_id": str})
    assert list(кадр.columns) == list(PREDICTION_COLUMNS)
    assert кадр["top1"].tolist() == [1, 0]
