"""Тесты сборки отчёта о прогоне ingest."""

import pandas as pd

from pipeline.ingest.build import BuildStats
from pipeline.schema import COLUMNS
from scripts.build_stanford_splits import render_report


def пустой() -> pd.DataFrame:
    return pd.DataFrame([], columns=list(COLUMNS))


def test_отчёт_называет_все_счётчики() -> None:
    статистика = BuildStats(
        seen=16185,
        rejected={"no_vehicle": 12, "too_small": 3},
        unmapped=7,
        deduped=41,
        iou_values=(0.8, 0.9, 0.95),
    )
    текст = render_report(статистика, {"train": пустой()}, classes=150)
    assert "16185" in текст
    assert "no_vehicle" in текст
    assert "150" in текст
    assert "0.9" in текст  # медиана


def test_отчёт_без_iou_не_падает() -> None:
    """Зеркало могло не отдать ручную разметку — сверки тогда нет (Р-8)."""
    текст = render_report(BuildStats(seen=1), {"train": пустой()}, classes=1)
    assert "нет разметки" in текст


def test_отчёт_перечисляет_сплиты() -> None:
    текст = render_report(BuildStats(seen=2), {"train": пустой(), "val": пустой()}, classes=1)
    assert "train" in текст
    assert "val" in текст
