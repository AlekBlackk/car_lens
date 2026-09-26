"""Парное сравнение двух моделей по кадрам релевантного набора."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.compare_relevance import PairedRow, paired_rows, sign_test_p


def ответы(*строки: tuple[str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [(p, t, k, 0.8) for p, t, k in строки], columns=["photo_id", "class_idx", "top1", "sim1"]
    )


def разметка(*строки: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(p, v, False) for p, v in строки], columns=["photo_id", "verdict", "top_down"]
    )


def test_знаковый_тест() -> None:
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(21, 6) == pytest.approx(0.00593, abs=1e-4)
    assert sign_test_p(8, 8) == 1.0


def test_строки_по_слабым_и_сильным_классам() -> None:
    база = ответы(("a", 0, 1), ("b", 0, 1), ("c", 1, 1), ("d", 1, 1), ("e", 1, 1))
    новая = ответы(("a", 0, 0), ("b", 0, 1), ("c", 1, 1), ("d", 1, 0), ("e", 1, 1))
    размечено = разметка(
        ("a", "relevant"),
        ("b", "relevant"),
        ("c", "relevant"),
        ("d", "relevant"),
        ("e", "off_street"),
    )
    строки = paired_rows(база, новая, размечено, weak_below=0.6)
    assert строки[0] == PairedRow("релевантный", 4, 0.5, 0.5, 1, 1, 1.0)
    assert [(с.name, с.n, с.wins, с.losses) for с in строки[1:]] == [
        ("слабые классы", 2, 1, 0),
        ("сильные классы", 2, 0, 1),
    ]


def test_разные_наборы_кадров_роняют_сравнение() -> None:
    with pytest.raises(ValueError, match="photo_id"):
        paired_rows(
            ответы(("a", 0, 0)), ответы(("b", 0, 0)), разметка(("a", "relevant")), weak_below=0.6
        )
