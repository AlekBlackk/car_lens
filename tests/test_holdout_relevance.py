"""Разметка holdout_ru по релевантности и маски наборов (спека holdout-relevance §3–4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.holdout_relevance import (
    CSV_HEADER,
    SET_FULL,
    SET_RELEVANT,
    SET_RELEVANT_LEVEL,
    RelevanceError,
    load_relevance,
    subsets,
)


def _csv(tmp_path: Path, text: str) -> Path:
    путь = tmp_path / "holdout_relevance.csv"
    путь.write_text(text, encoding="utf-8")
    return путь


def _разметка(*строки: tuple[str, str, bool]) -> pd.DataFrame:
    return pd.DataFrame(строки, columns=["photo_id", "verdict", "top_down"])


def test_формат_выгрузки_страницы_читается(tmp_path: Path) -> None:
    путь = _csv(tmp_path, f"{CSV_HEADER}\na,relevant,true\nb,off_street,false\n")
    разметка = load_relevance(путь)
    assert разметка["photo_id"].tolist() == ["a", "b"]
    assert разметка["top_down"].tolist() == [True, False]


def test_файл_после_excel_с_bom_и_crlf_читается(tmp_path: Path) -> None:
    путь = tmp_path / "holdout_relevance.csv"
    путь.write_bytes(b"\xef\xbb\xbfphoto_id,verdict,top_down\r\na,unsure,false\r\n")
    assert load_relevance(путь)["verdict"].tolist() == ["unsure"]


@pytest.mark.parametrize(
    ("текст", "фраза"),
    [
        ("photo_id,verdict\na,relevant\n", "колонки"),
        (f"{CSV_HEADER}\na,maybe,false\n", "неизвестные вердикты"),
        (f"{CSV_HEADER}\na,relevant,yes\n", "top_down"),
        (f"{CSV_HEADER}\na,relevant,false\na,unsure,false\n", "повторяется"),
    ],
)
def test_битая_разметка_отвергается(tmp_path: Path, текст: str, фраза: str) -> None:
    with pytest.raises(RelevanceError, match=фраза):
        load_relevance(_csv(tmp_path, текст))


def test_маски_трёх_наборов_в_порядке_эталона() -> None:
    разметка = _разметка(
        ("a", "relevant", True),
        ("b", "relevant", False),
        ("c", "off_street", False),
        ("d", "unsure", False),
        ("e", "wrong_label", True),
    )
    маски = subsets(["e", "d", "c", "b", "a"], разметка)
    assert маски[SET_FULL].tolist() == [True] * 5
    assert маски[SET_RELEVANT].tolist() == [False, False, False, True, True]
    assert маски[SET_RELEVANT_LEVEL].tolist() == [False, False, False, True, False]


def test_кадр_эталона_без_вердикта_роняет_замер() -> None:
    разметка = _разметка(("a", "relevant", False))
    with pytest.raises(RelevanceError, match="без вердикта 1"):
        subsets(["a", "b"], разметка)


def test_вердикт_для_чужого_кадра_роняет_замер() -> None:
    разметка = _разметка(("a", "relevant", False), ("z", "relevant", False))
    with pytest.raises(RelevanceError, match="лишних 1"):
        subsets(["a"], разметка)
