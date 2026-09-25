"""Разметка `holdout_ru` по релевантности и наборы приёмки (спека holdout-relevance §3–4).

Разметку ставит человек в `scripts/relevance_gallery.py`; здесь — только чтение и сборка
масок. Неполная разметка роняет замер: молча суженный набор выглядел бы как результат.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

VERDICTS = ("relevant", "off_street", "wrong_label", "unsure")
COLUMNS = ("photo_id", "verdict", "top_down")
# Заголовок выгрузки страницы берётся отсюда же: формат CSV задан в одном месте.
CSV_HEADER = ",".join(COLUMNS)

SET_FULL = "full"
SET_RELEVANT = "relevant"
SET_RELEVANT_LEVEL = "relevant_no_top_down"
SET_TITLES = {
    SET_FULL: "полный",
    SET_RELEVANT: "релевантный",
    SET_RELEVANT_LEVEL: "релевантный без «сверху»",
}


class RelevanceError(ValueError):
    """Разметка не годится для замера: формат, словарь или покрытие эталона."""


def load_relevance(path: Path) -> pd.DataFrame:
    """Прочитать и проверить разметку; `utf-8-sig` — на случай пересохранения в Excel."""
    кадр = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if tuple(кадр.columns) != COLUMNS:
        raise RelevanceError(f"{path}: колонки {list(кадр.columns)}, ожидались {list(COLUMNS)}")
    чужие = sorted(set(кадр["verdict"].tolist()) - set(VERDICTS))
    if чужие:
        raise RelevanceError(f"{path}: неизвестные вердикты {чужие}")
    флаги = sorted(set(кадр["top_down"].tolist()) - {"true", "false"})
    if флаги:
        raise RelevanceError(f"{path}: top_down должен быть true/false, а не {флаги}")
    повторы = sorted(кадр.loc[кадр["photo_id"].duplicated(), "photo_id"].tolist())
    if повторы:
        raise RelevanceError(f"{path}: photo_id повторяется: {повторы[:5]}")
    кадр["top_down"] = кадр["top_down"] == "true"
    return кадр


def subsets(photo_ids: Sequence[str], relevance: pd.DataFrame) -> dict[str, np.ndarray]:
    """Маски наборов §4 в порядке `photo_ids`; разметка обязана покрывать эталон ровно."""
    эталон = list(photo_ids)
    размечены = set(relevance["photo_id"].tolist())
    нет = sorted(set(эталон) - размечены)
    лишние = sorted(размечены - set(эталон))
    if нет or лишние:
        raise RelevanceError(
            f"разметка не совпадает с эталоном: без вердикта {len(нет)} {нет[:5]}, "
            f"лишних {len(лишние)} {лишние[:5]}"
        )
    по_id = relevance.set_index("photo_id")
    вердикт = по_id.loc[эталон, "verdict"].to_numpy()
    сверху = по_id.loc[эталон, "top_down"].to_numpy(dtype=bool)
    релевантный = вердикт == "relevant"
    return {
        SET_FULL: np.ones(len(эталон), dtype=bool),
        SET_RELEVANT: релевантный,
        SET_RELEVANT_LEVEL: релевантный & ~сверху,
    }
