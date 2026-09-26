"""Yandex MAD-Cars: соответствие каталогу и выборка кадров 360°-обхода.

`view_id` к ракурсу не привязан (карточка датасета), но кадры идут по кругу обхода, поэтому
равный шаг по порядку кадров даёт разные стороны машины.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.catalog import Catalog
from pipeline.ingest.external import ExternalIngestError

MAPPING_COLUMNS = ("mad_brand", "mad_model", "model_id")
SAMPLE_COLUMNS = ("model_id", "car_id", "view_id", "url")
FRAMES_PER_CAR = 3


def read_mad_mapping(path: Path, catalog: Catalog) -> dict[tuple[str, str], str]:
    """Таблица → `{(brand, model): model_id}`; чужой класс или повтор пары роняют загрузку."""
    with path.open(encoding="utf-8", newline="") as файл:
        читатель = csv.DictReader(файл)
        нет = sorted(set(MAPPING_COLUMNS) - set(читатель.fieldnames or ()))
        if нет:
            raise ExternalIngestError(f"{path}: нет колонок {', '.join(нет)}")
        таблица: dict[tuple[str, str], str] = {}
        for строка in читатель:
            пара = (строка["mad_brand"], строка["mad_model"])
            модель = строка["model_id"]
            if модель not in catalog:
                raise ExternalIngestError(f"{path}: {модель} нет в каталоге")
            if пара in таблица:
                raise ExternalIngestError(f"{path}: пара {','.join(пара)} повторяется")
            таблица[пара] = модель
    return таблица


def _кадры_обхода(число: int) -> list[int]:
    """Позиции кадров с шагом N/3; при коротком обходе — без повторов."""
    return sorted({(число * i) // FRAMES_PER_CAR for i in range(FRAMES_PER_CAR)})


def sample_frames(
    manifest: pd.DataFrame,
    mapping: Mapping[tuple[str, str], str],
    *,
    cars_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """До `cars_per_class` машин на класс, по 3 кадра; от порядка строк входа не зависит."""
    пары = pd.Series(list(zip(manifest["brand"], manifest["model"], strict=True)))
    кадр = manifest.assign(model_id=пары.map(dict(mapping)).to_numpy()).dropna(subset=["model_id"])
    генератор = np.random.default_rng(seed)
    строки: list[tuple[str, str, str, str]] = []
    for модель in sorted(кадр["model_id"].unique()):
        свои = кадр[кадр["model_id"] == модель]
        машины = sorted(свои["car_id"].unique(), key=int)
        взять = min(cars_per_class, len(машины))
        выбранные = sorted(генератор.choice(машины, size=взять, replace=False), key=int)
        for машина in выбранные:
            обход = свои[свои["car_id"] == машина].sort_values(
                "view_id", key=lambda s: s.astype(int)
            )
            for позиция in _кадры_обхода(len(обход)):
                кадр_обхода = обход.iloc[позиция]
                ракурс, url = str(кадр_обхода["view_id"]), str(кадр_обхода["url"])
                строки.append((модель, str(машина), ракурс, url))
    return pd.DataFrame(строки, columns=list(SAMPLE_COLUMNS))
