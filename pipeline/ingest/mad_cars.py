"""Yandex MAD-Cars: соответствие каталогу и выборка кадров 360°-обхода.

`view_id` к ракурсу не привязан (карточка датасета), но кадры идут по кругу обхода, поэтому
равный шаг по порядку кадров даёт разные стороны машины.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.catalog import Catalog
from pipeline.ingest.external import ExternalIngestError

MAPPING_COLUMNS = ("mad_brand", "mad_model", "model_id")
SAMPLE_COLUMNS = ("model_id", "car_id", "view_id", "url")
EXCLUDED_COLUMNS = ("model_id", "car_id", "reason")
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


@dataclass(frozen=True, slots=True)
class FetchReport:
    downloaded: int
    present: int
    removed: int
    failed: list[str]


def read_excluded(path: Path) -> frozenset[str]:
    """`car_id` выброшенных по листам машин; `car_id` в MAD-Cars глобален."""
    кадр = pd.read_csv(path, dtype=str, keep_default_na=False)
    нет = sorted(set(EXCLUDED_COLUMNS) - set(кадр.columns))
    if нет:
        raise ExternalIngestError(f"{path}: нет колонок {', '.join(нет)}")
    return frozenset(кадр["car_id"])


def frame_path(root: Path, model_id: str, car_id: str, view_id: str) -> Path:
    """Раскладка, которую читает `read_mad_cars`: `root/<model_id>/<car_id>_<view_id>.jpg`."""
    return root / model_id / f"{car_id}_{view_id}.jpg"


def fetch_frames(
    sample: pd.DataFrame,
    root: Path,
    excluded: frozenset[str],
    download: Callable[[str, Path], None],
    workers: int = 4,
) -> FetchReport:
    """Докачать недостающее, удалить кадры выброшенных машин; сбой не оставляет файла."""
    удалено = лежит = 0
    очередь: list[tuple[str, Path]] = []
    for модель, машина, ракурс, url in sample[list(SAMPLE_COLUMNS)].itertuples(index=False):
        путь = frame_path(root, модель, машина, ракурс)
        if машина in excluded:
            if путь.exists():
                путь.unlink()
                удалено += 1
            continue
        if путь.is_file() and путь.stat().st_size > 0:
            лежит += 1
            continue
        путь.parent.mkdir(parents=True, exist_ok=True)
        очередь.append((url, путь))

    def одно(задание: tuple[str, Path]) -> str | None:
        url, путь = задание
        try:
            download(url, путь)
        except OSError:
            путь.unlink(missing_ok=True)
            return url
        return None

    with ThreadPoolExecutor(max_workers=workers) as пул:
        сбои = [url for url in пул.map(одно, очередь) if url is not None]
    return FetchReport(len(очередь) - len(сбои), лежит, удалено, сбои)
