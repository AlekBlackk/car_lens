"""Yandex MAD-Cars: соответствие каталогу и выборка кадров 360°-обхода.

`view_id` к ракурсу не привязан (карточка датасета), но кадры идут по кругу обхода, поэтому
равный шаг по порядку кадров даёт разные стороны машины.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.catalog import Catalog
from pipeline.ingest.external import ExternalIngestError
from pipeline.negatives import catalog_needle_hits

MAPPING_COLUMNS = ("mad_brand", "mad_model", "model_id")
SAMPLE_COLUMNS = ("model_id", "car_id", "view_id", "url")
EXCLUDED_COLUMNS = ("model_id", "car_id", "reason")
DISPUTED_COLUMNS = ("mad_brand", "mad_model", "model_id", "reason")
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


def read_mad_disputed(path: Path, catalog: Catalog) -> frozenset[tuple[str, str]]:
    """Спорные пары: соседнее поколение, рестайл или кузов класса. Причина обязательна."""
    кадр = pd.read_csv(path, dtype=str, keep_default_na=False)
    нет = sorted(set(DISPUTED_COLUMNS) - set(кадр.columns))
    if нет:
        raise ExternalIngestError(f"{path}: нет колонок {', '.join(нет)}")
    for номер, (модель, причина) in enumerate(
        zip(кадр["model_id"], кадр["reason"], strict=True), start=2
    ):
        if модель not in catalog:
            raise ExternalIngestError(f"{path}:{номер}: {модель} нет в каталоге")
        if not причина.strip():
            raise ExternalIngestError(f"{path}:{номер}: спорная пара без причины")
    return frozenset(zip(кадр["mad_brand"], кадр["mad_model"], strict=True))


def off_catalog_reason(
    pair: tuple[str, str],
    exact: Collection[tuple[str, str]],
    disputed: frozenset[tuple[str, str]],
    needles: Mapping[str, Sequence[str]],
) -> str | None:
    """Причина не считать пару чужой (`exact`, `disputed`, `needle:<ids>`) или None."""
    if pair in exact:
        return "exact"
    if pair in disputed:
        return "disputed"
    совпали = catalog_needle_hits(" ".join(pair), needles)
    if совпали:
        return "needle:" + "|".join(совпали)
    return None


def select_off_catalog_pairs(
    pairs: Iterable[tuple[str, str]],
    exact: Collection[tuple[str, str]],
    disputed: frozenset[tuple[str, str]],
    needles: Mapping[str, Sequence[str]],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    """Пары MAD-Cars вне каталога и причины отброса остальных.

    Спорная пара, которой нет в MAD-Cars, роняет отбор: опечатка в списке тихо вернула бы
    соседнее поколение класса в чужие машины. Игла ищется в строке «марка модель» тем же
    фильтром, что у прежних негативов: лишний выброшенный негатив дешевле своей машины.
    """
    все = sorted(set(pairs))
    пропали = sorted(disputed - set(все))
    if пропали:
        raise ExternalIngestError(
            "спорных пар нет в MAD-Cars: " + ", ".join("/".join(п) for п in пропали)
        )
    чужие: list[tuple[str, str]] = []
    отброшены: dict[tuple[str, str], str] = {}
    for пара in все:
        причина = off_catalog_reason(пара, exact, disputed, needles)
        if причина is None:
            чужие.append(пара)
        else:
            отброшены[пара] = причина
    return чужие, отброшены


def sample_negative_frames(
    manifest: pd.DataFrame,
    pairs: Collection[tuple[str, str]],
    *,
    cars_per_model: int,
    total: int,
    seed: int,
) -> pd.DataFrame:
    """До `cars_per_model` машин на чужую модель, средний кадр обхода, всего до `total`.

    В колонке `model_id` — папка модели MAD-Cars `<brand>__<model>`: раскладка та же, что у
    `fetch_frames`, но это не `model_id` каталога. От порядка строк входа не зависит.
    """
    свои = set(pairs)
    пары = zip(manifest["brand"], manifest["model"], strict=True)
    кадр = manifest[[пара in свои for пара in пары]]
    кадр = кадр.assign(папка=кадр["brand"] + "__" + кадр["model"])
    генератор = np.random.default_rng(seed)
    строки: list[tuple[str, str, str, str]] = []
    for папка, модель in кадр.groupby("папка", sort=True):
        машины = sorted(модель["car_id"].unique(), key=int)
        взять = min(cars_per_model, len(машины))
        for машина in sorted(генератор.choice(машины, size=взять, replace=False), key=int):
            обход = модель[модель["car_id"] == машина].sort_values(
                "view_id", key=lambda s: s.astype(int)
            )
            середина = обход.iloc[len(обход) // 2]
            строки.append((str(папка), str(машина), str(середина["view_id"]), str(середина["url"])))
    if len(строки) > total:
        оставить = sorted(генератор.choice(len(строки), size=total, replace=False))
        строки = [строки[i] for i in оставить]
    return pd.DataFrame(строки, columns=list(SAMPLE_COLUMNS))


def _кадры_пула(число: int) -> list[int]:
    """Позиции `N // 4`, `N // 2`, `3N // 4`; при коротком обходе — без повторов."""
    return sorted({(число * четверть) // 4 for четверть in (1, 2, 3)})


def sample_reject_pool_frames(
    manifest: pd.DataFrame,
    pairs: Collection[tuple[str, str]],
    measured_models: Collection[str],
    *,
    cars_per_model: int,
    seed: int,
) -> pd.DataFrame:
    """Кадры пула отказных прототипов: чужие модели вне замерной выборки.

    `measured_models` — папки `<brand>__<model>` выборки замера: прототип, собранный на
    модели замера, мерился бы на её же соседях, и оценка вышла бы оптимистичной. До
    `cars_per_model` машин на модель, три кадра обхода на машину — разные ракурсы.
    От порядка строк входа не зависит.
    """
    чужие = {f"{марка}__{модель}" for марка, модель in pairs} - set(measured_models)
    кадр = manifest.assign(папка=manifest["brand"] + "__" + manifest["model"])
    кадр = кадр[кадр["папка"].isin(чужие)]
    генератор = np.random.default_rng(seed)
    строки: list[tuple[str, str, str, str]] = []
    for папка, модель in кадр.groupby("папка", sort=True):
        машины = sorted(модель["car_id"].unique(), key=int)
        взять = min(cars_per_model, len(машины))
        for машина in sorted(генератор.choice(машины, size=взять, replace=False), key=int):
            обход = модель[модель["car_id"] == машина].sort_values(
                "view_id", key=lambda s: s.astype(int)
            )
            for позиция in _кадры_пула(len(обход)):
                ракурс = обход.iloc[позиция]
                строки.append((str(папка), str(машина), str(ракурс["view_id"]), str(ракурс["url"])))
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
