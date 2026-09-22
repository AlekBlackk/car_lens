"""Приём внешних кадров: готовые наборы и ручной сбор владельца.

Спека: `docs/superpowers/specs/2026-09-22-external-data-design.md`. Источник отдаёт папки
фотографий, у каждой папки — `model_id` каталога: у ручного сбора это имя папки, у готового
набора — строка таблицы соответствия `catalog/external/<источник>.csv`.

Единица сопоставления — `model_id`, а не `class_id`, как в `ingest/build.py`: класс выводит
`member_index`, и группы пересобираются без пересборки данных (§13.2).

Стэнфордский оркестратор сюда не обобщён намеренно: он завязан на формат Stanford, а второй
вход не повод трогать рабочий путь.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.catalog import Catalog, class_index, member_index
from pipeline.clean import DEFAULT_PHASH_HAMMING, duplicate_indices, phash_of
from pipeline.crop import Detector, crop_image, select_primary
from pipeline.hashing import bucket_of, close_pairs
from pipeline.schema import COLUMNS, empty_frame

# Порядок — приоритет при склейке: съёмка с уровня глаз ценнее пресс-фото (фильтры 7–8).
SOURCE_PRIORITY: tuple[str, ...] = ("manual", "skillfactory", "autoevolution")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAPPING_COLUMNS = ("dir_path", "model_id")

# Тот же порог, что `min_short_side_px` в `thresholds.json` приложения: кроп мельче
# приложение не классифицирует, учить на нём незачем.
MIN_SHORT_SIDE_PX = 120

DEFAULT_MAX_PER_CLASS = 150
GAP_CLASSES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "catalog" / "external" / "gap_classes.txt"
)


class ExternalIngestError(ValueError):
    """Вход внешнего источника не соответствует контракту спеки."""


@dataclass(frozen=True, slots=True)
class ExternalPhoto:
    """Одна фотография внешнего источника до детектора."""

    path: Path
    model_id: str
    # Объявление §13.1: у набора — галерея (`dir_path`), у ручного сбора — сам файл.
    group: str
    raw_model_string: str | None


@dataclass(frozen=True, slots=True)
class ExternalResult:
    """Кадр §13.1 одного источника и счётчики: `rejected` — отказы по причинам спеки."""

    frame: pd.DataFrame
    seen: int
    rejected: dict[str, int]


def read_mapping(path: Path, catalog: Catalog) -> dict[str, str]:
    """Таблица соответствия → `{dir_path: model_id}`.

    Обязательны `dir_path` и `model_id`; остальные колонки — для глаз проверяющего и кодом
    не читаются. Битая строка роняет прогон целиком: пропущенная молча строка — это класс,
    недополучивший данные, и заметить его можно было бы только по итоговым числам.
    """
    with path.open(encoding="utf-8", newline="") as файл:
        читатель = csv.DictReader(файл)
        нет = sorted(set(MAPPING_COLUMNS) - set(читатель.fieldnames or ()))
        if нет:
            raise ExternalIngestError(f"{path}: нет колонок {', '.join(нет)}")
        таблица: dict[str, str] = {}
        for номер, строка in enumerate(читатель, start=2):
            папка = (строка["dir_path"] or "").strip()
            модель = (строка["model_id"] or "").strip()
            if not папка or not модель:
                raise ExternalIngestError(f"{path}:{номер}: пустое dir_path или model_id")
            if модель not in catalog:
                raise ExternalIngestError(f"{path}:{номер}: model_id {модель!r} нет в каталоге")
            if папка in таблица:
                raise ExternalIngestError(f"{path}:{номер}: dir_path {папка!r} повторяется")
            таблица[папка] = модель
    return таблица


def _картинки(папка: Path) -> list[Path]:
    """Файлы-картинки папки в порядке имён: порядок задаёт, какой из дублей выживет."""
    return sorted(
        путь for путь in папка.iterdir() if путь.is_file() and путь.suffix.lower() in IMAGE_SUFFIXES
    )


def read_mapped(root: Path, mapping: Mapping[str, str]) -> list[ExternalPhoto]:
    """Фотографии готового набора по таблице, в порядке `dir_path` и имён файлов."""
    фото: list[ExternalPhoto] = []
    for папка, модель in sorted(mapping.items()):
        путь = root / папка
        if not путь.is_dir():
            raise ExternalIngestError(f"{путь}: папки из таблицы соответствия нет в наборе")
        фото.extend(ExternalPhoto(файл, модель, папка, папка) for файл in _картинки(путь))
    return фото


def read_manual(root: Path, catalog: Catalog) -> list[ExternalPhoto]:
    """Ручной сбор: `root/<model_id>/*`. Неизвестная папка — опечатка, прогон падает."""
    фото: list[ExternalPhoto] = []
    for путь in sorted(папка for папка in root.iterdir() if папка.is_dir()):
        if путь.name not in catalog:
            raise ExternalIngestError(f"{путь}: папка не model_id каталога")
        фото.extend(ExternalPhoto(файл, путь.name, файл.name, None) for файл in _картинки(путь))
    return фото


def _близкие(кандидаты: Sequence[int], опорные: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Пары (индекс кандидата, индекс опорного) на расстоянии ≤ `DEFAULT_PHASH_HAMMING`.

    Через `close_pairs` по общему списку, а не перебором: тысячи на тысячи — десятки
    миллионов сравнений. `close_pairs` выдаёт пары с возрастающими индексами, поэтому
    смешанная пара — ровно та, где первый индекс слева от границы, второй справа.
    """
    граница = len(кандидаты)
    for i, j in close_pairs([*кандидаты, *опорные], DEFAULT_PHASH_HAMMING):
        if i < граница <= j:
            yield i, j - граница


def build_external_frame(
    photos: Iterable[ExternalPhoto],
    detector: Detector,
    catalog: Catalog,
    crops_root: Path,
    *,
    source: str,
    holdout_hashes: Sequence[int],
    pool: Sequence[tuple[str, int]],
) -> ExternalResult:
    """Фотографии одного источника → кадр §13.1 после фильтров 1–6 спеки.

    `holdout_hashes` — pHash всех кадров `holdout_ru`: близость к любому из них выбрасывает
    кадр без оглядки на класс. `pool` — пары `(class_id, phash)` обучающего пула: там
    выбрасывается только совпадение в своём классе.
    """
    индекс = class_index(catalog)
    класс_модели = member_index(catalog)
    отказы: Counter[str] = Counter()
    записи: list[dict[str, Any]] = []
    классы: list[str] = []
    хеши: list[int] = []
    отпечатки: set[str] = set()
    просмотрено = 0

    for фото in photos:
        просмотрено += 1
        байты = фото.path.read_bytes()
        отпечаток = hashlib.sha256(байты).hexdigest()[:16]
        if отпечаток in отпечатки:
            # Тот же файл второй раз записал бы кроп поверх первого, а дедуп ниже удалил бы
            # общий файл вместе с дублем.
            отказы["dup_source"] += 1
            continue
        отпечатки.add(отпечаток)
        # На пустом буфере `imdecode` бросает `cv2.error`, а не возвращает `None`.
        изображение = (
            cv2.imdecode(np.frombuffer(байты, np.uint8), cv2.IMREAD_COLOR) if байты else None
        )
        if изображение is None:
            отказы["unreadable"] += 1
            continue
        высота, ширина = изображение.shape[:2]
        решение = select_primary(detector.detect(изображение), ширина, высота)
        if решение.box is None:
            отказы[решение.reject_reason or "unknown"] += 1
            continue
        кроп = crop_image(изображение, решение.box)
        if min(кроп.shape[:2]) < MIN_SHORT_SIDE_PX:
            отказы["small"] += 1
            continue

        class_id = класс_модели[фото.model_id]
        относительный = f"external/{source}/{class_id}/{отпечаток}.jpg"
        путь_кропа = crops_root / относительный
        путь_кропа.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(путь_кропа), кроп):
            raise OSError(f"не удалось записать кроп {путь_кропа}")
        # Хеш — по перечитанному JPEG, как в `ingest/build.py`: иначе колонка `phash`
        # разойдётся с тем, что получит любой код, открывший файл заново.
        кроп_с_диска = cv2.imread(str(путь_кропа))
        if кроп_с_диска is None:
            raise OSError(f"кроп записан, но не читается обратно: {путь_кропа}")
        хеши.append(phash_of(кроп_с_диска))
        классы.append(class_id)
        записи.append(
            {
                "photo_id": f"{source}:{отпечаток}",
                "listing_id": f"{source}:{class_id}:{фото.group}",
                "source": source,
                "crop_path": относительный,
                "brand": catalog.by_id(фото.model_id).brand,
                "model_id": фото.model_id,
                "class_idx": индекс[class_id],
                "body_style": None,
                "year": None,
                "bbox": [
                    float(решение.box.x),
                    float(решение.box.y),
                    float(решение.box.w),
                    float(решение.box.h),
                ],
                "bbox_area_frac": float(решение.area_frac),
                "det_conf": float(решение.box.conf),
                "phash": 0,  # проставляется ниже, после фильтров
                "raw_model_string": фото.raw_model_string,
                "is_multi_car": решение.is_multi_car,
            }
        )

    # Причина — первая сработавшая: дубль внутри источника, утечка в эталон, повтор пула.
    отброшены: dict[int, str] = dict.fromkeys(duplicate_indices(хеши), "dup_source")
    for i, _ in _близкие(хеши, holdout_hashes):
        отброшены.setdefault(i, "holdout_leak")
    for i, j in _близкие(хеши, [хеш for _, хеш in pool]):
        if pool[j][0] == классы[i]:
            отброшены.setdefault(i, "dup_pool")
    for i, причина in отброшены.items():
        отказы[причина] += 1
        (crops_root / записи[i]["crop_path"]).unlink(missing_ok=True)

    оставшиеся = [
        {**запись, "phash": хеши[i]} for i, запись in enumerate(записи) if i not in отброшены
    ]
    кадр = pd.DataFrame(оставшиеся, columns=list(COLUMNS)) if оставшиеся else empty_frame()
    return ExternalResult(frame=кадр, seen=просмотрено, rejected=dict(отказы))


def merge_external(
    frames: Mapping[str, pd.DataFrame], *, max_per_class: int, seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Склеить кадры источников: межисточниковый дедуп и потолок на класс (фильтры 7–8).

    Источники обходятся в порядке `SOURCE_PRIORITY`: из близкой пары остаётся кадр более
    приоритетного (`duplicate_indices` оставляет самый ранний), и потолок заполняется им же
    первым. Внутри источника порядок задаёт `bucket_of(photo_id, seed)` — от порядка строк
    на входе результат не зависит.
    """
    чужие = sorted(set(frames) - set(SOURCE_PRIORITY))
    if чужие:
        raise ExternalIngestError(f"неизвестные источники: {', '.join(чужие)}")
    части = []
    for источник in SOURCE_PRIORITY:
        кадр = frames.get(источник)
        if кадр is None or кадр.empty:
            continue
        ключи = [bucket_of(str(photo_id), seed) for photo_id in кадр["photo_id"]]
        части.append(кадр.assign(_ключ=ключи).sort_values("_ключ", kind="stable"))
    if not части:
        return empty_frame(), {}

    всё = pd.concat(части, ignore_index=True)
    отказы: Counter[str] = Counter()
    дубли = duplicate_indices([int(хеш) for хеш in всё["phash"]])
    отказы["dup_cross"] = len(дубли)
    всё = всё.drop(index=sorted(дубли))
    сверх = всё.groupby("class_idx", sort=False).cumcount() >= max_per_class
    отказы["cap"] = int(сверх.sum())
    итог = всё.loc[~сверх, list(COLUMNS)].reset_index(drop=True)
    return итог, {причина: n for причина, n in отказы.items() if n}


def read_gap_classes(path: Path, catalog: Catalog) -> frozenset[str]:
    """Классы «дыры» из текстового списка; `#` — комментарий. Чужой `class_id` — ошибка."""
    известные = {класс.class_id for класс in catalog.classes}
    строки = (строка.split("#", 1)[0].strip() for строка in path.read_text("utf-8").splitlines())
    дыра = frozenset(строка for строка in строки if строка)
    чужие = sorted(дыра - известные)
    if чужие:
        raise ExternalIngestError(f"{path}: классов нет в каталоге: {', '.join(чужие)}")
    return дыра
