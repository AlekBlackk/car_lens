"""Негативы для подбора порогов отказа (§13.5 SPEC; спека 2026-09-23-calibration-ru §4).

Две ответственности: решить, какие папки и классы внешних наборов — машины вне каталога,
и превратить отобранные фото в строки `negatives.parquet`. Детектор приходит снаружи
(протокол `Detector`), поэтому тесты обходятся без весов YOLO.

Фильтр чужих машин консервативен намеренно: папка, где нашлась игла любого класса
каталога, выбрасывается целиком. Лишний выброшенный негатив дешевле машины каталога,
записанной в мусор, — она занизила бы допустимый порог для своего же класса.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.clean import phash_of
from pipeline.crop import VEHICLE_CLASSES, Detection, Detector, crop_image, select_primary
from pipeline.harvest.aliases import matches_needle, normalize_filename
from pipeline.harvest.cache import cache_path
from pipeline.ingest.build import iou
from pipeline.schema import COLUMNS, empty_frame


class NegativesError(ValueError):
    """Входы сборки негативов не годятся."""


def catalog_needle_hits(name: str, table: Mapping[str, Sequence[str]]) -> list[str]:
    """Классы каталога, чья игла нашлась в имени папки или класса внешнего набора."""
    нормальное = normalize_filename(name)
    return sorted(
        класс
        for класс, иглы in table.items()
        if any(matches_needle(нормальное, игла) for игла in иглы)
    )


@dataclass(frozen=True, slots=True)
class Exclusions:
    """Ручные исключения: имена папок/классов и `photo_id` уже нарезанных кропов."""

    names: frozenset[str]
    photos: frozenset[str]


def read_exclusions(path: Path) -> Exclusions:
    """Прочитать `kind,value,reason`. Причина обязательна: исключение без неё не проверить."""
    if not path.exists():
        return Exclusions(frozenset(), frozenset())
    имена: set[str] = set()
    фото: set[str] = set()
    строки = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    for номер, строка in enumerate(строки, start=2):
        вид = (строка.get("kind") or "").strip()
        значение = (строка.get("value") or "").strip()
        if not (строка.get("reason") or "").strip():
            raise NegativesError(f"{path}:{номер}: исключение {значение!r} без причины")
        if вид == "name":
            имена.add(значение)
        elif вид == "photo":
            фото.add(значение)
        else:
            raise NegativesError(f"{path}:{номер}: неизвестный kind {вид!r}, ждём name|photo")
    return Exclusions(frozenset(имена), frozenset(фото))


@dataclass(frozen=True, slots=True)
class OffCatalogGroup:
    """Одна папка autoevolution или один класс Stanford: кандидат в машины вне каталога."""

    origin: str
    name: str
    paths: tuple[Path, ...]


def select_off_catalog(
    groups: Iterable[OffCatalogGroup],
    table: Mapping[str, Sequence[str]],
    exclusions: Exclusions,
) -> tuple[list[OffCatalogGroup], list[tuple[OffCatalogGroup, str]]]:
    """Разделить группы на чужие и отброшенные; у отброшенной — причина для отчёта."""
    оставлены: list[OffCatalogGroup] = []
    отброшены: list[tuple[OffCatalogGroup, str]] = []
    for группа in groups:
        совпали = catalog_needle_hits(группа.name, table)
        if совпали:
            отброшены.append((группа, "needle:" + "|".join(совпали)))
        elif группа.name in exclusions.names:
            отброшены.append((группа, "excluded"))
        else:
            оставлены.append(группа)
    return оставлены, отброшены


def sample_candidates(
    groups: Sequence[OffCatalogGroup], per_group: int, seed: int
) -> list[tuple[OffCatalogGroup, Path]]:
    """Не больше `per_group` фото с группы, общий порядок перемешан по `seed`.

    Порядок групп фиксируется сортировкой по `origin` до перемешивания: иначе итог зависел
    бы от порядка обхода файловой системы.
    """
    генератор = random.Random(seed)
    кандидаты: list[tuple[OffCatalogGroup, Path]] = []
    for группа in sorted(groups, key=lambda г: г.origin):
        пути = sorted(группа.paths)
        генератор.shuffle(пути)
        кандидаты.extend((группа, путь) for путь in пути[:per_group])
    генератор.shuffle(кандидаты)
    return кандидаты


NEG_KINDS = ("off_catalog", "off_catalog_street", "detector_fp", "truck_bus")
# Кроп короче 120 px телефон не классифицирует — показывает «Подойдите ближе» (§6.4), и
# порог сходства такой кадр не видит никогда.
MIN_SHORT_SIDE_PX = 120
# Кадры харвеста, где при штатном пороге 0.5 машины не нашлось или CLIP счёл кадр не
# фотографией машины: всё транспортное, что YOLO найдёт там при 0.35, — ложное срабатывание
# в том смысле, какой имеет в виду §13.5.
REJECT_REASONS = frozenset({"no_vehicle", "not_a_car_photo"})
# COCO 2 — car; 5 и 7 — bus и truck (§4.1).
_ЛЕГКОВОЙ = 2
# Бокс детектора считается тем же объектом, что бокс разметки COCO, от этого перекрытия.
IOU_MATCH = 0.5

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CocoTruth:
    """Разметка картинки COCO в пикселях `(x, y, w, h)`: грузовики и автобусы, легковушки."""

    truck_bus: tuple[Box, ...]
    car: tuple[Box, ...]


def _xyxy(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    return (int(x), int(y), int(x + w), int(y + h))


def coco_box_kind(box: Detection, truth: CocoTruth) -> str | None:
    """Вид бокса детектора по разметке COCO; `None` — бокс не берётся.

    Легковушка проверяется первой: машина каталога в негативах занизила бы порог своего
    класса, а пикапы и фургоны COCO бывают размечены как car. Картинка без разметки
    транспорта даёт ложное срабатывание; бокс мимо разметки на картинке с транспортом
    неоднозначен и не берётся.
    """
    бокс = _xyxy(box.x, box.y, box.w, box.h)
    if any(iou(бокс, _xyxy(*р)) >= IOU_MATCH for р in truth.car):
        return None
    if any(iou(бокс, _xyxy(*р)) >= IOU_MATCH for р in truth.truck_bus):
        return "truck_bus"
    if not truth.truck_bus and not truth.car:
        return "detector_fp"
    return None


def neg_kind_of(listing_id: str) -> str:
    """Вид негатива хранится первым сегментом `listing_id`: у схемы §13.1 нет своей колонки."""
    вид = listing_id.split("/", 1)[0]
    if вид not in NEG_KINDS:
        raise NegativesError(f"{listing_id!r}: вид {вид!r} не из {NEG_KINDS}")
    return вид


@dataclass(frozen=True, slots=True)
class NegativeCandidate:
    """Фото-источник.

    `off_catalog` — один главный бокс вида `primary_kind`; иначе — каждый транспортный бокс,
    вид по классу детектора, а при `coco` — по разметке COCO (`coco_box_kind`).
    """

    path: Path
    off_catalog: bool
    origin: str
    raw_name: str
    primary_kind: str = "off_catalog"
    coco: CocoTruth | None = None


@dataclass(frozen=True, slots=True)
class NegativesResult:
    frame: pd.DataFrame
    kept: dict[str, int]
    rejected: dict[str, int]


def _боксы(
    кандидат: NegativeCandidate, детекции: list[Detection], ширина: int, высота: int
) -> tuple[list[tuple[str, Detection, float]], str | None]:
    """Боксы кандидата с видом и долей площади; вторым — причина, если боксов нет."""
    площадь = float(ширина * высота)
    if кандидат.off_catalog:
        решение = select_primary(детекции, ширина, высота)
        if решение.box is None:
            return [], решение.reject_reason or "no_vehicle"
        return [(кандидат.primary_kind, решение.box, решение.area_frac)], None
    транспорт = [д for д in детекции if д.cls in VEHICLE_CLASSES]
    if not транспорт:
        return [], "no_vehicle"
    if кандидат.coco is not None:
        правда = кандидат.coco
        виды = [(coco_box_kind(д, правда), д) for д in транспорт]
        взятые = [(вид, д, д.area / площадь) for вид, д in виды if вид is not None]
        return взятые, (None if взятые else "coco_unmatched")
    return [
        ("detector_fp" if д.cls == _ЛЕГКОВОЙ else "truck_bus", д, д.area / площадь)
        for д in транспорт
    ], None


def build_negative_frame(
    candidates: Iterable[NegativeCandidate],
    detector: Detector,
    crops_root: Path,
    *,
    off_catalog_cap: int,
    excluded_photos: frozenset[str],
    box_caps: Mapping[str, int] | None = None,
) -> NegativesResult:
    """Кандидаты → строки `negatives.parquet` и кропы под `crops_root/negatives/<вид>/`.

    `box_caps` — потолок по виду бокса; когда заполнены все, обход кандидатов прекращается:
    ленивый поток COCO не качает лишнего.
    """
    оставлено: Counter[str] = Counter()
    отказы: Counter[str] = Counter()
    записи: list[dict[str, Any]] = []
    отпечатки: set[str] = set()
    потолки = dict(box_caps or {})
    for кандидат in candidates:
        if потолки and all(оставлено[вид] >= n for вид, n in потолки.items()):
            break
        if кандидат.off_catalog and оставлено[кандидат.primary_kind] >= off_catalog_cap:
            continue
        байты = кандидат.path.read_bytes()
        отпечаток = hashlib.sha256(байты).hexdigest()[:16]
        if отпечаток in отпечатки:
            отказы["dup_source"] += 1
            continue
        отпечатки.add(отпечаток)
        изображение = (
            cv2.imdecode(np.frombuffer(байты, np.uint8), cv2.IMREAD_COLOR) if байты else None
        )
        if изображение is None:
            отказы["unreadable"] += 1
            continue
        высота, ширина = изображение.shape[:2]
        боксы, причина = _боксы(кандидат, detector.detect(изображение), ширина, высота)
        if причина is not None:
            отказы[причина] += 1
        for номер, (вид, бокс, доля) in enumerate(боксы):
            if вид in потолки and оставлено[вид] >= потолки[вид]:
                отказы["cap"] += 1
                continue
            photo_id = f"negatives:{отпечаток}:{номер}"
            if photo_id in excluded_photos:
                отказы["excluded"] += 1
                continue
            кроп = crop_image(изображение, бокс)
            if min(кроп.shape[:2]) < MIN_SHORT_SIDE_PX:
                отказы["small"] += 1
                continue
            относительный = f"negatives/{вид}/{отпечаток}_{номер}.jpg"
            путь = crops_root / относительный
            путь.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(путь), кроп):
                raise OSError(f"не удалось записать кроп {путь}")
            # Хеш — по перечитанному JPEG, как в `ingest/external.py`.
            перечитан = cv2.imread(str(путь))
            if перечитан is None:
                raise OSError(f"кроп записан, но не читается обратно: {путь}")
            оставлено[вид] += 1
            записи.append(
                {
                    "photo_id": photo_id,
                    "listing_id": f"{вид}/{кандидат.origin}",
                    "source": "negatives",
                    "crop_path": относительный,
                    "brand": None,
                    "model_id": None,
                    "class_idx": None,
                    "body_style": None,
                    "year": None,
                    "bbox": [float(бокс.x), float(бокс.y), float(бокс.w), float(бокс.h)],
                    "bbox_area_frac": float(доля),
                    "det_conf": float(бокс.conf),
                    "phash": phash_of(перечитан),
                    "raw_model_string": кандидат.raw_name,
                    "is_multi_car": False,
                }
            )
    кадр = pd.DataFrame(записи, columns=list(COLUMNS)) if записи else empty_frame()
    return NegativesResult(frame=кадр, kept=dict(оставлено), rejected=dict(отказы))


def reject_candidates(
    reject_log: Path, reject_index: Path, harvest_root: Path
) -> tuple[list[NegativeCandidate], int]:
    """Отбраковки харвеста с причинами `REJECT_REASONS`, у которых сырой кадр на диске.

    Журнал знает причину, индекс — sha256 сырого файла; связь — пара
    `(class_id, source_item_id)`. Вторым значением — сколько подходящих записей без файла:
    это число идёт в отчёт, а не теряется молча.
    """
    хеши = {
        (строка["class_id"], строка["source_item_id"]): строка["sha256"]
        for строка in csv.DictReader(reject_index.read_text(encoding="utf-8").splitlines())
    }
    кандидаты: list[NegativeCandidate] = []
    без_файла = 0
    for строка in csv.DictReader(reject_log.read_text(encoding="utf-8").splitlines()):
        if строка["reason"] not in REJECT_REASONS:
            continue
        ключ = (строка["class_id"], строка["source_item_id"])
        хеш = хеши.get(ключ)
        путь = cache_path(harvest_root, строка["source"], хеш, "jpg") if хеш else None
        if путь is None or not путь.is_file():
            без_файла += 1
            continue
        кандидаты.append(
            NegativeCandidate(
                путь, False, f"{строка['source']}/{строка['source_item_id']}", ":".join(ключ)
            )
        )
    return кандидаты, без_файла
