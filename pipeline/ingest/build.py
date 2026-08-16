"""Сборка кадра данных §13.1 из потока фотографий Stanford Cars.

Оркестратор соединяет уже готовые части: детектор выбирает главную
машину (§3.4), `crop` режет кроп с паддингом (§4.2), `clean` считает
перцептивный хеш, `normalize` сводит имя класса к `model_id` (§3.3).

Модуль сознательно завязан на `StanfordPhoto`: `build_frame` типизирован
`Iterable[StanfordPhoto]`, год разбирается из стэнфордской конвенции
имени класса (`year_of`), а `_разделить_имя` рассчитан на стэнфордские
строки вида «Audi A5 Coupe 2012». Независим он не от источника данных,
а от **детектора**: тот приходит параметром через протокол `Detector`,
а не импортируется. Без этого ни один тест не запустится без весов
YOLO, а тесты, требующие сети, перестают запускаться примерно через
неделю.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from pipeline.catalog import Catalog, class_index
from pipeline.clean import duplicate_indices, phash_of
from pipeline.crop import Detector, crop_image, select_primary
from pipeline.ingest.stanford import StanfordPhoto, year_of
from pipeline.normalize import normalize
from pipeline.schema import COLUMNS, empty_frame


@dataclass(frozen=True, slots=True)
class BuildStats:
    """Счётчики прогона. Идут в отчёт, а не в parquet.

    `iou_values` собирается до дедупа и до фильтрации сирот с диска —
    туда попадают и фотографии, впоследствии выброшенные как дубли.
    Поэтому `len(iou_values) != len(frame)` в общем случае, и это не
    баг: для отчёта о качестве детектора важна сверка со всей ручной
    разметкой, а не только с той частью, что выжила после дедупа.
    """

    seen: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    unmapped: int = 0
    deduped: int = 0
    iou_values: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Кадр данных §13.1 и статистика его сборки."""

    frame: pd.DataFrame
    stats: BuildStats


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Пересечение над объединением двух боксов `(x1, y1, x2, y2)`.

    Нужна не конвейеру, а отчёту: ручная разметка Stanford — бесплатный
    эталон, по которому видно, врёт ли детектор (решение Р-8). На финских
    объявлениях такой сверки не будет никогда.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    вx1, вy1 = max(ax1, bx1), max(ay1, by1)
    вx2, вy2 = min(ax2, bx2), min(ay2, by2)
    пересечение = max(0, вx2 - вx1) * max(0, вy2 - вy1)
    if пересечение == 0:
        return 0.0
    объединение = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - пересечение
    return пересечение / объединение if объединение > 0 else 0.0


def build_frame(
    photos: Iterable[StanfordPhoto],
    detector: Detector,
    catalog: Catalog,
    aliases: Mapping[str, str],
    crops_root: Path,
    source: str = "stanford",
) -> BuildResult:
    """Поток фотографий → кадр данных §13.1 и статистика прогона."""
    индекс = class_index(catalog)
    отбраковка: Counter[str] = Counter()
    несопоставленных = 0
    iou_значения: list[float] = []
    записи: list[dict[str, Any]] = []
    хеши: list[int] = []
    просмотрено = 0

    crops_root.mkdir(parents=True, exist_ok=True)

    for фото in photos:
        просмотрено += 1
        изображение = cv2.imread(str(фото.path))
        if изображение is None:
            отбраковка["unreadable"] += 1
            continue

        высота, ширина = изображение.shape[:2]
        решение = select_primary(detector.detect(изображение), ширина, высота)
        if решение.box is None:
            отбраковка[решение.reject_reason or "unknown"] += 1
            continue

        марка, остаток = _разделить_имя(фото.class_name, catalog)
        итог = normalize(остаток, марка, catalog, aliases)
        if итог.model_id is None or итог.model_id not in индекс:
            несопоставленных += 1
            continue

        if фото.manual_box is not None:
            бокс = решение.box
            iou_значения.append(
                iou(
                    (
                        round(бокс.x),
                        round(бокс.y),
                        round(бокс.x + бокс.w),
                        round(бокс.y + бокс.h),
                    ),
                    фото.manual_box,
                )
            )

        кроп = crop_image(изображение, решение.box)
        photo_id = f"{source}-{фото.path.stem}"
        относительный = f"{photo_id}.jpg"
        путь_кропа = crops_root / относительный
        if not cv2.imwrite(str(путь_кропа), кроп):
            raise OSError(f"не удалось записать кроп {путь_кропа}")

        # Хеш считается по кропу, только что перечитанному с диска, а не по
        # массиву до записи: `imwrite` сохраняет JPEG, и его квантование
        # чуть меняет пиксели. Если хешировать до сжатия, колонка `phash`
        # разойдётся с тем, что получит любой код, который позже откроет
        # тот же файл заново (например, повторный прогон дедупа с диска).
        кроп_с_диска = cv2.imread(str(путь_кропа))
        if кроп_с_диска is None:
            raise OSError(f"кроп записан, но не читается обратно: {путь_кропа}")
        хеши.append(phash_of(кроп_с_диска))
        записи.append(
            {
                "photo_id": photo_id,
                # У Stanford нет объявлений: каждое фото само себе объявление (Р-9).
                "listing_id": photo_id,
                "source": source,
                "crop_path": относительный,
                "brand": catalog.by_id(итог.model_id).brand,
                "model_id": итог.model_id,
                "class_idx": индекс[итог.model_id],
                "body_style": итог.body_style,
                "year": year_of(фото.class_name),
                "bbox": [
                    float(решение.box.x),
                    float(решение.box.y),
                    float(решение.box.w),
                    float(решение.box.h),
                ],
                "bbox_area_frac": float(решение.area_frac),
                "det_conf": float(решение.box.conf),
                "phash": 0,  # проставляется ниже, после дедупа
                "raw_model_string": фото.class_name,
                "is_multi_car": решение.is_multi_car,
            }
        )

    лишние = duplicate_indices(хеши)
    for i in лишние:
        # Кроп дубля уже на диске (писался до дедупа — хеш дубля неизвестен
        # заранее), поэтому сироту нужно убрать явно, а не просто выбросить
        # запись из кадра. На полном Stanford (16k фото) сирот без этого
        # накапливаются тысячи.
        (crops_root / записи[i]["crop_path"]).unlink(missing_ok=True)
    оставшиеся = [
        {**запись, "phash": хеши[i]} for i, запись in enumerate(записи) if i not in лишние
    ]

    кадр = pd.DataFrame(оставшиеся, columns=list(COLUMNS)) if оставшиеся else empty_frame()
    статистика = BuildStats(
        seen=просмотрено,
        rejected=dict(отбраковка),
        unmapped=несопоставленных,
        deduped=len(лишние),
        iou_values=tuple(iou_значения),
    )
    return BuildResult(frame=кадр, stats=статистика)


def _разделить_имя(class_name: str, catalog: Catalog) -> tuple[str, str]:
    """Имя класса → марка каталога и остаток строки.

    Марка ищется среди марок каталога, а не отрезается по первому
    пробелу: «AM General Hummer SUV 2000» и «Land Rover Range Rover SUV»
    начинаются с двух слов. Длинные имена проверяются первыми, иначе
    «Land» победил бы «Land Rover».
    """
    низкое = class_name.lower()
    for марка in sorted(catalog.brands(), key=len, reverse=True):
        if низкое.startswith(марка.lower() + " "):
            return марка, class_name[len(марка) :].strip()
    марка, _, остаток = class_name.partition(" ")
    return марка, остаток.strip()
