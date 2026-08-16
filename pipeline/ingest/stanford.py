"""Чтение раскладки Stanford Cars с диска.

Единственный модуль конвейера, знающий про этот датасет. Знает он ровно
контракт каталога:

    <корень>/
    ├── images/*.jpg
    └── annotations.csv   # image,x1,y1,x2,y2,class_name,official_split

Ни сети, ни детектора, ни parquet здесь нет. Всё, что выше этой черты, —
забота `scripts/fetch_stanford.py`, и именно поэтому тесты этого модуля
обходятся деревом из двух картинок 8×8.

Список колонок продублирован из качалки намеренно: `pipeline` не должен
зависеть от `scripts`. За расхождением копий следит
`test_контракт_совпадает_с_качалкой`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

# Приватный `_ГОД` берётся из normalize намеренно, а не дублируется своей
# регуляркой: расхождение двух определений года — тот же класс дефекта, что
# расхождение каталога с нормализатором (см. `scripts/gen_stanford_catalog.py`).
from pipeline.normalize import _ГОД

ANNOTATION_COLUMNS: tuple[str, ...] = (
    "image",
    "x1",
    "y1",
    "x2",
    "y2",
    "class_name",
    "official_split",
)

_КООРДИНАТЫ = ("x1", "y1", "x2", "y2")

# Хвостовой якорь `\b(...)\s*$` — надстройка над общим определением года из
# `_ГОД`, а не второе независимое определение того, что им является: год
# ищется не где угодно в строке, а строго в конце имени класса.
_ГОД_В_ХВОСТЕ = re.compile(rf"\b({_ГОД.pattern})\s*$")


class IngestError(ValueError):
    """Раскладка датасета не соответствует контракту."""


@dataclass(frozen=True, slots=True)
class StanfordPhoto:
    """Одна фотография датасета вместе с её разметкой."""

    path: Path
    class_name: str
    manual_box: tuple[int, int, int, int] | None
    official_split: str


def year_of(class_name: str) -> int | None:
    """Модельный год из хвоста имени класса.

    Год вынимает ingest, а не нормализатор: `NormalizeResult` поля `year`
    не имеет и не получит — год не участвует в определении класса, он
    просто едет в колонку §13.1 для будущей версии.
    """
    совпадение = _ГОД_В_ХВОСТЕ.search(class_name)
    return int(совпадение.group(1)) if совпадение else None


def _разобрать_бокс(строка: dict[str, str], где: str) -> tuple[int, int, int, int] | None:
    значения = [(строка.get(к) or "").strip() for к in _КООРДИНАТЫ]
    if not any(значения):
        return None
    if not all(значения):
        raise IngestError(f"{где}: bbox заполнен наполовину: {значения}")
    try:
        x1, y1, x2, y2 = (int(float(v)) for v in значения)
    except ValueError as ошибка:
        raise IngestError(f"{где}: bbox не разбирается: {значения}") from ошибка
    return x1, y1, x2, y2


def read_annotations(dataset_dir: Path) -> list[StanfordPhoto]:
    """Прочитать `annotations.csv` и проверить, что все картинки на месте."""
    разметка = dataset_dir / "annotations.csv"
    if not разметка.is_file():
        raise IngestError(f"нет {разметка}")

    изображения = dataset_dir / "images"
    итог: list[StanfordPhoto] = []

    with разметка.open(encoding="utf-8", newline="") as ф:
        читатель = csv.DictReader(ф)
        колонки = tuple(читатель.fieldnames or ())
        лишние = sorted(set(колонки) - set(ANNOTATION_COLUMNS))
        нет = sorted(set(ANNOTATION_COLUMNS) - set(колонки))
        if нет:
            raise IngestError(f"{разметка}: нет колонок: {', '.join(нет)}")
        if лишние:
            raise IngestError(f"{разметка}: лишние колонки: {', '.join(лишние)}")

        for номер, строка in enumerate(читатель, start=2):
            где = f"{разметка}:{номер}"
            имя = (строка.get("image") or "").strip()
            if not имя:
                raise IngestError(f"{где}: пустое имя файла")
            путь = изображения / имя
            if not путь.is_file():
                raise IngestError(f"{где}: нет файла {имя}")
            итог.append(
                StanfordPhoto(
                    path=путь,
                    class_name=(строка.get("class_name") or "").strip(),
                    manual_box=_разобрать_бокс(строка, где),
                    official_split=(строка.get("official_split") or "").strip(),
                )
            )
    return итог
