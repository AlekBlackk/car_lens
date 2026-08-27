"""Провенанс Commons-выборки: юридический след замера и его же валидатор.

Состав зафиксирован до отбора фотографий, поэтому класс вне списка — ошибка загрузки, а
не повод расширить список: иначе выборка подстраивается под то, что нашлось, и замер
меряет удачу поиска. Недобор (меньше восьми фото на класс) не ошибка — он фиксируется
отчётом, а вот перекос сверх восьми ошибка, он утяжеляет вклад класса в общую долю.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from spike.clip_zeroshot.commons_classes import COMMONS_CLASSES

DEFAULT_COMMONS_MANIFEST = Path(__file__).resolve().parent / "commons_manifest.csv"
ФОТО_НА_КЛАСС = 8
ОБЯЗАТЕЛЬНЫЕ = ("file", "class_id", "url", "author", "license", "checked", "basis")
ДОПУСТИМЫЕ_ЛИЦЕНЗИИ = frozenset(
    {
        "CC0",
        "PD",
        "CC BY 2.0",
        "CC BY 3.0",
        "CC BY 4.0",
        "CC BY-SA 2.0",
        "CC BY-SA 3.0",
        "CC BY-SA 4.0",
    }
)


class CommonsError(ValueError):
    """Выборка не годится: замер по ней недействителен."""


@dataclass(frozen=True, slots=True)
class CommonsPhoto:
    """Одна фотография с полным провенансом."""

    file: str
    class_id: str
    url: str
    author: str
    license: str
    checked: str
    basis: str


def load_commons_manifest(path: Path) -> list[CommonsPhoto]:
    """Прочитать манифест фотографий и провалидировать его целиком."""
    строки = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not строки:
        raise CommonsError(f"{path}: манифест пуст")

    фото: list[CommonsPhoto] = []
    файлы: set[str] = set()
    for номер, строка in enumerate(строки, start=2):
        for поле in ОБЯЗАТЕЛЬНЫЕ:
            if not (строка.get(поле) or "").strip():
                raise CommonsError(f"{path}:{номер}: пустое поле {поле}")
        class_id = строка["class_id"].strip()
        if class_id not in COMMONS_CLASSES:
            raise CommonsError(
                f"{path}:{номер}: класс {class_id} вне зафиксированного состава §5.2"
            )
        лицензия = строка["license"].strip()
        if лицензия not in ДОПУСТИМЫЕ_ЛИЦЕНЗИИ:
            raise CommonsError(f"{path}:{номер}: лицензия {лицензия!r} не свободная")
        имя = строка["file"].strip()
        if имя in файлы:
            raise CommonsError(f"{path}:{номер}: повтор файла {имя!r}")
        файлы.add(имя)
        фото.append(
            CommonsPhoto(
                file=имя,
                class_id=class_id,
                url=строка["url"].strip(),
                author=строка["author"].strip(),
                license=лицензия,
                checked=строка["checked"].strip(),
                basis=строка["basis"].strip(),
            )
        )

    счёт = Counter(ф.class_id for ф in фото)
    перекос = sorted(c for c, n in счёт.items() if n > ФОТО_НА_КЛАСС)
    if перекос:
        raise CommonsError(f"{path}: больше {ФОТО_НА_КЛАСС} фото на класс: {', '.join(перекос)}")
    return фото
