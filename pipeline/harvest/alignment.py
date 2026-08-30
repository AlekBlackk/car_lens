"""Триаж выборки по именам файлов: кто уходит автоматически, кто идёт к человеку.

Модуль работает с манифестом целиком и потому отделён от `aliases.py`, который отвечает
на вопрос об одном имени и одном классе и нужен шлюзу на приёме — там нет ни манифеста,
ни отчёта. Разъём между модулями один: `alias_verdict`.

Сети здесь нет ни в одном пути: триаж считается по колонкам манифеста, а значит целиком
воспроизводим и целиком тестируем.

Автомат допущен только к однозначному случаю — совпала ровно одна чужая игла и не
совпала своя. Человеку нечего добавить к слову `cressida` в имени файла. Два и более
чужих класса означают, что иглы спорят между собой, и вердикт за человека выдавать
нельзя; `unknown` означает, что имя молчит, и решает тоже человек.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.harvest.aliases import AliasTable, alias_verdict, normalize_filename
from pipeline.harvest.manifest import HarvestRow

SEARCH_BASIS_PREFIX = "search:"


@dataclass(frozen=True, slots=True)
class RowVerdict:
    """Что имя файла говорит о строке манифеста."""

    file: str
    class_id: str
    kind: str
    foreign: tuple[str, ...]


def triage(rows: Sequence[HarvestRow], table: AliasTable) -> list[RowVerdict]:
    """Вердикт на каждую строку поискового базиса. Строки из категорий не трогаются.

    Имя берётся из `page_url`: колонки `title` в манифесте нет, а `page_url` несёт то же
    имя, только процент-кодированным — это снимает `normalize_filename`.
    """
    итог: list[RowVerdict] = []
    for строка in rows:
        if not строка.basis.startswith(SEARCH_BASIS_PREFIX):
            continue
        вердикт = alias_verdict(normalize_filename(строка.page_url), строка.class_id, table)
        итог.append(RowVerdict(строка.file, строка.class_id, вердикт.kind, вердикт.foreign))
    return итог


def auto_drops(verdicts: Sequence[RowVerdict]) -> list[RowVerdict]:
    """Строки, которые программа вправе выкинуть сама: ровно один чужой класс."""
    return [в for в in verdicts if в.kind == "foreign" and len(в.foreign) == 1]


def review_queue(verdicts: Sequence[RowVerdict]) -> list[RowVerdict]:
    """Строки на глаз человеку: имя молчит либо иглы спорят между собой."""
    return [
        в for в in verdicts if в.kind == "unknown" or (в.kind == "foreign" and len(в.foreign) > 1)
    ]
