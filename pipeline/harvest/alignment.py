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

from pipeline.harvest.aliases import (
    AliasesError,
    AliasKind,
    AliasTable,
    alias_verdict,
    normalize_filename,
)
from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.sources import SEARCH_BASIS_PREFIX

__all__ = ["SEARCH_BASIS_PREFIX", "RowVerdict", "auto_drops", "review_queue", "triage"]


@dataclass(frozen=True, slots=True)
class RowVerdict:
    """Что имя файла говорит о строке манифеста."""

    file: str
    class_id: str
    kind: AliasKind
    foreign: tuple[str, ...]


def triage(rows: Sequence[HarvestRow], table: AliasTable) -> list[RowVerdict]:
    """Вердикт на каждую строку поискового базиса. Строки из категорий не трогаются.

    Имя берётся из `page_url`: колонки `title` в манифесте нет, а `page_url` несёт то же
    имя, только процент-кодированным — это снимает `normalize_filename`. Форму URL
    `normalize_filename` знает одну — commons (`/wiki/File:`), и сегодня в манифесте
    других источников нет. Второй источник (Flickr, Openverse по Р-11) отдаст URL, от
    которого в токены уйдёт весь путь, и слово из домена сможет совпасть с иглой —
    к тому времени имя придётся брать из колонки источника, а не из адреса.

    Класс вне таблицы игл — не `unknown`, а остановка: молчаливый пропуск означал бы
    вердикт, посчитанный не по тому классу. `KeyError` из `alias_verdict` перевыбрасывается
    с именем файла и класса, потому что на манифесте в три с половиной тысячи строк голый
    `KeyError('lada__granta')` не говорит ни какая это строка, ни что править надо
    `catalog/model_aliases.yaml`.
    """
    итог: list[RowVerdict] = []
    for строка in rows:
        if not строка.basis.startswith(SEARCH_BASIS_PREFIX):
            continue
        try:
            вердикт = alias_verdict(normalize_filename(строка.page_url), строка.class_id, table)
        except KeyError as ошибка:
            raise AliasesError(
                f"{строка.file}: класса {строка.class_id} нет в таблице игл "
                f"(catalog/model_aliases.yaml)"
            ) from ошибка
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
