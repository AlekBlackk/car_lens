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
    broad_needles,
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
    needle: str


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
        итог.append(
            RowVerdict(строка.file, строка.class_id, вердикт.kind, вердикт.foreign, вердикт.needle)
        )
    return итог


def auto_drops(verdicts: Sequence[RowVerdict], table: AliasTable) -> list[RowVerdict]:
    """Строки, которые программа вправе выкинуть сама.

    Три условия: чужой класс ровно один, своя игла не совпала (это уже `kind`), и
    сработавшая чужая игла не шире собственных игл класса.

    Третье условие добавлено после ревью и стоило половины автомата. На заполненной
    таблице 67 автодропов из 131 держались на широкой игле соседа: `Lada_Niva_2.jpg` в
    классе `lada__niva_2131` ловится голой иглой `niva` соседнего `lada__niva_2121`, а
    своя игла (`niva 2131`, `2131`) совпасть с таким именем не может по устройству. §11
    дизайна оправдывает автомат тем, что «человеку нечего добавить к слову `cressida` в
    имени файла» — но к слову `niva` человеку есть что добавить: он посмотрит на кадр и
    посчитает двери. Такие строки уходят в очередь, а не под нож.
    """
    широкие = _широкие_пары(table)
    return [в for в in verdicts if _автомат_вправе(в, широкие)]


def review_queue(verdicts: Sequence[RowVerdict], table: AliasTable) -> list[RowVerdict]:
    """Строки на глаз человеку: всё, что не своё и не отдано автомату.

    Дополнение считается, а не перечисляется: `own` плюс автодропы плюс очередь обязаны
    давать полный список вердиктов, иначе охват `G_AL2` закрыть нечем.
    """
    широкие = _широкие_пары(table)
    return [в for в in verdicts if в.kind != "own" and not _автомат_вправе(в, широкие)]


def _широкие_пары(table: AliasTable) -> frozenset[tuple[str, str, str]]:
    """Тройки «сосед, его широкая игла, наш класс» — на них автомат не допускается."""
    return frozenset(broad_needles(table))


def _автомат_вправе(verdict: RowVerdict, широкие: frozenset[tuple[str, str, str]]) -> bool:
    """Один предикат на два списка: разойтись им нельзя."""
    if verdict.kind != "foreign" or len(verdict.foreign) != 1:
        return False
    return (verdict.foreign[0], verdict.needle, verdict.class_id) not in широкие
