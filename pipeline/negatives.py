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
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.harvest.aliases import matches_needle, normalize_filename


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
