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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pipeline.catalog import Catalog

# Порядок — приоритет при склейке: съёмка с уровня глаз ценнее пресс-фото (фильтры 7–8).
SOURCE_PRIORITY: tuple[str, ...] = ("manual", "skillfactory", "autoevolution")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAPPING_COLUMNS = ("dir_path", "model_id")


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
