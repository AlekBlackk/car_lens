"""Термины поиска и категории Commons на каждый класс каталога.

Отдельная таблица, а не переиспользование `latin_names.yaml`: там группы записаны как
«Geely Coolray or Belgee X50» — годный промпт для текстового энкодера и негодный
поисковый запрос. Искать надо по каждому члену группы отдельно, а результаты складывать
в один `class_id` (правило 4 Р-11).

Полнота таблицы держится тестом против каталога, а не вниманием автора: класс без
запроса не соберёт ни одной фотографии и уйдёт в дефицит по причине, которой нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pipeline.harvest.aliases import normalize_filename

_КОРЕНЬ = Path(__file__).resolve().parent.parent.parent
DEFAULT_QUERIES_PATH = _КОРЕНЬ / "catalog" / "harvest_queries.yaml"
DEFAULT_LATIN_NAMES_PATH = _КОРЕНЬ / "catalog" / "latin_names.yaml"

SUPPORTED_VERSION = 1


class QueriesError(ValueError):
    """Таблица запросов не проходит валидацию."""


@dataclass(frozen=True, slots=True)
class ClassQueries:
    """Чем искать один класс: категории Commons, текстовые термины и правила обхода."""

    class_id: str
    categories: tuple[str, ...]
    terms: tuple[str, ...]
    depth: int = 1
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


def load_queries(path: Path) -> dict[str, ClassQueries]:
    """Прочитать таблицу запросов и провалидировать её целиком."""
    сырое: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if сырое.get("version") != SUPPORTED_VERSION:
        raise QueriesError(f"{path}: поддерживается только version: {SUPPORTED_VERSION}")
    таблица = сырое.get("classes")
    if not isinstance(таблица, dict):
        raise QueriesError(f"{path}: classes должно быть отображением")

    результат: dict[str, ClassQueries] = {}
    for class_id, тело in таблица.items():
        if not isinstance(тело, dict):
            raise QueriesError(f"{path}: {class_id} должен быть отображением")
        категории = tuple(str(x).strip() for x in (тело.get("categories") or []) if str(x).strip())
        термины = tuple(str(x).strip() for x in (тело.get("terms") or []) if str(x).strip())
        if not категории and not термины:
            raise QueriesError(f"{path}: у {class_id} нет ни категорий, ни терминов")
        глубина = тело.get("depth", 1)
        if not isinstance(глубина, int) or isinstance(глубина, bool) or not 0 <= глубина <= 2:
            raise QueriesError(
                f"{path}: у {class_id} depth должен быть целым 0..2, получено {глубина!r}"
            )
        включения = tuple(str(x).strip() for x in (тело.get("include") or []) if str(x).strip())
        исключения = tuple(str(x).strip() for x in (тело.get("exclude") or []) if str(x).strip())
        результат[str(class_id)] = ClassQueries(
            str(class_id),
            категории,
            термины,
            depth=глубина,
            include=включения,
            exclude=исключения,
        )
    return результат


def claimed_categories(table: dict[str, ClassQueries]) -> dict[str, frozenset[str]]:
    """Нормализованное имя заявленной категории → множество заявивших её `class_id`.

    Материал для `admission.build_admit`: правило допуска подкатегории (§5 дизайна C1)
    первым делом бракует полку, заявленную другим классом, даже если её имя проходит по
    иглам, — а знать «другим» без этой карты не из чего.
    """
    итог: dict[str, set[str]] = {}
    for class_id, запросы in table.items():
        for категория in запросы.categories:
            итог.setdefault(normalize_filename(категория), set()).add(class_id)
    return {имя: frozenset(классы) for имя, классы in итог.items()}


def load_latin_names(path: Path) -> dict[str, str]:
    """Прочитать `class_id` → латинское имя класса."""
    сырое: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    таблица = сырое.get("names")
    if not isinstance(таблица, dict):
        raise QueriesError(f"{path}: names должно быть отображением")
    имена: dict[str, str] = {}
    for class_id, имя in таблица.items():
        текст = str(имя or "").strip()
        if not текст:
            raise QueriesError(f"{path}: пустое имя у {class_id}")
        имена[str(class_id)] = текст
    return имена
