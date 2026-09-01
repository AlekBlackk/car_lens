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
from pipeline.harvest.categories import MAX_DEPTH

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


def _строки(path: Path, class_id: str, тело: dict[str, Any], ключ: str) -> tuple[str, ...]:
    """Прочитать поле-список непустых строк (§6.1 дизайна C1) у одного класса.

    Проверка списка целиком и поэлементно, а не `str(x) for x in ...`, потому что таблицу
    правят руками. Скаляр вместо списка (`exclude: Abt Q5` без скобок) — законный YAML, но
    итерация пошла бы по строке и дала бы шесть односимвольных значений; `exclude` и
    `include` сравниваются по целому нормализованному имени, так что исключение молча
    перестало бы исключать что-либо. Та же защита с тем же обоснованием стоит в
    `aliases._иглы_класса`. От элемента требуется строка, а не `str(элемент)`: он превратил
    бы `null` в живое значение `none`.

    Значение, от которого после обрезки пробелов ничего не осталось, отбрасывается молча:
    отказ здесь дал бы ложную тревогу на лишнем пробеле, а пустой итог поля законен —
    непустоту `categories` и `terms` вместе проверяет вызывающий.
    """
    значение = тело.get(ключ)
    перечень = [] if значение is None else значение
    if not isinstance(перечень, list):
        raise QueriesError(f"{path}: {ключ} у {class_id} должно быть списком строк")
    значения: list[str] = []
    for элемент in перечень:
        if not isinstance(элемент, str):
            raise QueriesError(f"{path}: {ключ} у {class_id}: {элемент!r} не строка")
        обрезанное = элемент.strip()
        if обрезанное:
            значения.append(обрезанное)
    return tuple(значения)


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
        имя_класса = str(class_id)
        категории = _строки(path, имя_класса, тело, "categories")
        термины = _строки(path, имя_класса, тело, "terms")
        if not категории and not термины:
            raise QueriesError(f"{path}: у {имя_класса} нет ни категорий, ни терминов")
        глубина = тело.get("depth", 1)
        if (
            not isinstance(глубина, int)
            or isinstance(глубина, bool)
            or not 0 <= глубина <= MAX_DEPTH
        ):
            raise QueriesError(
                f"{path}: у {имя_класса} depth должен быть целым 0..{MAX_DEPTH},"
                f" получено {глубина!r}"
            )
        результат[имя_класса] = ClassQueries(
            имя_класса,
            категории,
            термины,
            depth=глубина,
            include=_строки(path, имя_класса, тело, "include"),
            exclude=_строки(path, имя_класса, тело, "exclude"),
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
