"""Тесты валидности и полноты таблицы игл моделей (G_AL1)."""

from __future__ import annotations

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import (
    DEFAULT_ALIASES_PATH,
    load_alias_table,
    needle_conflicts,
    render_alias_table,
)


def test_таблица_покрывает_весь_каталог() -> None:
    """Класс без игл превращает поисковый путь в глухую стену (§9 дизайна)."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    таблица = load_alias_table(DEFAULT_ALIASES_PATH)
    assert set(таблица) == {класс.class_id for класс in каталог.classes}


def test_взаимно_неразличимых_игл_нет() -> None:
    """Это и есть G_AL1: пока он красный, дальние числа посчитаны не по тем иглам."""
    конфликты = needle_conflicts(load_alias_table(DEFAULT_ALIASES_PATH))
    assert конфликты == [], f"неразличимые иглы: {конфликты[:5]}"


def test_файл_таблицы_в_каноническом_виде() -> None:
    """Ямл-дампер однажды переписал таблицу целиком и снёс шапку про ручное ведение.

    Диффа на 566 строк при четырнадцати новых иглах по смыслу человек не читает, а вместе
    с шапкой уходит и указание на этот самый гейт. Формат держится тестом, потому что
    держать его больше нечем: файл правят руками.
    """
    ожидание = render_alias_table(load_alias_table(DEFAULT_ALIASES_PATH))
    assert DEFAULT_ALIASES_PATH.read_text(encoding="utf-8") == ожидание
