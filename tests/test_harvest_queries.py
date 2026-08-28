from pathlib import Path

import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.queries import (
    DEFAULT_LATIN_NAMES_PATH,
    DEFAULT_QUERIES_PATH,
    QueriesError,
    load_latin_names,
    load_queries,
)


def _записать(tmp_path: Path, тело: str) -> Path:
    путь = tmp_path / "harvest_queries.yaml"
    путь.write_text(тело, encoding="utf-8")
    return путь


def test_класс_с_категорией_и_терминами(tmp_path: Path) -> None:
    путь = _записать(
        tmp_path,
        "version: 1\n"
        "classes:\n"
        "  lada__granta:\n"
        "    categories: [Lada Granta]\n"
        "    terms: [\"Lada Granta\"]\n",
    )
    запросы = load_queries(путь)
    assert запросы["lada__granta"].categories == ("Lada Granta",)
    assert запросы["lada__granta"].terms == ("Lada Granta",)


def test_класс_без_категорий_и_терминов_отвергается(tmp_path: Path) -> None:
    """Пустая строка таблицы — молчаливый пропуск класса в прогоне."""
    путь = _записать(
        tmp_path,
        "version: 1\nclasses:\n  lada__granta:\n    categories: []\n    terms: []\n",
    )
    with pytest.raises(QueriesError, match="lada__granta"):
        load_queries(путь)


def test_таблица_проекта_покрывает_весь_каталог() -> None:
    """Класс без запроса не соберёт ни одной фотографии и уйдёт в дефицит молча."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    запросы = load_queries(DEFAULT_QUERIES_PATH)
    assert set(запросы) == {класс.class_id for класс in каталог.classes}


def test_группы_ищутся_по_каждому_члену() -> None:
    """Правило 4 Р-11: группа — единый класс, но искать надо по обоим именам."""
    запросы = load_queries(DEFAULT_QUERIES_PATH)
    термины = запросы["grp__coolray"].terms
    assert any("Coolray" in т for т in термины)
    assert any("Belgee" in т for т in термины)
    assert not any(" or " in т for т in термины)


def test_латинские_имена_покрывают_каталог() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES_PATH)
    assert set(имена) == {класс.class_id for класс in каталог.classes}
    for имя in имена.values():
        assert имя.isascii()
