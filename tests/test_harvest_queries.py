from pathlib import Path

import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.categories import MAX_DEPTH
from pipeline.harvest.queries import (
    DEFAULT_LATIN_NAMES_PATH,
    DEFAULT_QUERIES_PATH,
    ClassQueries,
    QueriesError,
    claimed_categories,
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
        '    terms: ["Lada Granta"]\n',
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


def test_новые_поля_читаются(tmp_path: Path) -> None:
    путь = tmp_path / "q.yaml"
    путь.write_text(
        "version: 1\n"
        "classes:\n"
        "  hyundai__accent:\n"
        "    categories: [Hyundai Accent]\n"
        '    terms: ["Hyundai Accent"]\n'
        "    depth: 2\n"
        '    exclude: ["Hyundai Accent (RB)"]\n'
        '    include: ["Hyundai Accent (X3)"]\n',
        encoding="utf-8",
    )

    класс = load_queries(путь)["hyundai__accent"]

    assert класс.depth == 2
    assert класс.exclude == ("Hyundai Accent (RB)",)
    assert класс.include == ("Hyundai Accent (X3)",)


def test_без_новых_полей_действуют_умолчания(tmp_path: Path) -> None:
    путь = tmp_path / "q.yaml"
    путь.write_text(
        'version: 1\nclasses:\n  kia__rio:\n    categories: [Kia Rio]\n    terms: ["Kia Rio"]\n',
        encoding="utf-8",
    )

    класс = load_queries(путь)["kia__rio"]

    assert (класс.depth, класс.include, класс.exclude) == (1, (), ())


def test_глубина_вне_диапазона_роняет(tmp_path: Path) -> None:
    путь = tmp_path / "q.yaml"
    путь.write_text(
        "version: 1\nclasses:\n  kia__rio:\n    categories: [Kia Rio]\n    depth: 5\n",
        encoding="utf-8",
    )

    with pytest.raises(QueriesError, match="depth"):
        load_queries(путь)


@pytest.mark.parametrize("значение", ["3", "true", '"2"', "null"])
def test_негодная_глубина_роняет(tmp_path: Path, значение: str) -> None:
    """Не только выход за диапазон: `true` в Python — целое, а глубиной обхода не является."""
    путь = _записать(
        tmp_path,
        f"version: 1\nclasses:\n  kia__rio:\n    categories: [Kia Rio]\n    depth: {значение}\n",
    )

    with pytest.raises(QueriesError, match="depth"):
        load_queries(путь)


def test_нулевая_глубина_допустима(tmp_path: Path) -> None:
    """0 — легальное значение: «в подкатегории не ходить вовсе»."""
    путь = _записать(
        tmp_path,
        "version: 1\nclasses:\n  kia__rio:\n    categories: [Kia Rio]\n    depth: 0\n",
    )

    assert load_queries(путь)["kia__rio"].depth == 0


def test_граница_глубины_взята_из_обхода(tmp_path: Path) -> None:
    """Верхняя граница — `categories.MAX_DEPTH`, а не литерал, дублирующий его в двух файлах."""
    путь = _записать(
        tmp_path,
        f"version: 1\nclasses:\n  kia__rio:\n    categories: [Kia Rio]\n    depth: {MAX_DEPTH}\n",
    )

    assert load_queries(путь)["kia__rio"].depth == MAX_DEPTH


@pytest.mark.parametrize("поле", ["categories", "terms", "include", "exclude"])
def test_скаляр_вместо_списка_роняет(tmp_path: Path, поле: str) -> None:
    """Скаляр — законный YAML, но итерация пошла бы по строке, по одной букве на значение.

    Тише всех отказ у `exclude`: сравнение идёт по целому нормализованному имени, поэтому
    `exclude: "Abt Q5"` без скобок превратился бы в шесть односимвольных исключений и
    перестал бы исключать что-либо, не сказав ни слова. §6.1 требует проверку явно.
    """
    поля = {"categories": "[Audi Q5]", "terms": '["Audi Q5"]'}
    поля[поле] = '"Abt Q5"'
    тело = "".join(f"    {ключ}: {значение}\n" for ключ, значение in поля.items())
    путь = _записать(tmp_path, f"version: 1\nclasses:\n  audi__q5:\n{тело}")

    with pytest.raises(QueriesError, match=поле):
        load_queries(путь)


def test_не_строка_в_списке_роняет(tmp_path: Path) -> None:
    """`str(элемент)` превратил бы `null` в живое исключение `none`."""
    путь = _записать(
        tmp_path,
        "version: 1\n"
        "classes:\n"
        "  audi__q5:\n"
        "    categories: [Audi Q5]\n"
        '    terms: ["Audi Q5"]\n'
        "    exclude: [Abt Q5, null]\n",
    )

    with pytest.raises(QueriesError, match="не строка"):
        load_queries(путь)


def test_пробельное_значение_отбрасывается_молча(tmp_path: Path) -> None:
    """Лишний пробел в ручной таблице — не повод ронять прогон, в отличие от скаляра."""
    путь = _записать(
        tmp_path,
        "version: 1\n"
        "classes:\n"
        "  audi__q5:\n"
        "    categories: [Audi Q5]\n"
        '    terms: ["Audi Q5"]\n'
        '    exclude: ["Abt Q5", "   "]\n',
    )

    assert load_queries(путь)["audi__q5"].exclude == ("Abt Q5",)


def test_категория_заявленная_двумя_классами_отдаёт_оба() -> None:
    таблица = {
        "geely__coolray": ClassQueries("geely__coolray", ("Geely Coolray",), ("Geely Coolray",)),
        "grp__coolray": ClassQueries("grp__coolray", ("Geely Coolray",), ("Belgee X50",)),
    }

    заявители = claimed_categories(таблица)

    assert заявители["geely coolray"] == frozenset({"geely__coolray", "grp__coolray"})


def test_имена_заявленных_категорий_нормализованы() -> None:
    таблица = {
        "kia__rio": ClassQueries("kia__rio", ("Kia Rio (UB)",), ("Kia Rio",)),
    }

    заявители = claimed_categories(таблица)

    assert заявители == {"kia rio ub": frozenset({"kia__rio"})}
