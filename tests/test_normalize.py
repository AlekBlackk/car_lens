"""Тесты нормализации свободного текста модели (§3.3)."""

from pathlib import Path

import pytest

from pipeline.catalog import Catalog, load_catalog
from pipeline.normalize import BODY_STYLES, model_core, normalize

YAML = """
version: 1
brands:
  Volkswagen:
    models:
      golf: {display: Volkswagen Golf}
      passat: {display: Volkswagen Passat}
  Audi:
    models:
      a4: {display: Audi A4}
      a6: {display: Audi A6}
  Mercedes-Benz:
    models:
      c_class: {display: Mercedes-Benz C-Class}
  Citroen:
    models:
      c3: {display: Citroën C3}
"""


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    путь = tmp_path / "canonical.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return load_catalog(путь)


@pytest.mark.parametrize(
    "сырое",
    ["Golf", "Golf Variant", "GOLF 1.6 TDI", "golf gti", "  Golf   VII  "],
)
def test_варианты_из_спецификации_сходятся_в_один_id(каталог: Catalog, сырое: str) -> None:
    итог = normalize(сырое, "Volkswagen", каталог, {})
    assert итог.model_id == "volkswagen__golf"


def test_кузовной_суффикс_сохраняется_отдельным_полем(каталог: Catalog) -> None:
    итог = normalize("Golf Variant", "Volkswagen", каталог, {})
    assert итог.model_id == "volkswagen__golf"
    assert итог.body_style == "wagon"
    assert итог.reason == "fuzzy"
    assert итог.score >= 88


def test_кузовной_суффикс_не_создаёт_отдельный_класс(каталог: Catalog) -> None:
    обычный = normalize("Golf", "Volkswagen", каталог, {})
    универсал = normalize("Golf Kombi", "Volkswagen", каталог, {})
    assert обычный.model_id == "volkswagen__golf"
    assert универсал.model_id == "volkswagen__golf"


def test_все_кузова_из_словаря_допустимы_по_схеме() -> None:
    assert {
        "hatchback",
        "sedan",
        "wagon",
        "suv",
        "coupe",
        "van",
        "pickup",
    } == BODY_STYLES


@pytest.mark.parametrize(
    ("сырое", "ожидание"),
    [
        ("A4 2.0 TDI quattro", "audi__a4"),
        ("A6 Avant 3.0 TDI", "audi__a6"),
        ("A4 S-line", "audi__a4"),
    ],
)
def test_отсекает_объём_топливо_привод_и_трим(каталог: Catalog, сырое: str, ожидание: str) -> None:
    assert normalize(сырое, "Audi", каталог, {}).model_id == ожидание


def test_диакритика_снимается(каталог: Catalog) -> None:
    assert normalize("C3 Élégance", "Citroën", каталог, {}).model_id == "citroen__c3"


def test_немецкое_написание_класса_ловится_алиасом(каталог: Catalog) -> None:
    """`C-Klasse` фонетически далёк от `c_class`, fuzzy его не возьмёт.

    Это ровно тот случай, ради которого §3.3 держит карту алиасов:
    отсечение объёма и топлива работает как обычно, а последний шаг
    делает явная запись в `aliases.yaml`.
    """
    итог = normalize(
        "C-Klasse 220 CDI", "Mercedes-Benz", каталог, {"c-klasse": "mercedes_benz__c_class"}
    )
    assert итог.model_id == "mercedes_benz__c_class"
    assert итог.reason == "alias"


def test_алиас_имеет_приоритет_над_fuzzy(каталог: Catalog) -> None:
    итог = normalize("Kolmonen", "Volkswagen", каталог, {"kolmonen": "volkswagen__passat"})
    assert итог.model_id == "volkswagen__passat"
    assert итог.reason == "alias"


def test_алиас_чужой_марки_не_применяется(каталог: Catalog) -> None:
    """Карта алиасов общая на весь каталог, а марка в объявлении задана.

    `3er` в объявлении Volkswagen не должен превратиться в `bmw__3_series`:
    это та же межмарочная утечка, от которой fuzzy защищён фильтром
    кандидатов по марке. Чужой алиас отбрасывается, строка идёт в fuzzy
    и там не набирает порога.
    """
    итог = normalize("3er", "Volkswagen", каталог, {"3er": "bmw__3_series"})
    assert итог.model_id is None
    assert итог.reason == "below_threshold"


def test_ниже_порога_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("Kolmonen", "Volkswagen", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "below_threshold"


def test_неизвестная_марка_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("Golf", "Lada", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "unknown_brand"


def test_пустая_строка_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("   ", "Volkswagen", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "empty"


def test_строка_из_одних_отсекаемых_токенов_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("1.6 TDI", "Volkswagen", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "empty_after_strip"


def test_сырое_значение_всегда_сохраняется(каталог: Catalog) -> None:
    assert normalize("GOLF 1.6 TDI", "Volkswagen", каталог, {}).raw == "GOLF 1.6 TDI"


def test_кандидат_ограничен_моделями_своей_марки(каталог: Catalog) -> None:
    """A4 не должен утечь в Volkswagen, даже если строка похожа."""
    итог = normalize("A4", "Volkswagen", каталог, {})
    assert итог.model_id is None


def test_хвостовой_год_отбрасывается(каталог: Catalog) -> None:
    """Имя класса Stanford — «Audi A5 Coupe 2012», год к классу отношения не имеет."""
    итог = normalize("A4 Sedan 2012", "Audi", каталог, {})
    assert итог.model_id == "audi__a4"
    assert итог.body_style == "sedan"


def test_год_остаётся_если_он_единственный_токен(каталог: Catalog) -> None:
    """«Peugeot 2008» — название модели, а не год выпуска.

    Правило снимает год только с хвоста и только если в ядре есть что-то
    ещё. Иначе нормализатор съел бы половину модельного ряда Peugeot.
    """
    ядро, _ = model_core("2008")
    assert ядро == "2008"


def test_год_в_середине_не_трогается() -> None:
    ядро, _ = model_core("Silverado 1500 2012")
    assert ядро == "silverado 1500"


@pytest.mark.parametrize(
    ("сырое", "кузов"),
    [
        ("Silverado 1500 Extended Cab 2012", "pickup"),
        ("Silverado 1500 Regular Cab 2012", "pickup"),
        ("F-150 SuperCab 2012", "pickup"),
        ("Ram 1500 Crew Cab 2012", "pickup"),
        ("Frontier King Cab 2012", "pickup"),
        ("Dakota Club Cab 2012", "pickup"),
        ("Caravan Minivan 2012", "van"),
        ("Outback Wagon 2012", "wagon"),
    ],
)
def test_американские_кузова_распознаются(сырое: str, кузов: str) -> None:
    _, найденный = model_core(сырое)
    assert найденный == кузов


def test_кузов_из_кабины_не_остаётся_в_ядре() -> None:
    """Иначе «extended» уедет в fuzzy и утащит сопоставление в сторону."""
    ядро, кузов = model_core("Silverado 1500 Extended Cab 2012")
    assert ядро == "silverado 1500"
    assert кузов == "pickup"


def test_model_core_публичен_и_повторяет_внутреннее_правило() -> None:
    """Генератор каталога скелета обязан использовать ровно эти правила.

    Если он начнёт разбирать имена своей регуляркой, каталог и
    нормализатор разойдутся, и часть классов перестанет находиться.
    """
    assert model_core("Golf Variant 1.6 TDI") == ("golf", "wagon")
