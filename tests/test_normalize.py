"""Тесты нормализации свободного текста модели (§3.3)."""

from pathlib import Path

import pytest

from pipeline.catalog import Catalog, load_catalog
from pipeline.normalize import BODY_STYLES, normalize

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


def test_кузовной_суффикс_не_создаёт_отдельный_класс(каталог: Catalog) -> None:
    обычный = normalize("Golf", "Volkswagen", каталог, {})
    универсал = normalize("Golf Kombi", "Volkswagen", каталог, {})
    assert обычный.model_id == универсал.model_id


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
    assert normalize("C3", "Citroën", каталог, {}).model_id == "citroen__c3"


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
