"""Тесты генератора расходного каталога скелета (решение Р-7)."""

from pathlib import Path

import pytest
import yaml

from pipeline.catalog import class_index, load_catalog
from scripts.gen_stanford_catalog import build_catalog, display_of, split_brand

ИМЕНА = [
    "Audi A5 Coupe 2012",
    "Audi A5 Cabriolet 2012",
    "AM General Hummer SUV 2000",
    "Land Rover Range Rover SUV 2012",
    "Chevrolet Silverado 1500 Extended Cab 2012",
]


@pytest.mark.parametrize(
    ("имя", "марка", "остаток"),
    [
        ("Audi A5 Coupe 2012", "Audi", "A5 Coupe 2012"),
        ("AM General Hummer SUV 2000", "AM General", "Hummer SUV 2000"),
        ("Land Rover Range Rover SUV 2012", "Land Rover", "Range Rover SUV 2012"),
        ("Rolls-Royce Ghost Sedan 2012", "Rolls-Royce", "Ghost Sedan 2012"),
        ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin", "V8 Vantage Coupe 2012"),
    ],
)
def test_составные_марки_не_разрезаются(имя: str, марка: str, остаток: str) -> None:
    """«AM General Hummer» — это марка из двух слов, а не марка AM и модель General."""
    assert split_brand(имя) == (марка, остаток)


def test_display_снимает_год_и_кузов() -> None:
    assert display_of("Audi A5 Coupe 2012") == "Audi A5"
    assert display_of("Chevrolet Silverado 1500 Extended Cab 2012") == "Chevrolet Silverado 1500"


def test_кузовные_варианты_схлопываются_в_один_класс() -> None:
    """§3.3: кузов — отдельное поле, но не отдельный класс."""
    каталог = build_catalog(ИМЕНА)
    модели = каталог["brands"]["Audi"]["models"]
    assert list(модели) == ["a5"]


def test_каждая_марка_попала_в_каталог() -> None:
    каталог = build_catalog(ИМЕНА)
    assert set(каталог["brands"]) == {"Audi", "AM General", "Land Rover", "Chevrolet"}


def test_версия_схемы_проставлена() -> None:
    assert build_catalog(ИМЕНА)["version"] == 1


def test_результат_грузится_load_catalog(tmp_path: Path) -> None:
    """Главная проверка: генератор и валидатор каталога согласны друг с другом."""
    путь = tmp_path / "stanford.yaml"
    путь.write_text(
        yaml.safe_dump(build_catalog(ИМЕНА), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    каталог = load_catalog(путь)
    assert len(каталог) == 4
    assert "am_general__hummer" in каталог
    assert class_index(каталог)["am_general__hummer"] == 0


def test_генератор_детерминирован() -> None:
    assert build_catalog(ИМЕНА) == build_catalog(list(reversed(ИМЕНА)))


def test_имя_без_модели_отвергается() -> None:
    """«Audi 2012» — марка, год и ничего больше: сопоставлять не с чем."""
    with pytest.raises(ValueError, match="пустое ядро"):
        build_catalog(["Audi 2012"])


def test_реальный_каталог_скелета_валиден() -> None:
    """Файл в репозитории обязан грузиться и не расходиться с генератором."""
    from pipeline.catalog import STANFORD_CATALOG_PATH

    каталог = load_catalog(STANFORD_CATALOG_PATH)
    assert len(каталог) >= 120
    assert len(каталог.brands()) >= 40
    индекс = class_index(каталог)
    assert len(индекс) == len(каталог)
