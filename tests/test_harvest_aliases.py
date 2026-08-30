import pytest

from pipeline.harvest.aliases import matches_needle, normalize_filename


@pytest.mark.parametrize(
    ("сырое", "ожидание"),
    [
        ("File:2019_Lada_Granta_SW_219410_grey_rear.jpg", "2019 lada granta sw 219410 grey rear"),
        (
            "https://commons.wikimedia.org/wiki/File:Moscow%2C_Lada_Granta_black.jpg",
            "moscow lada granta black",
        ),
        ("File:ВАЗ-2114_ёлка.JPEG", "ваз 2114 елка"),
        ("File:Chery_A15__front..png", "chery a15 front"),
    ],
)
def test_нормализация_имени(сырое: str, ожидание: str) -> None:
    assert normalize_filename(сырое) == ожидание


@pytest.mark.parametrize(
    ("игла", "имя", "ожидание"),
    [
        ("fx", "infiniti fx35 s50 washington dc", True),
        ("lx", "lexus lx570 japan", True),
        ("lx", "bmw 320i deluxe", False),
        ("469", "uaz 4690 concept", False),
        ("469", "uaz 469 moscow", True),
        ("on do", "datsun on do front", True),
        ("on do", "datsun mi do front", False),
        ("2114", "lada 21145 tuning", False),
        ("samara 2", "lada samara baltic blue", False),
        ("cressida", "1984 toyota cressida mx73 glx", True),
    ],
)
def test_правило_совпадения(игла: str, имя: str, ожидание: bool) -> None:
    assert matches_needle(имя, игла) is ожидание
