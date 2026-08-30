"""Тесты нормализатора имён файлов Commons и правила совпадения игл моделей."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.harvest.aliases import (
    AliasesError,
    load_alias_table,
    matches_needle,
    normalize_filename,
)


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
        ("File:Renault_M%C3%A9gane_II.jpg", "renault megane ii"),
        ("File:Škoda_Octavia_III.jpg", "skoda octavia iii"),
        ("File:Citro%C3%ABn_C4_front.jpg", "citroen c4 front"),
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
        ("", "lada 2114 tuning", False),
        ("lada samara sedan", "lada samara", False),
        ("ваз 2114", "ваз 2114 елка", True),
    ],
)
def test_правило_совпадения(игла: str, имя: str, ожидание: bool) -> None:
    assert matches_needle(имя, игла) is ожидание


_ДВЕ_ИГЛЫ = 'version: 1\nclasses:\n  toyota__mark_2:\n    needles: ["Mark II", "Cressida"]\n'


def _таблица(tmp_path: Path, тело: str) -> Path:
    путь = tmp_path / "model_aliases.yaml"
    путь.write_text(тело, encoding="utf-8")
    return путь


def test_загрузка_нормализует_иглы(tmp_path: Path) -> None:
    таблица = load_alias_table(_таблица(tmp_path, _ДВЕ_ИГЛЫ))
    assert таблица["toyota__mark_2"] == ("mark ii", "cressida")


def test_чужая_версия_отвергается(tmp_path: Path) -> None:
    путь = _таблица(tmp_path, "version: 2\nclasses: {}\n")
    with pytest.raises(AliasesError, match="version"):
        load_alias_table(путь)


def test_класс_без_игл_отвергается(tmp_path: Path) -> None:
    """Класс без игл превращает поисковый путь в глухую стену — молча этого нельзя."""
    путь = _таблица(tmp_path, "version: 1\nclasses:\n  lada__granta:\n    needles: []\n")
    with pytest.raises(AliasesError, match="lada__granta"):
        load_alias_table(путь)


def test_игла_из_одной_пунктуации_отвергается(tmp_path: Path) -> None:
    тело = 'version: 1\nclasses:\n  lada__granta:\n    needles: ["---"]\n'
    with pytest.raises(AliasesError, match="lada__granta"):
        load_alias_table(_таблица(tmp_path, тело))


def test_needles_скаляром_отвергается(tmp_path: Path) -> None:
    """`needles: "Mark II"` без дефиса — законный YAML, но иглами стали бы буквы."""
    тело = 'version: 1\nclasses:\n  toyota__mark_2:\n    needles: "Mark II"\n'
    with pytest.raises(AliasesError, match="toyota__mark_2"):
        load_alias_table(_таблица(tmp_path, тело))


def test_needles_отображением_отвергается(tmp_path: Path) -> None:
    тело = 'version: 1\nclasses:\n  toyota__mark_2:\n    needles:\n      ru: "Марк 2"\n'
    with pytest.raises(AliasesError, match="toyota__mark_2"):
        load_alias_table(_таблица(tmp_path, тело))


def test_needles_числом_отвергается(tmp_path: Path) -> None:
    тело = "version: 1\nclasses:\n  lada__granta:\n    needles: 5\n"
    with pytest.raises(AliasesError, match="lada__granta"):
        load_alias_table(_таблица(tmp_path, тело))


def test_игла_null_отвергается(tmp_path: Path) -> None:
    """`str(игла)` превратил бы `null` в живую иглу `none` рядом с настоящей."""
    тело = 'version: 1\nclasses:\n  toyota__mark_2:\n    needles: [null, "Mark II"]\n'
    with pytest.raises(AliasesError, match=r"None.+toyota__mark_2"):
        load_alias_table(_таблица(tmp_path, тело))


def test_игла_числом_отвергается(tmp_path: Path) -> None:
    тело = "version: 1\nclasses:\n  lada__granta:\n    needles: [2114]\n"
    with pytest.raises(AliasesError, match=r"2114.+lada__granta"):
        load_alias_table(_таблица(tmp_path, тело))


def test_битый_yaml_отвергается(tmp_path: Path) -> None:
    """Шлюз ловит объявленный контрактом тип: трейсбек из PyYAML оборвал бы прогон."""
    with pytest.raises(AliasesError, match="YAML"):
        load_alias_table(_таблица(tmp_path, 'version: "1\n'))


@pytest.mark.parametrize("тело", ["- version: 1\n", "просто строка\n"])
def test_корень_не_отображение_отвергается(tmp_path: Path, тело: str) -> None:
    with pytest.raises(AliasesError, match="корень"):
        load_alias_table(_таблица(tmp_path, тело))


def test_classes_не_отображение_отвергается(tmp_path: Path) -> None:
    тело = "version: 1\nclasses:\n  - lada__granta\n"
    with pytest.raises(AliasesError, match="classes"):
        load_alias_table(_таблица(tmp_path, тело))


def test_тело_класса_не_отображение_отвергается(tmp_path: Path) -> None:
    тело = 'version: 1\nclasses:\n  lada__granta: "Granta"\n'
    with pytest.raises(AliasesError, match="lada__granta"):
        load_alias_table(_таблица(tmp_path, тело))
