"""Тесты нормализатора имён файлов Commons и правила совпадения игл моделей."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.harvest.aliases import (
    AliasesError,
    AliasVerdict,
    alias_verdict,
    broad_needles,
    load_alias_table,
    matches_needle,
    needle_conflicts,
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


def test_загрузка_дедуплицирует_иглы_совпавшие_после_нормализации(tmp_path: Path) -> None:
    """Две записи, отличающиеся только регистром или диакритикой, — законная ручная правка.

    Без дедупликации одна и та же тройка попадает в отчёты `needle_conflicts` и
    `broad_needles` дважды. Порядок первых появлений сохраняется: он идёт человеку.
    """
    тело = (
        "version: 1\nclasses:\n  citroen__c4:\n"
        '    needles: ["Citroen C4", "C4 Picasso", "Citroën C4", "CITROEN C4"]\n'
    )
    таблица = load_alias_table(_таблица(tmp_path, тело))
    assert таблица["citroen__c4"] == ("citroen c4", "c4 picasso")


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


_ТАБЛИЦА = {
    "lexus__lx": ("lx", "lx570"),
    "toyota__land_cruiser": ("land cruiser", "fj62v"),
    "lada__samara": ("2108", "2109"),
    "lada__samara_2": ("2113", "2114", "2115"),
}


def test_вердикт_свой_класс() -> None:
    вердикт = alias_verdict("2019 lexus lx570 front", "lexus__lx", _ТАБЛИЦА)
    assert вердикт.kind == "own"
    assert вердикт.foreign == ()


def test_вердикт_чужой_класс_называет_его_поимённо() -> None:
    """Ровно этот случай — 21 строка `lexus__lx`, названная Land Cruiser."""
    вердикт = alias_verdict("1982 toyota land cruiser bj41", "lexus__lx", _ТАБЛИЦА)
    assert вердикт.kind == "foreign"
    assert вердикт.foreign == ("toyota__land_cruiser",)


def test_вердикт_ничего_не_совпало() -> None:
    вердикт = alias_verdict("img 0412", "lexus__lx", _ТАБЛИЦА)
    assert вердикт.kind == "unknown"
    assert вердикт.foreign == ()


def test_своя_игла_побеждает_чужую() -> None:
    """Имя, называющее оба класса, остаётся своим: чужое имя рядом со своим — контекст."""
    вердикт = alias_verdict("lexus lx570 vs land cruiser", "lexus__lx", _ТАБЛИЦА)
    assert вердикт.kind == "own"


def test_класс_вне_таблицы_это_ошибка() -> None:
    with pytest.raises(KeyError):
        alias_verdict("что угодно", "нет__такого", _ТАБЛИЦА)


def test_конфликтов_в_чистой_таблице_нет() -> None:
    assert needle_conflicts(_ТАБЛИЦА) == []
    assert broad_needles(_ТАБЛИЦА) == []


def test_одинаковая_игла_у_двух_классов_это_конфликт() -> None:
    """Взаимно неразличимые иглы делают вердикт `foreign` бессмысленным в обе стороны."""
    таблица = {"chevrolet__niva": ("niva",), "lada__niva_2121": ("niva",)}
    assert needle_conflicts(таблица) == [("chevrolet__niva", "lada__niva_2121", "niva")]


def test_широкая_игла_соседа_не_блокирует_но_видна() -> None:
    """Голая «niva» накрывает «niva 2131»: не симметрия, а повод человеку взглянуть."""
    таблица = {"chevrolet__niva": ("niva",), "lada__niva_2131": ("niva 2131", "21310")}
    assert needle_conflicts(таблица) == []
    assert broad_needles(таблица) == [("chevrolet__niva", "niva", "lada__niva_2131")]


def test_более_конкретная_игла_соседа_законна() -> None:
    """ASX в США зовут Outlander Sport. Запрещать это значило бы терять живые кадры."""
    таблица = {
        "mitsubishi__asx": ("asx", "outlander sport"),
        "mitsubishi__outlander": ("outlander",),
    }
    assert needle_conflicts(таблица) == []
    assert (
        alias_verdict("2012 mitsubishi outlander sport es", "mitsubishi__asx", таблица).kind
        == "own"
    )
    чужой = alias_verdict("mitsubishi outlander phev", "mitsubishi__asx", таблица)
    assert чужой == AliasVerdict("foreign", ("mitsubishi__outlander",), "outlander")


def test_вердикт_foreign_полон_и_упорядочен() -> None:
    """Неполный `foreign` дал бы автодроп задачи 6 на неоднозначной строке.

    Автоматический дроп разрешён только при `foreign` ровно с одним чужим классом, поэтому
    потерянный второй класс превратил бы неоднозначное имя в основание для дропа. Порядок
    кортежа лексикографический, а не порядок правок в YAML: он идёт человеку в отчёт.
    """
    таблица = {
        "toyota__land_cruiser": ("land cruiser",),
        "lada__niva_2121": ("niva 2121",),
        "lexus__lx": ("lx570",),
    }
    вердикт = alias_verdict("lada niva 2121 vs toyota land cruiser", "lexus__lx", таблица)
    assert вердикт.kind == "foreign"
    assert вердикт.foreign == ("lada__niva_2121", "toyota__land_cruiser")


def test_три_класса_с_одной_иглой_дают_все_три_пары() -> None:
    """Выдача полная и упорядочена лексикографически по паре классов."""
    таблица = {
        "lada__niva_2121": ("niva",),
        "chevrolet__niva": ("niva",),
        "vaz__niva_travel": ("niva",),
    }
    assert needle_conflicts(таблица) == [
        ("chevrolet__niva", "lada__niva_2121", "niva"),
        ("chevrolet__niva", "vaz__niva_travel", "niva"),
        ("lada__niva_2121", "vaz__niva_travel", "niva"),
    ]


def test_вердикт_пустого_имени_это_unknown() -> None:
    """Пустое имя не ошибка, а `unknown`: свидетельствовать ему нечем."""
    assert alias_verdict("", "lexus__lx", _ТАБЛИЦА) == AliasVerdict("unknown", ())


def test_таблица_из_одного_класса_чиста() -> None:
    """Сравнивать не с кем: ни конфликтов, ни широких игл."""
    таблица = {"chevrolet__niva": ("niva",)}
    assert needle_conflicts(таблица) == []
    assert broad_needles(таблица) == []
