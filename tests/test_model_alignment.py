# tests/test_model_alignment.py
"""Триаж манифеста: кто уходит автоматом, кто идёт человеку, кого не трогают вовсе."""

from __future__ import annotations

import pytest

from pipeline.harvest.aliases import AliasesError
from pipeline.harvest.alignment import auto_drops, review_queue, triage
from pipeline.harvest.manifest import HarvestRow

_ТАБЛИЦА = {
    "lexus__lx": ("lx",),
    "toyota__land_cruiser": ("land cruiser",),
    "lada__samara": ("2108", "2109"),
    "lada__samara_2": ("2113", "2114"),
}


def _строка(файл: str, класс: str, имя_файла: str, базис: str) -> HarvestRow:
    основа = файл.removesuffix(".jpg")
    return HarvestRow(
        file=файл,
        class_id=класс,
        source="commons",
        source_item_id=основа,
        page_url=f"https://commons.wikimedia.org/wiki/File:{имя_файла}",
        file_url="https://upload.wikimedia.org/x.jpg",
        author="автор",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis=базис,
        sha256=основа,
        phash=1,
        clip_rank=1,
        clip_margin=0.2,
        plates=0,
        faces=0,
        bbox_x=0.0,
        bbox_y=0.0,
        bbox_w=100.0,
        bbox_h=80.0,
        bbox_area_frac=0.5,
        det_conf=0.9,
        audit="ok",
    )


def test_категории_триаж_не_трогает() -> None:
    строки = [_строка("a.jpg", "lexus__lx", "IMG_0412.jpg", "category:Lexus LX")]
    assert triage(строки, _ТАБЛИЦА) == []


def test_чужой_класс_ровно_один_уходит_в_автодроп() -> None:
    строки = [_строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser_BJ41.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert [в.kind for в in вердикты] == ["foreign"]
    assert [в.file for в in auto_drops(вердикты, _ТАБЛИЦА)] == ["a.jpg"]
    assert review_queue(вердикты, _ТАБЛИЦА) == []


def test_молчащее_имя_уходит_человеку() -> None:
    строки = [_строка("b.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert [в.kind for в in вердикты] == ["unknown"]
    assert auto_drops(вердикты, _ТАБЛИЦА) == []
    assert [в.file for в in review_queue(вердикты, _ТАБЛИЦА)] == ["b.jpg"]


def test_два_чужих_класса_уходят_человеку() -> None:
    """Иглы спорят между собой — вердикт за человека выдавать нельзя."""
    строки = [_строка("c.jpg", "lexus__lx", "Lada_2108_and_2113_meet.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert вердикты[0].foreign == ("lada__samara", "lada__samara_2")
    assert auto_drops(вердикты, _ТАБЛИЦА) == []
    assert [в.file for в in review_queue(вердикты, _ТАБЛИЦА)] == ["c.jpg"]


def test_своё_имя_не_попадает_ни_в_один_список() -> None:
    строки = [_строка("d.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert [в.kind for в in вердикты] == ["own"]
    assert auto_drops(вердикты, _ТАБЛИЦА) == []
    assert review_queue(вердикты, _ТАБЛИЦА) == []


def test_пустой_манифест_даёт_пустой_вердикт() -> None:
    assert triage([], _ТАБЛИЦА) == []


def test_категория_с_явно_чужим_именем_всё_равно_не_трогается() -> None:
    """Категорию размечали люди: имя Land Cruiser в классе LX её не отменяет."""
    строки = [
        _строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser_BJ41.jpg", "category:Lexus LX")
    ]
    assert triage(строки, _ТАБЛИЦА) == []


def test_класс_вне_таблицы_игл_останавливает_триаж_с_именем_файла() -> None:
    """Молчаливый `unknown` означал бы вердикт, посчитанный не по тому классу."""
    строки = [_строка("a.jpg", "нет__игл", "2019_Lexus_LX570.jpg", "search:Нет игл")]
    with pytest.raises(AliasesError) as авария:
        triage(строки, _ТАБЛИЦА)
    assert "a.jpg" in str(авария.value)
    assert "нет__игл" in str(авария.value)


def test_порядок_вердиктов_повторяет_порядок_манифеста() -> None:
    """Задачи 7 и 8 считают числа и пишут CSV по этому порядку."""
    строки = [
        _строка("c.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX"),
        _строка("a.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX"),
        _строка("b.jpg", "lexus__lx", "IMG_0413.jpg", "category:Lexus LX"),
        _строка("d.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX"),
    ]
    assert [в.file for в in triage(строки, _ТАБЛИЦА)] == ["c.jpg", "a.jpg", "d.jpg"]


_ТАБЛИЦА_НИВЫ = {
    "lada__niva_2121": ("2121", "niva", "vaz 2121"),
    "lada__niva_2131": ("2131", "niva 2131", "vaz 2131"),
}


def test_широкая_игла_соседа_не_даёт_автомату_права() -> None:
    """`Lada_Niva_2.jpg` в классе 2131: голая `niva` соседа шире всех своих игл."""
    строки = [_строка("a.jpg", "lada__niva_2131", "Lada_Niva_2.jpg", "search:Lada Niva 2131")]
    вердикты = triage(строки, _ТАБЛИЦА_НИВЫ)
    assert (вердикты[0].kind, вердикты[0].foreign, вердикты[0].needle) == (
        "foreign",
        ("lada__niva_2121",),
        "niva",
    )
    assert auto_drops(вердикты, _ТАБЛИЦА_НИВЫ) == []
    assert [в.file for в in review_queue(вердикты, _ТАБЛИЦА_НИВЫ)] == ["a.jpg"]


def test_узкая_чужая_игла_автомат_не_останавливает() -> None:
    """Свидетельство однозначно: имя называет кузов соседа, а не общее слово."""
    строки = [_строка("b.jpg", "lada__niva_2131", "VAZ_2121_Niva.jpg", "search:Lada Niva 2131")]
    вердикты = triage(строки, _ТАБЛИЦА_НИВЫ)
    assert вердикты[0].needle == "2121"
    assert [в.file for в in auto_drops(вердикты, _ТАБЛИЦА_НИВЫ)] == ["b.jpg"]
    assert review_queue(вердикты, _ТАБЛИЦА_НИВЫ) == []


def test_три_списка_дают_полный_охват() -> None:
    """`own` плюс автомат плюс очередь — все вердикты: иначе `G_AL2` нечем закрыть."""
    строки = [
        _строка("a.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX"),
        _строка("b.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX"),
        _строка("c.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX"),
        _строка("d.jpg", "lexus__lx", "Lada_2108_and_2113_meet.jpg", "search:Lexus LX"),
    ]
    вердикты = triage(строки, _ТАБЛИЦА)
    свои = [в for в in вердикты if в.kind == "own"]
    покрытие = (
        len(свои) + len(auto_drops(вердикты, _ТАБЛИЦА)) + len(review_queue(вердикты, _ТАБЛИЦА))
    )
    assert покрытие == len(вердикты) == 4
