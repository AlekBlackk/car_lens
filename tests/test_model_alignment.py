# tests/test_model_alignment.py
"""Триаж манифеста: кто уходит автоматом, кто идёт человеку, кого не трогают вовсе."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, AliasesError
from pipeline.harvest.alignment import (
    auto_drops,
    check_alias_gate,
    render_alignment_report,
    review_queue,
    triage,
)
from pipeline.harvest.manifest import HarvestRow, write_manifest
from scripts.audit_model_alignment import main as триаж_main
from scripts.audit_model_alignment import проверить_категории

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


_КЛАССЫ = ("lexus__lx", "toyota__land_cruiser", "lada__samara", "lada__samara_2")


def test_гейт_проходит_когда_всё_помеченное_выкинуто() -> None:
    строки = [
        _строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX"),
        _строка("b.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX"),
    ]
    вердикты = {"a.jpg": "drop:wrong_model"}
    гейт = check_alias_gate(строки, _ТАБЛИЦА, вердикты, вердикты, _КЛАССЫ)
    assert гейт.passed
    assert гейт.blocking_failures == ()


def test_строка_без_вердикта_роняет_g_al2() -> None:
    """«Ноль найденной грязи» одинаково верно и когда её не смотрели."""
    строки = [_строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX")]
    гейт = check_alias_gate(строки, _ТАБЛИЦА, {}, {}, _КЛАССЫ)
    assert not гейт.passed
    assert any("G_AL2" in причина for причина in гейт.blocking_failures)


def test_строка_без_алиас_вердикта_роняет_g_al2() -> None:
    """Вердикт H3/privacy в all_verdicts не закрывает G_AL2: модель не смотрели."""
    строки = [_строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX")]
    все_вердикты = {"a.jpg": "ok"}
    гейт = check_alias_gate(строки, _ТАБЛИЦА, все_вердикты, {}, _КЛАССЫ)
    assert not гейт.passed
    assert any("G_AL2" in причина for причина in гейт.blocking_failures)


def test_вердикт_ok_на_молчащем_имени_не_закрывает_g_al3() -> None:
    """Кадр оставили — значит имя надо объяснить иглой, а не вердиктом."""
    строки = [_строка("a.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX")]
    вердикты = {"a.jpg": "ok"}
    гейт = check_alias_gate(строки, _ТАБЛИЦА, вердикты, вердикты, _КЛАССЫ)
    assert not гейт.passed
    assert any("G_AL3" in причина for причина in гейт.blocking_failures)


def test_неполная_таблица_роняет_g_al1_и_рендер_отчёта_не_падает() -> None:
    """Строки непустые намеренно: на пустых гейт не дошёл бы до `triage` и дефект бы уцелел."""
    строки = [_строка("a.jpg", "нет__игл", "2019_Lexus_LX570.jpg", "search:Нет игл")]
    все_классы = (*_КЛАССЫ, "нет__игл")
    гейт = check_alias_gate(строки, _ТАБЛИЦА, {}, {}, все_классы)
    assert not гейт.passed
    assert any("G_AL1" in причина for причина in гейт.blocking_failures)
    assert гейт.measured == ()

    отчёт = render_alignment_report(строки, _ТАБЛИЦА, {}, гейт, "abc1234", "2026-08-31")
    assert "НЕ ПРОЙДЕН" in отчёт
    assert "G_AL1" in отчёт
    assert "Нет." in отчёт
    assert "не посчитано, таблица игл неполна" in отчёт
    assert "помечено правилом игл: 0" not in отчёт, "ноль читался бы как «выборка чиста»"


def test_конфликт_игл_роняет_g_al1() -> None:
    """Взаимно неразличимые иглы у двух классов блокируют G_AL1."""
    конфликтная_таблица = {
        "chevrolet__niva": ("niva",),
        "lada__niva_2121": ("niva",),
    }
    классы = ("chevrolet__niva", "lada__niva_2121")
    гейт = check_alias_gate([], конфликтная_таблица, {}, {}, классы)
    assert not гейт.passed
    assert any("G_AL1: взаимно неразличимых игл" in причина for причина in гейт.blocking_failures)


def test_арифметика_автомата_и_широких_игл_пинится() -> None:
    """Широкие иглы не идут в автодроп, но считаются в broad_needles и measured."""
    строки = [
        _строка("a.jpg", "lada__niva_2131", "VAZ_2121_Niva.jpg", "search:Lada Niva 2131"),
        _строка("b.jpg", "lada__niva_2131", "Lada_Niva_2.jpg", "search:Lada Niva 2131"),
    ]
    вердикты = triage(строки, _ТАБЛИЦА_НИВЫ)
    авто = auto_drops(вердикты, _ТАБЛИЦА_НИВЫ)
    очередь = review_queue(вердикты, _ТАБЛИЦА_НИВЫ)
    assert [в.file for в in авто] == ["a.jpg"]
    assert [в.file for в in очередь] == ["b.jpg"]

    вердикты_карта = {"a.jpg": "drop:wrong_model", "b.jpg": "drop:wrong_model"}
    гейт = check_alias_gate(
        строки, _ТАБЛИЦА_НИВЫ, вердикты_карта, вердикты_карта, tuple(_ТАБЛИЦА_НИВЫ)
    )
    assert гейт.passed
    assert "помечено: 2, из них автоматом 1, человеку 1" in гейт.measured[1]
    assert "широких игл (не блокирует): 1 (пар: 1)" in гейт.measured[2]


def test_отчёт_несёт_числа_и_вердикт() -> None:
    строки = [
        _строка(
            "a.jpg", "lada__niva_2131", "1982_Toyota_Land_Cruiser.jpg", "search:Lada Niva 2131"
        ),
        _строка("b.jpg", "lada__niva_2131", "Lada_Niva_2.jpg", "search:Lada Niva 2131"),
    ]
    вердикты = {"a.jpg": "drop:wrong_model", "b.jpg": "drop:wrong_model"}
    гейт = check_alias_gate(строки, _ТАБЛИЦА_НИВЫ, вердикты, вердикты, tuple(_ТАБЛИЦА_НИВЫ))
    отчёт = render_alignment_report(строки, _ТАБЛИЦА_НИВЫ, вердикты, гейт, "abc1234", "2026-08-31")
    assert "abc1234" in отчёт
    assert "lada__niva_2131" in отчёт
    assert "пройден" in отчёт
    assert "- строк в манифесте до чистки: 2" in отчёт
    assert "- строк поискового базиса: 2" in отчёт
    assert "- помечено правилом игл: 2" in отчёт
    assert "- строк осталось после применения вердиктов: 0" in отчёт
    assert "- `lada__niva_2121`: «niva» накрывает класс `lada__niva_2131`" in отчёт
    assert "- широких игл (не блокирует): 1 (пар: 1)" in отчёт


def test_cli_пишет_автовердикты_очередь_и_отчёт(tmp_path: Path) -> None:
    манифест = tmp_path / "harvest_manifest.csv"
    write_manifest(
        манифест,
        [
            _строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX"),
            _строка("b.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX"),
            _строка("c.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX"),
        ],
    )
    таблица = tmp_path / "model_aliases.yaml"
    таблица.write_text(
        "version: 1\nclasses:\n"
        "  lexus__lx:\n    needles: [lx]\n"
        "  toyota__land_cruiser:\n    needles: [land cruiser]\n",
        encoding="utf-8",
    )
    вердикты_файл = tmp_path / "verdicts.alias.csv"
    очередь = tmp_path / "alias_queue.txt"
    отчёт = tmp_path / "model-alignment.md"

    код = триаж_main(
        [
            "--manifest",
            str(манифест),
            "--aliases",
            str(таблица),
            "--out-verdicts",
            str(вердикты_файл),
            "--queue",
            str(очередь),
            "--report",
            str(отчёт),
        ]
    )

    assert код == 1, "без вердиктов человека G_AL2 обязан быть красным"
    assert вердикты_файл.read_text(encoding="utf-8").splitlines() == [
        "file,verdict",
        "a.jpg,drop:wrong_model",
    ]
    assert очередь.read_text(encoding="utf-8").split() == ["b.jpg"]
    assert "НЕ ПРОЙДЕН" in отчёт.read_text(encoding="utf-8")


def test_cli_успешный_прогон_с_вердиктами(tmp_path: Path) -> None:
    манифест = tmp_path / "harvest_manifest.csv"
    write_manifest(
        манифест,
        [
            _строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser.jpg", "search:Lexus LX"),
            _строка("b.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX"),
            _строка("c.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX"),
        ],
    )
    вердикты_ревью = tmp_path / "verdicts.alias-review.csv"
    вердикты_ревью.write_text("file,verdict\nb.jpg,drop:wrong_model\n", encoding="utf-8")
    вердикты_файл = tmp_path / "verdicts.alias.csv"
    очередь = tmp_path / "alias_queue.txt"
    отчёт = tmp_path / "model-alignment.md"

    код = триаж_main(
        [
            "--manifest",
            str(манифест),
            "--aliases",
            str(DEFAULT_ALIASES_PATH),
            "--out-verdicts",
            str(вердикты_файл),
            "--queue",
            str(очередь),
            "--report",
            str(отчёт),
            "--verdicts",
            str(вердикты_файл),
            str(вердикты_ревью),
            "--alias-verdicts",
            str(вердикты_ревью),
        ]
    )

    assert код == 0
    assert "пройден" in отчёт.read_text(encoding="utf-8")


def test_check_categories_требует_contact(tmp_path: Path) -> None:
    манифест = tmp_path / "harvest_manifest.csv"
    write_manifest(манифест, [])
    with pytest.raises(SystemExit):
        триаж_main(["--manifest", str(манифест), "--check-categories"])


def test_проверить_категории_пропускает_классы_с_категориями_и_опрашивает_остальные(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    запросы_файл = tmp_path / "harvest_queries.yaml"
    запросы_файл.write_text(
        "version: 1\nclasses:\n"
        "  lexus__lx:\n"
        "    terms: [lx]\n"
        "    categories: [Lexus LX]\n"
        "  toyota__land_cruiser:\n"
        "    terms: [land cruiser]\n"
        "    categories: [Toyota Land Cruiser]\n",
        encoding="utf-8",
    )
    строки = [_строка("a.jpg", "lexus__lx", "IMG_0412.jpg", "category:Lexus LX")]
    опрошенные: list[str] = []

    def mock_populated(transport: object, category: str, sleep: object = None) -> bool:
        опрошенные.append(category)
        return category == "Toyota Land Cruiser"

    monkeypatch.setattr("pipeline.harvest.commons.category_is_populated", mock_populated)
    паузы: list[float] = []
    итог = проверить_категории(
        строки,
        contact="user@test.org",
        queries_path=запросы_файл,
        sleep=паузы.append,
    )
    assert итог == {"toyota__land_cruiser": True}
    assert опрошенные == ["Toyota Land Cruiser"]
    assert len(паузы) == 1
