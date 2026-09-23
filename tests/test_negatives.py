"""Негативы для подбора порогов: отбор чужих машин и сборка кадра (спека calibration-ru §4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.negatives import (
    Exclusions,
    NegativesError,
    OffCatalogGroup,
    catalog_needle_hits,
    read_exclusions,
    sample_candidates,
    select_off_catalog,
)

ТАБЛИЦА = {
    "hyundai__solaris": ("solaris",),
    "renault__logan": ("logan",),
    "audi__a4": ("a4",),
}


def _группа(имя: str, файлов: int = 5) -> OffCatalogGroup:
    return OffCatalogGroup(
        origin=f"autoevolution/{имя}",
        name=имя,
        paths=tuple(Path(f"/x/{имя}/{н}.jpg") for н in range(файлов)),
    )


def test_игла_находит_модель_каталога_в_имени_папки() -> None:
    assert catalog_needle_hits("HYUNDAI_Solaris_20172020", ТАБЛИЦА) == ["hyundai__solaris"]
    assert catalog_needle_hits("DACIA_Logan_MCV_2013", ТАБЛИЦА) == ["renault__logan"]


def test_чужая_модель_без_совпадений() -> None:
    assert catalog_needle_hits("ALFA_ROMEO_Giulia_2016", ТАБЛИЦА) == []


def test_цифровая_игла_не_цепляет_продолжение() -> None:
    # «a4» кончается цифрой — продолжение «a4l» ей не совпадение (правило matches_needle).
    assert catalog_needle_hits("AUDI_A4L_20122016", ТАБЛИЦА) == []


def test_отбор_делит_группы_и_называет_причину() -> None:
    чужая = _группа("ALFA_ROMEO_Giulia_2016")
    своя = _группа("HYUNDAI_Solaris_2017")
    вручную = _группа("HYUNDAI_Accent_2011")
    исключения = Exclusions(names=frozenset({"HYUNDAI_Accent_2011"}), photos=frozenset())

    оставлены, отброшены = select_off_catalog([чужая, своя, вручную], ТАБЛИЦА, исключения)

    assert оставлены == [чужая]
    assert [(г.name, причина) for г, причина in отброшены] == [
        ("HYUNDAI_Solaris_2017", "needle:hyundai__solaris"),
        ("HYUNDAI_Accent_2011", "excluded"),
    ]


def test_выборка_детерминирована_и_держит_потолок_на_группу() -> None:
    группы = [_группа("A", 10), _группа("B", 2)]

    первая = sample_candidates(группы, per_group=3, seed=7)
    вторая = sample_candidates(группы, per_group=3, seed=7)

    assert первая == вторая
    assert sum(1 for г, _ in первая if г.name == "A") == 3
    assert sum(1 for г, _ in первая if г.name == "B") == 2


def test_исключения_читаются_по_видам(tmp_path: Path) -> None:
    путь = tmp_path / "exclude.csv"
    путь.write_text(
        "kind,value,reason\n"
        "name,HYUNDAI_Accent_2011,Accent RB = Solaris\n"
        "photo,negatives:abc:0,на кропе Веста\n",
        encoding="utf-8",
    )

    исключения = read_exclusions(путь)

    assert исключения.names == frozenset({"HYUNDAI_Accent_2011"})
    assert исключения.photos == frozenset({"negatives:abc:0"})


def test_исключение_без_причины_роняет(tmp_path: Path) -> None:
    путь = tmp_path / "exclude.csv"
    путь.write_text("kind,value,reason\nname,X,\n", encoding="utf-8")

    with pytest.raises(NegativesError, match="причин"):
        read_exclusions(путь)


def test_неизвестный_вид_исключения_роняет(tmp_path: Path) -> None:
    путь = tmp_path / "exclude.csv"
    путь.write_text("kind,value,reason\ndir,X,опечатка\n", encoding="utf-8")

    with pytest.raises(NegativesError, match="dir"):
        read_exclusions(путь)


def test_нет_файла_исключений_значит_пусто(tmp_path: Path) -> None:
    assert read_exclusions(tmp_path / "нет.csv") == Exclusions(frozenset(), frozenset())
