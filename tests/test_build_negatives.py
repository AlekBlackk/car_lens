"""Источники чужих машин для `scripts/build_negatives.py`."""

from __future__ import annotations

from pathlib import Path

from scripts.build_negatives import autoevolution_groups, stanford_groups


def test_autoevolution_берёт_только_неприсоединённые_папки(tmp_path: Path) -> None:
    for папка in ("AUDI_A4_19942001", "ALFA_ROMEO_Giulia_2016"):
        (tmp_path / "ae" / папка).mkdir(parents=True)
        (tmp_path / "ae" / папка / "1.jpg").write_bytes(b"x")
    (tmp_path / "ae" / "ALFA_ROMEO_Giulia_2016" / "readme.txt").write_text("x")
    таблица = tmp_path / "map.csv"
    таблица.write_text(
        "dir_path,model_id,brand,model,years,images\n"
        "AUDI_A4_19942001,audi__a4,AUDI,AUDI A4,1994-2024,1\n",
        encoding="utf-8",
    )

    группы = autoevolution_groups(tmp_path / "ae", таблица)

    assert [г.name for г in группы] == ["ALFA_ROMEO_Giulia_2016"]
    assert группы[0].origin == "autoevolution/ALFA_ROMEO_Giulia_2016"
    assert [п.name for п in группы[0].paths] == ["1.jpg"]


def test_stanford_группирует_по_классу(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    for имя in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / "images" / имя).write_bytes(b"x")
    (tmp_path / "annotations.csv").write_text(
        "image,x1,y1,x2,y2,class_name,official_split\n"
        "a.jpg,1,1,5,5,Bugatti Veyron 16.4 Coupe 2009,train\n"
        "b.jpg,1,1,5,5,Bugatti Veyron 16.4 Coupe 2009,test\n"
        "c.jpg,1,1,5,5,Hyundai Accent Sedan 2012,train\n",
        encoding="utf-8",
    )

    группы = stanford_groups(tmp_path)

    assert {г.name: len(г.paths) for г in группы} == {
        "Bugatti Veyron 16.4 Coupe 2009": 2,
        "Hyundai Accent Sedan 2012": 1,
    }
