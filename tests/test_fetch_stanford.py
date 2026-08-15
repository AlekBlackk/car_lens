"""Тесты приведения источника к контракту на диске.

Сеть здесь не участвует: проверяется только формат CSV и поведение при
недоступном источнике.
"""

import csv
from pathlib import Path

import pytest

from scripts.fetch_stanford import (
    ANNOTATION_COLUMNS,
    STANFORD_DIRNAME,
    dataset_dir,
    write_annotations,
)


def test_каталог_датасета_лежит_под_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", "/tmp/cl")
    assert dataset_dir() == Path("/tmp/cl/raw") / STANFORD_DIRNAME


def test_колонки_записаны_в_объявленном_порядке(tmp_path: Path) -> None:
    путь = tmp_path / "annotations.csv"
    write_annotations([], путь)
    с_диска = путь.read_text(encoding="utf-8").splitlines()[0]
    assert с_диска.split(",") == list(ANNOTATION_COLUMNS)


def test_строка_записывается_целиком(tmp_path: Path) -> None:
    путь = tmp_path / "annotations.csv"
    write_annotations(
        [
            {
                "image": "00001.jpg",
                "x1": 10,
                "y1": 20,
                "x2": 300,
                "y2": 200,
                "class_name": "Audi A5 Coupe 2012",
                "official_split": "train",
            }
        ],
        путь,
    )
    with путь.open(encoding="utf-8", newline="") as ф:
        строки = list(csv.DictReader(ф))
    assert строки == [
        {
            "image": "00001.jpg",
            "x1": "10",
            "y1": "20",
            "x2": "300",
            "y2": "200",
            "class_name": "Audi A5 Coupe 2012",
            "official_split": "train",
        }
    ]


def test_пустой_bbox_записывается_пустой_строкой(tmp_path: Path) -> None:
    """Зеркало может не отдать ручную разметку — это допустимо (Р-8)."""
    путь = tmp_path / "annotations.csv"
    write_annotations(
        [
            {
                "image": "00002.jpg",
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "class_name": "Audi A5 Coupe 2012",
                "official_split": "test",
            }
        ],
        путь,
    )
    with путь.open(encoding="utf-8", newline="") as ф:
        строки = list(csv.DictReader(ф))
    assert строки[0]["x1"] == ""


def test_лишнее_поле_отвергается(tmp_path: Path) -> None:
    """Контракт §4 дизайна закрытый: колонок ровно семь."""
    with pytest.raises(ValueError, match="лишн"):
        write_annotations(
            [
                {
                    "image": "00003.jpg",
                    "x1": 1,
                    "y1": 1,
                    "x2": 2,
                    "y2": 2,
                    "class_name": "Audi A5 Coupe 2012",
                    "official_split": "train",
                    "color": "red",
                }
            ],
            tmp_path / "annotations.csv",
        )


def test_инструкция_называет_целевой_каталог() -> None:
    from scripts.fetch_stanford import ИНСТРУКЦИЯ

    assert "images/" in ИНСТРУКЦИЯ
    assert "annotations.csv" in ИНСТРУКЦИЯ
