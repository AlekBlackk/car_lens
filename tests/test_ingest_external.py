"""Приём внешних кадров: чтение папок, фильтры, склейка источников."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.catalog import Catalog, load_catalog
from pipeline.ingest.external import (
    ExternalIngestError,
    read_manual,
    read_mapped,
    read_mapping,
)

YAML = """
version: 1
brands:
  Audi:
    models:
      a5: {display: Audi A5}
  BMW:
    models:
      x5: {display: BMW X5}
  Volkswagen:
    models:
      golf: {display: Volkswagen Golf}
groups:
  grp__x5_golf:
    members: [bmw__x5, volkswagen__golf]
    display: "BMW X5 / Volkswagen Golf"
"""


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    путь = tmp_path / "catalog.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return load_catalog(путь)


def шум(путь: Path, зерно: int, размер: tuple[int, int] = (240, 320)) -> Path:
    """Случайная картинка: у разных зёрен pHash расходится далеко за порог дедупа."""
    путь.parent.mkdir(parents=True, exist_ok=True)
    поле = np.random.default_rng(зерно).integers(0, 256, (*размер, 3), dtype=np.uint8)
    assert cv2.imwrite(str(путь), поле)
    return путь


def таблица(tmp_path: Path, текст: str) -> Path:
    путь = tmp_path / "mapping.csv"
    путь.write_text(текст, encoding="utf-8")
    return путь


def test_таблица_соответствия_читается(tmp_path: Path, каталог: Catalog) -> None:
    путь = таблица(tmp_path, "dir_path,model_id,title\nAUDI/A5,audi__a5,A5 2016\n")
    assert read_mapping(путь, каталог) == {"AUDI/A5": "audi__a5"}


@pytest.mark.parametrize(
    ("текст", "фрагмент"),
    [
        ("dir_path,title\nAUDI/A5,x\n", "нет колонок model_id"),
        ("dir_path,model_id\nAUDI/A5,audi__a6\n", ":2: model_id 'audi__a6'"),
        ("dir_path,model_id\n,audi__a5\n", ":2: пустое"),
        ("dir_path,model_id\nA,audi__a5\nA,bmw__x5\n", ":3: dir_path 'A' повторяется"),
    ],
)
def test_битая_таблица_роняет_прогон(
    tmp_path: Path, каталог: Catalog, текст: str, фрагмент: str
) -> None:
    with pytest.raises(ExternalIngestError, match=фрагмент):
        read_mapping(таблица(tmp_path, текст), каталог)


def test_набор_читается_по_таблице(tmp_path: Path) -> None:
    корень = tmp_path / "набор"
    шум(корень / "AUDI" / "A5" / "b.jpg", 1)
    шум(корень / "AUDI" / "A5" / "a.png", 2)
    (корень / "AUDI" / "A5" / "readme.txt").write_text("не картинка", encoding="utf-8")
    шум(корень / "BMW" / "M3" / "c.jpg", 3)  # папки нет в таблице — не берётся

    фото = read_mapped(корень, {"AUDI/A5": "audi__a5"})

    assert [ф.path.name for ф in фото] == ["a.png", "b.jpg"]
    assert {ф.model_id for ф in фото} == {"audi__a5"}
    assert {ф.group for ф in фото} == {"AUDI/A5"}
    assert {ф.raw_model_string for ф in фото} == {"AUDI/A5"}


def test_папка_из_таблицы_обязана_быть(tmp_path: Path) -> None:
    (tmp_path / "набор").mkdir()
    with pytest.raises(ExternalIngestError, match="папки из таблицы"):
        read_mapped(tmp_path / "набор", {"AUDI/A5": "audi__a5"})


def test_ручной_сбор_читается_по_model_id(tmp_path: Path, каталог: Catalog) -> None:
    корень = tmp_path / "manual"
    шум(корень / "volkswagen__golf" / "1.webp", 1)
    шум(корень / "audi__a5" / "2.jpg", 2)

    фото = read_manual(корень, каталог)

    assert [(ф.model_id, ф.path.name) for ф in фото] == [
        ("audi__a5", "2.jpg"),
        ("volkswagen__golf", "1.webp"),
    ]
    # У ручного сбора каждое фото — своё объявление: экземпляры разные по правилу сбора.
    assert [ф.group for ф in фото] == ["2.jpg", "1.webp"]
    assert all(ф.raw_model_string is None for ф in фото)


def test_неизвестная_папка_ручного_сбора_роняет_прогон(tmp_path: Path, каталог: Catalog) -> None:
    # Кириллическая «а» в «аudi» — ровно та опечатка, которую глазом не видно.
    шум(tmp_path / "manual" / "аudi__a5" / "1.jpg", 1)
    with pytest.raises(ExternalIngestError, match="не model_id каталога"):
        read_manual(tmp_path / "manual", каталог)
