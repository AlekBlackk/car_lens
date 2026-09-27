"""Приём внешних кадров: чтение папок, фильтры, склейка источников."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.catalog import Catalog, class_index, load_catalog, member_index
from pipeline.crop import Detection
from pipeline.ingest.external import (
    ExternalIngestError,
    ExternalPhoto,
    ExternalResult,
    build_external_frame,
    read_mad_cars,
    read_manual,
    read_mapped,
    read_mapping,
)
from pipeline.schema import validate_frame

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


def test_mad_cars_группирует_кадры_по_машине(tmp_path: Path, каталог: Catalog) -> None:
    корень = tmp_path / "mad"
    шум(корень / "audi__a5" / "17_0.jpg", 1)
    шум(корень / "audi__a5" / "17_42.jpg", 2)
    шум(корень / "audi__a5" / "905_3.jpg", 3)

    фото = read_mad_cars(корень, каталог)

    # Три ракурса одной машины — одно объявление: иначе они разошлись бы как разные экземпляры.
    assert [(ф.model_id, ф.path.name, ф.group) for ф in фото] == [
        ("audi__a5", "17_0.jpg", "17"),
        ("audi__a5", "17_42.jpg", "17"),
        ("audi__a5", "905_3.jpg", "905"),
    ]
    assert all(ф.raw_model_string is None for ф in фото)


def test_mad_cars_без_номера_машины_в_имени_роняет_прогон(tmp_path: Path, каталог: Catalog) -> None:
    шум(tmp_path / "mad" / "audi__a5" / "фото.jpg", 1)
    with pytest.raises(ExternalIngestError, match="<car_id>_<view_id>"):
        read_mad_cars(tmp_path / "mad", каталог)


def test_неизвестная_папка_ручного_сбора_роняет_прогон(tmp_path: Path, каталог: Catalog) -> None:
    # Кириллическая «а» в «аudi» — ровно та опечатка, которую глазом не видно.
    шум(tmp_path / "manual" / "аudi__a5" / "1.jpg", 1)
    with pytest.raises(ExternalIngestError, match="не model_id каталога"):
        read_manual(tmp_path / "manual", каталог)


class ФейковыйДетектор:
    """Отдаёт заранее заданные боксы, ничего не считая."""

    def __init__(self, боксы: list[Detection]) -> None:
        self._боксы = боксы

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self._боксы)


# Бокс на почти весь кадр 240×320: кроп выходит ~240 px по короткой стороне.
ВЕСЬ_КАДР = ФейковыйДетектор([Detection(x=10.0, y=10.0, w=300.0, h=220.0, conf=0.9, cls=2)])


def принять(
    фото: list[ExternalPhoto],
    каталог: Catalog,
    кропы: Path,
    *,
    детектор: ФейковыйДетектор = ВЕСЬ_КАДР,
    эталон: tuple[int, ...] = (),
    пул: tuple[tuple[str, int], ...] = (),
) -> ExternalResult:
    return build_external_frame(
        фото, детектор, каталог, кропы, source="manual", holdout_hashes=эталон, pool=пул
    )


def одно(tmp_path: Path, зерно: int, модель: str = "audi__a5", имя: str = "") -> ExternalPhoto:
    путь = шум(tmp_path / "in" / модель / (имя or f"{зерно}.jpg"), зерно)
    return ExternalPhoto(путь, модель, путь.name, None)


def перекодировать(исходник: Path, куда: Path) -> Path:
    """Тот же массив в другом кодеке: байты другие, pHash почти тот же."""
    поле = cv2.imread(str(исходник))
    assert поле is not None
    куда.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(куда), поле)
    return куда


def test_кадр_проходит_схему_и_несёт_поля(tmp_path: Path, каталог: Catalog) -> None:
    итог = принять([одно(tmp_path, 1, "volkswagen__golf")], каталог, tmp_path / "crops")

    validate_frame(итог.frame)
    строка = итог.frame.iloc[0]
    класс = member_index(каталог)["volkswagen__golf"]
    assert строка["source"] == "manual"
    assert строка["model_id"] == "volkswagen__golf"
    assert строка["brand"] == "Volkswagen"
    assert строка["class_idx"] == class_index(каталог)[класс]
    assert строка["photo_id"].startswith("manual:")
    assert строка["listing_id"] == f"manual:{класс}:1.jpg"
    assert строка["crop_path"].startswith(f"external/manual/{класс}/")
    assert (tmp_path / "crops" / строка["crop_path"]).is_file()
    assert итог.seen == 1
    assert итог.rejected == {}


def test_нет_машины_и_мелкий_кроп(tmp_path: Path, каталог: Catalog) -> None:
    пусто = принять([одно(tmp_path, 1)], каталог, tmp_path / "c1", детектор=ФейковыйДетектор([]))
    assert пусто.rejected == {"no_vehicle": 1}

    мелкое = одно(tmp_path, 2)
    шум(мелкое.path, 2, размер=(100, 100))
    бокс = ФейковыйДетектор([Detection(x=5.0, y=5.0, w=90.0, h=90.0, conf=0.9, cls=2)])
    итог = принять([мелкое], каталог, tmp_path / "c2", детектор=бокс)
    assert итог.rejected == {"small": 1}
    assert итог.frame.empty


def test_дубли_внутри_источника(tmp_path: Path, каталог: Catalog) -> None:
    первое = одно(tmp_path, 1, имя="a.jpg")
    побайтно = tmp_path / "in" / "audi__a5" / "b.jpg"
    побайтно.write_bytes(первое.path.read_bytes())
    перекодировано = перекодировать(первое.path, tmp_path / "in" / "audi__a5" / "c.png")
    фото = [первое] + [
        ExternalPhoto(путь, "audi__a5", путь.name, None) for путь in (побайтно, перекодировано)
    ]

    итог = принять(фото, каталог, tmp_path / "crops")

    assert len(итог.frame) == 1
    assert итог.frame.iloc[0]["listing_id"].endswith(":a.jpg")
    assert итог.rejected == {"dup_source": 2}
    # Кропы отброшенных удалены: на диске ровно то, что в кадре.
    assert len(list((tmp_path / "crops").rglob("*.jpg"))) == 1


def test_близкие_кадры_разных_классов_внутри_источника_не_дубли(
    tmp_path: Path, каталог: Catalog
) -> None:
    # Разные машины на одном стенде в одном ракурсе дают близкий pHash; дубль — только в
    # своём классе, как у совпадения с пулом.
    первое = одно(tmp_path, 1)
    похожее = перекодировать(первое.path, tmp_path / "in" / "volkswagen__golf" / "b.png")
    фото = [первое, ExternalPhoto(похожее, "volkswagen__golf", похожее.name, None)]

    итог = принять(фото, каталог, tmp_path / "crops")

    assert sorted(итог.frame["model_id"]) == ["audi__a5", "volkswagen__golf"]
    assert итог.rejected == {}


def test_побайтная_копия_в_чужом_классе_остаётся_дублем(tmp_path: Path, каталог: Catalog) -> None:
    # Один и тот же файл под двумя классами — не две машины, а конфликт меток.
    первое = одно(tmp_path, 1)
    копия = tmp_path / "in" / "volkswagen__golf" / "b.jpg"
    копия.parent.mkdir(parents=True)
    копия.write_bytes(первое.path.read_bytes())
    фото = [первое, ExternalPhoto(копия, "volkswagen__golf", копия.name, None)]

    итог = принять(фото, каталог, tmp_path / "crops")

    assert list(итог.frame["model_id"]) == ["audi__a5"]
    assert итог.rejected == {"dup_source": 1}


def test_утечка_в_эталон_без_оглядки_на_класс(tmp_path: Path, каталог: Catalog) -> None:
    фото = [одно(tmp_path, 1)]
    хеш = int(принять(фото, каталог, tmp_path / "c1").frame.iloc[0]["phash"])

    итог = принять(фото, каталог, tmp_path / "c2", эталон=(хеш,))

    assert итог.rejected == {"holdout_leak": 1}
    assert not list((tmp_path / "c2").rglob("*.jpg"))


def test_совпадение_с_пулом_только_в_своём_классе(tmp_path: Path, каталог: Catalog) -> None:
    фото = [одно(tmp_path, 1)]
    хеш = int(принять(фото, каталог, tmp_path / "c1").frame.iloc[0]["phash"])

    свой = принять(фото, каталог, tmp_path / "c2", пул=(("audi__a5", хеш),))
    чужой = принять(фото, каталог, tmp_path / "c3", пул=(("grp__x5_golf", хеш),))

    assert свой.rejected == {"dup_pool": 1}
    assert len(чужой.frame) == 1


def test_cli_manual_без_таблицы_а_набор_с_таблицей(tmp_path: Path) -> None:
    from scripts.ingest_external import main

    with pytest.raises(SystemExit, match="нужна --mapping"):
        main(["--source", "autoevolution", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="у manual нет таблицы"):
        main(["--source", "manual", "--root", str(tmp_path), "--mapping", "x.csv"])
    with pytest.raises(SystemExit, match="у mad_cars нет таблицы"):
        main(["--source", "mad_cars", "--root", str(tmp_path), "--mapping", "x.csv"])
