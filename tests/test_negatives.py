"""Негативы для подбора порогов: отбор чужих машин и сборка кадра (спека calibration-ru §4)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.crop import Detection
from pipeline.negatives import (
    Exclusions,
    NegativeCandidate,
    NegativesError,
    OffCatalogGroup,
    build_negative_frame,
    catalog_needle_hits,
    neg_kind_of,
    read_exclusions,
    reject_candidates,
    sample_candidates,
    select_off_catalog,
)
from pipeline.schema import read_split, write_split

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


def _фото(путь: Path, цвет: int) -> Path:
    путь.parent.mkdir(parents=True, exist_ok=True)
    картинка = np.full((400, 600, 3), цвет, dtype=np.uint8)
    картинка[100:300, 100:500] = 255 - цвет
    cv2.imwrite(str(путь), картинка)
    return путь


class _ДетекторПоПути:
    """Боксы по порядку вызовов: `build_negative_frame` читает файлы сам."""

    def __init__(self, по_порядку: list[list[Detection]]) -> None:
        self._очередь = list(по_порядку)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self._очередь.pop(0)


МАШИНА = Detection(x=100, y=100, w=400, h=200, conf=0.9, cls=2)
АВТОБУС = Detection(x=120, y=110, w=300, h=180, conf=0.4, cls=5)
КРОШКА = Detection(x=0, y=0, w=60, h=40, conf=0.6, cls=2)


def test_чужая_машина_даёт_один_кроп_главного_бокса(tmp_path: Path) -> None:
    фото = _фото(tmp_path / "src" / "a.jpg", 40)
    кандидат = NegativeCandidate(фото, True, "autoevolution/ALFA_Giulia", "ALFA_Giulia")

    итог = build_negative_frame(
        [кандидат],
        _ДетекторПоПути([[МАШИНА, КРОШКА]]),
        tmp_path / "crops",
        off_catalog_cap=10,
        excluded_photos=frozenset(),
    )

    assert итог.kept == {"off_catalog": 1}
    строка = итог.frame.iloc[0]
    assert строка["source"] == "negatives"
    assert строка["model_id"] is None and строка["brand"] is None
    assert neg_kind_of(строка["listing_id"]) == "off_catalog"
    assert (tmp_path / "crops" / строка["crop_path"]).is_file()


def test_отбраковка_харвеста_режет_каждый_бокс_по_виду(tmp_path: Path) -> None:
    фото = _фото(tmp_path / "src" / "b.jpg", 90)
    кандидат = NegativeCandidate(фото, False, "commons/123", "commons:123")

    итог = build_negative_frame(
        [кандидат],
        _ДетекторПоПути([[МАШИНА, АВТОБУС, КРОШКА]]),
        tmp_path / "crops",
        off_catalog_cap=10,
        excluded_photos=frozenset(),
    )

    assert итог.kept == {"detector_fp": 1, "truck_bus": 1}
    assert итог.rejected == {"small": 1}


def test_потолок_чужих_машин_держится(tmp_path: Path) -> None:
    кандидаты = [
        NegativeCandidate(_фото(tmp_path / "src" / f"{н}.jpg", 10 + н), True, f"ae/{н}", str(н))
        for н in range(3)
    ]

    итог = build_negative_frame(
        кандидаты,
        _ДетекторПоПути([[МАШИНА]] * 3),
        tmp_path / "crops",
        off_catalog_cap=2,
        excluded_photos=frozenset(),
    )

    assert итог.kept == {"off_catalog": 2}


def test_исключённый_кроп_не_пишется(tmp_path: Path) -> None:
    фото = _фото(tmp_path / "src" / "c.jpg", 70)
    кандидат = NegativeCandidate(фото, True, "ae/c", "c")
    первый = build_negative_frame(
        [кандидат],
        _ДетекторПоПути([[МАШИНА]]),
        tmp_path / "crops",
        off_catalog_cap=10,
        excluded_photos=frozenset(),
    )
    лишний = str(первый.frame.iloc[0]["photo_id"])

    второй = build_negative_frame(
        [кандидат],
        _ДетекторПоПути([[МАШИНА]]),
        tmp_path / "crops2",
        off_catalog_cap=10,
        excluded_photos=frozenset({лишний}),
    )

    assert второй.frame.empty
    assert второй.rejected == {"excluded": 1}


def test_кадр_проходит_схему_сплита(tmp_path: Path) -> None:
    фото = _фото(tmp_path / "src" / "d.jpg", 20)
    итог = build_negative_frame(
        [NegativeCandidate(фото, True, "ae/d", "d")],
        _ДетекторПоПути([[МАШИНА]]),
        tmp_path / "crops",
        off_catalog_cap=10,
        excluded_photos=frozenset(),
    )

    write_split(итог.frame, tmp_path / "negatives.parquet")
    прочитан = read_split(tmp_path / "negatives.parquet")
    assert прочитан["class_idx"].isna().all()


def test_отбраковки_находят_файл_по_индексу(tmp_path: Path) -> None:
    журнал = tmp_path / "reject_log.csv"
    журнал.write_text(
        "class_id,basis,source,source_item_id,reason,checked\n"
        "lada__niva,q,commons,1,no_vehicle,2026-09-01\n"
        "lada__niva,q,commons,2,redaction_failed,2026-09-01\n"
        "lada__niva,q,commons,3,not_a_car_photo,2026-09-01\n",
        encoding="utf-8",
    )
    индекс = tmp_path / "reject_index.csv"
    индекс.write_text(
        "class_id,source_item_id,sha256\n"
        f"lada__niva,1,{'a' * 64}\n"
        f"lada__niva,2,{'b' * 64}\n"
        f"lada__niva,3,{'c' * 64}\n",
        encoding="utf-8",
    )
    сырое = tmp_path / "raw" / "commons" / "aa" / f"{'a' * 64}.jpg"
    сырое.parent.mkdir(parents=True)
    сырое.write_bytes(b"x")

    кандидаты, без_файла = reject_candidates(журнал, индекс, tmp_path)

    assert [к.path for к in кандидаты] == [сырое]
    assert кандидаты[0].off_catalog is False
    assert без_файла == 1  # запись 3 есть в индексе, файла нет; запись 2 — чужая причина
