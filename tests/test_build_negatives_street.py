"""Оркестрация уличных негативов: кандидаты MAD-Cars и ленивый поток COCO."""

from __future__ import annotations

import http.client
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.crop import Detection
from pipeline.negatives import CocoTruth, NegativeCandidate, build_negative_frame
from pipeline.negatives_coco import CocoImage
from pipeline.schema import write_split
from scripts.build_negatives_street import coco_candidates, street_candidates

ПУСТО = CocoTruth(truck_bus=(), car=())


def _кадр(путь: Path) -> Path:
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(b"jpeg")
    return путь


def test_уличные_кандидаты_по_папкам_моделей(tmp_path: Path) -> None:
    _кадр(tmp_path / "bmw__x3" / "12_40.jpg")
    _кадр(tmp_path / "bmw__x3" / "7_3.jpg")
    _кадр(tmp_path / "audi__q7" / "5_1.jpg")
    _кадр(tmp_path / "vaz__granta_x" / "9_1.jpg")

    кандидаты = street_candidates(tmp_path, frozenset({"vaz__granta_x"}))

    assert [(к.origin, к.path.name) for к in кандидаты] == [
        ("mad_cars/audi__q7", "5_1.jpg"),
        ("mad_cars/bmw__x3", "12_40.jpg"),
        ("mad_cars/bmw__x3", "7_3.jpg"),
    ]
    assert all(к.off_catalog and к.primary_kind == "off_catalog_street" for к in кандидаты)
    assert кандидаты[0].raw_name == "audi__q7"


def test_coco_качает_лениво_и_берёт_из_кэша(tmp_path: Path) -> None:
    _кадр(tmp_path / "cache" / "a.jpg")
    картинки = [CocoImage("a.jpg", "u/a", ПУСТО, False), CocoImage("b.jpg", "u/b", ПУСТО, False)]
    скачано: list[str] = []

    def скачать(url: str, путь: Path) -> None:
        скачано.append(url)
        путь.write_bytes(b"jpeg")

    поток = coco_candidates(картинки, tmp_path / "cache", скачать, [])
    первый = next(поток)

    assert первый.path.name == "a.jpg" and скачано == []  # лежит — не качается
    assert next(поток).path.name == "b.jpg" and скачано == ["u/b"]
    assert первый.coco == ПУСТО and первый.origin == "coco/a.jpg"


@pytest.mark.parametrize("ошибка", [OSError("сеть"), http.client.IncompleteRead(b"")])
def test_сбой_скачивания_coco_пропускает_картинку(tmp_path: Path, ошибка: Exception) -> None:
    картинки = [CocoImage("a.jpg", "u/a", ПУСТО, False), CocoImage("b.jpg", "u/b", ПУСТО, False)]
    сбои: list[str] = []

    def скачать(url: str, путь: Path) -> None:
        путь.write_bytes(b"")
        if url == "u/a":
            raise ошибка
        путь.write_bytes(b"jpeg")

    кандидаты = list(coco_candidates(картинки, tmp_path / "cache", скачать, сбои))

    assert [к.path.name for к in кандидаты] == ["b.jpg"]
    assert сбои == ["u/a"]
    assert not (tmp_path / "cache" / "a.jpg").exists()


class _ОдинБокс:
    def detect(self, image: np.ndarray) -> list[Detection]:
        return [Detection(x=20, y=20, w=260, h=180, conf=0.9, cls=2)]


def test_листы_читают_выбранный_сплит(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.negatives_sheets as листы

    фото = tmp_path / "src" / "bmw__x3" / "1_2.jpg"
    фото.parent.mkdir(parents=True)
    cv2.imwrite(str(фото), np.full((240, 320, 3), 128, dtype=np.uint8))
    кандидат = NegativeCandidate(
        фото, True, "mad_cars/bmw__x3", "bmw__x3", primary_kind="off_catalog_street"
    )
    итог = build_negative_frame(
        [кандидат],
        _ОдинБокс(),
        tmp_path / "crops",
        off_catalog_cap=10,
        excluded_photos=frozenset(),
    )
    write_split(итог.frame, tmp_path / "negatives-street.parquet")
    monkeypatch.setattr(листы, "splits_dir", lambda: tmp_path)
    monkeypatch.setattr(листы, "harvest_dir", lambda: tmp_path)

    assert листы.main(["--split", "negatives-street", "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "off_catalog_street_00.png").is_file()
    assert "bmw__x3" in (tmp_path / "out" / "off_catalog_street_00.txt").read_text("utf-8")
