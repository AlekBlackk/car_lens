"""Разбор разметки COCO 2017 в кандидатов не-машин (спека open-set-street §3)."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.negatives_coco import CocoImage, coco_pools, read_coco_images


def разметка(tmp_path: Path) -> Path:
    документ = {
        "images": [
            {"id": 3, "file_name": "3.jpg", "coco_url": "u/3.jpg"},
            {"id": 1, "file_name": "1.jpg", "coco_url": "u/1.jpg"},
            {"id": 2, "file_name": "2.jpg", "coco_url": "u/2.jpg"},
            {"id": 4, "file_name": "4.jpg", "coco_url": "u/4.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 8, "bbox": [1, 2, 3, 4], "iscrowd": 0},
            {"image_id": 1, "category_id": 3, "bbox": [5, 6, 7, 8], "iscrowd": 0},
            {"image_id": 2, "category_id": 6, "bbox": [0, 0, 9, 9], "iscrowd": 1},
            {"image_id": 4, "category_id": 4, "bbox": [0, 0, 5, 5], "iscrowd": 0},
            {"image_id": 3, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
        ],
    }
    путь = tmp_path / "instances.json"
    путь.write_text(json.dumps(документ), encoding="utf-8")
    return путь


def test_разметка_читается_по_картинкам(tmp_path: Path) -> None:
    картинки = read_coco_images(разметка(tmp_path))
    assert [к.file_name for к in картинки] == ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    первая = картинки[0]
    assert первая.url == "u/1.jpg"
    assert первая.truth.truck_bus == ((1.0, 2.0, 3.0, 4.0),)
    assert первая.truth.car == ((5.0, 6.0, 7.0, 8.0),)
    assert картинки[1].truth.truck_bus == ((0.0, 0.0, 9.0, 9.0),)  # iscrowd тоже транспорт
    assert картинки[3].has_motorcycle


def test_стопки_грузовых_и_пустых(tmp_path: Path) -> None:
    грузовые, пустые = coco_pools(read_coco_images(разметка(tmp_path)), seed=1)
    assert sorted(к.file_name for к in грузовые) == ["1.jpg", "2.jpg"]
    assert [к.file_name for к in пустые] == ["3.jpg"]  # мотоцикл — не пустая


def test_стопки_не_зависят_от_порядка_входа(tmp_path: Path) -> None:
    картинки = read_coco_images(разметка(tmp_path))
    прямые = coco_pools(картинки, seed=7)
    обратные = coco_pools(list(reversed(картинки)), seed=7)
    assert прямые == обратные
    assert isinstance(прямые[0][0], CocoImage)
