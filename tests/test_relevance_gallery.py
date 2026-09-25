"""Слепая галерея разметки holdout_ru (спека holdout-relevance §5, §8.1)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pipeline.holdout_relevance import CSV_HEADER
from scripts.relevance_gallery import GalleryFrame, frames_from, render_gallery, stage_images

МЕТА = {
    "classes": [
        {"class_id": "lada__vesta", "display": "Лада Веста"},
        {"class_id": "audi__a4", "display": "Audi A4"},
    ]
}


def _эталон() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "photo_id": ["p3", "p1", "p2", "p9"],
            "class_idx": [0, 1, 0, 0],
            "crop_path": ["lada__vesta/p3.jpg", "audi__a4/p1.jpg", "lada__vesta/p2.jpg", "x"],
            "is_multi_car": [False, False, False, True],
        }
    )


def test_кадры_без_многомашинных_по_классу_и_photo_id() -> None:
    кадры = frames_from(_эталон(), МЕТА)
    assert [к.photo_id for к in кадры] == ["p1", "p2", "p3"]
    assert кадры[0] == GalleryFrame("p1", "audi__a4", "Audi A4", "audi__a4/p1.jpg")


def test_каждый_кадр_на_странице_ровно_раз() -> None:
    страница = render_gallery(frames_from(_эталон(), МЕТА), {})
    for pid in ("p1", "p2", "p3"):
        assert страница.count(f'data-id="{pid}"') == 1
    assert "p9" not in страница


def test_страница_не_знает_ответов_модели() -> None:
    страница = render_gallery(frames_from(_эталон(), МЕТА), {}).lower()
    for запретное in ("sim", "top1", "recall", "score"):
        assert запретное not in страница


def test_клавиши_по_коду_хранилище_в_try_и_заголовок_из_константы() -> None:
    страница = render_gallery(frames_from(_эталон(), МЕТА), {})
    for код in ("KeyR", "KeyN", "KeyW", "KeyU", "KeyT"):
        assert код in страница
    assert страница.count("try{") >= 2
    assert CSV_HEADER in страница


def test_прошлые_вердикты_попадают_в_начальное_состояние() -> None:
    страница = render_gallery(frames_from(_эталон(), МЕТА), {"p2": ("off_street", True)})
    assert '"p2": {"v": "off_street", "t": true}' in страница


def _кроп(путь: Path, сторона: int) -> None:
    путь.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(путь), np.full((сторона, сторона, 3), 128, dtype=np.uint8))


def test_миниатюра_ужата_полный_кроп_скопирован(tmp_path: Path) -> None:
    _кроп(tmp_path / "crops" / "audi__a4" / "p1.jpg", 1000)
    кадр = GalleryFrame("p1", "audi__a4", "Audi A4", "audi__a4/p1.jpg")
    stage_images([кадр], tmp_path / "crops", tmp_path / "out", thumb_px=200)
    миниатюра = cv2.imread(str(tmp_path / "out" / "thumbs" / "p1.jpg"))
    assert max(миниатюра.shape[:2]) == 200
    assert (tmp_path / "out" / "full" / "p1.jpg").exists()


def test_пропавший_кроп_роняет_сборку_до_записи(tmp_path: Path) -> None:
    _кроп(tmp_path / "crops" / "audi__a4" / "p1.jpg", 300)
    кадры = [
        GalleryFrame("p1", "audi__a4", "Audi A4", "audi__a4/p1.jpg"),
        GalleryFrame("p2", "lada__vesta", "Лада Веста", "lada__vesta/p2.jpg"),
    ]
    with pytest.raises(FileNotFoundError, match="p2"):
        stage_images(кадры, tmp_path / "crops", tmp_path / "out", thumb_px=200)
    assert not (tmp_path / "out").exists()
