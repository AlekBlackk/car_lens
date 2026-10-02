"""Свежие не-машины: отбор картинок, которых прежняя сборка не касалась."""

from __future__ import annotations

from pipeline.negatives import CocoTruth
from pipeline.negatives_coco import CocoImage
from scripts.build_noncar_fresh import unseen_images


def _картинка(имя: str) -> CocoImage:
    return CocoImage(имя, f"u/{имя}", CocoTruth(truck_bus=(), car=()), has_motorcycle=False)


def test_свежие_это_картинки_без_файла_в_кэше_в_порядке_пула() -> None:
    пул = [_картинка(имя) for имя in ("c.jpg", "a.jpg", "b.jpg", "d.jpg")]

    свежие = unseen_images(пул, {"a.jpg", "d.jpg"})

    assert [к.file_name for к in свежие] == ["c.jpg", "b.jpg"]
