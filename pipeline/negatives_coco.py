"""COCO 2017 как источник не-машин (спека 2026-09-27-open-set-street §3).

Модуль читает только разметку: картинки качает оркестрация лениво, по одной, чтобы поток
остановился, как только наберутся потолки видов. Боксы разметки нужны, чтобы решить вид бокса
детектора (`pipeline.negatives.coco_box_kind`), а не чтобы резать по ним.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.negatives import Box, CocoTruth

COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK = 3, 4, 6, 8


@dataclass(frozen=True, slots=True)
class CocoImage:
    file_name: str
    url: str
    truth: CocoTruth
    has_motorcycle: bool


def read_coco_images(path: Path) -> list[CocoImage]:
    """Картинки разметки с боксами транспорта, по возрастанию `id`.

    `iscrowd`-боксы считаются транспортом: толпа машин — всё равно машины.
    """
    документ = json.loads(path.read_text(encoding="utf-8"))
    грузовые: dict[int, list[Box]] = defaultdict(list)
    легковые: dict[int, list[Box]] = defaultdict(list)
    с_мотоциклом: set[int] = set()
    for разметка in документ["annotations"]:
        категория = int(разметка["category_id"])
        картинка = int(разметка["image_id"])
        x, y, w, h = (float(v) for v in разметка["bbox"])
        if категория in (COCO_BUS, COCO_TRUCK):
            грузовые[картинка].append((x, y, w, h))
        elif категория == COCO_CAR:
            легковые[картинка].append((x, y, w, h))
        elif категория == COCO_MOTORCYCLE:
            с_мотоциклом.add(картинка)
    return [
        CocoImage(
            str(к["file_name"]),
            str(к["coco_url"]),
            CocoTruth(tuple(грузовые[int(к["id"])]), tuple(легковые[int(к["id"])])),
            int(к["id"]) in с_мотоциклом,
        )
        for к in sorted(документ["images"], key=lambda к: int(к["id"]))
    ]


def coco_pools(images: Sequence[CocoImage], seed: int) -> tuple[list[CocoImage], list[CocoImage]]:
    """Картинки с грузовиками/автобусами и картинки без транспорта, перемешанные по `seed`.

    Порядок входа фиксируется сортировкой по имени файла до перемешивания. Картинка с
    мотоциклом не пустая: детектор может найти его транспортом, и это не ложное срабатывание.
    """
    по_имени = sorted(images, key=lambda к: к.file_name)
    грузовые = [к for к in по_имени if к.truth.truck_bus]
    пустые = [
        к for к in по_имени if not к.truth.truck_bus and not к.truth.car and not к.has_motorcycle
    ]
    генератор = random.Random(seed)
    генератор.shuffle(грузовые)
    генератор.shuffle(пустые)
    return грузовые, пустые
