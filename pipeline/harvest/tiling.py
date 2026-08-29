"""Нарезка кадра на перекрывающиеся тайлы для детекции мелких объектов.

Снимок 4000 px, поданный в детектор с входом 300–640 px, теряет номер физически —
ужатие стирает символы раньше, чем сработает любой порог. Нарезка гоняет вложенный
детектор по окнам в нативном разрешении и сливает боксы обратно по NMS.

Обёртка тонкая и без знания предметной области: `inner` — любой `Detector`, поэтому одна
и та же нарезка накрывает и barrier-0106, и YOLO-LP. `nms_merge` вынесен сюда же и
переиспользуется редакцией для слияния находок ансамбля детекторов.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from pipeline.crop import Detection, Detector


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    """Левые (или верхние) края тайлов вдоль одной оси."""
    if size <= tile:
        return [0]
    шаг = max(1, int(tile * (1.0 - overlap)))
    начала = list(range(0, size - tile + 1, шаг))
    последний = size - tile
    if начала[-1] != последний:
        начала.append(последний)
    return начала


def nms_merge(dets: Sequence[Detection], iou: float = 0.45) -> list[Detection]:
    """Снять перекрывающиеся боксы по non-maximum suppression."""
    if len(dets) < 2:
        return list(dets)
    прямоугольники = [[int(d.x), int(d.y), int(d.w), int(d.h)] for d in dets]
    оценки = [float(d.conf) for d in dets]
    индексы = cv2.dnn.NMSBoxes(прямоугольники, оценки, score_threshold=0.0, nms_threshold=iou)
    оставить = {int(i) for i in np.array(индексы).reshape(-1)}
    return [d for н, d in enumerate(dets) if н in оставить]


class TiledDetector:
    """Прогон вложенного детектора по тайлам с NMS-слиянием результатов."""

    def __init__(
        self,
        inner: Detector,
        tile: int = 1280,
        overlap: float = 0.25,
        iou_merge: float = 0.45,
    ) -> None:
        self._inner = inner
        self._tile = tile
        self._overlap = overlap
        self._iou = iou_merge

    def detect(self, image: np.ndarray) -> list[Detection]:
        высота, ширина = image.shape[:2]
        собрано: list[Detection] = []
        for y0 in tile_origins(высота, self._tile, self._overlap):
            for x0 in tile_origins(ширина, self._tile, self._overlap):
                y1 = min(высота, y0 + self._tile)
                x1 = min(ширина, x0 + self._tile)
                тайл = image[y0:y1, x0:x1]
                for б in self._inner.detect(тайл):
                    собрано.append(Detection(б.x + x0, б.y + y0, б.w, б.h, б.conf, б.cls))
        return nms_merge(собрано, self._iou)
