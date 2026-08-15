"""Адаптер ultralytics YOLO11n к протоколу `Detector`.

Тонкая обёртка без логики: вся логика выбора бокса живёт в `crop.py` и
тестируется без весов. Здесь только перевод формата.

Лицензия ultralytics — AGPL-3.0. Ограничения и план замены — §15.
"""

from __future__ import annotations

import numpy as np

from pipeline.crop import Detection


class YoloDetector:
    """Детектор на весах COCO, без дообучения (§4.1)."""

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 416,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Найти боксы. Вход — BGR-массив OpenCV, выход — координаты исходного фото."""
        результаты = self._model.predict(
            image,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        боксы: list[Detection] = []
        for результат in результаты:
            if результат.boxes is None:
                continue
            xyxy = результат.boxes.xyxy.cpu().numpy()
            confs = результат.boxes.conf.cpu().numpy()
            classes = результат.boxes.cls.cpu().numpy().astype(int)
            for (x0, y0, x1, y1), conf, cls in zip(xyxy, confs, classes, strict=True):
                боксы.append(
                    Detection(
                        x=float(x0),
                        y=float(y0),
                        w=float(x1 - x0),
                        h=float(y1 - y0),
                        conf=float(conf),
                        cls=int(cls),
                    )
                )
        return боксы
