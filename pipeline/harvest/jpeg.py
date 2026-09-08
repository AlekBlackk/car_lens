"""Единое кодирование JPEG для кропов харвеста и переderivации."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class EncodedJpeg:
    """Байты JPEG и результат декодирования именно этих байтов."""

    data: bytes
    decoded: np.ndarray


def encode_jpeg(image: np.ndarray) -> EncodedJpeg:
    """Один раз закодировать кроп и проверить декодирование полученного буфера."""
    удалось, буфер = cv2.imencode(".jpg", image)
    if not удалось or буфер is None:
        raise RuntimeError("не удалось закодировать JPEG")
    данные = буфер.tobytes()
    декодированное = cv2.imdecode(np.frombuffer(данные, dtype=np.uint8), cv2.IMREAD_COLOR)
    if декодированное is None:
        raise RuntimeError("не удалось декодировать JPEG")
    return EncodedJpeg(данные, декодированное)


def write_jpeg_atomic(path: Path, encoded: EncodedJpeg) -> None:
    """Атомарно записать тот же JPEG-буфер, который уже прошёл декодирование."""
    временный = path.with_suffix(path.suffix + ".tmp")
    try:
        with временный.open("wb") as файл:
            файл.write(encoded.data)
            файл.flush()
        временный.replace(path)
    finally:
        временный.unlink(missing_ok=True)


def observed_pixels_equal(old_path: Path, new: EncodedJpeg) -> bool:
    """Сравнить декодированные старые пиксели с проверенным новым JPEG."""
    try:
        данные = old_path.read_bytes()
        if not данные:
            return False
        старое = cv2.imdecode(np.frombuffer(данные, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return False
    return старое is not None and np.array_equal(старое, new.decoded)
