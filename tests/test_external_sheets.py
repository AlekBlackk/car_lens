"""Листы-превью кропов с впаянными номерами."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.external_sheets import CELL, COLUMNS_PER_SHEET, render_sheet


def test_лист_раскладывает_кадры_сеткой_с_номерами(tmp_path: Path) -> None:
    пути = []
    for номер in range(3):
        путь = tmp_path / f"{номер}.jpg"
        Image.fromarray(np.full((50, 80, 3), 60 * номер, dtype=np.uint8)).save(путь)
        пути.append(путь)

    лист = render_sheet(пути)

    assert лист.size == (CELL[0] * COLUMNS_PER_SHEET, CELL[1])
    # Плашка номера в левом верхнем углу каждой ячейки — жёлтая.
    for номер in range(3):
        assert лист.getpixel((номер * CELL[0] + 1, 1)) == (255, 255, 0)
