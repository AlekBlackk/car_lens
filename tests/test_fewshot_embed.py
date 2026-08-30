"""Проход энкодера по кропам (`pipeline/fewshot/embed.py`)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.fewshot.embed import EmbedError, embed_crops


class _Энкодер:
    """Подставной энкодер: красный канал в первую координату, чтобы порядок был виден."""

    def __init__(self, нулевой: bool = False) -> None:
        self.видел: list[tuple[int, int, int]] = []
        self._нулевой = нулевой

    def encode_rgb(self, image: np.ndarray) -> np.ndarray:
        self.видел.append((int(image[0, 0, 0]), int(image[0, 0, 1]), int(image[0, 0, 2])))
        if self._нулевой:
            return np.zeros(3, dtype=np.float32)
        return np.array([float(image[0, 0, 0]), 1.0, 0.0], dtype=np.float32)


def _кроп(каталог: Path, имя: str, bgr: tuple[int, int, int]) -> Path:
    """PNG, а не JPEG: сжатие с потерями сдвинуло бы значения каналов."""
    путь = каталог / имя
    картинка = np.zeros((8, 8, 3), dtype=np.uint8)
    картинка[:, :] = bgr
    cv2.imwrite(str(путь), картинка)
    return путь


def test_bgr_конвертируется_в_rgb(tmp_path: Path) -> None:
    энкодер = _Энкодер()
    embed_crops([_кроп(tmp_path, "a.png", (10, 20, 30))], энкодер)
    assert энкодер.видел == [(30, 20, 10)]


def test_порядок_путей_сохранён(tmp_path: Path) -> None:
    пути = [
        _кроп(tmp_path, "первый.png", (0, 0, 7)),
        _кроп(tmp_path, "второй.png", (0, 0, 3)),
        _кроп(tmp_path, "третий.png", (0, 0, 5)),
    ]
    энкодер = _Энкодер()
    матрица = embed_crops(пути, энкодер)
    assert [красный for красный, _, _ in энкодер.видел] == [7, 3, 5]
    assert матрица.shape == (3, 3)


def test_вектора_нормализованы(tmp_path: Path) -> None:
    пути = [_кроп(tmp_path, f"{n}.png", (0, 0, n + 1)) for n in range(4)]
    матрица = embed_crops(пути, _Энкодер())
    assert np.allclose(np.linalg.norm(матрица, axis=1), 1.0, atol=1e-6)
    assert матрица.dtype == np.float32


def test_нечитаемый_файл_отвергается(tmp_path: Path) -> None:
    битый = tmp_path / "битый.png"
    битый.write_bytes("не картинка".encode())
    with pytest.raises(EmbedError, match="не читается"):
        embed_crops([битый], _Энкодер())


def test_пустой_список_отвергается() -> None:
    with pytest.raises(EmbedError, match="пустой список"):
        embed_crops([], _Энкодер())


def test_нулевой_эмбеддинг_отвергается(tmp_path: Path) -> None:
    with pytest.raises(EmbedError, match="нулевой эмбеддинг"):
        embed_crops([_кроп(tmp_path, "z.png", (1, 1, 1))], _Энкодер(нулевой=True))
