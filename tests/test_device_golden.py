"""Эталон паритета для устройства (§9 дизайна M-1.5).

Тест не грузит модель: он проверяет структуру эталона и согласованность верхнего
уровня — размерность, число записей, нормированность эмбеддингов. Сам паритет
считает инструментальный тест на телефоне (задача 11), а здесь ловится фикстура,
собранная не тем скриптом или уехавшая при правке генератора.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ЭТАЛОН = Path("tests/fixtures/device/golden.json")


@pytest.fixture(scope="module")
def эталон() -> dict[str, Any]:
    if not ЭТАЛОН.exists():
        pytest.fail("нет tests/fixtures/device/golden.json: собери scripts.gen_device_golden")
    документ: dict[str, Any] = json.loads(ЭТАЛОН.read_text(encoding="utf-8"))
    return документ


def test_в_эталоне_двенадцать_записей_каждого_уровня(эталон: dict[str, Any]) -> None:
    assert len(эталон["tensor"]) == 12
    assert len(эталон["frame"]) == 12


def test_эмбеддинги_нормированы_и_нужной_размерности(эталон: dict[str, Any]) -> None:
    размерность = эталон["embedding_dim"]
    for уровень in ("tensor", "frame"):
        for запись in эталон[уровень]:
            вектор = np.asarray(запись["embedding"], dtype=np.float32)
            assert вектор.shape == (размерность,)
            assert np.isfinite(вектор).all()
            assert float(np.linalg.norm(вектор)) == pytest.approx(1.0, abs=1e-3)


def test_все_png_на_месте_и_нужного_размера(эталон: dict[str, Any]) -> None:
    from PIL import Image

    корень = ЭТАЛОН.parent
    for запись in эталон["tensor"]:
        with Image.open(корень / запись["png"]) as картинка:
            assert картинка.size == (320, 320)
            assert картинка.mode == "RGB"
    for запись in эталон["frame"]:
        with Image.open(корень / запись["png"]) as картинка:
            assert картинка.mode == "RGB"
            assert min(картинка.size) >= 320


def test_top5_это_пять_разных_классов(эталон: dict[str, Any]) -> None:
    for уровень in ("tensor", "frame"):
        for запись in эталон[уровень]:
            assert len(запись["top5"]) == 5
            assert len(set(запись["top5"])) == 5


def test_боксы_кадров_положительны_и_внутри_картинки(эталон: dict[str, Any]) -> None:
    from PIL import Image

    корень = ЭТАЛОН.parent
    for запись in эталон["frame"]:
        бокс = запись["box"]
        with Image.open(корень / запись["png"]) as картинка:
            ширина, высота = картинка.size
        assert бокс["w"] > 0 and бокс["h"] > 0
        assert 0 <= бокс["x"] <= ширина
        assert бокс["y"] + бокс["h"] <= высота + 1e-3


def test_эталон_называет_происхождение(эталон: dict[str, Any]) -> None:
    """Без хешей эталон нельзя привязать к моделям, на которых он собран.

    Расходятся они молча: чужой чекпоинт даёт другие эмбеддинги, и паритетный тест
    на устройстве покажет расхождение конвейеров там, где разошлись веса.
    """
    for ключ in ("checkpoint_sha256", "classifier_sha256"):
        assert len(эталон[ключ]) == 64
        assert set(эталон[ключ]) <= set("0123456789abcdef")


def test_боксы_детектора_согласованы_с_кропом(эталон: dict[str, Any]) -> None:
    """`box` — то, по чему резался кроп, `detector_box` — то, что нашёл детектор.

    На текущем конвейере это один и тот же бокс: кроп берётся с главного бокса
    детектора. Тест держит связь явной — если генератор начнёт писать в `box`
    что-то своё, инструментальный тест уровня B поедет на другом входе, чем
    считался эталон, и объяснить расхождение будет нечем.
    """
    for запись in эталон["frame"]:
        for поле in ("x", "y", "w", "h"):
            assert запись["box"][поле] == pytest.approx(запись["detector_box"][поле], abs=1e-3)


def test_кадр_ссылается_на_свой_тензор(эталон: dict[str, Any]) -> None:
    """Уровень B сравнивает свой кроп с этим PNG: ссылка обязана быть в самой записи."""
    for номер, запись in enumerate(эталон["frame"]):
        assert запись["tensor_png"] == эталон["tensor"][номер]["png"]
        assert (ЭТАЛОН.parent / запись["tensor_png"]).is_file()


def test_записи_ссылаются_на_разные_картинки(эталон: dict[str, Any]) -> None:
    for уровень in ("tensor", "frame"):
        пути = [запись["png"] for запись in эталон[уровень]]
        assert len(set(пути)) == len(пути)
