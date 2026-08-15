"""Тесты выбора главного автомобиля и нарезки кропа (§3.4, §4.2)."""

import numpy as np

from pipeline.crop import (
    VEHICLE_CLASSES,
    Detection,
    crop_image,
    pad_box,
    select_primary,
)

КАДР_W, КАДР_H = 1000, 1000


def детекция(x: float, y: float, w: float, h: float, conf: float = 0.9, cls: int = 2) -> Detection:
    return Detection(x=x, y=y, w=w, h=h, conf=conf, cls=cls)


def test_классы_транспорта_из_coco() -> None:
    # Порядок операндов — автофикс ruff (SIM300): имя из ЗАГЛАВНЫХ он считает
    # константой и требует держать её справа.
    assert {2, 5, 7} == VEHICLE_CLASSES


def test_без_детекций_отвергается() -> None:
    итог = select_primary([], КАДР_W, КАДР_H)
    assert итог.box is None
    assert итог.reject_reason == "no_vehicle"


def test_нетранспортные_классы_игнорируются() -> None:
    итог = select_primary([детекция(0, 0, 900, 900, cls=0)], КАДР_W, КАДР_H)
    assert итог.reject_reason == "no_vehicle"


def test_низкая_уверенность_игнорируется() -> None:
    итог = select_primary([детекция(0, 0, 900, 900, conf=0.4)], КАДР_W, КАДР_H)
    assert итог.reject_reason == "no_vehicle"


def test_мелкий_бокс_отвергается_как_салон_или_деталь() -> None:
    """Площадь 0.09 кадра — ниже порога 0.15 из §3.4."""
    итог = select_primary([детекция(0, 0, 300, 300)], КАДР_W, КАДР_H)
    assert итог.box is None
    assert итог.reject_reason == "too_small"
    assert итог.area_frac == 0.09


def test_берётся_самый_крупный_бокс() -> None:
    боксы = [детекция(0, 0, 400, 400), детекция(500, 500, 500, 500)]
    итог = select_primary(боксы, КАДР_W, КАДР_H)
    assert итог.box is not None
    assert итог.box.w == 500
    assert итог.reject_reason is None


def test_соперник_меньше_сорока_процентов_не_мешает() -> None:
    """500x500 = 250000, соперник 300x300 = 90000, это 36 процентов."""
    боксы = [детекция(0, 0, 500, 500), детекция(600, 600, 300, 300)]
    итог = select_primary(боксы, КАДР_W, КАДР_H)
    assert итог.is_multi_car is False
    assert итог.reject_reason is None


def test_соперник_от_сорока_процентов_помечает_кадр() -> None:
    """500x500 = 250000, соперник 400x400 = 160000, это 64 процента."""
    боксы = [детекция(0, 0, 500, 500), детекция(550, 550, 400, 400)]
    итог = select_primary(боксы, КАДР_W, КАДР_H)
    assert итог.is_multi_car is True
    assert итог.box is not None
    assert итог.reject_reason is None


def test_автобус_и_грузовик_считаются_транспортом() -> None:
    for cls in (5, 7):
        итог = select_primary([детекция(0, 0, 900, 900, cls=cls)], КАДР_W, КАДР_H)
        assert итог.box is not None, f"класс {cls} должен приниматься"


def test_паддинг_расширяет_на_восемь_процентов() -> None:
    x0, y0, x1, y1 = pad_box(детекция(400, 400, 200, 100), КАДР_W, КАДР_H, pad=0.08)
    assert (x0, y0, x1, y1) == (384, 392, 616, 508)


def test_паддинг_обрезается_границами_кадра() -> None:
    x0, y0, x1, y1 = pad_box(детекция(0, 0, 1000, 1000), КАДР_W, КАДР_H, pad=0.08)
    assert (x0, y0, x1, y1) == (0, 0, 1000, 1000)


def test_кроп_имеет_ожидаемый_размер() -> None:
    картинка = np.zeros((КАДР_H, КАДР_W, 3), dtype=np.uint8)
    кроп = crop_image(картинка, детекция(400, 400, 200, 100), pad=0.08)
    assert кроп.shape == (116, 232, 3)


def test_кроп_берёт_правильную_область() -> None:
    картинка = np.zeros((КАДР_H, КАДР_W, 3), dtype=np.uint8)
    картинка[400:500, 400:600] = 255
    кроп = crop_image(картинка, детекция(400, 400, 200, 100), pad=0.0)
    assert кроп.shape == (100, 200, 3)
    assert int(кроп.mean()) == 255
