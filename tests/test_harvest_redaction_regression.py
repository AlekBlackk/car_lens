"""Регресс двух дефектов прогона 2026-08-29: пропуск номера детектором и читаемый
номер фоновой машины в сохранённом кропе."""

import numpy as np

from pipeline.crop import Detection, crop_image, select_primary
from pipeline.harvest.redact import redact_frame


class Список:
    def __init__(self, s: list[list[Detection]]) -> None:
        self.s = list(s)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.s.pop(0) if self.s else []


def test_номер_фоновой_машины_не_попадает_в_кроп() -> None:
    кадр = np.full((600, 1200, 3), 130, dtype=np.uint8)
    главная = Detection(60.0, 100.0, 520.0, 420.0, 0.96, 2)
    фоновая = Detection(560.0, 120.0, 300.0, 200.0, 0.7, 2)
    номер_фоновой = Detection(600.0, 300.0, 70.0, 20.0, 0.9, 0)

    итог = redact_frame(
        кадр,
        [главная, фоновая],
        [Список([[номер_фоновой], [], [], [], [], []])],
        Список([[]]),
    )
    assert итог.ok
    решение = select_primary([главная, фоновая], 1200, 600)
    assert решение.box is not None
    кроп = crop_image(итог.image, решение.box)
    x0 = int(600 - (решение.box.x - решение.box.w * 0.08))
    y0 = int(300 - (решение.box.y - решение.box.h * 0.08))
    assert 0 <= x0 < кроп.shape[1] and 0 <= y0 < кроп.shape[0]
    assert int(кроп[y0, x0].max()) == 0, "номер фоновой машины обязан быть залит"


def test_второй_детектор_ловит_то_что_пропустил_первый() -> None:
    кадр = np.full((500, 900, 3), 120, dtype=np.uint8)
    машина = Detection(50.0, 50.0, 500.0, 380.0, 0.95, 2)
    номер = Detection(150.0, 360.0, 90.0, 26.0, 0.9, 0)

    слепой = Список([[], [], []])  # барьер-0106 промахнулся
    зрячий = Список([[номер], [], []])  # plate_hires нашёл
    итог = redact_frame(кадр, [машина], [слепой, зрячий], Список([[]]))
    assert итог.ok and итог.plates == 1
    assert int(итог.image[370, 180].max()) == 0


def test_остаток_на_собственной_заливке_не_отбивает_кадр() -> None:
    кадр = np.full((400, 800, 3), 130, dtype=np.uint8)
    машина = Detection(60.0, 100.0, 520.0, 220.0, 0.96, 2)
    номер = Detection(200.0, 250.0, 120.0, 30.0, 0.9, 0)
    заливка_как_номер = Detection(205.0, 252.0, 100.0, 26.0, 0.5, 0)

    итог = redact_frame(
        кадр,
        [машина],
        [Список([[номер], [], [заливка_как_номер], []])],
        Список([[]]),
    )

    assert итог.ok
    assert итог.plates == 1


def test_остаток_той_же_формы_в_стороне_отбивает_кадр() -> None:
    кадр = np.full((400, 800, 3), 130, dtype=np.uint8)
    машина = Detection(60.0, 100.0, 520.0, 220.0, 0.96, 2)
    номер = Detection(200.0, 250.0, 120.0, 30.0, 0.9, 0)
    другой_номер = Detection(600.0, 40.0, 100.0, 26.0, 0.5, 0)

    итог = redact_frame(
        кадр,
        [машина],
        [Список([[номер], [], [другой_номер], []])],
        Список([[]]),
    )

    assert not итог.ok
