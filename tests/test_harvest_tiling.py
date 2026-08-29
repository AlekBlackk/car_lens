import numpy as np

from pipeline.crop import Detection
from pipeline.harvest.tiling import TiledDetector, nms_merge, tile_origins


class ЗаписнойДетектор:
    """Возвращает бокс в центре каждого поданного выреза, пишет размеры вырезов."""

    def __init__(self) -> None:
        self.поданные: list[tuple[int, int]] = []

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        self.поданные.append((w, h))
        return [Detection(w / 2 - 5, h / 2 - 5, 10, 10, 0.9, 0)]


def test_начала_тайлов_покрывают_размер_с_перекрытием() -> None:
    начала = tile_origins(3000, 1280, 0.25)
    assert начала[0] == 0
    assert начала[-1] + 1280 >= 3000
    шаги = [начала[i + 1] - начала[i] for i in range(len(начала) - 1)]
    assert all(0 < шаг < 1280 for шаг in шаги)


def test_маленький_кадр_один_тайл() -> None:
    assert tile_origins(800, 1280, 0.25) == [0]


def test_nms_merge_снимает_совпадающие_боксы() -> None:
    a = Detection(100.0, 100.0, 50.0, 20.0, 0.9, 0)
    почти = Detection(103.0, 101.0, 50.0, 20.0, 0.8, 0)
    далеко = Detection(400.0, 400.0, 50.0, 20.0, 0.7, 0)
    итог = nms_merge([a, почти, далеко])
    assert len(итог) == 2


def test_детектор_зовётся_по_каждому_тайлу_и_боксы_в_координатах_кадра() -> None:
    кадр = np.zeros((2000, 3000, 3), dtype=np.uint8)
    внутренний = ЗаписнойДетектор()
    боксы = TiledDetector(внутренний, tile=1280, overlap=0.25).detect(кадр)
    assert len(внутренний.поданные) > 1, "большой кадр обязан резаться"
    for б in боксы:
        assert 0 <= б.x <= 3000 and 0 <= б.y <= 2000


def test_дубли_из_перекрытия_снимаются_nms() -> None:
    кадр = np.zeros((600, 1600, 3), dtype=np.uint8)
    кадр[280:320, 760:840] = 255  # белый прямоугольник в зоне перекрытия тайлов

    class БелыйКвадрат:
        def detect(self, image: np.ndarray) -> list[Detection]:
            маска = image.max(axis=2) > 200
            if not маска.any():
                return []
            ys, xs = np.where(маска)
            return [
                Detection(
                    float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min()), 0.9, 0,
                )
            ]

    боксы = TiledDetector(БелыйКвадрат(), tile=1000, overlap=0.3).detect(кадр)
    assert len(боксы) == 1
