"""Тесты фильтров харвеста: YOLO-гейт, CLIP-негативы и мягкий ранг (Task 8)."""

from __future__ import annotations

import numpy as np

from pipeline.crop import Detection
from pipeline.harvest.filters import NEGATIVE_PROMPTS, clip_verdict, vehicle_gate


class ФиксированныйДетектор:
    def __init__(self, боксы: list[Detection]) -> None:
        self.боксы = боксы

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.боксы


class ЗаданныйЭнкодер:
    """Отдаёт заранее назначенный вектор и запоминает, в каком порядке пришли каналы."""

    def __init__(self, вектор: np.ndarray) -> None:
        self.вектор = вектор
        self.первый_пиксель: tuple[int, int, int] | None = None

    def encode_rgb(self, image: np.ndarray) -> np.ndarray:
        синий, зелёный, красный = (int(x) for x in image[0, 0])
        self.первый_пиксель = (синий, зелёный, красный)
        return self.вектор


def _кадр(цвет: tuple[int, int, int] = (10, 20, 240)) -> np.ndarray:
    кадр = np.zeros((200, 300, 3), dtype=np.uint8)
    кадр[:, :] = цвет
    return кадр


def test_машина_найдена() -> None:
    бокс = Detection(x=30.0, y=30.0, w=200.0, h=120.0, conf=0.9, cls=2)
    вердикт = vehicle_gate(_кадр(), ФиксированныйДетектор([бокс]))
    assert вердикт.reason is None
    assert вердикт.box == бокс


def test_машины_нет() -> None:
    assert vehicle_gate(_кадр(), ФиксированныйДетектор([])).reason == "no_vehicle"


def test_машина_слишком_мелкая() -> None:
    бокс = Detection(x=0.0, y=0.0, w=40.0, h=30.0, conf=0.9, cls=2)
    assert vehicle_gate(_кадр(), ФиксированныйДетектор([бокс])).reason == "too_small"


def test_две_сопоставимые_машины_отбраковываются() -> None:
    """Метка класса относится к галерее, а не к боксу: какая из двух — неизвестно."""
    первая = Detection(x=0.0, y=0.0, w=200.0, h=120.0, conf=0.9, cls=2)
    вторая = Detection(x=200.0, y=0.0, w=180.0, h=110.0, conf=0.9, cls=2)
    вердикт = vehicle_gate(_кадр(), ФиксированныйДетектор([первая, вторая]))
    assert вердикт.reason == "multi_car"


def test_негативные_промпты_зафиксированы() -> None:
    assert len(NEGATIVE_PROMPTS) == 7
    assert all(текст.startswith("a photo of") for текст in NEGATIVE_PROMPTS)


def test_энкодер_получает_rgb() -> None:
    """Единственная конверсия BGR → RGB в конвейере живёт здесь и обязана случиться."""
    энкодер = ЗаданныйЭнкодер(np.array([1.0, 0.0], dtype=np.float32))
    классы = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    негативы = np.array([[0.0, 1.0]], dtype=np.float32)
    clip_verdict(_кадр((10, 20, 240)), энкодер, классы, негативы, target_index=0)
    assert энкодер.первый_пиксель == (240, 20, 10), "каналы обязаны развернуться"


def test_мусорный_кадр_отсекается() -> None:
    энкодер = ЗаданныйЭнкодер(np.array([0.0, 1.0], dtype=np.float32))
    классы = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    негативы = np.array([[0.0, 1.0]], dtype=np.float32)
    вердикт = clip_verdict(_кадр(), энкодер, классы, негативы, target_index=0)
    assert not вердикт.is_car_photo


def test_ранг_и_отрыв_считаются_но_не_отбраковывают() -> None:
    """Ранг задаёт порядок аудита. Отбраковки по нему нет — см. 0.511 на RU+CN (Р-10)."""
    энкодер = ЗаданныйЭнкодер(np.array([0.6, 0.8], dtype=np.float32))
    классы = np.array([[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071]], dtype=np.float32)
    негативы = np.array([[-1.0, 0.0]], dtype=np.float32)
    вердикт = clip_verdict(_кадр(), энкодер, классы, негативы, target_index=1)
    assert вердикт.is_car_photo
    assert вердикт.rank == 3, "целевой класс третий по косинусу из трёх"
    assert вердикт.margin > 0.1
