import numpy as np
import pytest

from pipeline.harvest.backends import USER_AGENT_TEMPLATE, decode_ssd_output


def test_разбор_ssd_переводит_доли_в_пиксели() -> None:
    """Выход SSD — доли кадра; конвейер работает в пикселях исходного фото."""
    сырое = np.array([[[[0.0, 2.0, 0.9, 0.25, 0.5, 0.75, 1.0]]]], dtype=np.float32)
    боксы = decode_ssd_output(сырое, frame_w=400, frame_h=200, conf=0.5)
    assert len(боксы) == 1
    бокс = боксы[0]
    assert (бокс.x, бокс.y) == (100.0, 100.0)
    assert (бокс.w, бокс.h) == (200.0, 100.0)
    assert бокс.conf == pytest.approx(0.9)


def test_разбор_ssd_режет_по_порогу() -> None:
    сырое = np.array([[[[0.0, 2.0, 0.2, 0.25, 0.5, 0.75, 1.0]]]], dtype=np.float32)
    assert decode_ssd_output(сырое, 400, 200, conf=0.5) == []


def test_разбор_ssd_обрезает_выход_за_кадр() -> None:
    сырое = np.array([[[[0.0, 2.0, 0.9, -0.1, -0.2, 1.4, 1.3]]]], dtype=np.float32)
    бокс = decode_ssd_output(сырое, 400, 200, conf=0.5)[0]
    assert (бокс.x, бокс.y) == (0.0, 0.0)
    assert (бокс.w, бокс.h) == (400.0, 200.0)


def test_user_agent_несёт_контакт() -> None:
    """§3.2: обезличенный User-Agent — это то, за что хранилища банят по IP."""
    строка = USER_AGENT_TEMPLATE.format(contact="mail@example.com")
    assert "mail@example.com" in строка
    assert "carlens" in строка.lower()
