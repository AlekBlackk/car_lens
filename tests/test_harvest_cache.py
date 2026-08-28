import io
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.harvest.cache import crop_path, decode_image, digest, store


def _jpeg(ширина: int, высота: int, orientation: int | None = None) -> bytes:
    картинка = Image.new("RGB", (ширина, высота), (10, 20, 240))
    буфер = io.BytesIO()
    if orientation is None:
        картинка.save(буфер, format="JPEG")
    else:
        exif = Image.Exif()
        exif[274] = orientation  # тег Orientation
        картинка.save(буфер, format="JPEG", exif=exif)
    return буфер.getvalue()


def test_декодирование_даёт_bgr() -> None:
    """BGR — сквозной формат конвейера: синий пиксель обязан прийти первым каналом."""
    массив = decode_image(_jpeg(40, 20))
    assert массив.shape == (20, 40, 3)
    синий, _, красный = массив[0, 0]
    assert синий > красный


def test_exif_поворот_применяется() -> None:
    """Кадр, оставшийся боком, теряет машину на YOLO-гейте и уводит класс в дефицит."""
    массив = decode_image(_jpeg(40, 20, orientation=6))
    assert массив.shape[:2] == (40, 20), "ориентация 6 меняет высоту и ширину местами"


def test_store_идемпотентен(tmp_path: Path) -> None:
    данные = _jpeg(40, 20)
    первый = store(tmp_path, "commons", данные, "jpg")
    второй = store(tmp_path, "commons", данные, "jpg")
    assert первый == второй
    assert (first_bytes := первый.read_bytes())
    assert first_bytes == данные


def test_store_раскладывает_по_двум_символам_хеша(tmp_path: Path) -> None:
    данные = _jpeg(40, 20)
    путь = store(tmp_path, "commons", данные, "jpg")
    sha = digest(данные)
    assert путь == tmp_path / "raw" / "commons" / sha[:2] / f"{sha}.jpg"


def test_временные_файлы_не_остаются(tmp_path: Path) -> None:
    """Оборванная закачка не должна оставлять битый JPEG, который сойдёт за готовый."""
    store(tmp_path, "commons", _jpeg(40, 20), "jpg")
    assert not list(tmp_path.rglob("*.tmp"))


def test_путь_кропа_по_классу(tmp_path: Path) -> None:
    assert crop_path(tmp_path, "lada__granta", "ab" * 32) == (
        tmp_path / "crops" / "lada__granta" / f"{'ab' * 32}.jpg"
    )


def test_декодирование_не_меняет_данные_дважды() -> None:
    данные = _jpeg(40, 20)
    assert np.array_equal(decode_image(данные), decode_image(данные))
