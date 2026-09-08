from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.harvest.jpeg import encode_jpeg, observed_pixels_equal, write_jpeg_atomic


def _декодировать(data: bytes) -> np.ndarray:
    изображение = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert изображение is not None
    return изображение


def test_записаны_ровно_проверенные_байты(tmp_path: Path) -> None:
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    encoded = encode_jpeg(image)
    path = tmp_path / "crop.jpg"

    write_jpeg_atomic(path, encoded)

    assert path.read_bytes() == encoded.data
    assert np.array_equal(_декодировать(path.read_bytes()), encoded.decoded)


def test_audit_переносится_только_для_равных_наблюдаемых_пикселей(tmp_path: Path) -> None:
    encoded = encode_jpeg(np.full((40, 60, 3), 127, np.uint8))
    path = tmp_path / "old.jpg"
    write_jpeg_atomic(path, encoded)

    assert observed_pixels_equal(path, encoded)
    changed = encode_jpeg(np.full((40, 60, 3), 128, np.uint8))
    assert not observed_pixels_equal(path, changed)
    assert not observed_pixels_equal(tmp_path / "missing.jpg", encoded)


def test_пустой_старый_jpeg_сбрасывает_audit(tmp_path: Path) -> None:
    path = tmp_path / "old.jpg"
    path.write_bytes(b"")

    encoded = encode_jpeg(np.full((40, 60, 3), 127, np.uint8))

    assert not observed_pixels_equal(path, encoded)


def test_битый_старый_jpeg_сбрасывает_audit(tmp_path: Path) -> None:
    path = tmp_path / "old.jpg"
    path.write_bytes(b"not a jpeg")

    encoded = encode_jpeg(np.full((40, 60, 3), 127, np.uint8))

    assert not observed_pixels_equal(path, encoded)


def test_неудача_кодирования_не_возвращает_непроверенный_буфер(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, None))

    with pytest.raises(RuntimeError, match="закодировать JPEG"):
        encode_jpeg(np.zeros((20, 30, 3), dtype=np.uint8))


def test_неудача_декодирования_не_возвращает_буфер(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imdecode", lambda *_: None)

    with pytest.raises(RuntimeError, match="декодировать JPEG"):
        encode_jpeg(np.zeros((20, 30, 3), dtype=np.uint8))


def test_неудача_замены_удаляет_временный_файл(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "crop.jpg"
    path.write_bytes(b"old")
    encoded = encode_jpeg(np.full((20, 30, 3), 127, np.uint8))
    настоящий_replace = Path.replace

    def сломать_replace(source: Path, target: Path) -> Path:
        if source.suffix == ".tmp":
            raise OSError("replace failed")
        return настоящий_replace(source, target)

    monkeypatch.setattr(Path, "replace", сломать_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_jpeg_atomic(path, encoded)

    assert path.read_bytes() == b"old"
    assert not path.with_suffix(".jpg.tmp").exists()
