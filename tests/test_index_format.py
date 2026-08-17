"""Тесты формата §13.2."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from pipeline.index_format import (
    DTYPE_FLOAT32,
    HEADER_SIZE,
    IndexFormatError,
    read_index,
    verify_meta,
    write_index,
)

МЕТА = b'{"format_version": 1}'


def _центроиды(классов: int = 2, центроидов: int = 4, размерность: int = 8) -> np.ndarray:
    значения = np.arange(классов * центроидов * размерность, dtype=np.float32)
    return значения.reshape(классов, центроидов, размерность)


def test_раунд_трип(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    ожидаемые = _центроиды()
    write_index(файл, ожидаемые, meta_bytes=МЕТА)
    заголовок, прочитанные = read_index(файл)
    assert заголовок.num_classes == 2
    assert заголовок.centroids_per_class == 4
    assert заголовок.embedding_dim == 8
    assert заголовок.dtype == DTYPE_FLOAT32
    assert заголовок.normalized is True
    np.testing.assert_array_equal(прочитанные, ожидаемые)


def test_поля_заголовка_лежат_по_смещениям_спеки(tmp_path: Path) -> None:
    """Проверка по байтам, а не по структуре: перестановка полей обязана ронять тест."""
    файл = tmp_path / "catalog_index.bin"
    write_index(файл, _центроиды(классов=3, размерность=16), meta_bytes=МЕТА)
    сырое = файл.read_bytes()[:HEADER_SIZE]
    assert сырое[0:4] == b"CLIX"
    assert struct.unpack_from("<H", сырое, 4)[0] == 1
    assert struct.unpack_from("<H", сырое, 6)[0] == 16
    assert struct.unpack_from("<I", сырое, 8)[0] == 3
    assert struct.unpack_from("<I", сырое, 12)[0] == 4
    assert сырое[16] == DTYPE_FLOAT32
    assert сырое[17] == 1
    assert сырое[18:20] == b"\x00\x00"
    assert сырое[20:52] == hashlib.sha256(МЕТА).digest()
    assert сырое[52:64] == b"\x00" * 12


def test_порядок_тела_class_major(tmp_path: Path) -> None:
    """Значение кодирует свои координаты: (класс, центроид, измерение)."""
    файл = tmp_path / "catalog_index.bin"
    исходные = np.zeros((2, 4, 8), dtype=np.float32)
    for c in range(2):
        for k in range(4):
            for d in range(8):
                исходные[c, k, d] = c * 100 + k * 10 + d
    write_index(файл, исходные, meta_bytes=МЕТА)
    тело = np.frombuffer(файл.read_bytes()[HEADER_SIZE:], dtype="<f4")
    # Первые восемь значений — класс 0, центроид 0.
    np.testing.assert_array_equal(тело[:8], np.arange(8, dtype=np.float32))
    # Следующие восемь — класс 0, центроид 1.
    np.testing.assert_array_equal(тело[8:16], np.arange(8, dtype=np.float32) + 10)
    # Смещение центроида 2 класса 1 по формуле §13.2.
    начало = (1 * 4 + 2) * 8
    np.testing.assert_array_equal(тело[начало : начало + 8], np.arange(8, dtype=np.float32) + 120)


def test_файл_короче_заголовка(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    файл.write_bytes(b"CLIX" + b"\x00" * 10)
    with pytest.raises(IndexFormatError, match="короче заголовка"):
        read_index(файл)


def test_чужой_magic(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    write_index(файл, _центроиды(), meta_bytes=МЕТА)
    сырое = bytearray(файл.read_bytes())
    сырое[0:4] = b"CLXX"
    файл.write_bytes(bytes(сырое))
    with pytest.raises(IndexFormatError, match="magic"):
        read_index(файл)


def test_чужая_версия_формата(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    write_index(файл, _центроиды(), meta_bytes=МЕТА)
    сырое = bytearray(файл.read_bytes())
    struct.pack_into("<H", сырое, 4, 2)
    файл.write_bytes(bytes(сырое))
    with pytest.raises(IndexFormatError, match="format_version"):
        read_index(файл)


def test_неизвестный_dtype(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    write_index(файл, _центроиды(), meta_bytes=МЕТА)
    сырое = bytearray(файл.read_bytes())
    сырое[16] = 7
    файл.write_bytes(bytes(сырое))
    with pytest.raises(IndexFormatError, match="dtype"):
        read_index(файл)


def test_ненулевой_reserved(tmp_path: Path) -> None:
    """Ненулевой reserved значит чужую версию формата, а не наш файл."""
    файл = tmp_path / "catalog_index.bin"
    write_index(файл, _центроиды(), meta_bytes=МЕТА)
    сырое = bytearray(файл.read_bytes())
    сырое[18] = 1
    файл.write_bytes(bytes(сырое))
    with pytest.raises(IndexFormatError, match="reserved"):
        read_index(файл)


def test_длина_файла_не_сходится_с_заголовком(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    write_index(файл, _центроиды(), meta_bytes=МЕТА)
    файл.write_bytes(файл.read_bytes()[:-4])
    with pytest.raises(IndexFormatError, match="длина"):
        read_index(файл)


def test_verify_meta_ловит_чужую_мету(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    заголовок = write_index(файл, _центроиды(), meta_bytes=МЕТА)
    verify_meta(заголовок, МЕТА)
    with pytest.raises(IndexFormatError, match="sha256"):
        verify_meta(заголовок, МЕТА + b" ")


def test_форма_центроидов_проверяется(tmp_path: Path) -> None:
    файл = tmp_path / "catalog_index.bin"
    with pytest.raises(IndexFormatError, match="три измерения"):
        write_index(файл, np.zeros((2, 8), dtype=np.float32), meta_bytes=МЕТА)
