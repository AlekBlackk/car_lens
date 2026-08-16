"""Тесты перцептивного хеша и дедупа (§3.4)."""

import numpy as np

from pipeline.clean import DEFAULT_PHASH_HAMMING, duplicate_indices, phash_of
from pipeline.hashing import hamming


def градиент(сдвиг: int = 0) -> np.ndarray:
    """Картинка 64×64 BGR с диагональным градиентом."""
    оси = np.arange(64, dtype=np.int64)
    поле = ((оси[:, None] * 3 + оси[None, :] * 3 + сдвиг) % 256).astype(np.uint8)
    return np.stack([поле, поле, поле], axis=-1)


def test_хеш_помещается_в_int64() -> None:
    значение = phash_of(градиент())
    assert -(2**63) <= значение < 2**63


def test_одинаковые_картинки_дают_одинаковый_хеш() -> None:
    assert phash_of(градиент()) == phash_of(градиент())


def test_слегка_изменённая_картинка_остаётся_близкой() -> None:
    шум = градиент()
    шум[0, 0] = [255, 255, 255]
    assert hamming(phash_of(градиент()), phash_of(шум)) <= DEFAULT_PHASH_HAMMING


def test_разные_картинки_расходятся() -> None:
    полосы = np.zeros((64, 64, 3), dtype=np.uint8)
    полосы[::2] = 255
    assert hamming(phash_of(градиент()), phash_of(полосы)) > DEFAULT_PHASH_HAMMING


def test_дедуп_оставляет_первый_из_группы() -> None:
    assert duplicate_indices([7, 7, 7]) == {1, 2}


def test_дедуп_не_трогает_далёкие() -> None:
    assert duplicate_indices([0, (1 << 63) - 1]) == set()


def test_дедуп_на_пустом_входе() -> None:
    assert duplicate_indices([]) == set()


def test_дедуп_транзитивен_по_цепочке() -> None:
    """A близок к B, B близок к C, A от C далёк — удаляются оба хвоста.

    Иначе в выборке остались бы две почти одинаковые картинки, и весь
    смысл дедупа потерялся бы на границе порога.
    """
    a = 0
    b = 0b1111
    c = 0b11111111
    assert hamming(a, c) > DEFAULT_PHASH_HAMMING
    assert duplicate_indices([a, b, c]) == {1, 2}
