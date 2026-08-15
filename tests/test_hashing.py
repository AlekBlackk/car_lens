"""Тесты битовых операций над перцептивными хешами."""

import pytest

from pipeline.hashing import HASH_BITS, close_pairs, hamming


def test_хэмминг_на_равных_нулевой() -> None:
    assert hamming(0b1010, 0b1010) == 0


def test_хэмминг_считает_биты() -> None:
    assert hamming(0b1010, 0b1011) == 1
    assert hamming(0b0000, 0b1111) == 4


def test_хэмминг_смотрит_только_на_младшие_64_бита() -> None:
    """`phash` хранится в int64, и у отрицательных значений старшие биты бесконечны."""
    assert hamming(0, -1) == HASH_BITS


def test_близкие_пары_находятся() -> None:
    """Третий хеш лежит за порогом, хотя и попадает с остальными в общие корзины."""
    далёкий = 0xFF << 40
    assert hamming(0, далёкий) == 8
    хеши = [0, 0b111, далёкий]
    assert set(close_pairs(хеши, max_distance=6)) == {(0, 1)}


def test_пара_ровно_на_пороге_находится() -> None:
    """Шесть различающихся битов, разложенных по разным кускам, — самый тяжёлый случай."""
    далёкий = 0b1 | (1 << 9) | (1 << 18) | (1 << 27) | (1 << 36) | (1 << 45)
    assert hamming(0, далёкий) == 6
    assert set(close_pairs([0, далёкий], max_distance=6)) == {(0, 1)}


def test_пара_за_порогом_не_находится() -> None:
    далёкий = 0b1 | (1 << 9) | (1 << 18) | (1 << 27) | (1 << 36) | (1 << 45) | (1 << 54)
    assert hamming(0, далёкий) == 7
    assert set(close_pairs([0, далёкий], max_distance=6)) == set()


def test_отрицательные_хеши_обрабатываются() -> None:
    """int64 из parquet приходит со знаком; биты от этого не меняются."""
    assert set(close_pairs([-1, -2], max_distance=6)) == {(0, 1)}


def test_каждая_пара_выдаётся_один_раз() -> None:
    """Два одинаковых хеша попадают сразу в восемь общих корзин."""
    пары = list(close_pairs([7, 7, 7], max_distance=6))
    assert sorted(пары) == [(0, 1), (0, 2), (1, 2)]


def test_индексы_возрастают() -> None:
    for a, b in close_pairs([5, 5, 5], max_distance=6):
        assert a < b


def test_пустой_вход_даёт_пустой_выход() -> None:
    assert list(close_pairs([], max_distance=6)) == []


def test_слишком_большой_порог_отвергается() -> None:
    """При пороге от восьми битов раскладка по восьми кускам перестаёт быть точной."""
    with pytest.raises(ValueError, match="порог"):
        list(close_pairs([1, 2], max_distance=8))
