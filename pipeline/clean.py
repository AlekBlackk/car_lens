"""Перцептивный хеш и дедуп фотографий (§3.4).

Хеш считается по **кропу**, а не по исходному кадру (решение Р-11).
Причина: дедуп должен ловить одну и ту же машину, снятую дважды, а
одинаковый фон дилерского салона на двух разных машинах ближе по хешу
кадра, чем две фотографии одной машины на разном фоне.

`imagehash` возвращает 64 бита без знака, parquet хранит их в int64 с той
же битовой картиной (§13.1), поэтому значение сворачивается в знаковый
диапазон здесь, а не размазывается по вызывающему коду.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import imagehash
import numpy as np
from PIL import Image

from pipeline.hashing import HASH_BITS, close_pairs

DEFAULT_PHASH_HAMMING = 6

_ЗНАКОВЫЙ_ПОТОЛОК = 1 << (HASH_BITS - 1)
_ПОЛНЫЙ = 1 << HASH_BITS


def phash_of(image: np.ndarray) -> int:
    """Перцептивный хеш изображения как знаковое 64-битное целое.

    На входе BGR — то, что отдаёт `cv2.imread` и что режет `crop_image`.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    беззнаковый = int(str(imagehash.phash(Image.fromarray(rgb))), 16)
    return беззнаковый - _ПОЛНЫЙ if беззнаковый >= _ЗНАКОВЫЙ_ПОТОЛОК else беззнаковый


def duplicate_indices(hashes: Sequence[int], max_distance: int = DEFAULT_PHASH_HAMMING) -> set[int]:
    """Индексы записей-дублей. Из каждой группы остаётся самый ранний.

    Группа собирается транзитивно: если A близок к B, а B к C, то из
    выборки уходят и B, и C, даже когда сами A и C за порогом. Иначе на
    границе порога в данных остались бы две почти одинаковые фотографии,
    и смысл дедупа пропал бы.
    """
    лишние: set[int] = set()
    for _, второй in close_pairs(hashes, max_distance):
        лишние.add(второй)
    return лишние
