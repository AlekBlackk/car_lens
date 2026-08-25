"""Бинарный формат индекса эталонов `catalog_index.bin` (§13.2).

Единственный артефакт, пересекающий границу Python → TypeScript, поэтому формат
описан побайтово, а не структурой языка. Little-endian везде.

Модуль не импортирует torch: формат — самая ответственная часть вехи, и он обязан
проверяться в CI, где стоит только группа `dev`.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.catalog import CarClass

MAGIC = b"CLIX"
FORMAT_VERSION = 1
HEADER_SIZE = 64
DTYPE_FLOAT16 = 0
DTYPE_FLOAT32 = 1

# 4s magic, H version, H dim, I classes, I centroids, B dtype, B normalized,
# H reserved, 32s sha256 меты, 12s reserved. Ровно 64 байта.
_HEADER = struct.Struct("<4sHHIIBBH32s12s")

# Аннотация обязательна: без неё mypy выводит общий тип двух разных
# `np.dtype[...]` как `object`, и `itemsize` перестаёт существовать.
_DTYPES: dict[int, np.dtype[Any]] = {
    DTYPE_FLOAT16: np.dtype("<f2"),
    DTYPE_FLOAT32: np.dtype("<f4"),
}


class IndexFormatError(ValueError):
    """Файл индекса не соответствует §13.2."""


@dataclass(frozen=True, slots=True)
class IndexHeader:
    """Разобранный заголовок §13.2."""

    embedding_dim: int
    num_classes: int
    centroids_per_class: int
    dtype: int
    normalized: bool
    meta_sha256: bytes

    def body_size(self) -> int:
        """Размер тела в байтах по полям заголовка."""
        значений = self.num_classes * self.centroids_per_class * self.embedding_dim
        return значений * _DTYPES[self.dtype].itemsize


def write_index(
    path: Path,
    centroids: np.ndarray,
    *,
    meta_bytes: bytes,
    normalized: bool = True,
) -> IndexHeader:
    """Записать индекс. `centroids` — `(классы, центроиды, измерения)`, float32.

    `meta_bytes` — байты `catalog_meta.json` **как он записан на диск**: их sha256
    идёт в заголовок, и клиент сверяет прочитанное с диска, а не строку в памяти.
    """
    if centroids.ndim != 3:
        raise IndexFormatError(f"центроиды должны иметь три измерения, получено {centroids.ndim}")
    классов, центроидов, размерность = centroids.shape
    if классов == 0 or центроидов == 0 or размерность == 0:
        raise IndexFormatError(f"пустая форма центроидов: {centroids.shape}")
    заголовок = IndexHeader(
        embedding_dim=int(размерность),
        num_classes=int(классов),
        centroids_per_class=int(центроидов),
        dtype=DTYPE_FLOAT32,
        normalized=normalized,
        meta_sha256=hashlib.sha256(meta_bytes).digest(),
    )
    упакованный = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        заголовок.embedding_dim,
        заголовок.num_classes,
        заголовок.centroids_per_class,
        заголовок.dtype,
        int(заголовок.normalized),
        0,
        заголовок.meta_sha256,
        b"\x00" * 12,
    )
    тело = np.ascontiguousarray(centroids, dtype="<f4")
    path.write_bytes(упакованный + тело.tobytes())
    return заголовок


def read_index(path: Path) -> tuple[IndexHeader, np.ndarray]:
    """Прочитать индекс, проверив заголовок по §13.2.

    Каждая проверка падает своим сообщением: по общему «файл повреждён» нельзя
    отличить чужую версию формата от обрезанной записи.
    """
    сырое = path.read_bytes()
    if len(сырое) < HEADER_SIZE:
        raise IndexFormatError(f"{path}: файл короче заголовка ({len(сырое)} < {HEADER_SIZE})")
    (
        magic,
        версия,
        размерность,
        классов,
        центроидов,
        dtype,
        нормализован,
        резерв_короткий,
        хеш_меты,
        резерв_длинный,
    ) = _HEADER.unpack_from(сырое)
    if magic != MAGIC:
        raise IndexFormatError(f"{path}: magic {magic!r} вместо {MAGIC!r}")
    if версия != FORMAT_VERSION:
        raise IndexFormatError(f"{path}: format_version {версия}, поддерживается {FORMAT_VERSION}")
    if dtype not in _DTYPES:
        raise IndexFormatError(f"{path}: неизвестный dtype {dtype}")
    if резерв_короткий != 0 or резерв_длинный != b"\x00" * 12:
        raise IndexFormatError(f"{path}: reserved не нулевой — файл писала другая версия формата")
    заголовок = IndexHeader(
        embedding_dim=int(размерность),
        num_classes=int(классов),
        centroids_per_class=int(центроидов),
        dtype=int(dtype),
        normalized=bool(нормализован),
        meta_sha256=bytes(хеш_меты),
    )
    ожидалось = HEADER_SIZE + заголовок.body_size()
    if len(сырое) != ожидалось:
        raise IndexFormatError(f"{path}: длина {len(сырое)} против {ожидалось} по заголовку")
    тело = np.frombuffer(сырое, dtype=_DTYPES[заголовок.dtype], offset=HEADER_SIZE)
    форма = (заголовок.num_classes, заголовок.centroids_per_class, заголовок.embedding_dim)
    return заголовок, тело.reshape(форма).astype(np.float32)


def verify_meta(header: IndexHeader, meta_bytes: bytes) -> None:
    """Сверить sha256 меты с полем заголовка (§13.2, обязанность клиента)."""
    фактический = hashlib.sha256(meta_bytes).digest()
    if фактический != header.meta_sha256:
        raise IndexFormatError(
            f"sha256 меты {фактический.hex()} против {header.meta_sha256.hex()} в заголовке"
        )


def meta_json(
    classes: Sequence[CarClass],
    *,
    embedding_dim: int,
    centroids_per_class: int,
    model_weights_sha256: str,
    generated_at: str,
) -> bytes:
    """Собрать `catalog_meta.json` (§13.2). `classes` идут в порядке индексов классов.

    Запись описывает класс, а не модель: `model_id` в ней нет, состав
    группы лежит в `members[]`, а `brand` пуст у разнобрендовой группы.
    Поля `idx` нет намеренно — индексом класса служит позиция в массиве.

    Сериализация фиксирована — `ensure_ascii=False`, `indent=2`, UTF-8, как у
    `manifest.json` прогона обучения. Поменять её значит поменять sha256 в
    заголовке при том же содержимом, поэтому форма закреплена тестом.
    """
    if not classes:
        raise IndexFormatError("пустой список классов: индекс без классов не имеет смысла")
    документ = {
        "format_version": FORMAT_VERSION,
        "generated_at": generated_at,
        "model_weights_sha256": model_weights_sha256,
        "embedding_dim": embedding_dim,
        "centroids_per_class": centroids_per_class,
        "classes": [
            {
                "class_id": класс.class_id,
                "members": list(класс.members),
                "brand": класс.brand,
                "display": класс.display,
            }
            for класс in classes
        ],
    }
    return json.dumps(документ, ensure_ascii=False, indent=2).encode("utf-8")
