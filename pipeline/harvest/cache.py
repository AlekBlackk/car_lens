"""Контентно-адресуемый кэш оригиналов и декодирование с учётом EXIF.

В кэше лежит **оригинал** — файл ровно в том виде, в каком он уже опубликован публично
под свободной лицензией. Редакция применяется на выходе, в кроп (§8 дизайна): смена
детектора или порогов обязана пересчитываться по кэшу, а не перекачиванием десятков
тысяч файлов.

Адресация по SHA256 содержимого даёт три вещи разом: файл, найденный по двум запросам,
скачивается однажды; повторный прогон видит его на диске; переименование на стороне
хранилища ничего не ломает.

**EXIF-ориентация применяется здесь и только здесь.** `cv2.imdecode` тег `Orientation`
игнорирует, а часть файлов Commons залита боком: такой кадр теряет машину на YOLO-гейте
и уводит класс в дефицит по причине, которой нет.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def digest(data: bytes) -> str:
    """SHA256 содержимого — он же имя файла в кэше."""
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes) -> np.ndarray:
    """Декодировать байты в BGR-массив с применённой EXIF-ориентацией."""
    картинка = Image.open(io.BytesIO(data))
    выпрямленная = ImageOps.exif_transpose(картинка)
    rgb = np.asarray((выпрямленная or картинка).convert("RGB"))
    # Единственный переход RGB → BGR в модуле: дальше по конвейеру всё в BGR.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cache_path(root: Path, source: str, sha: str, ext: str) -> Path:
    """Путь файла в кэше. Два символа хеша каталогом — чтобы не класть 30 тысяч в один."""
    return root / "raw" / source / sha[:2] / f"{sha}.{ext}"


def crop_path(root: Path, class_id: str, sha: str) -> Path:
    """Путь редактированного кропа. Имя то же, что у оригинала, — связь без отдельного поля."""
    return root / "crops" / class_id / f"{sha}.jpg"


def store(root: Path, source: str, data: bytes, ext: str) -> Path:
    """Положить файл в кэш атомарно. Повторный вызов ничего не перезаписывает."""
    путь = cache_path(root, source, digest(data), ext)
    if путь.exists():
        return путь
    путь.parent.mkdir(parents=True, exist_ok=True)
    временный = путь.with_suffix(путь.suffix + ".tmp")
    временный.write_bytes(data)
    временный.replace(путь)
    return путь
