"""Пути к каталогу данных.

Единственный модуль, знающий структуру каталога данных. Остальной код спрашивает
пути здесь и никогда не собирает их сам: данные лежат в файловой системе
WSL, а репозиторий на диске Windows (§16.1), и захардкоженный путь
сломается при первом же переносе.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / "carlens-data"

_SUBDIRS = ("raw", "crops", "splits", "runs")


def data_root() -> Path:
    """Корень каталога данных из `CARLENS_DATA_ROOT` или умолчание."""
    raw = os.environ.get("CARLENS_DATA_ROOT")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_DATA_ROOT


def raw_dir() -> Path:
    """Исходные скачанные фотографии."""
    return data_root() / "raw"


def crops_dir() -> Path:
    """Кропы автомобилей, нарезанные детектором."""
    return data_root() / "crops"


def splits_dir() -> Path:
    """Parquet-файлы сплитов по схеме §13.1."""
    return data_root() / "splits"


def runs_dir() -> Path:
    """Карточки прогонов обучения."""
    return data_root() / "runs"


def harvest_dir() -> Path:
    """Кэш оригиналов, кропы и состояние харвеста (§8 дизайна M0-harvest)."""
    return data_root() / "harvest"


def ensure_dirs() -> None:
    """Создать всё дерево каталогов данных, если его нет."""
    root = data_root()
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
