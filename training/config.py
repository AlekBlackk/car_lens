"""Конфиг прогона обучения и его валидация.

Проверка жёсткая, как у схемы данных: лишний ключ — такая же ошибка, как
пропущенный. Опечатка `epoch: 60` вместо `epochs: 60` иначе обернулась бы
часовым прогоном на значении по умолчанию, и заметить подмену можно было бы
только по времени прогона.

Модуль не импортирует torch (Р-14): проверки гоняются в CI.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

SKELETON_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "skeleton_stanford.yaml"

OPTIMIZERS = frozenset({"AdamW"})
SCHEDULERS = frozenset({"cosine"})
AMP_MODES = frozenset({"bf16"})


class ConfigError(ValueError):
    """Конфиг прогона не проходит валидацию."""


@dataclass(frozen=True, slots=True)
class ArcFaceConfig:
    """Параметры головы ArcFace и расписание маржи."""

    s: float
    margin_start: float
    margin_end: float
    margin_warmup_epochs: int


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Полный набор гиперпараметров прогона (§4.2)."""

    backbone: str
    embedding_dim: int
    img_size: int
    batch_size: int
    epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    arcface: ArcFaceConfig
    amp: str
    ema_decay: float
    grad_clip: float
    backbone_lr_mult: float
    backbone_lr_mult_epochs: int
    num_workers: int
    seed: int


def load_config(path: Path) -> TrainConfig:
    """Прочитать YAML и собрать `TrainConfig`, проверив ключи и значения."""
    сырое = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(сырое, dict):
        raise ConfigError(f"{path}: корень должен быть отображением")

    arcface = _разобрать_arcface(path, сырое.get("arcface"))
    остальное = {ключ: значение for ключ, значение in сырое.items() if ключ != "arcface"}
    ожидаемые = {поле.name for поле in fields(TrainConfig)} - {"arcface"}
    _сверить_ключи(path, set(остальное), ожидаемые)

    конфиг = TrainConfig(
        backbone=str(остальное["backbone"]),
        embedding_dim=int(остальное["embedding_dim"]),
        img_size=int(остальное["img_size"]),
        batch_size=int(остальное["batch_size"]),
        epochs=int(остальное["epochs"]),
        optimizer=str(остальное["optimizer"]),
        lr=float(остальное["lr"]),
        weight_decay=float(остальное["weight_decay"]),
        scheduler=str(остальное["scheduler"]),
        warmup_epochs=int(остальное["warmup_epochs"]),
        arcface=arcface,
        amp=str(остальное["amp"]),
        ema_decay=float(остальное["ema_decay"]),
        grad_clip=float(остальное["grad_clip"]),
        backbone_lr_mult=float(остальное["backbone_lr_mult"]),
        backbone_lr_mult_epochs=int(остальное["backbone_lr_mult_epochs"]),
        num_workers=int(остальное["num_workers"]),
        seed=int(остальное["seed"]),
    )
    _проверить_значения(path, конфиг)
    return конфиг


def _разобрать_arcface(path: Path, сырое: Any) -> ArcFaceConfig:
    if not isinstance(сырое, dict):
        raise ConfigError(f"{path}: arcface должен быть отображением")
    ожидаемые = {поле.name for поле in fields(ArcFaceConfig)}
    _сверить_ключи(path, set(сырое), ожидаемые, где="arcface")
    return ArcFaceConfig(
        s=float(сырое["s"]),
        margin_start=float(сырое["margin_start"]),
        margin_end=float(сырое["margin_end"]),
        margin_warmup_epochs=int(сырое["margin_warmup_epochs"]),
    )


def _сверить_ключи(path: Path, есть: set[str], ожидаемые: set[str], *, где: str = "") -> None:
    место = f"{path}: {где}" if где else f"{path}:"
    пропущены = sorted(ожидаемые - есть)
    if пропущены:
        raise ConfigError(f"{место} нет обязательных ключей: {', '.join(пропущены)}")
    лишние = sorted(есть - ожидаемые)
    if лишние:
        raise ConfigError(f"{место} лишние ключи: {', '.join(лишние)}")


def _проверить_значения(path: Path, конфиг: TrainConfig) -> None:
    if конфиг.optimizer not in OPTIMIZERS:
        raise ConfigError(f"{path}: optimizer={конфиг.optimizer!r}, допустим только AdamW")
    if конфиг.scheduler not in SCHEDULERS:
        raise ConfigError(f"{path}: scheduler={конфиг.scheduler!r}, допустим только cosine")
    if конфиг.amp not in AMP_MODES:
        raise ConfigError(f"{path}: amp={конфиг.amp!r}, допустим только bf16")
    if конфиг.epochs <= 0 or конфиг.batch_size <= 0 or конфиг.img_size <= 0:
        raise ConfigError(f"{path}: epochs, batch_size и img_size должны быть положительными")
    if конфиг.warmup_epochs >= конфиг.epochs:
        raise ConfigError(
            f"{path}: warmup_epochs={конфиг.warmup_epochs} не меньше epochs={конфиг.epochs}"
        )
