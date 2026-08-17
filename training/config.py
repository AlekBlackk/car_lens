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
        embedding_dim=_целое(path, "embedding_dim", остальное["embedding_dim"]),
        img_size=_целое(path, "img_size", остальное["img_size"]),
        batch_size=_целое(path, "batch_size", остальное["batch_size"]),
        epochs=_целое(path, "epochs", остальное["epochs"]),
        optimizer=str(остальное["optimizer"]),
        lr=_дробное(path, "lr", остальное["lr"]),
        weight_decay=_дробное(path, "weight_decay", остальное["weight_decay"]),
        scheduler=str(остальное["scheduler"]),
        warmup_epochs=_целое(path, "warmup_epochs", остальное["warmup_epochs"]),
        arcface=arcface,
        amp=str(остальное["amp"]),
        ema_decay=_дробное(path, "ema_decay", остальное["ema_decay"]),
        grad_clip=_дробное(path, "grad_clip", остальное["grad_clip"]),
        backbone_lr_mult=_дробное(path, "backbone_lr_mult", остальное["backbone_lr_mult"]),
        backbone_lr_mult_epochs=_целое(
            path, "backbone_lr_mult_epochs", остальное["backbone_lr_mult_epochs"]
        ),
        num_workers=_целое(path, "num_workers", остальное["num_workers"]),
        seed=_целое(path, "seed", остальное["seed"]),
    )
    _проверить_значения(path, конфиг)
    return конфиг


def validate_epoch_budgets(config: TrainConfig, *, source: str) -> None:
    """Проверить, что каждый эпоховый бюджет строго меньше `epochs`.

    Функция публичная, потому что вызывается дважды: из `load_config` и из
    `train.py` после CLI-override'а `--epochs`, который перекрывает конфиг уже
    после валидации. Бюджет, не меньший числа эпох, означает, что режим, ради
    которого он введён (полный lr после разогрева, конечная маржа, снятый
    множитель бэкбона), целиком не наступит — прогон проверяет не то, что
    задумано, и молча.

    `source` — что показать в сообщении: путь конфига или имя флага.
    """
    бюджеты = (
        ("warmup_epochs", config.warmup_epochs),
        ("margin_warmup_epochs", config.arcface.margin_warmup_epochs),
        ("backbone_lr_mult_epochs", config.backbone_lr_mult_epochs),
    )
    for имя, значение in бюджеты:
        if значение >= config.epochs:
            raise ConfigError(f"{source}: {имя}={значение} не меньше epochs={config.epochs}")


def _разобрать_arcface(path: Path, сырое: Any) -> ArcFaceConfig:
    if not isinstance(сырое, dict):
        raise ConfigError(f"{path}: arcface должен быть отображением")
    ожидаемые = {поле.name for поле in fields(ArcFaceConfig)}
    _сверить_ключи(path, set(сырое), ожидаемые, где="arcface")
    return ArcFaceConfig(
        s=_дробное(path, "arcface.s", сырое["s"]),
        margin_start=_дробное(path, "arcface.margin_start", сырое["margin_start"]),
        margin_end=_дробное(path, "arcface.margin_end", сырое["margin_end"]),
        margin_warmup_epochs=_целое(
            path, "arcface.margin_warmup_epochs", сырое["margin_warmup_epochs"]
        ),
    )


def _целое(path: Path, ключ: str, значение: Any) -> int:
    """Целочисленное поле. Строка, дробное и `bool` отвергаются.

    `int()` вместо этой проверки давал `epochs: 60.9` → 60 и `batch_size: "48"`
    → 48, то есть ровно то молчаливое приведение, от которого докстринг модуля
    обещает защиту, а на мусоре (`epochs: "шестьдесят"`) пробивал наружу голый
    `ValueError` мимо `ConfigError`. `bool` отвергается отдельно, потому что в
    Python он подкласс `int`, и `epochs: yes` иначе прошло бы как 1.
    """
    if isinstance(значение, bool) or not isinstance(значение, int | float):
        raise ConfigError(f"{path}: {ключ}={значение!r} должно быть целым числом")
    if isinstance(значение, float) and not значение.is_integer():
        raise ConfigError(f"{path}: {ключ}={значение!r} должно быть целым, дробное недопустимо")
    return int(значение)


def _дробное(path: Path, ключ: str, значение: Any) -> float:
    """Поле с плавающей точкой. Строка и `bool` отвергаются, целое годится.

    Целое пропускается сознательно: `s: 32` и `s: 32.0` — одно значение, а вот
    `lr: "3e-4"` из YAML приходит строкой, и `float()` съел бы её молча.
    """
    if isinstance(значение, bool) or not isinstance(значение, int | float):
        raise ConfigError(f"{path}: {ключ}={значение!r} должно быть числом")
    return float(значение)


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
    validate_epoch_budgets(конфиг, source=str(path))
