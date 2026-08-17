"""Тесты конфига прогона обучения (M-1.2, §6 дизайна)."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from training.config import (
    SKELETON_CONFIG_PATH,
    ConfigError,
    load_config,
    validate_epoch_budgets,
)

ЭТАЛОН: dict[str, object] = {
    "backbone": "mobilenetv4_conv_medium.e500_r256_in1k",
    "embedding_dim": 256,
    "img_size": 320,
    "batch_size": 128,
    "epochs": 60,
    "optimizer": "AdamW",
    "lr": 3.0e-4,
    "weight_decay": 0.05,
    "scheduler": "cosine",
    "warmup_epochs": 3,
    "arcface": {
        "s": 32.0,
        "margin_start": 0.2,
        "margin_end": 0.5,
        "margin_warmup_epochs": 10,
    },
    "amp": "bf16",
    "ema_decay": 0.9998,
    "grad_clip": 1.0,
    "backbone_lr_mult": 0.1,
    "backbone_lr_mult_epochs": 5,
    "num_workers": 8,
    "seed": 1337,
}


def записать(tmp_path: Path, данные: dict[str, object]) -> Path:
    файл = tmp_path / "config.yaml"
    файл.write_text(yaml.safe_dump(данные, sort_keys=False), encoding="utf-8")
    return файл


def test_эталонный_конфиг_читается(tmp_path: Path) -> None:
    конфиг = load_config(записать(tmp_path, ЭТАЛОН))
    assert конфиг.epochs == 60
    assert конфиг.img_size == 320
    assert конфиг.arcface.margin_end == 0.5
    assert конфиг.backbone_lr_mult_epochs == 5


def test_конфиг_скелета_в_репозитории_валиден() -> None:
    конфиг = load_config(SKELETON_CONFIG_PATH)
    assert конфиг.backbone == "mobilenetv4_conv_medium.e500_r256_in1k"
    assert конфиг.seed == 1337


def test_неизвестный_ключ_отвергается(tmp_path: Path) -> None:
    данные = dict(ЭТАЛОН)
    данные["epoch"] = 60
    with pytest.raises(ConfigError, match="лишние ключи"):
        load_config(записать(tmp_path, данные))


def test_пропущенный_ключ_отвергается(tmp_path: Path) -> None:
    данные = dict(ЭТАЛОН)
    del данные["epochs"]
    with pytest.raises(ConfigError, match="нет обязательных ключей"):
        load_config(записать(tmp_path, данные))


def test_неизвестный_ключ_внутри_arcface_отвергается(tmp_path: Path) -> None:
    данные = dict(ЭТАЛОН)
    данные["arcface"] = dict(ЭТАЛОН["arcface"], margin=0.5)  # type: ignore[call-overload]
    with pytest.raises(ConfigError, match="arcface"):
        load_config(записать(tmp_path, данные))


@pytest.mark.parametrize(
    ("ключ", "значение"),
    [("optimizer", "SGD"), ("scheduler", "step"), ("amp", "fp16")],
)
def test_перечислимые_поля_проверяются(tmp_path: Path, ключ: str, значение: str) -> None:
    данные = dict(ЭТАЛОН)
    данные[ключ] = значение
    with pytest.raises(ConfigError, match=ключ):
        load_config(записать(tmp_path, данные))


@pytest.mark.parametrize("бюджет", ["warmup_epochs", "backbone_lr_mult_epochs"])
def test_эпоховый_бюджет_не_меньше_epochs_отвергается(tmp_path: Path, бюджет: str) -> None:
    """Бюджет длиной во весь прогон означает, что режим целиком не наступит.

    `epochs=12` взято так, чтобы остальные два бюджета эталона (3 и 10) остались
    в допуске и падал ровно проверяемый.
    """
    данные = dict(ЭТАЛОН, epochs=12)
    данные[бюджет] = 12
    with pytest.raises(ConfigError, match=f"{бюджет}=12 не меньше epochs=12"):
        load_config(записать(tmp_path, данные))


def test_margin_warmup_epochs_не_меньше_epochs_отвергается(tmp_path: Path) -> None:
    секция = dict(ЭТАЛОН["arcface"], margin_warmup_epochs=12)  # type: ignore[call-overload]
    данные = dict(ЭТАЛОН, epochs=12, arcface=секция)
    with pytest.raises(ConfigError, match="margin_warmup_epochs=12 не меньше epochs=12"):
        load_config(записать(tmp_path, данные))


def test_override_epochs_перепроверяет_бюджеты(tmp_path: Path) -> None:
    """`--epochs` перекрывает конфиг после валидации — отсюда публичная функция.

    Сам флаг разбирается в `train.py` (там нужен torch), здесь проверяется
    функция, которую он вызывает.
    """
    конфиг = load_config(записать(tmp_path, ЭТАЛОН))
    validate_epoch_budgets(конфиг, source="конфиг")
    с_двумя_эпохами = replace(конфиг, epochs=2)
    with pytest.raises(ConfigError, match=r"--epochs: warmup_epochs=3 не меньше epochs=2"):
        validate_epoch_budgets(с_двумя_эпохами, source="--epochs")


@pytest.mark.parametrize(
    ("ключ", "значение"),
    [
        ("epochs", "шестьдесят"),  # мусор: раньше пробивал наружу голый ValueError
        ("epochs", 60.9),  # молчаливое усечение до 60
        ("batch_size", "48"),  # строка вместо числа
        ("num_workers", True),  # bool — подкласс int, раньше проходил как 1
        ("seed", None),  # ключ есть, значение пустое
        ("img_size", [320]),  # список
    ],
)
def test_целочисленные_поля_не_приводятся_молча(
    tmp_path: Path, ключ: str, значение: object
) -> None:
    данные = dict(ЭТАЛОН)
    данные[ключ] = значение
    with pytest.raises(ConfigError, match=ключ):
        load_config(записать(tmp_path, данные))


@pytest.mark.parametrize(
    ("ключ", "значение"),
    [("lr", "3e-4"), ("weight_decay", "0.05"), ("grad_clip", True)],
)
def test_дробные_поля_отвергают_строку_и_bool(tmp_path: Path, ключ: str, значение: object) -> None:
    данные = dict(ЭТАЛОН)
    данные[ключ] = значение
    with pytest.raises(ConfigError, match=ключ):
        load_config(записать(tmp_path, данные))


def test_целое_вместо_дробного_допустимо(tmp_path: Path) -> None:
    """`s: 32` и `s: 32.0` — одно значение, ужесточать до `float` не за что."""
    данные = dict(ЭТАЛОН)
    данные["arcface"] = dict(ЭТАЛОН["arcface"], s=32)  # type: ignore[call-overload]
    assert load_config(записать(tmp_path, данные)).arcface.s == pytest.approx(32.0)


@pytest.mark.parametrize(
    ("ключ", "значение"),
    [("margin_warmup_epochs", 10.5), ("margin_warmup_epochs", "десять"), ("s", "32")],
)
def test_поля_arcface_тоже_проверяются(tmp_path: Path, ключ: str, значение: object) -> None:
    данные = dict(ЭТАЛОН)
    данные["arcface"] = dict(ЭТАЛОН["arcface"], **{ключ: значение})  # type: ignore[call-overload]
    with pytest.raises(ConfigError, match=f"arcface.{ключ}"):
        load_config(записать(tmp_path, данные))
