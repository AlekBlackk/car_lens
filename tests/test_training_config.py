"""Тесты конфига прогона обучения (M-1.2, §6 дизайна)."""

from pathlib import Path

import pytest
import yaml

from training.config import SKELETON_CONFIG_PATH, ConfigError, load_config

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
