"""Смоук-прогон обучения (M-1.2, §9 п.12 дизайна). Требует torch."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

import cv2
import torch

from pipeline.schema import COLUMNS, write_split

КЛАССЫ = {"acura__rl": 0, "acura__tl": 1, "bmw__x5": 2}


def каталог(tmp_path: Path) -> Path:
    файл = tmp_path / "catalog.yaml"
    файл.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "brands": {
                    "Acura": {
                        "models": {"rl": {"display": "Acura RL"}, "tl": {"display": "Acura TL"}}
                    },
                    "BMW": {"models": {"x5": {"display": "BMW X5"}}},
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return файл


def конфиг(tmp_path: Path) -> Path:
    файл = tmp_path / "config.yaml"
    файл.write_text(
        yaml.safe_dump(
            {
                "backbone": "mobilenetv4_conv_medium.e500_r256_in1k",
                "embedding_dim": 32,
                "img_size": 64,
                "batch_size": 2,
                "epochs": 2,
                "optimizer": "AdamW",
                "lr": 1.0e-3,
                "weight_decay": 0.05,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "arcface": {
                    "s": 32.0,
                    "margin_start": 0.2,
                    "margin_end": 0.5,
                    "margin_warmup_epochs": 1,
                },
                "amp": "bf16",
                "ema_decay": 0.9998,
                "grad_clip": 1.0,
                "backbone_lr_mult": 0.1,
                "backbone_lr_mult_epochs": 1,
                "num_workers": 0,
                "seed": 1337,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return файл


def данные(корень: Path) -> None:
    """20 синтетических кропов, разложенных по train/val/test."""
    кропы = корень / "crops" / "stanford"
    кропы.mkdir(parents=True)
    сплиты = корень / "splits"
    сплиты.mkdir(parents=True)
    генератор = np.random.default_rng(0)
    номер = 0
    for имя, сколько in (("train", 12), ("val", 4), ("test", 4)):
        строки = []
        for _ in range(сколько):
            номер += 1
            файл = f"stanford-{номер:05d}.jpg"
            cv2.imwrite(
                str(кропы / файл),
                генератор.integers(0, 255, (80, 80, 3), dtype=np.uint8),
            )
            модель_id, индекс = list(КЛАССЫ.items())[номер % len(КЛАССЫ)]
            строки.append(
                {
                    "photo_id": f"0f8e4f70-aadc-4f70-9f8e-{номер:012d}",
                    "listing_id": f"stanford_{номер:06d}",
                    "source": "stanford",
                    "crop_path": файл,
                    "brand": модель_id.split("__")[0],
                    "model_id": модель_id,
                    "class_idx": индекс,
                    "body_style": "sedan",
                    "year": 2012,
                    "bbox": [1.0, 2.0, 30.0, 20.0],
                    "bbox_area_frac": 0.4,
                    "det_conf": 0.9,
                    "phash": 1234567890123456789 + номер,
                    "raw_model_string": модель_id,
                    "is_multi_car": False,
                }
            )
        write_split(pd.DataFrame(строки, columns=list(COLUMNS)), сплиты / f"{имя}.parquet")


def test_смоук_прогон_пишет_чекпоинт_и_карточку(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.train import main

    данные(tmp_path)
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    код = main(
        [
            "--config",
            str(конфиг(tmp_path)),
            "--catalog",
            str(каталог(tmp_path)),
            "--device",
            "cpu",
            "--no-pretrained",
        ]
    )
    assert код == 0

    прогоны = sorted((tmp_path / "runs").glob("m1_2_train_*"))
    assert len(прогоны) == 1
    прогон = прогоны[0]
    for имя in ("config.yaml", "manifest.json", "metrics.json", "report.md", "best.pt", "last.pt"):
        assert (прогон / имя).is_file(), f"нет артефакта {имя}"

    метрики = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))
    assert len(метрики["epochs"]) == 2
    assert метрики["epochs"][0]["loss"] > 0

    чекпоинт = torch.load(прогон / "best.pt", map_location="cpu", weights_only=False)
    for ключ in (
        "model_state",
        "ema_state",
        "head_state",
        "class_index",
        "img_size",
        "mean",
        "std",
        "config",
        "epoch",
        "metrics",
        "git_sha",
    ):
        assert ключ in чекпоинт, f"в чекпоинте нет поля {ключ}"
    assert чекпоинт["class_index"] == КЛАССЫ
