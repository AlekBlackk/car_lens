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
# Базовый lr конфига ниже: тест расписания сверяется с ним, а не с литералом,
# чтобы не стать пустым, если конфиг теста однажды поправят.
БАЗОВЫЙ_LR = 1.0e-3


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
                "lr": БАЗОВЫЙ_LR,
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
    путь_конфига = конфиг(tmp_path)

    код = main(
        [
            "--config",
            str(путь_конфига),
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

    # Копия конфига в каталоге прогона обязана быть побайтной: пересборка YAML
    # из `TrainConfig` дала бы файл с другим порядком ключей и форматом чисел,
    # и сравнить два прогона глазами стало бы труднее.
    assert (прогон / "config.yaml").read_bytes() == путь_конфига.read_bytes()

    метрики = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))
    assert len(метрики["epochs"]) == 2
    assert метрики["epochs"][0]["loss"] > 0
    # EMA оценивается один раз в конце прогона: без этих чисел решать «сырые
    # веса или EMA» в M-1.3 нечем.
    for ключ in ("val", "test", "val_ema", "test_ema"):
        assert "top1_model" in метрики[ключ], f"в metrics.json нет {ключ}.top1_model"

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
    # `best.pt` пишется по улучшению, а не всегда: в нём должна лежать эпоха с
    # максимальным `val top1_model` (при равенстве — первая из них, отбор идёт
    # по строгому `>`), а не последняя.
    журнал = метрики["epochs"]
    лучшая = max(range(len(журнал)), key=lambda номер: журнал[номер]["val_top1_model"])
    assert чекпоинт["epoch"] == лучшая


def test_лимит_батчей_учитывается_в_знаменателе_расписания(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--limit-batches` должен резать и число шагов, и знаменатель расписания.

    В конфиге теста 12 кропов и batch 2, то есть 6 шагов в полной эпохе. Если
    знаменатель расписания остаётся равным 6, а фактически делается 2 шага, то
    прогон целиком сидит в разогреве: `warmup_epochs=1` растянут на 6 шагов, а
    случится их всего 2 — lr нулевой эпохи не доходит до `base_lr`, и косинус
    не начинается.
    """
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
            "--limit-batches",
            "2",
        ]
    )
    assert код == 0

    прогон = sorted((tmp_path / "runs").glob("m1_2_train_*"))[0]
    журнал = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))["epochs"]
    assert len(журнал) == 2
    # `warmup_epochs=1`: последний шаг нулевой эпохи — последний шаг разогрева,
    # а разогрев по `lr_at` заканчивается ровно на `base_lr`.
    assert журнал[0]["lr"] == pytest.approx(БАЗОВЫЙ_LR)
    # Дальше идёт косинус вниз, поэтому эпоха 1 обязана уйти ниже `base_lr`.
    assert журнал[1]["lr"] < БАЗОВЫЙ_LR
