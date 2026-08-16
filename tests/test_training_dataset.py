"""Тесты датасета кропов (M-1.2, §4.1 дизайна). Требуют torch."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import cv2

from training.dataset import CropDataset, build_transforms, make_loader
from training.samples import Sample

СРЕДНЕЕ = (0.485, 0.456, 0.406)
ОТКЛОНЕНИЕ = (0.229, 0.224, 0.225)


def кроп(tmp_path: Path, имя: str = "stanford-00001.jpg") -> Path:
    файл = tmp_path / имя
    картинка = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    cv2.imwrite(str(файл), картинка)
    return файл


def test_val_трансформы_дают_тензор_нужной_формы(tmp_path: Path) -> None:
    образцы = [Sample(path=кроп(tmp_path), class_idx=3, is_multi_car=False)]
    датасет = CropDataset(образцы, build_transforms(320, train=False, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ))
    тензор, метка = датасет[0]
    assert тензор.shape == (3, 320, 320)
    assert тензор.dtype.is_floating_point
    assert метка == 3


def test_val_трансформы_детерминированы(tmp_path: Path) -> None:
    образцы = [Sample(path=кроп(tmp_path), class_idx=0, is_multi_car=False)]
    трансформы = build_transforms(320, train=False, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ)
    датасет = CropDataset(образцы, трансформы)
    первый, _ = датасет[0]
    второй, _ = датасет[0]
    assert bool((первый == второй).all())


def test_train_трансформы_меняют_картинку(tmp_path: Path) -> None:
    образцы = [Sample(path=кроп(tmp_path), class_idx=0, is_multi_car=False)]
    датасет = CropDataset(образцы, build_transforms(320, train=True, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ))
    выходы = [датасет[0][0] for _ in range(8)]
    различия = [bool((выходы[0] != другой).any()) for другой in выходы[1:]]
    assert any(различия), "аугментации не применились ни разу за восемь попыток"


def test_загрузчик_собирает_батч(tmp_path: Path) -> None:
    образцы = [
        Sample(
            path=кроп(tmp_path, f"stanford-{номер:05d}.jpg"),
            class_idx=номер,
            is_multi_car=False,
        )
        for номер in range(4)
    ]
    загрузчик = make_loader(
        образцы,
        build_transforms(64, train=False, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    картинки, метки = next(iter(загрузчик))
    assert картинки.shape == (2, 3, 64, 64)
    assert метки.tolist() == [0, 1]
