"""Тесты датасета кропов (M-1.2, §4.1 дизайна). Требуют torch."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import cv2
from torch import Tensor

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
    """Две независимо собранные val-цепочки дают побитово один тензор.

    Дважды дёрнуть один `датасет[0]` недостаточно: у val-цепочки нет ни одного
    случайного трансформа, но тот же результат дал бы и случайный конвейер с
    зафиксированным сидом. Проверять надо именно два разных `A.Compose` без
    `seed` — тогда совпадение говорит про отсутствие случайности, а не про сид.
    """
    образец = [Sample(path=кроп(tmp_path), class_idx=0, is_multi_car=False)]
    первый, _ = CropDataset(
        образец, build_transforms(320, train=False, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ)
    )[0]
    второй, _ = CropDataset(
        образец, build_transforms(320, train=False, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ)
    )[0]
    assert bool((первый == второй).all())


def test_train_трансформы_меняют_картинку(tmp_path: Path) -> None:
    образцы = [Sample(path=кроп(tmp_path), class_idx=0, is_multi_car=False)]
    датасет = CropDataset(образцы, build_transforms(320, train=True, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ))
    выходы = [датасет[0][0] for _ in range(8)]
    различия = [bool((выходы[0] != другой).any()) for другой in выходы[1:]]
    assert any(различия), "аугментации не применились ни разу за восемь попыток"


def test_train_трансформы_воспроизводимы_по_сиду(tmp_path: Path) -> None:
    """Один сид — одна последовательность аугментаций, другой сид — другая."""
    образцы = [Sample(path=кроп(tmp_path), class_idx=0, is_multi_car=False)]

    def последовательность(сид: int) -> list[Tensor]:
        датасет = CropDataset(
            образцы, build_transforms(64, train=True, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ, seed=сид)
        )
        return [датасет[0][0] for _ in range(4)]

    первая = последовательность(1337)
    вторая = последовательность(1337)
    другая = последовательность(7331)
    assert all(bool((левый == правый).all()) for левый, правый in zip(первая, вторая, strict=True))
    assert any(bool((левый != правый).any()) for левый, правый in zip(первая, другая, strict=True))


def test_воркеры_аугментируют_по_разному_и_воспроизводимо(tmp_path: Path) -> None:
    """Один и тот же кроп в двух воркерах — разные аугментации, при повторе те же.

    Fork-копии загрузчика получают один и тот же объект `A.Compose`, поэтому без
    пересева в `worker_init_fn` все воркеры идут одной последовательностью
    параметров аугментаций: восемь воркеров дают не восемь разных картинок, а
    восемь копий одной.
    """
    один_и_тот_же = кроп(tmp_path)
    образцы = [
        Sample(path=один_и_тот_же, class_idx=номер, is_multi_car=False) for номер in range(4)
    ]

    def прогон() -> list[Tensor]:
        загрузчик = make_loader(
            образцы,
            build_transforms(64, train=True, mean=СРЕДНЕЕ, std=ОТКЛОНЕНИЕ, seed=1337),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            seed=1337,
        )
        return [картинки for картинки, _ in загрузчик]

    первый = прогон()
    второй = прогон()
    assert bool((первый[0] != первый[1]).any()), "два воркера дали одну и ту же аугментацию"
    assert all(bool((левый == правый).all()) for левый, правый in zip(первый, второй, strict=True))


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
