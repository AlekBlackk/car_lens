"""Датасет кропов и аугментации (§4.2).

Разделение с `training.samples` не косметическое: там чистое чтение parquet,
которое гоняется в CI без torch, здесь — обёртка, которой нужны torch,
albumentations и cv2. Импорт этого модуля в CI упал бы.

Для валидации аугментаций нет — только `Resize` и нормализация: метрика
должна мериться на одном и том же изображении, иначе её нельзя сравнивать
между эпохами.
"""

from __future__ import annotations

import albumentations as A  # noqa: N812
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from training.samples import DatasetError, Sample


def build_transforms(
    img_size: int,
    *,
    train: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    seed: int | None = None,
) -> A.Compose:
    """Собрать конвейер преобразований. Аугментации §4.2 — только при `train`.

    `seed` идёт в `A.Compose`, а не в глобальные `random`/`numpy`: у каждого
    трансформа albumentations 2.x свои `py_random` и `random_generator`, и от
    энтропии ОС они не отвязываются ни `random.seed`, ни `np.random.seed`.
    Без `seed` два прогона под одним конфигом расходятся по аугментациям.
    """
    if not train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            seed=seed,
        )
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.55, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.Perspective(scale=(0.02, 0.09), p=0.4),
            A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-8, 8), p=0.5),
            A.OneOf(
                [A.MotionBlur(blur_limit=7), A.GaussianBlur(blur_limit=5), A.Defocus()], p=0.35
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.OneOf([A.ISONoise(), A.GaussNoise()], p=0.3),
            A.ImageCompression(quality_range=(35, 90), p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 4), hole_height_range=(16, 48), hole_width_range=(16, 48), p=0.3
            ),
            A.RandomShadow(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        seed=seed,
    )


class CropDataset(Dataset[tuple[Tensor, int]]):  # type: ignore[misc]
    """Кроп с диска, BGR→RGB, преобразования, метка из parquet."""

    def __init__(self, samples: list[Sample], transforms: A.Compose) -> None:
        self._samples = samples
        self._transforms = transforms

    def reseed(self, seed: int) -> None:
        """Пересеять RNG аугментаций у этой копии датасета."""
        self._transforms.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        образец = self._samples[index]
        картинка = cv2.imread(str(образец.path), cv2.IMREAD_COLOR)
        if картинка is None:
            raise DatasetError(f"не читается кроп {образец.path}")
        картинка = cv2.cvtColor(картинка, cv2.COLOR_BGR2RGB)
        тензор: Tensor = self._transforms(image=картинка)["image"]
        return тензор, образец.class_idx


def _reseed_worker(worker_id: int) -> None:
    """Дать воркеру свою последовательность параметров аугментаций.

    Воркеры получают копию датасета вместе с уже засеянным `A.Compose`, то есть
    все идут одними и теми же параметрами: при `num_workers: 8` вместо восьми
    разных аугментаций одной картинки выходит восемь одинаковых. Сид воркера
    torch выводит из генератора загрузчика, поэтому пересев не ломает
    воспроизводимость — он её и обеспечивает.
    """
    информация = get_worker_info()
    if информация is None:  # pragma: no cover — вне воркера функция не вызывается
        return
    датасет = информация.dataset
    if isinstance(датасет, CropDataset):
        датасет.reseed(int(информация.seed) % 2**31)


def make_loader(
    samples: list[Sample],
    transforms: A.Compose,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader[tuple[Tensor, int]]:
    """Загрузчик поверх `CropDataset`.

    `drop_last=shuffle`: в обучении последний неполный батч ломает
    `BatchNorm1d` при единственном элементе, в оценке терять данные нельзя.

    `seed` засевает генератор загрузчика: от него зависят и порядок перемешивания,
    и сиды воркеров, из которых `_reseed_worker` выводит аугментации. Без `seed`
    порядок и аугментации берутся из глобального состояния и от прогона к прогону
    меняются.
    """
    генератор: torch.Generator | None = None
    if seed is not None:
        генератор = torch.Generator()
        генератор.manual_seed(seed)
    return DataLoader(
        CropDataset(samples, transforms),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        generator=генератор,
        worker_init_fn=_reseed_worker if seed is not None else None,
    )
