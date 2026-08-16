"""Чтение сплита §13.1 в список образцов для обучения.

Модуль не импортирует torch намеренно (Р-14): проверки ниже гоняются в CI,
где стоит только группа `dev`. Всё, что требует torch, живёт в
`training.dataset` и импортирует этот модуль, а не наоборот.

Битые строки не пропускаются молча. Датасет, который сам себя уменьшил на
десяток строк, даёт метрику, посчитанную не на том, на чём кажется, и
заметить это по числам невозможно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.schema import read_split


@dataclass(frozen=True, slots=True)
class Sample:
    """Один кроп с меткой: путь на диске, индекс класса, признак §3.5."""

    path: Path
    class_idx: int
    is_multi_car: bool


class DatasetError(ValueError):
    """Сплит не годится для обучения."""


def read_samples(
    parquet: Path,
    crops_root: Path,
    *,
    num_classes: int,
    skip_multi_car: bool,
) -> list[Sample]:
    """Прочитать сплит и превратить в образцы.

    `crops_root` — каталог, относительно которого раскрывается `crop_path`
    (`crops_dir() / "stanford"`). Он параметр, а не вызов `pipeline.paths`
    внутри, чтобы функция тестировалась на `tmp_path`.

    Падает с `DatasetError`, если `class_idx` пуст, выходит за границу
    `num_classes` или файл кропа отсутствует.
    """
    кадр = read_split(parquet)
    if skip_multi_car:
        кадр = кадр[~кадр["is_multi_car"].astype(bool)]

    пустых = int(кадр["class_idx"].isna().sum())
    if пустых:
        raise DatasetError(f"{parquet}: {пустых} строк без class_idx")

    образцы: list[Sample] = []
    for сырой_путь, сырой_индекс, многомашинный in zip(
        кадр["crop_path"], кадр["class_idx"], кадр["is_multi_car"], strict=True
    ):
        индекс = int(сырой_индекс)
        if not 0 <= индекс < num_classes:
            raise DatasetError(f"{parquet}: class_idx {индекс} вне диапазона 0..{num_classes - 1}")
        файл = crops_root / str(сырой_путь)
        if not файл.is_file():
            raise DatasetError(f"{parquet}: нет файла кропа {файл}")
        образцы.append(Sample(path=файл, class_idx=индекс, is_multi_car=bool(многомашинный)))
    return образцы
