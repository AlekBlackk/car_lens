"""Отбор representative dataset для PTQ (§4.3 дизайна M-1.4).

Только из `train`: калибровка по `val` или `test` подгоняла бы квантование под ту
же выборку, на которой потом меряется приёмка. Стратификация нужна потому, что
диапазоны активаций формирует разнообразие классов, а случайные 512 кропов из
почти двенадцати тысяч покрыли бы 171 класс как попало.

Модуль не импортирует torch: он решает, что подать на калибровку, а подаёт —
`export/to_tflite.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from training.samples import Sample


class CalibrationError(ValueError):
    """Калибровочную выборку собрать нельзя."""


@dataclass(frozen=True, slots=True)
class Calibration:
    """Отобранные кропы и то, как они распределились по классам."""

    samples: tuple[Sample, ...]
    per_class_counts: dict[str, int]
    requested_total: int
    seed: int


def select_calibration(
    samples: Sequence[Sample],
    *,
    class_index: Mapping[str, int],
    total: int = 512,
    seed: int = 1337,
) -> Calibration:
    """Отобрать `total` кропов стратифицированно по классам.

    `total` — цель, а не гарантия: класс, у которого кропов меньше квоты, отдаёт
    все свои, и соседи за него не добирают. Фактический итог тогда меньше цели и
    виден в `per_class_counts`.

    `class_index` передаётся явно, потому что по одному списку образцов класс без
    единого кропа неотличим от несуществующего.
    """
    классов = len(class_index)
    if классов == 0:
        raise CalibrationError("пустой class_index: калибровать нечего")
    if total < классов:
        raise CalibrationError(
            f"total {total} меньше числа классов {классов}: часть классов осталась бы без кропов"
        )
    по_индексу = {индекс: имя for имя, индекс in class_index.items()}
    if len(по_индексу) != классов:
        raise CalibrationError("индексы классов в class_index не уникальны")

    группы: dict[int, list[Sample]] = {индекс: [] for индекс in по_индексу}
    for образец in samples:
        свои = группы.get(образец.class_idx)
        if свои is None:
            raise CalibrationError(f"метка {образец.class_idx} вне class_index")
        свои.append(образец)

    пустые = sorted(по_индексу[индекс] for индекс, свои in группы.items() if not свои)
    if пустые:
        raise CalibrationError(f"классы без кропов: {пустые[:5]}")

    база, остаток = divmod(total, классов)
    генератор = np.random.default_rng(seed)
    отобранные: list[Sample] = []
    счётчики: dict[str, int] = {}
    for позиция, индекс in enumerate(sorted(группы)):
        квота = база + (1 if позиция < остаток else 0)
        # Сортировка до розыгрыша: иначе сид фиксирует не выбор, а порядок строк
        # в parquet. `as_posix` — чтобы ключ не зависел от разделителя пути.
        свои = sorted(группы[индекс], key=lambda образец: образец.path.as_posix())
        сколько = min(квота, len(свои))
        выбор = генератор.choice(len(свои), size=сколько, replace=False)
        отобранные.extend(свои[int(позиция_в_классе)] for позиция_в_классе in sorted(выбор))
        счётчики[по_индексу[индекс]] = сколько

    return Calibration(
        samples=tuple(отобранные),
        per_class_counts=счётчики,
        requested_total=total,
        seed=seed,
    )
