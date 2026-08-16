"""Метрики закрытого множества: top-1/top-5 по модели и top-1 по марке.

Марка выводится из `model_id` по префиксу до `__`, отдельного поля нет — по
той же причине, по которой каталог не хранит `model_id` рядом с ключом:
второе поле с той же информацией рассинхронизируется с первым.

Метрик открытого множества (AUROC, FPR@TPR95, ECE) здесь нет: без негативов
они не определены, а посчитанные «как получится» ввели бы в заблуждение.
Долг зафиксирован в §11 дизайна.

Модуль не импортирует torch (Р-14): на вход идут готовые индексы top-5.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metrics:
    """Метрики одного прогона по сплиту."""

    top1_model: float
    top5_model: float
    top1_brand: float
    per_class_recall: dict[str, float]


def brand_of(model_id: str) -> str:
    """Марка из `model_id`: часть до разделителя `__`."""
    марка, _, _ = model_id.partition("__")
    return марка


def aggregate(
    top5: Sequence[Sequence[int]],
    targets: Sequence[int],
    *,
    idx_to_model_id: Mapping[int, str],
) -> Metrics:
    """Свести предсказания в метрики.

    `top5[i]` — индексы классов по убыванию уверенности, `top5[i][0]` —
    предсказание top-1. `targets[i]` — истинный индекс класса.
    `KeyError`, если встретился индекс, которого нет в `idx_to_model_id`:
    разъехавшийся каталог обязан ронять оценку, а не занижать её молча.
    """
    if len(top5) != len(targets):
        raise ValueError(f"несовпадение длин: {len(top5)} предсказаний, {len(targets)} меток")
    if not targets:
        raise ValueError("пустая выборка")

    попаданий_1 = 0
    попаданий_5 = 0
    попаданий_марка = 0
    верных_в_классе: Counter[int] = Counter()
    всего_в_классе: Counter[int] = Counter()

    for предсказания, цель in zip(top5, targets, strict=True):
        истинный_id = idx_to_model_id[цель]
        предсказанный_id = idx_to_model_id[предсказания[0]]
        всего_в_классе[цель] += 1
        if предсказания[0] == цель:
            попаданий_1 += 1
            верных_в_классе[цель] += 1
        if цель in предсказания:
            попаданий_5 += 1
        if brand_of(предсказанный_id) == brand_of(истинный_id):
            попаданий_марка += 1

    сколько = len(targets)
    return Metrics(
        top1_model=попаданий_1 / сколько,
        top5_model=попаданий_5 / сколько,
        top1_brand=попаданий_марка / сколько,
        per_class_recall={
            idx_to_model_id[индекс]: верных_в_классе[индекс] / всего
            for индекс, всего in sorted(всего_в_классе.items())
        },
    )
