"""Метрики few-shot замера Р-11: top-1, top-3 и поклассовый top-3.

Это не `training.metrics.aggregate` с другим `k`. `Metrics` — контракт M-1.2: он пишется в
карточки прогонов обучения и содержит `top5_model` и `top1_brand`. Пороги Р-11 просят
другого набора — top-3 вместо top-5 и **поклассовый** top-3, которого в `Metrics` нет.
Дописать `k` в `aggregate` значило бы, что поле `top5_model` иногда хранит top-3, то есть
переименовать метрику параметром.

Марки здесь нет намеренно: G_FS1–G_FS3 говорят о классе, и `top1_brand` «заодно» завёл бы
число, которое никакой порог не проверяет.

Модуль не импортирует torch: на вход идут готовые индексы ранжирования.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

FEWSHOT_TOP_K = 3


@dataclass(frozen=True, slots=True)
class FewShotMetrics:
    """Метрики одного замера по `holdout_ru`.

    `per_class_support` идёт рядом с `per_class_top3` потому, что доля без числа кадров не
    читается: 1.0 на одном кадре и 1.0 на двенадцати — разные утверждения. В отчёт
    попадают оба.
    """

    top1: float
    top3: float
    per_class_top3: dict[str, float]
    per_class_support: dict[str, int]


def aggregate_fewshot(
    ranking: Sequence[Sequence[int]],
    targets: Sequence[int],
    *,
    idx_to_class_id: Mapping[int, str],
) -> FewShotMetrics:
    """Свести ранжирование в метрики Р-11.

    `ranking[i]` — индексы классов по убыванию сходства, не короче `FEWSHOT_TOP_K`: иначе
    top-3 посчитался бы по двум кандидатам и вышел заниженным под верным именем.

    В `per_class_top3` попадают только классы, встретившиеся в `targets`. Класс без
    эталона в отчёте отсутствует, и это не пропуск: приписать ему долю значило бы выдумать
    измерение. Как считать такой класс в пороге, решает гейт.

    `KeyError`, если метка кадра не найдена в `idx_to_class_id`: разъехавшийся каталог
    обязан ронять оценку, а не занижать её молча. `ValueError` — на неинъективной карте
    имён: два индекса с одним `class_id` перетёрли бы друг друга в `per_class_top3`.
    """
    if len(ranking) != len(targets):
        raise ValueError(f"несовпадение длин: {len(ranking)} ранжирований, {len(targets)} меток")
    if not targets:
        raise ValueError("пустая выборка")
    повторы = sorted(
        class_id for class_id, сколько in Counter(idx_to_class_id.values()).items() if сколько > 1
    )
    if повторы:
        raise ValueError(f"idx_to_class_id неинъективен: {повторы} под разными индексами")

    попаданий_1 = 0
    попаданий_3 = 0
    в_топ3: Counter[int] = Counter()
    всего: Counter[int] = Counter()

    for номер, (кандидаты, цель) in enumerate(zip(ranking, targets, strict=True)):
        if len(кандидаты) < FEWSHOT_TOP_K:
            raise ValueError(
                f"строка {номер}: {len(кандидаты)} кандидатов, "
                f"а top-{FEWSHOT_TOP_K} требует {FEWSHOT_TOP_K}"
            )
        всего[цель] += 1
        if кандидаты[0] == цель:
            попаданий_1 += 1
        if цель in кандидаты[:FEWSHOT_TOP_K]:
            попаданий_3 += 1
            в_топ3[цель] += 1

    сколько = len(targets)
    return FewShotMetrics(
        top1=попаданий_1 / сколько,
        top3=попаданий_3 / сколько,
        per_class_top3={
            idx_to_class_id[индекс]: в_топ3[индекс] / сколько_в_классе
            for индекс, сколько_в_классе in sorted(всего.items())
        },
        per_class_support={
            idx_to_class_id[индекс]: сколько_в_классе
            for индекс, сколько_в_классе in sorted(всего.items())
        },
    )
