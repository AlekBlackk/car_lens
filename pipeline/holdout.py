"""Отбор отложенного эталона `holdout_ru` (правило 2 Р-11, §3.5).

Единица отбора — не фотография, а `listing_id`: `{source}:{class_id}:{author}`. Автор
Commons заливает серию ракурсов одного экземпляра машины одним пакетом, и разнесение
такой серии между эталоном и обучением даёт ту же утечку, от которой §3.5 защищается
сплитом по объявлению. Класс входит в ключ потому, что один автор снимает машины разных
классов, а один `listing_id` в двух сплитах — нарушение инварианта 1 §13.1, даже когда
речь о разных фотографиях.

В эталон идут только кадры с вердиктом человека `ok`. `conflict` — это «человек не
уверен», и место такому кадру в обучающем пуле, а не в мере, по которой судят модель.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.harvest.manifest import HarvestRow
from pipeline.hashing import bucket_of

HOLDOUT_TARGET = 10
HOLDOUT_MIN = 5
# Насколько группе позволено перебрать цель. Без допуска серия из шести кадров никогда
# не влезет в остаток из четырёх, и класс останется без эталона при полном наборе фото.
OVERSHOOT = 2


@dataclass(frozen=True, slots=True)
class HoldoutSelection:
    """Что ушло в эталон и каким классам его не хватило."""

    files: frozenset[str]
    short_classes: tuple[str, ...]


def listing_id_of(row: HarvestRow) -> str:
    """Ключ физической машины: источник, класс и автор."""
    return f"{row.source}:{row.class_id}:{row.author}"


def select_holdout(
    rows: Sequence[HarvestRow],
    target: int = HOLDOUT_TARGET,
    minimum: int = HOLDOUT_MIN,
    seed: int = 1337,
) -> HoldoutSelection:
    """Выбрать до `target` эталонных кадров на класс, не разрывая серии автора.

    Группа (listing_id) неделима: если в серии автора есть кадры без вердикта `ok`
    (например, `conflict`), вся группа остаётся в пуле, чтобы не допустить утечки
    автора между эталоном и обучением (§13.1, инвариант 1).
    """
    по_классам: dict[str, dict[str, list[HarvestRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    все_классы: set[str] = set()
    for строка in rows:
        все_классы.add(строка.class_id)
        по_классам[строка.class_id][listing_id_of(строка)].append(строка)

    выбранные: set[str] = set()
    короткие: list[str] = []
    for class_id in sorted(все_классы):
        группы = по_классам.get(class_id, {})
        чистые_группы = {
            ключ: группа
            for ключ, группа in группы.items()
            if all(строка.audit == "ok" for строка in группа)
        }
        порядок = sorted(чистые_группы, key=lambda ключ: bucket_of(ключ, seed))
        набрано: list[str] = []
        for ключ in порядок:
            if len(набрано) >= target:
                break
            группа = чистые_группы[ключ]
            if len(набрано) + len(группа) > target + OVERSHOOT:
                continue
            набрано += [строка.file for строка in группа]
        if len(набрано) < minimum:
            короткие.append(class_id)
            continue
        выбранные.update(набрано)

    return HoldoutSelection(frozenset(выбранные), tuple(короткие))
