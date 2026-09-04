"""Журнал причин отказа, переживающий процесс.

`ClassOutcome` считает дельту прогона, и этого мало: прогон 2026-08-31 просмотрел ноль
кандидатов (все метки уже лежали в `done_items`) и отчитался пустой таблицей причин. За три
прогона в репозитории не осталось ни одной причины отказа, и диагностировать пустой класс
стало нечем — §2.5 дизайна C1.

Формат — CSV рядом с `alias_rejects.csv`. Дублирование с ним намеренное: там подробности
вердикта игл, здесь единая линейка причин для отчёта. Тело `append_reject_row` — третья копия
одного и того же писателя (семь строк, как в `aliases.append_reject` и `manifest.append_row`):
общий писатель здесь не выносится, чтобы не связывать модули обёрткой вокруг `csv.DictWriter`
посреди этой задачи, но дублирование не бесплатно — сведение всех трёх в один модуль
заведено отдельной задачей.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Reject:
    """Один отказ цепочки решений."""

    class_id: str
    basis: str
    source: str
    source_item_id: str
    reason: str
    checked: str


REJECT_COLUMNS = tuple(поле.name for поле in fields(Reject))


def append_reject_row(path: Path, reject: Reject) -> None:
    """Дописать строку журнала, создав файл с шапкой при первом вызове."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Файл нулевой длины — файл без шапки: журнал укорачивают `> reject_log.csv`, и без
    # проверки размера первая дописанная строка молча стала бы шапкой.
    новый = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(REJECT_COLUMNS))
        if новый:
            писатель.writeheader()
        писатель.writerow(asdict(reject))


def load_rejects(path: Path) -> list[Reject]:
    """Прочитать журнал целиком; отсутствие файла — пустой журнал, а не ошибка."""
    if not path.exists():
        return []
    return [
        Reject(
            class_id=строка["class_id"],
            basis=строка["basis"],
            source=строка["source"],
            source_item_id=строка["source_item_id"],
            reason=строка["reason"],
            checked=строка["checked"],
        )
        for строка in csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    ]


def reason_counts(rejects: Sequence[Reject]) -> dict[str, int]:
    """Сколько кандидатов съела каждая причина."""
    return dict(Counter(отказ.reason for отказ in rejects))


def unique_by_reason(rejects: Sequence[Reject], reason: str) -> list[Reject]:
    """Уникальные кандидаты одной причины, в порядке первой встречи.

    Журнал накопительный: один и тот же кандидат отбит в каждом прогоне заново, и без
    снятия повторов вызывающий прогнал бы половину кадров дважды.
    """
    видели: set[str] = set()
    отобранные: list[Reject] = []
    for отказ in rejects:
        if отказ.reason != reason or отказ.source_item_id in видели:
            continue
        видели.add(отказ.source_item_id)
        отобранные.append(отказ)
    return отобранные
