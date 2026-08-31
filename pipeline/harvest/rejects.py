"""Журнал причин отказа, переживающий процесс.

`ClassOutcome` считает дельту прогона, и этого мало: прогон 2026-08-31 просмотрел ноль
кандидатов (все метки уже лежали в `done_items`) и отчитался пустой таблицей причин. За три
прогона в репозитории не осталось ни одной причины отказа, и диагностировать пустой класс
стало нечем — §2.5 дизайна C1.

Формат — CSV рядом с `alias_rejects.csv`. Дублирование с ним намеренное: там подробности
вердикта игл, здесь единая линейка причин для отчёта. Тело `append_reject_row` тоже
намеренно повторяет `aliases.append_reject`: выносить общий писатель значило бы связать два
модуля обёрткой вокруг `csv.DictWriter` ради трёх строк.
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
    новый = not path.exists()
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
