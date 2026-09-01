"""Журнал обхода категорий, переживающий процесс.

`ExpandResult` (`categories.py`) считает потери обхода — сколько подкатегорий срезал
каждый стоп-маркер, `exclude` и предикат допуска (`admit`, §5 дизайна C1) — но живёт
внутри одного прогона:
объект пропадает вместе с процессом. Отчёт C1b, напротив, читает только файлы с диска
(манифест, `reject_log.csv`, `verdicts.dedup.csv`), и без этого журнала посчитанные
`ExpandResult` числа до отчёта попросту не доезжают — задача 12.

Формат — CSV по образцу `rejects.py`: четвёртая копия того же писателя (после
`aliases.append_reject`, `manifest.append_row`, `rejects.append_reject_row`), и это
осознанно — сведение писателей в один модуль заведено отдельной задачей (Task 5 minor
deferred).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExpandRow:
    """Одна строка журнала: итог обхода одной заявленной категории одного класса."""

    class_id: str
    category: str
    categories: int
    requests: int
    rejected_stopword: int
    rejected_exclude: int
    rejected_not_own: int
    budget_exhausted: bool
    checked: str
    # Колонка добавлена позже остальных (M6 фикс-волны), поэтому у неё умолчание, а читатель
    # терпит её отсутствие: журналы прежних прогонов лежат на диске без неё.
    truncated: int = 0


EXPAND_LOG_COLUMNS = tuple(поле.name for поле in fields(ExpandRow))


def append_expand_row(path: Path, row: ExpandRow) -> None:
    """Дописать строку журнала, создав файл с шапкой при первом вызове."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Файл нулевой длины — файл без шапки: журнал укорачивают `> expand_log.csv`, и без
    # проверки размера первая дописанная строка молча стала бы шапкой (урок коммита
    # 8b8f58d, `rejects.append_reject_row`).
    новый = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(EXPAND_LOG_COLUMNS))
        if новый:
            писатель.writeheader()
        писатель.writerow(asdict(row))


def load_expand_rows(path: Path) -> list[ExpandRow]:
    """Прочитать журнал целиком; отсутствие файла — пустой журнал, а не ошибка.

    Колонка `truncated` читается терпимо: журналы прогонов до M6 фикс-волны её не имеют, а
    отчёт обязан собираться и на них — ровно так же, как он собирается на данных до задачи 12.
    """
    if not path.exists():
        return []
    return [
        ExpandRow(
            class_id=строка["class_id"],
            category=строка["category"],
            categories=int(строка["categories"]),
            requests=int(строка["requests"]),
            rejected_stopword=int(строка["rejected_stopword"]),
            rejected_exclude=int(строка["rejected_exclude"]),
            rejected_not_own=int(строка["rejected_not_own"]),
            budget_exhausted=строка["budget_exhausted"] == "True",
            checked=строка["checked"],
            truncated=int(строка.get("truncated") or 0),
        )
        for строка in csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    ]


def expand_totals(rows: Sequence[ExpandRow]) -> dict[str, int]:
    """Сумма счётчиков обхода по всем строкам — материал таблицы отчёта C1.

    Контракт тотальный: все шесть ключей возвращаются нулями и на пустом `rows`, чтобы
    вызывающему (`scripts/report_c1.py`) не приходилось подстраховываться `.get(..., 0)`.
    """
    итог = {
        "requests": 0,
        "rejected_stopword": 0,
        "rejected_exclude": 0,
        "rejected_not_own": 0,
        "budget_exhausted": 0,
        "truncated": 0,
    }
    for строка in rows:
        итог["requests"] += строка.requests
        итог["rejected_stopword"] += строка.rejected_stopword
        итог["rejected_exclude"] += строка.rejected_exclude
        итог["rejected_not_own"] += строка.rejected_not_own
        итог["budget_exhausted"] += int(строка.budget_exhausted)
        итог["truncated"] += строка.truncated
    return итог
