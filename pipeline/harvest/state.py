# pipeline/harvest/state.py
"""Состояние прогона: пагинация, счётчики и уже обработанные элементы.

Счётчики живут здесь, а не в памяти прогона, и это принципиально: обрыв сети на сто
восьмидесятом кандидате иначе обнулил бы бюджет, и рестарт просмотрел бы те же двести
заново — то есть двести лишних запросов к чужому хранилищу.

Стоп-критерий класса — 200 просмотренных ИЛИ 15 принятых, что раньше. Пятнадцать это
5 в `holdout_ru` (правило 2 Р-11) плюс 10 на центроиды; класс, не добравший их,
помечается дефицитным и уходит в синтетический слой — это исход, а не ошибка.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SEEN_BUDGET = 400
ACCEPTED_TARGET = 30


@dataclass(slots=True)
class HarvestState:
    """Всё, что нужно, чтобы продолжить прогон с той же точки."""

    seen: dict[str, int]
    accepted: dict[str, int]
    tokens: dict[str, str]
    done_items: set[str]


def load_state(path: Path) -> HarvestState:
    """Прочитать состояние; отсутствие файла — это пустое состояние, а не ошибка."""
    if not path.exists():
        return HarvestState({}, {}, {}, set())
    данные = json.loads(path.read_text(encoding="utf-8"))
    return HarvestState(
        seen=dict(данные.get("seen", {})),
        accepted=dict(данные.get("accepted", {})),
        tokens=dict(данные.get("tokens", {})),
        done_items=set(данные.get("done_items", [])),
    )


def save_state(path: Path, state: HarvestState) -> None:
    """Записать состояние атомарно: во временный файл и `rename`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    временный = path.with_suffix(".tmp")
    временный.write_text(
        json.dumps(
            {
                "seen": state.seen,
                "accepted": state.accepted,
                "tokens": state.tokens,
                "done_items": sorted(state.done_items),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    временный.replace(path)


def is_class_done(state: HarvestState, class_id: str) -> bool:
    """Исчерпан ли класс: бюджет просмотра или набранная цель."""
    return (
        state.seen.get(class_id, 0) >= SEEN_BUDGET
        or state.accepted.get(class_id, 0) >= ACCEPTED_TARGET
    )
