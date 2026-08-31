# pipeline/harvest/state.py
"""Состояние прогона: пагинация, счётчики и уже обработанные элементы.

Счётчики живут здесь, а не в памяти прогона, и это принципиально: обрыв сети на исходе
бюджета иначе обнулил бы его, и рестарт просмотрел бы те же `SEEN_BUDGET` кандидатов
заново — то есть отдал бы чужому хранилищу целый бюджет лишних запросов.

Стоп-критерий класса — `SEEN_BUDGET = 400` просмотренных ИЛИ `ACCEPTED_TARGET = 30`
принятых, что раньше. Числа названы константами, а не в тексте: прежняя редакция
докстринга говорила «200 или 15», разойдясь с кодом и пережив три прогона незамеченной.
Класс, не добравший цель, помечается дефицитным и уходит в синтетический слой — это исход,
а не ошибка.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    # Развёрнутые деревья категорий, ключ `class_id|категория`. Поле со значением по
    # умолчанию: состояние, записанное до задачи C1, обязано читаться без правки файла.
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
        categories={
            ключ: tuple(значения) for ключ, значения in (данные.get("categories") or {}).items()
        },
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
                "categories": {ключ: list(значения) for ключ, значения in state.categories.items()},
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
