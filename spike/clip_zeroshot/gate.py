"""Пороги гейта §7 спеки, зафиксированные ДО запуска замера.

Критерий, назначенный после того, как увидены числа, — подгонка критерия под результат.
Поэтому пороги живут константами в коде, а не суждением читателя отчёта.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

G1_TOP1 = 0.75
G1_TOP3 = 0.85
G2_CONTROL = 0.80
G3_RU_CN = 0.50
G4_CORE = 0.40
G5_MB = 40.0
G6_MS = 48.7
G7A_COSINE = 0.9999
G7B_COSINE = 0.99
G7C_COSINE = 0.9


class Outcome(Enum):
    """Три исхода §8 спеки, равноправные по форме записи."""

    GO = "go"
    NO_GO = "no-go"
    PARTIAL = "частично"


@dataclass(frozen=True, slots=True)
class PhaseOneResult:
    """Числа фазы 1 по одному кандидату."""

    stanford_top1: float
    stanford_top3: float
    control_top3: float
    ru_cn_top3: float
    core_top3: dict[str, float] = field(default_factory=dict)


def check_phase_one(итог: PhaseOneResult) -> tuple[bool, list[str]]:
    """Прошёл ли кандидат фазу 1. Второй элемент — причины отказа, все сразу."""
    причины: list[str] = []
    if итог.control_top3 < G2_CONTROL:
        причины.append(
            f"G2: контроль {итог.control_top3:.3f} < {G2_CONTROL} — замер недействителен, "
            "выборку надо переделать"
        )
    if итог.stanford_top1 < G1_TOP1:
        причины.append(f"G1: Stanford top-1 {итог.stanford_top1:.3f} < {G1_TOP1}")
    if итог.stanford_top3 < G1_TOP3:
        причины.append(f"G1: Stanford top-3 {итог.stanford_top3:.3f} < {G1_TOP3}")
    if итог.ru_cn_top3 < G3_RU_CN:
        причины.append(f"G3: RU+CN top-3 {итог.ru_cn_top3:.3f} < {G3_RU_CN}")
    for class_id, доля in sorted(итог.core_top3.items()):
        if доля < G4_CORE:
            причины.append(f"G4: {class_id} top-3 {доля:.3f} < {G4_CORE}")
    return not причины, причины


def decide(
    фаза1: bool,
    размер_мб: float | None,
    латентность_мс: float | None,
    паритет: dict[str, float] | None,
) -> Outcome:
    """Исход по кандидату. `None` означает «фаза 2 не запускалась»."""
    if not фаза1:
        return Outcome.NO_GO
    if размер_мб is None or латентность_мс is None or паритет is None:
        return Outcome.NO_GO
    # Паритет проверяется раньше размера: модель, считающая не то же самое, не годится
    # ни в каком смысле, и «частично» о ней сказать нельзя.
    if (
        паритет.get("g7a", 0.0) < G7A_COSINE
        or паритет.get("g7b", 0.0) < G7B_COSINE
        or паритет.get("g7c", 0.0) < G7C_COSINE
    ):
        return Outcome.NO_GO
    if размер_мб > G5_MB or латентность_мс > G6_MS:
        return Outcome.PARTIAL
    return Outcome.GO
