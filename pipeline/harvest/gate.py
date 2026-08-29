# pipeline/harvest/gate.py
"""Пороги прогона H1–H4 и отчёт (§14 дизайна).

Пороги назначены до запуска: критерий, выбранный после того, как увидены числа, — это
подгонка критерия под результат, а не решение.

Блокирующих порогов два, и разделение — суть гейта. H3 и H4 говорят «конвейер нельзя
пускать в данные»: первый это правило 1 Р-11 и 152-ФЗ, второй — выборка без
юридического следа. Охват аудита и разбор вердиктов — `pipeline/harvest/audit.py`;
числа приходят из CSV, а не хардкодятся. H3 блокирует по двум разным поводам: найденный
dirty-кадр и **незакрытый охват**. Второй повод существует потому, что `audited_dirty == 0`
одинаково верно и когда аудит прошёл чисто, и когда его не было вовсе — на этой
неоднозначности прогон 2026-08-29 отрапортовал H3 «ПРОЙДЕН» на двух проверенных кадрах из
двадцати восьми. H1 и H2 не блокируют: их провал это **измеренный
дефицит**, который открывает Flickr, Openverse и синтетический слой, а не переделку
харвестера. Требовать от H1 прохождения значило бы требовать от кода того, что зависит
от содержимого чужого хранилища.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.harvest.audit import audit_gap
from pipeline.harvest.manifest import HarvestRow, ManifestError, validate
from pipeline.harvest.run import ClassOutcome

H1_MIN_CLASSES = 100
CLASS_TARGET = 15
H2_CORE_CLASSES = ("lada__2107", "lada__granta", "lada__niva_2121", "uaz__patriot")


@dataclass(frozen=True, slots=True)
class GateResult:
    """Вердикт прогона: что блокирует и что просто измерено."""

    passed: bool
    blocking_failures: tuple[str, ...]
    advisory_failures: tuple[str, ...]


def check_gate(
    outcomes: Sequence[ClassOutcome],
    rows: Sequence[HarvestRow],
    audited_total: int,
    audited_dirty: int,
    allow_sa: bool,
) -> GateResult:
    """Проверить четыре порога и разделить провалы на блокирующие и измерительные."""
    блокирующие: list[str] = []
    измерительные: list[str] = []

    добрали = sum(1 for итог in outcomes if итог.accepted >= CLASS_TARGET)
    if добрали < H1_MIN_CLASSES:
        измерительные.append(
            f"H1: классов с {CLASS_TARGET} принятыми — {добрали} < {H1_MIN_CLASSES}"
        )

    по_классам = {итог.class_id: итог.accepted for итог in outcomes}
    for class_id in H2_CORE_CLASSES:
        принято = по_классам.get(class_id, 0)
        if принято < CLASS_TARGET:
            измерительные.append(f"H2: ядровой {class_id} — {принято} < {CLASS_TARGET}")

    if audited_dirty > 0:
        блокирующие.append(
            f"H3: аудит нашёл {audited_dirty} из {audited_total} кадров с читаемым номером "
            f"или различимым лицом — правится редакция, а не список файлов"
        )

    недостача = audit_gap(rows)
    if недостача:
        пример = sorted(недостача)[0]
        блокирующие.append(
            f"H3: аудит не закрыт — {len(недостача)} обязательных кадров без вердикта "
            f"(например {пример}); «0 dirty» на непроверенной выборке ничего не значит"
        )

    try:
        validate(rows, allow_sa=allow_sa)
    except ManifestError as ошибка:
        блокирующие.append(f"H4: манифест не проходит валидацию — {ошибка}")

    return GateResult(not блокирующие, tuple(блокирующие), tuple(измерительные))


def render_report(
    outcomes: Sequence[ClassOutcome],
    rows: Sequence[HarvestRow],
    gate: GateResult,
    git_sha: str,
    ran_at: str,
    audited_total: int,
    audited_dirty: int,
) -> str:
    """Собрать `bench/harvest-m0.md`: числа прогона в отслеживаемом git месте."""
    дефицитные = sorted(
        (итог for итог in outcomes if итог.accepted < CLASS_TARGET),
        key=lambda итог: итог.accepted,
    )
    вердикт_гейта = (
        "пройден (автоматический контур + регресс, ожидает закрытия ручного аудита)"
        if gate.passed
        else "НЕ ПРОЙДЕН"
    )
    строки = [
        "# Харвест открытого провенанса, M0 — прогон",
        "",
        f"Прогон от {ran_at}, `git_sha` — `{git_sha}`.",
        "",
        f"- классов в прогоне: {len(outcomes)}",
        f"- принято фотографий: {len(rows)}",
        f"- классов, добравших {CLASS_TARGET}: "
        f"{sum(1 for итог in outcomes if итог.accepted >= CLASS_TARGET)}",
        (
            f"- аудит H3: проверено {audited_total}, с дефектами {audited_dirty} "
            f"(ручной покадровый просмотр в процессе)"
        ),
        f"- вердикт гейта: {вердикт_гейта}",
        "",
        "## Блокирующие провалы",
        "",
        *([f"- {причина}" for причина in gate.blocking_failures] or ["Нет."]),
        "",
        "## Измеренный дефицит (не блокирует)",
        "",
        *([f"- {причина}" for причина in gate.advisory_failures] or ["Нет."]),
        "",
        "## Дефицитные классы",
        "",
        "Уходят в синтетический слой Р-11. Это исход, а не ошибка прогона.",
        "",
        "| Класс | Принято | Просмотрено |",
        "|---|---|---|",
        *[f"| `{и.class_id}` | {и.accepted} | {и.seen} |" for и in дефицитные],
    ]
    return "\n".join(строки) + "\n"
