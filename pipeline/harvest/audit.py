"""Протокол ручного аудита H3 и заведение его чисел в гейт.

Прогон 2026-08-29 провалил H3 вакуумно: гейт получил `audited_dirty=0`, потому что
аудит покрыл 2 кадра из 28 с номерами, а не потому что выборка чиста. Здесь охват задан
явно и проверяем: все кадры с найденными номерами/лицами, все кадры ядровых классов,
плюс случайные 200 из тех, где детектор не нашёл ничего, — контроль ложноотрицательных
пропусков.

`drop:plate` и `drop:face` — нарушение приватности: такие кадры удаляются из манифеста
**и** засчитываются в `audited_dirty`, потому что хотя бы один такой кадр означает, что
автоматика протекает системно (гейт блокирует — правится редакция, а не список файлов).
`drop:<прочее>` (не та модель, мусор) приватности не касается: кадр удаляется, в dirty
не идёт.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from pipeline.harvest.manifest import HarvestRow

AUDIT_DIRTY = frozenset({"drop:plate", "drop:face"})
CORE_CLASSES = ("lada__2107", "lada__granta", "lada__niva_2121", "uaz__patriot")
RANDOM_CLEAN_SAMPLE = 200


class AuditError(ValueError):
    """CSV вердиктов не годится."""


def _годен(вердикт: str) -> bool:
    return вердикт in {"ok", "conflict"} or вердикт.startswith("drop:")


def load_audit(path: Path) -> dict[str, str]:
    """Прочитать CSV `file,verdict` и провалидировать вердикты."""
    вердикты: dict[str, str] = {}
    for строка in csv.DictReader(path.read_text(encoding="utf-8").splitlines()):
        файл = (строка.get("file") or "").strip()
        вердикт = (строка.get("verdict") or "").strip()
        if not файл:
            continue
        if not _годен(вердикт):
            raise AuditError(f"{path}: непонятный вердикт {вердикт!r} у {файл}")
        вердикты[файл] = вердикт
    return вердикты


def apply_audit(
    rows: Sequence[HarvestRow], verdicts: Mapping[str, str]
) -> list[HarvestRow]:
    """Проставить колонку `audit` и выкинуть строки с любым `drop:*`."""
    итог: list[HarvestRow] = []
    for строка in rows:
        вердикт = verdicts.get(строка.file, строка.audit)
        if вердикт.startswith("drop:"):
            continue
        итог.append(replace(строка, audit=вердикт))
    return итог


def audit_counts(rows: Sequence[HarvestRow]) -> tuple[int, int]:
    """`(audited_total, audited_dirty)` по строкам до удаления dirty."""
    total = sum(1 for с in rows if с.audit)
    dirty = sum(1 for с in rows if с.audit in AUDIT_DIRTY)
    return total, dirty


def required_audit_files(rows: Sequence[HarvestRow], seed: int = 0) -> set[str]:
    """Какие `file` обязаны быть в аудите по протоколу §14."""
    надо: set[str] = set()
    чистые: list[str] = []
    for строка in rows:
        if строка.plates > 0 or строка.faces > 0 or строка.class_id in CORE_CLASSES:
            надо.add(строка.file)
        else:
            чистые.append(строка.file)
    выборка = random.Random(seed).sample(чистые, min(RANDOM_CLEAN_SAMPLE, len(чистые)))
    надо.update(выборка)
    return надо


def audit_gap(rows: Sequence[HarvestRow], seed: int = 0) -> set[str]:
    """Обязательные по §14 кадры, у которых нет вердикта.

    Отличает «аудит прошёл и чист» от «аудита не было»: `audited_dirty == 0` верно в обоих
    случаях, и на этой неоднозначности прогон 2026-08-29 отрапортовал H3 вакуумно. Пустая
    недостача — единственное доказательство, что протокол закрыт, а не пропущен.
    """
    надо = required_audit_files(rows, seed=seed)
    проверено = {строка.file for строка in rows if строка.audit}
    return надо - проверено
