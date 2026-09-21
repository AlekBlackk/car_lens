"""Модуль для очистки пула по списку кандидатов чистоты классов.

Содержит типы данных, чтение CSV-кандидатов, фильтрацию по tier,
проверку целостности относительно пула и расчёт влияния на классы.
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DROP_COLUMNS: tuple[str, ...] = (
    "photo_id",
    "model_id",
    "sheet",
    "cell",
    "tier",
    "category",
    "confidence",
    "reason",
    "crop_path",
)

TIERS: tuple[int, ...] = (1, 2, 3)
CONFIDENCES: tuple[str, ...] = ("high", "medium")

# Ярлыки для отчёта о перекосе источника: хватит ли классу кадров после
# гипотетического прореживания. Порог — параметр вызывающего кода, не зашит здесь.
LABEL_CAN_THIN = "можно_проредить"
LABEL_COVERAGE_HOLE = "дыра_покрытия"


class ClassPurityError(ValueError):
    """Ошибка валидации или целостности списка кандидатов на исключение."""


@dataclass(frozen=True, slots=True)
class DropRow:
    """Строка кандидата на исключение из пула."""

    photo_id: str
    model_id: str
    sheet: int
    cell: int
    tier: int
    category: str
    confidence: str
    reason: str
    crop_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.photo_id, str) or not self.photo_id.strip():
            raise ClassPurityError("photo_id не может быть пустым")
        if type(self.sheet) is not int:
            raise ClassPurityError(f"sheet должен быть int, получено {type(self.sheet).__name__}")
        if type(self.cell) is not int:
            raise ClassPurityError(f"cell должен быть int, получено {type(self.cell).__name__}")
        if type(self.tier) is not int or self.tier not in TIERS:
            raise ClassPurityError(f"tier {self.tier} вне допустимых {TIERS}")
        if self.confidence not in CONFIDENCES:
            raise ClassPurityError(f"confidence '{self.confidence}' вне допустимых {CONFIDENCES}")


def read_drop_list(path: Path | str) -> list[DropRow]:
    """Прочитать CSV-файл со списком кандидатов на исключение.

    Проверяет точное совпадение шапки с DROP_COLUMNS и отсутствие
    повторов photo_id. Строки с tier=0 пропускаются (вычеркнуты).
    """
    rows: list[DropRow] = []
    seen_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ClassPurityError("Файл пуст, шапка отсутствует") from None

        if tuple(header) != DROP_COLUMNS:
            raise ClassPurityError(
                f"Неверная шапка CSV: {tuple(header)}, ожидалась: {DROP_COLUMNS}"
            )

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(DROP_COLUMNS):
                raise ClassPurityError(
                    f"Строка {line_num}: неверное число колонок ({len(row)} != {len(DROP_COLUMNS)})"
                )

            pid = row[0]
            if pid in seen_ids:
                raise ClassPurityError(f"photo_id {pid} встречается дважды (строка {line_num})")
            seen_ids.add(pid)

            try:
                sheet = int(row[2])
                cell = int(row[3])
                tier = int(row[4])
            except ValueError as e:
                raise ClassPurityError(
                    f"Строка {line_num}: ошибка преобразования чисел ({e})"
                ) from e

            if tier == 0:
                continue

            rows.append(
                DropRow(
                    photo_id=pid,
                    model_id=row[1],
                    sheet=sheet,
                    cell=cell,
                    tier=tier,
                    category=row[5],
                    confidence=row[6],
                    reason=row[7],
                    crop_path=row[8],
                )
            )

    return rows


def select_tiers(rows: Iterable[DropRow], tiers: Collection[int]) -> frozenset[str]:
    """Выбрать множество photo_id для заданных уровней tier."""
    target_tiers = frozenset(tiers)
    return frozenset(r.photo_id for r in rows if r.tier in target_tiers)


def unknown_ids(rows_ids: Iterable[str], pool_ids: Iterable[str]) -> tuple[str, ...]:
    """Вернуть отсортированный кортеж id кандидатов, которых нет в пуле."""
    pool_set = set(pool_ids)
    missing = set(rows_ids) - pool_set
    return tuple(sorted(missing))


def cleaned_pool(pool: pd.DataFrame, drop_ids: Collection[str]) -> pd.DataFrame:
    """Вернуть копию пула без указанных кадров с сохранением колонок и порядка."""
    drop_set = set(drop_ids)
    mask = ~pool["photo_id"].isin(drop_set)
    return pool.loc[mask].reset_index(drop=True)


def class_impact(pool: pd.DataFrame, drop_ids: Collection[str]) -> dict[str, tuple[int, int]]:
    """Посчитать (было, стало) только для классов, затронутых чисткой."""
    drop_set = set(drop_ids)
    if not drop_set or pool.empty:
        return {}

    dropped_mask = pool["photo_id"].isin(drop_set)
    dropped = pool.loc[dropped_mask]
    if dropped.empty:
        return {}

    affected_models = sorted(dropped["model_id"].dropna().unique())
    pool_counts = pool["model_id"].value_counts()
    dropped_counts = dropped["model_id"].value_counts()

    impact: dict[str, tuple[int, int]] = {}
    for model in affected_models:
        model_str = str(model)
        before = int(pool_counts.get(model, 0))
        dropped_count = int(dropped_counts.get(model, 0))
        impact[model_str] = (before, before - dropped_count)

    return impact


def skew_label(remaining: int, min_street: int) -> str:
    """Ярлык класса по числу кадров, которые останутся после прореживания.

    `remaining >= min_street` — прореживать безопасно (в классе хватит кадров);
    иначе класс — дыра покрытия, удаление сделает хуже. Ярлык говорит только о
    достаточности остатка, а не о том, что остаток снят с уровня глаз.
    """
    return LABEL_CAN_THIN if remaining >= min_street else LABEL_COVERAGE_HOLE
