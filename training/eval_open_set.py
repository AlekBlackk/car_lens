"""Честный замер открытого множества (спека 2026-09-27-open-set-street §4).

Главная метрика — покрытие своих машин при FPR ≤ 0.15 на каждой группе негативов сразу:
в приложении порог один, и он обязан держать и чужие машины, и не-машины. Правило решения и
сетка — те же, что у подбора порогов (`training.calibration`) и на телефоне.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from training.calibration import MARGIN_GRID, SIM_GRID, Scores, operating_point

MAX_ERROR = 0.15
MAX_FPR = 0.15
STREET = "чужие машины"
NONCAR = "не-машины"
GROUP_KINDS: dict[str, tuple[str, ...]] = {
    STREET: ("off_catalog_street",),
    NONCAR: ("truck_bus", "detector_fp"),
}
REFERENCE_KIND = "off_catalog"
DEFAULT_SEED = 20260927


@dataclass(frozen=True, slots=True)
class OpenSetPoint:
    """Пара порогов: покрытие и ошибка на знакомых, FPR — на каждой группе негативов."""

    sim_min: float
    margin_min: float
    coverage: float
    error: float
    shown: int
    fpr: dict[str, float]


def _проверить_группы(groups: Mapping[str, Scores]) -> None:
    """FPR пустой группы равен нулю и «проходит» любое ограничение — такой замер ложен."""
    пустые = [имя for имя, негативы in groups.items() if len(негативы.sim1) == 0]
    if not groups or пустые:
        raise ValueError(f"пустая группа негативов: {', '.join(пустые) or 'групп нет'}")


def open_set_point(
    known: Scores, groups: Mapping[str, Scores], sim_min: float, margin_min: float
) -> OpenSetPoint:
    точки = {
        имя: operating_point(known, негативы, sim_min, margin_min)
        for имя, негативы in groups.items()
    }
    любая = next(iter(точки.values()))
    return OpenSetPoint(
        sim_min=float(sim_min),
        margin_min=float(margin_min),
        coverage=любая.coverage,
        error=любая.error,
        shown=любая.shown,
        fpr={имя: точка.fpr for имя, точка in точки.items()},
    )


def select_open_set_pair(
    known: Scores,
    groups: Mapping[str, Scores],
    *,
    max_error: float = MAX_ERROR,
    max_fpr: float = MAX_FPR,
    sim_grid: np.ndarray = SIM_GRID,
    margin_grid: np.ndarray = MARGIN_GRID,
) -> OpenSetPoint | None:
    """Пара с наибольшим покрытием, держащая ошибку и FPR всех групп; тай-брейк — сетка.

    Порядок обхода и замена только на строго большее покрытие — как в `select_pair`: при
    равенстве остаётся меньший `sim_min`, затем меньший `margin_min`.
    """
    _проверить_группы(groups)
    лучшая: OpenSetPoint | None = None
    for sim_min in sim_grid:
        for margin_min in margin_grid:
            точка = open_set_point(known, groups, float(sim_min), float(margin_min))
            if точка.shown == 0 or точка.error > max_error:
                continue
            if any(fpr > max_fpr for fpr in точка.fpr.values()):
                continue
            if лучшая is None or точка.coverage > лучшая.coverage:
                лучшая = точка
    return лучшая


def binding_constraint(
    point: OpenSetPoint, *, max_error: float = MAX_ERROR, max_fpr: float = MAX_FPR
) -> str:
    """Ограничение с наименьшим запасом до предела: оно и держит покрытие."""
    запасы = {"ошибка": max_error - point.error}
    запасы.update({f"FPR {имя}": max_fpr - fpr for имя, fpr in point.fpr.items()})
    return min(запасы, key=lambda имя: запасы[имя])


def split_by_group(groups: np.ndarray, seed: int) -> np.ndarray:
    """Маска половины A: группы целиком, через одну после перемешивания по `seed`.

    Чужая модель не должна попасть в обе половины: пара, подобранная на её кадрах, мерилась
    бы на её же соседях, и оценка вышла бы оптимистичной.
    """
    уникальные = np.unique(groups)
    np.random.default_rng(seed).shuffle(уникальные)
    в_a = set(уникальные[::2].tolist())
    return np.array([группа in в_a for группа in groups.tolist()], dtype=bool)


@dataclass(frozen=True, slots=True)
class OpenSetFold:
    tuned_on: str
    chosen: OpenSetPoint | None
    held_out: OpenSetPoint | None


def cross_fit_open_set(
    known: Scores,
    groups: Mapping[str, Scores],
    known_a: np.ndarray,
    groups_a: Mapping[str, np.ndarray],
    *,
    max_error: float = MAX_ERROR,
    max_fpr: float = MAX_FPR,
) -> list[OpenSetFold]:
    """Подбор на одной половине, замер той же пары на другой; обе стороны."""
    стороны: list[OpenSetFold] = []
    for имя, сторона_a in (("A", True), ("B", False)):
        свои = known_a if сторона_a else ~known_a
        учёба = {г: s.subset(groups_a[г] if сторона_a else ~groups_a[г]) for г, s in groups.items()}
        проверка = {
            г: s.subset(~groups_a[г] if сторона_a else groups_a[г]) for г, s in groups.items()
        }
        точка = select_open_set_pair(
            known.subset(свои), учёба, max_error=max_error, max_fpr=max_fpr
        )
        if точка is None:
            стороны.append(OpenSetFold(имя, None, None))
            continue
        отложенная = open_set_point(known.subset(~свои), проверка, точка.sim_min, точка.margin_min)
        стороны.append(OpenSetFold(имя, точка, отложенная))
    return стороны
