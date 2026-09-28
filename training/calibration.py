"""Подбор порогов отказа и изотонической шкалы уверенности (§6.4, §13.5, §13.6 SPEC;
спека 2026-09-23-calibration-ru §5).

Чистый numpy без torch и sklearn: модуль гоняется в CI группой `dev`. Правило решения —
копия `app/src/vision/decide.ts`: подобранный здесь порог должен означать ровно то, что
делает приложение, иначе в отчёт уедет правдоподобное неверное число.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

REJECT, BRAND, MODEL = 0, 1, 2
SIM_GRID = np.round(np.arange(0.30, 0.9001, 0.01), 2)
MARGIN_GRID = np.round(np.arange(0.0, 0.1501, 0.005), 3)
# Ветка отказного прототипа (спека 2026-09-28 §3): −∞ выключает её без особого случая —
# разность сходств никогда не меньше −∞, и решения совпадают с 0.5 побайтно.
REJECT_OFF = float("-inf")
REJECT_GRID = np.concatenate([[REJECT_OFF], np.round(np.arange(-0.05, 0.1001, 0.01), 2)])
CLIP = (0.01, 0.98)
# Не подбираются: офлайн нет ни треков (наблюдения), ни кропов короче 120 px (§7 спеки).
MIN_OBSERVATIONS = 3
MIN_SHORT_SIDE_PX = 120


class CalibrationError(ValueError):
    """Шкалу по этим данным построить нельзя."""


@dataclass(frozen=True, slots=True)
class Scores:
    """Всё, что решение §6.4 знает о кадре, и истина. У негативов оба `*_hit` — False.

    `sim_reject` — лучшее сходство с отказными прототипами; `None` — прототипов нет.
    """

    sim1: np.ndarray
    sim2: np.ndarray
    same_brand: np.ndarray
    class_hit: np.ndarray
    brand_hit: np.ndarray
    sim_reject: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> Scores:
        return Scores(
            self.sim1[mask],
            self.sim2[mask],
            self.same_brand[mask],
            self.class_hit[mask],
            self.brand_hit[mask],
            None if self.sim_reject is None else self.sim_reject[mask],
        )


def decide(
    scores: Scores, sim_min: float, margin_min: float, reject_margin: float = REJECT_OFF
) -> np.ndarray:
    """Порядок проверок `decide.ts`: sim, отказной прототип, разрыв; сравнения строгие.

    Все три отказа дают одно и то же `REJECT`, поэтому порядок присваиваний ниже важен только
    для «марки»: прототип обязан отменять её, а не наоборот.
    """
    решения = np.full(len(scores.sim1), MODEL, dtype=np.int8)
    близко = (scores.sim1 - scores.sim2) < margin_min
    решения[близко & scores.same_brand] = BRAND
    решения[близко & ~scores.same_brand] = REJECT
    if scores.sim_reject is not None:
        решения[(scores.sim1 - scores.sim_reject) < reject_margin] = REJECT
    решения[scores.sim1 < sim_min] = REJECT
    return решения


@dataclass(frozen=True, slots=True)
class Operating:
    """Рабочая точка: пара порогов и что она даёт на знакомых и негативах."""

    sim_min: float
    margin_min: float
    coverage: float
    error: float
    fpr: float
    shown: int
    reject_margin: float = REJECT_OFF


def operating_point(
    known: Scores,
    negatives: Scores,
    sim_min: float,
    margin_min: float,
    reject_margin: float = REJECT_OFF,
) -> Operating:
    решения = decide(known, sim_min, margin_min, reject_margin)
    показаны = решения != REJECT
    верно = np.where(решения == MODEL, known.class_hit, known.brand_hit)
    показано = int(показаны.sum())
    ошибка = float((показаны & ~верно).sum() / показано) if показано else 0.0
    принято = decide(negatives, sim_min, margin_min, reject_margin) != REJECT
    return Operating(
        sim_min=float(sim_min),
        margin_min=float(margin_min),
        coverage=float(показаны.mean()),
        error=ошибка,
        fpr=float(принято.mean()) if len(принято) else 0.0,
        shown=показано,
        reject_margin=float(reject_margin),
    )


def select_pair(
    known: Scores,
    negatives: Scores,
    *,
    max_error: float,
    max_fpr: float,
    sim_grid: np.ndarray = SIM_GRID,
    margin_grid: np.ndarray = MARGIN_GRID,
) -> Operating | None:
    """Допустимая пара с наибольшим покрытием; при равенстве — первая по сетке.

    Сетка обходится по возрастанию `sim_min`, затем `margin_min`, а замена — только на
    строго большее покрытие: так тай-брейк «меньший порог» выходит из порядка обхода.
    Точка без единого показанного ответа недопустима — «молчать всегда» ошибок не делает.
    """
    лучшая: Operating | None = None
    for sim_min in sim_grid:
        for margin_min in margin_grid:
            точка = operating_point(known, negatives, float(sim_min), float(margin_min))
            if точка.shown == 0 or точка.error > max_error or точка.fpr > max_fpr:
                continue
            if лучшая is None or точка.coverage > лучшая.coverage:
                лучшая = точка
    return лучшая


def _pav(xs: np.ndarray, ys: np.ndarray, веса: np.ndarray) -> list[list[float]]:
    """Pool-adjacent-violators: блоки [сумма x·w, сумма y·w, вес], y блоков строго растёт."""
    блоки: list[list[float]] = []
    for x, y, w in zip(xs, ys, веса, strict=True):
        блоки.append([float(x * w), float(y * w), float(w)])
        while len(блоки) > 1 and блоки[-2][1] / блоки[-2][2] >= блоки[-1][1] / блоки[-1][2]:
            sx, sy, sw = блоки.pop()
            блоки[-1][0] += sx
            блоки[-1][1] += sy
            блоки[-1][2] += sw
    return блоки


def calibration_knots(
    sims: np.ndarray, hits: np.ndarray, sim_min: float, max_knots: int = 32
) -> list[tuple[float, float]]:
    """Изотоническая `P(модель угадана | sim)` как узлы §13.6.

    Одинаковые `sim` сводятся в одну точку до PAV: иначе два блока с равным x дали бы
    таблицу, которую `parseCalibration` отвергнет. Больше `max_knots` блоков — соседние
    сливаются группами взвешенным средним; средние соседних групп строго растут по обеим
    осям, потому что растут у блоков. Первый узел переносится на `sim_min`: ниже первого
    узла `confidenceFromSim` отдаёт `clipLo`, и ответ с `sim` чуть выше порога показал бы 1%.
    """
    xs, обратный, веса = np.unique(
        np.asarray(sims, dtype=np.float64), return_inverse=True, return_counts=True
    )
    попадания = np.bincount(обратный, weights=np.asarray(hits, dtype=np.float64))
    блоки = _pav(xs, попадания / веса, веса.astype(np.float64))
    if len(блоки) < 2:
        raise CalibrationError("изотоническая регрессия дала одну ступень: шкалы нет")
    группы = np.array_split(np.arange(len(блоки)), min(max_knots, len(блоки)))
    узлы: list[tuple[float, float]] = []
    for группа in группы:
        sx = sum(блоки[i][0] for i in группа)
        sy = sum(блоки[i][1] for i in группа)
        sw = sum(блоки[i][2] for i in группа)
        узлы.append((round(sx / sw, 6), round(sy / sw, 6)))
    узлы[0] = (float(sim_min), узлы[0][1])
    for (x0, y0), (x1, y1) in pairwise(узлы):
        if not (x1 > x0 and y1 > y0):
            raise CalibrationError(f"узлы не строго возрастают: {(x0, y0)} → {(x1, y1)}")
    return узлы


def confidence_from_sim(
    sims: np.ndarray, knots: list[tuple[float, float]], clip: tuple[float, float] = CLIP
) -> np.ndarray:
    """Копия `confidenceFromSim`: на первом узле и ниже — `clipLo`, дальше — интерполяция."""
    xs = np.array([x for x, _ in knots])
    ys = np.array([y for _, y in knots])
    значения = np.interp(sims, xs, ys)
    значения = np.where(np.asarray(sims) <= xs[0], clip[0], значения)
    return np.clip(значения, clip[0], clip[1])


def ece(confidence: np.ndarray, hits: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error по равным корзинам на [0, 1]."""
    корзины = np.minimum((np.asarray(confidence) * bins).astype(int), bins - 1)
    итог = 0.0
    for корзина in range(bins):
        маска = корзины == корзина
        if маска.any():
            итог += abs(float(confidence[маска].mean()) - float(hits[маска].mean())) * (
                маска.sum() / len(confidence)
            )
    return итог


def split_halves(groups: np.ndarray, seed: int) -> np.ndarray:
    """Маска половины A: внутри каждой группы перемешать и брать через одного."""
    генератор = np.random.default_rng(seed)
    маска = np.zeros(len(groups), dtype=bool)
    for группа in np.unique(groups):
        позиции = np.flatnonzero(groups == группа)
        генератор.shuffle(позиции)
        маска[позиции[::2]] = True
    return маска


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Площадь под ROC через ранги (Манн — Уитни), связи — средним рангом."""
    ранги = pd.Series(np.concatenate([positive, negative])).rank().to_numpy()
    сумма = float(ранги[: len(positive)].sum())
    return (сумма - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))


def fpr_at_tpr(positive: np.ndarray, negative: np.ndarray, tpr: float = 0.95) -> float:
    """Доля негативов не ниже порога, пропускающего долю `tpr` знакомых."""
    порог = float(np.quantile(positive, 1.0 - tpr))
    return float((np.asarray(negative) >= порог).mean())


def thresholds_doc(point: Operating, val_n: int, neg_n: int) -> dict[str, Any]:
    """`thresholds.json` §13.5; `achieved_tpr` — доля знакомых, получивших любой ответ.

    `reject_margin` пишется только при включённой ветке: без поля приложение ведёт себя как
    0.5, и плейсхолдер порогов остаётся той же формы.
    """
    документ: dict[str, Any] = {
        "version": 1,
        "tuned": True,
        "sim_min": point.sim_min,
        "margin_min": point.margin_min,
    }
    if np.isfinite(point.reject_margin):
        документ["reject_margin"] = point.reject_margin
    документ.update(
        {
            "min_observations": MIN_OBSERVATIONS,
            "min_short_side_px": MIN_SHORT_SIDE_PX,
            "tuned_on": {
                "val_n": val_n,
                "neg_n": neg_n,
                "achieved_tpr": round(point.coverage, 4),
                "achieved_fpr": round(point.fpr, 4),
            },
        }
    )
    return документ


def calibration_doc(
    knots: list[tuple[float, float]], clip: tuple[float, float] = CLIP
) -> dict[str, Any]:
    """`calibration.json` §13.6."""
    return {
        "version": 1,
        "method": "isotonic_piecewise_linear",
        "tuned": True,
        "knots": [[x, y] for x, y in knots],
        "clip": [clip[0], clip[1]],
    }
