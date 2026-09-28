"""Замер отказных прототипов против 0.5 (спека 2026-09-28-open-set-reject-prototypes §4).

Классификатор один и тот же, эмбеддинги считаются один раз: 0.5 — это те же оценки без
`sim_reject`. Сравнение — главной метрикой прошлого замера (`training.eval_open_set`), плюс
точка «не хуже 0.5» и парный счёт показанных знакомых в ней. Гейт доставки записан до замера
и проверяется здесь же; ассеты порогов пишутся только при пройденном гейте.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.pool_audit import file_sha256
from pipeline.reject_index import REJECT_INDEX_NAME, read_reject_index
from pipeline.schema import read_split
from scripts.compare_relevance import sign_test_p
from training.calibration import (
    CLIP,
    MODEL,
    REJECT,
    REJECT_GRID,
    Operating,
    Scores,
    auroc,
    calibration_doc,
    calibration_knots,
    decide,
    split_halves,
    thresholds_doc,
)
from training.eval_open_set import (
    DEFAULT_SEED,
    NONCAR,
    STREET,
    OpenSetFold,
    OpenSetPoint,
    binding_constraint,
    cross_fit_open_set,
    load_model_assets,
    model_scores,
    open_set_point,
    relevant_known,
    select_open_set_pair,
    street_groups,
)
from training.tune_thresholds import check_holdout_slots, write_generated_section

GATE_HELD_OUT_GAIN = 0.02
GATE_P = 0.05


@dataclass(frozen=True, slots=True)
class Paired:
    wins: int
    losses: int
    p: float


def _показаны(known: Scores, point: OpenSetPoint) -> np.ndarray:
    показаны: np.ndarray = (
        decide(known, point.sim_min, point.margin_min, point.reject_margin) != REJECT
    )
    return показаны


def paired_shown(known: Scores, base: OpenSetPoint, new: OpenSetPoint) -> Paired:
    """Знакомые, показанные только новой точкой (выигрыши) и только базовой (проигрыши)."""
    старые, новые = _показаны(known, base), _показаны(known, new)
    выигрыши = int((новые & ~старые).sum())
    проигрыши = int((старые & ~новые).sum())
    return Paired(выигрыши, проигрыши, sign_test_p(выигрыши, проигрыши))


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    checks: list[GateCheck]


def gate(
    base_folds: Sequence[OpenSetFold],
    new_folds: Sequence[OpenSetFold],
    base_phone: OpenSetPoint,
    no_worse: OpenSetPoint | None,
    paired: Paired | None,
) -> GateResult:
    """Три условия предрегистрации, по порядку; каждое — с числами, на которых решено."""
    приросты: list[str] = []
    первое = True
    for база, новая in zip(base_folds, new_folds, strict=True):
        if база.held_out is None or новая.held_out is None:
            первое = False
            приросты.append(f"{новая.tuned_on}: —")
            continue
        # Округление до 1e-9: 0.826 − 0.806 во float64 меньше 0.02, а по смыслу равно.
        прирост = round(новая.held_out.coverage - база.held_out.coverage, 9)
        приросты.append(f"{новая.tuned_on}: {прирост:+.3f}")
        первое = первое and прирост >= GATE_HELD_OUT_GAIN
    второе = (
        no_worse is not None
        and no_worse.coverage > base_phone.coverage
        and bool(np.isfinite(no_worse.reject_margin))
    )
    третье = paired is not None and paired.wins > paired.losses and paired.p < GATE_P
    проверки = [
        GateCheck(
            f"покрытие обеих отложенных половин выросло на ≥ {GATE_HELD_OUT_GAIN}",
            первое,
            ", ".join(приросты),
        ),
        GateCheck(
            "точка «не хуже 0.5» есть, покрытие выше телефона 0.5, ветка прототипа включена",
            второе,
            "нет точки"
            if no_worse is None
            else f"{no_worse.coverage:.3f} против {base_phone.coverage:.3f}, "
            f"reject_margin {_порог(no_worse.reject_margin)}",
        ),
        GateCheck(
            f"парный счёт: выигрышей больше, p < {GATE_P}",
            третье,
            "нет точки"
            if paired is None
            else f"выигрышей {paired.wins}, проигрышей {paired.losses}, p = {paired.p:.2g}",
        ),
    ]
    return GateResult(all(п.passed for п in проверки), проверки)


@dataclass(frozen=True, slots=True)
class RejectResult:
    base_folds: list[OpenSetFold]
    new_folds: list[OpenSetFold]
    base_final: OpenSetPoint | None
    new_final: OpenSetPoint | None
    base_phone: OpenSetPoint
    no_worse: OpenSetPoint | None
    paired: Paired | None
    separability: dict[str, tuple[float, float]]
    gate: GateResult


def _порог(reject_margin: float) -> str:
    return f"{reject_margin:.2f}" if np.isfinite(reject_margin) else "выкл"


def _точка_md(точка: OpenSetPoint | None) -> str:
    if точка is None:
        return "— | — | — | — | — | — | —"
    return (
        f"{точка.sim_min:.2f} | {точка.margin_min:.3f} | {_порог(точка.reject_margin)} | "
        f"{точка.coverage:.3f} | {точка.error:.3f} | {точка.fpr[STREET]:.3f} | "
        f"{точка.fpr[NONCAR]:.3f}"
    )


_ШАПКА = (
    f"| sim_min | margin_min | reject_margin | покрытие | ошибка | FPR {STREET} | FPR {NONCAR} |"
)


def render_reject_report(
    result: RejectResult, counts: Mapping[str, int], provenance: Mapping[str, str]
) -> str:
    строки = ["## Происхождение", ""]
    строки += [f"- {ключ}: `{значение}`" for ключ, значение in provenance.items()]
    строки += ["", "## Выборки", ""]
    строки += [f"- {ключ}: {значение}" for ключ, значение in counts.items()]
    строки += [
        "",
        "## Главная метрика: ошибка и FPR каждой группы ≤ 0.15",
        "",
        "| вариант | сторона | где " + _ШАПКА,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for имя, стороны, итог in (
        ("0.5", result.base_folds, result.base_final),
        ("с прототипами", result.new_folds, result.new_final),
    ):
        for с in стороны:
            строки.append(f"| {имя} | {с.tuned_on} | подбор | {_точка_md(с.chosen)} |")
            строки.append(f"| {имя} | {с.tuned_on} | проверка | {_точка_md(с.held_out)} |")
        строки.append(f"| {имя} | итог | — | {_точка_md(итог)} |")
    узкое = "—" if result.new_final is None else binding_constraint(result.new_final)
    строки += [
        "",
        f"Итог с прототипами — узкое ограничение: {узкое}.",
        "",
        "## Точка телефона 0.5 и точка «не хуже 0.5»",
        "",
        "«Не хуже 0.5» — максимум покрытия при ошибке и FPR каждой группы не выше, чем у",
        "точки телефона 0.5 в этом же прогоне.",
        "",
        "| точка " + _ШАПКА,
        "|---|---|---|---|---|---|---|---|",
        f"| телефон 0.5 | {_точка_md(result.base_phone)} |",
        f"| «не хуже 0.5» | {_точка_md(result.no_worse)} |",
        "",
    ]
    if result.paired is not None:
        строки += [
            f"Парный счёт знакомых («не хуже 0.5» против телефона 0.5): выигрышей "
            f"{result.paired.wins}, проигрышей {result.paired.losses}, "
            f"p = {result.paired.p:.2g}.",
            "",
        ]
    строки += [
        "## Разделимость",
        "",
        "| группа | AUROC (sim1) | AUROC (sim1 − r) |",
        "|---|---|---|",
    ]
    строки += [f"| {г} | {a:.3f} | {б:.3f} |" for г, (a, б) in result.separability.items()]
    строки += ["", "## Гейт доставки", ""]
    строки += [f"- {'✓' if п.passed else '✗'} {п.name}: {п.detail}" for п in result.gate.checks]
    строки += [
        "",
        f"**Вердикт:** {'доставка' if result.gate.passed else 'СТОП'}.",
        "",
    ]
    return "\n".join(строки)


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Замер отказных прототипов против 0.5")
    парсер.add_argument("--models", type=Path, default=Path("app/android/models"))
    парсер.add_argument("--expected-sha", required=True, help="sha256 классификатора")
    парсер.add_argument("--reject-dir", type=Path, required=True)
    парсер.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    парсер.add_argument("--relevance", type=Path, default=Path("catalog/holdout_relevance.csv"))
    парсер.add_argument("--street", type=Path, default=splits_dir() / "negatives-street.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--batch-size", type=int, default=32)
    парсер.add_argument("--max-knots", type=int, default=32)
    парсер.add_argument("--report", type=Path, default=Path("bench/open-set-reject.md"))
    парсер.add_argument("--out-dir", type=Path, default=None)
    аргументы = парсер.parse_args(argv)

    эталон = relevant_known(аргументы.holdout, аргументы.relevance)
    улица_кадры = read_split(аргументы.street)
    активы = load_model_assets("0.5", аргументы.models, аргументы.expected_sha)
    check_holdout_slots(list(эталон["model_id"]), list(эталон["class_idx"]), активы.meta)
    прототипы = read_reject_index(аргументы.reject_dir, активы.classifier_sha256)
    истина = эталон["class_idx"].to_numpy(dtype=np.int64)
    знакомые = model_scores(
        активы,
        эталон,
        истина,
        crops_root=аргументы.crops_root,
        batch_size=аргументы.batch_size,
        reject_centroids=прототипы,
    )
    улица = model_scores(
        активы,
        улица_кадры,
        None,
        crops_root=аргументы.crops_root,
        batch_size=аргументы.batch_size,
        reject_centroids=прототипы,
    )
    уличные = street_groups(улица_кадры, улица, аргументы.seed)
    половина = split_halves(истина, аргументы.seed)

    # 0.5 — те же оценки без прототипов: ветка выключена данными, а не порогом.
    база_знакомые = replace(знакомые, sim_reject=None)
    база_группы = {г: replace(s, sim_reject=None) for г, s in уличные.groups.items()}
    sim_min = float(активы.thresholds["sim_min"])
    margin_min = float(активы.thresholds["margin_min"])
    телефон = open_set_point(база_знакомые, база_группы, sim_min, margin_min)
    не_хуже = select_open_set_pair(
        знакомые,
        уличные.groups,
        max_error=телефон.error,
        max_fpr=dict(телефон.fpr),
        reject_grid=REJECT_GRID,
    )
    парный = None if не_хуже is None else paired_shown(знакомые, телефон, не_хуже)
    база_стороны = cross_fit_open_set(база_знакомые, база_группы, половина, уличные.halves)
    новые_стороны = cross_fit_open_set(
        знакомые, уличные.groups, половина, уличные.halves, reject_grid=REJECT_GRID
    )
    assert знакомые.sim_reject is not None
    разрыв_знакомых = знакомые.sim1 - знакомые.sim_reject
    разделимость: dict[str, tuple[float, float]] = {}
    for группа, s in уличные.groups.items():
        assert s.sim_reject is not None
        разделимость[группа] = (
            auroc(знакомые.sim1, s.sim1),
            auroc(разрыв_знакомых, s.sim1 - s.sim_reject),
        )
    результат = RejectResult(
        base_folds=база_стороны,
        new_folds=новые_стороны,
        base_final=select_open_set_pair(база_знакомые, база_группы),
        new_final=select_open_set_pair(знакомые, уличные.groups, reject_grid=REJECT_GRID),
        base_phone=телефон,
        no_worse=не_хуже,
        paired=парный,
        separability=разделимость,
        gate=gate(база_стороны, новые_стороны, телефон, не_хуже, парный),
    )
    счётчики = {"знакомые (релевантный набор)": len(эталон)}
    счётчики.update({вид: int((уличные.kinds == вид).sum()) for вид in sorted(set(уличные.kinds))})
    происхождение = {
        "classifier_sha256": активы.classifier_sha256,
        "reject_index_sha256": file_sha256(аргументы.reject_dir / REJECT_INDEX_NAME),
        "holdout_sha256": file_sha256(аргументы.holdout),
        "relevance_sha256": file_sha256(аргументы.relevance),
        "negatives_street_sha256": file_sha256(аргументы.street),
        "seed": str(аргументы.seed),
        "reject_grid": f"{REJECT_GRID[1]}…{REJECT_GRID[-1]} и выкл",
    }
    write_generated_section(
        аргументы.report, render_reject_report(результат, счётчики, происхождение)
    )
    print(f"гейт: {'пройден' if результат.gate.passed else 'СТОП'}")
    for проверка in результат.gate.checks:
        print(f"  {'✓' if проверка.passed else '✗'} {проверка.name}: {проверка.detail}")
    if not результат.gate.passed or не_хуже is None:
        return 0

    рабочая = Operating(
        sim_min=не_хуже.sim_min,
        margin_min=не_хуже.margin_min,
        coverage=не_хуже.coverage,
        error=не_хуже.error,
        fpr=max(не_хуже.fpr.values()),
        shown=не_хуже.shown,
        reject_margin=не_хуже.reject_margin,
    )
    # Шкала учится на ответах моделью нового правила: при другом `sim_min` старая неверна.
    модель = decide(знакомые, рабочая.sim_min, рабочая.margin_min, рабочая.reject_margin) == MODEL
    узлы = calibration_knots(
        знакомые.sim1[модель], знакомые.class_hit[модель], рабочая.sim_min, аргументы.max_knots
    )
    выход = аргументы.out_dir or (
        data_root() / "calibration" / f"reject_{datetime.now():%Y%m%d_%H%M%S}"
    )
    выход.mkdir(parents=True, exist_ok=True)
    for имя, документ in (
        ("thresholds.json", thresholds_doc(рабочая, len(эталон), len(улица_кадры))),
        ("calibration.json", calibration_doc(узлы, CLIP)),
    ):
        (выход / имя).write_text(
            json.dumps(документ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"ассеты: {выход}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
