"""Честный замер открытого множества (спека 2026-09-27-open-set-street §4).

Главная метрика — покрытие своих машин при FPR ≤ 0.15 на каждой группе негативов сразу:
в приложении порог один, и он обязан держать и чужие машины, и не-машины. Правило решения и
сетка — те же, что у подбора порогов (`training.calibration`) и на телефоне.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.holdout_relevance import SET_RELEVANT, load_relevance, subsets
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import max_sim, top2_with_sims
from pipeline.negatives import neg_kind_of
from pipeline.paths import harvest_dir, splits_dir
from pipeline.pool_audit import file_sha256
from pipeline.schema import read_split
from training.calibration import (
    MARGIN_GRID,
    REJECT_OFF,
    SIM_GRID,
    Scores,
    auroc,
    fpr_at_tpr,
    operating_point,
    split_halves,
)
from training.tune_thresholds import (
    check_holdout_slots,
    embed_crops,
    scores_from,
    write_generated_section,
)

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
NO_REJECT = np.array([REJECT_OFF])
# Предел FPR — один на все группы или свой у каждой: точка «не хуже 0.5» держит FPR каждой
# группы на уровне телефона 0.5, а не общий потолок.
FprLimits = float | Mapping[str, float]


def _предел(max_fpr: FprLimits, группа: str) -> float:
    return float(max_fpr[группа]) if isinstance(max_fpr, Mapping) else float(max_fpr)


@dataclass(frozen=True, slots=True)
class OpenSetPoint:
    """Пара порогов: покрытие и ошибка на знакомых, FPR — на каждой группе негативов."""

    sim_min: float
    margin_min: float
    coverage: float
    error: float
    shown: int
    fpr: dict[str, float]
    reject_margin: float = REJECT_OFF


def _проверить_группы(groups: Mapping[str, Scores]) -> None:
    """FPR пустой группы равен нулю и «проходит» любое ограничение — такой замер ложен."""
    пустые = [имя for имя, негативы in groups.items() if len(негативы.sim1) == 0]
    if not groups or пустые:
        raise ValueError(f"пустая группа негативов: {', '.join(пустые) or 'групп нет'}")


def open_set_point(
    known: Scores,
    groups: Mapping[str, Scores],
    sim_min: float,
    margin_min: float,
    reject_margin: float = REJECT_OFF,
) -> OpenSetPoint:
    точки = {
        имя: operating_point(known, негативы, sim_min, margin_min, reject_margin)
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
        reject_margin=float(reject_margin),
    )


def select_open_set_pair(
    known: Scores,
    groups: Mapping[str, Scores],
    *,
    max_error: float = MAX_ERROR,
    max_fpr: FprLimits = MAX_FPR,
    sim_grid: np.ndarray = SIM_GRID,
    margin_grid: np.ndarray = MARGIN_GRID,
    reject_grid: np.ndarray = NO_REJECT,
) -> OpenSetPoint | None:
    """Пара с наибольшим покрытием, держащая ошибку и FPR всех групп; тай-брейк — сетка.

    Порядок обхода и замена только на строго большее покрытие — как в `select_pair`: при
    равенстве остаётся меньший `sim_min`, затем меньший `margin_min`, затем меньший
    `reject_margin` — первым в `REJECT_GRID` стоит выключенная ветка, и прототип выбирается,
    только когда он даёт больше покрытия.
    """
    _проверить_группы(groups)
    лучшая: OpenSetPoint | None = None
    for sim_min in sim_grid:
        for margin_min in margin_grid:
            for reject_margin in reject_grid:
                точка = open_set_point(
                    known, groups, float(sim_min), float(margin_min), float(reject_margin)
                )
                if точка.shown == 0 or точка.error > max_error:
                    continue
                if any(fpr > _предел(max_fpr, имя) for имя, fpr in точка.fpr.items()):
                    continue
                if лучшая is None or точка.coverage > лучшая.coverage:
                    лучшая = точка
    return лучшая


def binding_constraint(
    point: OpenSetPoint, *, max_error: float = MAX_ERROR, max_fpr: FprLimits = MAX_FPR
) -> str:
    """Ограничение с наименьшим запасом до предела: оно и держит покрытие."""
    запасы = {"ошибка": max_error - point.error}
    запасы.update({f"FPR {имя}": _предел(max_fpr, имя) - fpr for имя, fpr in point.fpr.items()})
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
    max_fpr: FprLimits = MAX_FPR,
    reject_grid: np.ndarray = NO_REJECT,
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
            known.subset(свои),
            учёба,
            max_error=max_error,
            max_fpr=max_fpr,
            reject_grid=reject_grid,
        )
        if точка is None:
            стороны.append(OpenSetFold(имя, None, None))
            continue
        отложенная = open_set_point(
            known.subset(~свои),
            проверка,
            точка.sim_min,
            точка.margin_min,
            точка.reject_margin,
        )
        стороны.append(OpenSetFold(имя, точка, отложенная))
    return стороны


@dataclass(frozen=True, slots=True)
class ModelResult:
    label: str
    classifier_sha256: str
    folds: list[OpenSetFold]
    final: OpenSetPoint | None
    phone: OpenSetPoint
    reference_fpr_at_phone: float
    separability: dict[str, tuple[float, float]]


def _точка_md(точка: OpenSetPoint | None) -> str:
    if точка is None:
        return "— | — | — | — | — | —"
    return (
        f"{точка.sim_min:.2f} | {точка.margin_min:.3f} | {точка.coverage:.3f} | "
        f"{точка.error:.3f} | {точка.fpr[STREET]:.3f} | {точка.fpr[NONCAR]:.3f}"
    )


def render_open_set_report(
    results: Sequence[ModelResult], counts: Mapping[str, int], provenance: Mapping[str, str]
) -> str:
    строки = ["## Происхождение", ""]
    строки += [f"- {ключ}: `{значение}`" for ключ, значение in provenance.items()]
    строки += ["", "## Выборки", ""]
    строки += [f"- {ключ}: {значение}" for ключ, значение in counts.items()]
    шапка = f"| sim_min | margin_min | покрытие | ошибка | FPR {STREET} | FPR {NONCAR} |"
    for р in results:
        строки += [
            "",
            f"## Модель {р.label}",
            "",
            f"Классификатор `{р.classifier_sha256}`. Знакомые — релевантный набор.",
            "",
            "| сторона | где " + шапка,
            "|---|---|---|---|---|---|---|---|",
        ]
        for с in р.folds:
            строки.append(f"| {с.tuned_on} | подбор | {_точка_md(с.chosen)} |")
            строки.append(f"| {с.tuned_on} | проверка | {_точка_md(с.held_out)} |")
        узкое = "—" if р.final is None else binding_constraint(р.final)
        строки += [
            "",
            "Итог на всём наборе:",
            "",
            шапка,
            "|---|---|---|---|---|---|",
            f"| {_точка_md(р.final)} |",
            "",
            f"Итог — узкое ограничение: {узкое}.",
            "",
            "Рабочая точка телефона:",
            "",
            шапка,
            "|---|---|---|---|---|---|",
            f"| {_точка_md(р.phone)} |",
            "",
            f"Проверка кода — справочный off_catalog в точке телефона: FPR "
            f"{р.reference_fpr_at_phone:.3f}.",
            "",
            "| группа | AUROC (sim1) | FPR@TPR95 |",
            "|---|---|---|",
        ]
        строки += [f"| {г} | {a:.3f} | {f:.3f} |" for г, (a, f) in р.separability.items()]
    строки.append("")
    return "\n".join(строки)


@dataclass(frozen=True, slots=True)
class ModelAssets:
    """Ассеты модели, сверенные по sha классификатора и sha меты индекса."""

    label: str
    classifier: Path
    classifier_sha256: str
    manifest: dict[str, Any]
    centroids: np.ndarray
    meta: dict[str, Any]
    brands: list[str | None]
    thresholds: dict[str, Any]


def load_model_assets(label: str, models: Path, expected_sha: str) -> ModelAssets:
    манифест = json.loads((models / "models_manifest.json").read_text("utf-8"))
    классификатор = models / манифест["files"]["classifier"]["name"]
    sha = file_sha256(классификатор)
    if sha != expected_sha:
        raise SystemExit(f"{label}: sha256 классификатора {sha}, ожидался {expected_sha}")
    байты_меты = (models / "catalog_meta.json").read_bytes()
    заголовок, центроиды = read_index(models / "catalog_index.bin")
    verify_meta(заголовок, байты_меты)
    мета = json.loads(байты_меты)
    return ModelAssets(
        label=label,
        classifier=классификатор,
        classifier_sha256=sha,
        manifest=манифест,
        centroids=центроиды,
        meta=мета,
        brands=[класс["brand"] for класс in мета["classes"]],
        thresholds=json.loads((models / "thresholds.json").read_text("utf-8")),
    )


def model_scores(
    assets: ModelAssets,
    frame: pd.DataFrame,
    truth: np.ndarray | None,
    *,
    crops_root: Path,
    batch_size: int,
    reject_centroids: np.ndarray | None = None,
) -> Scores:
    """Оценки кадров; с прототипами — ещё и `sim_reject` по тем же эмбеддингам."""
    пути = [crops_root / п for п in frame["crop_path"]]
    эмбеддинги = embed_crops(assets.classifier, пути, assets.manifest, batch_size)
    оценки = scores_from(*top2_with_sims(эмбеддинги, assets.centroids), truth, assets.brands)
    if reject_centroids is None:
        return оценки
    return replace(оценки, sim_reject=max_sim(эмбеддинги, reject_centroids))


@dataclass(frozen=True, slots=True)
class StreetGroups:
    kinds: np.ndarray
    groups: dict[str, Scores]
    halves: dict[str, np.ndarray]


def street_groups(street: pd.DataFrame, scores: Scores, seed: int) -> StreetGroups:
    """Группы негативов по видам и их деление: чужие машины — моделями, не-машины — по виду."""
    виды = np.array([neg_kind_of(str(л)) for л in street["listing_id"]])
    маски = {г: np.isin(виды, GROUP_KINDS[г]) for г in (STREET, NONCAR)}
    return StreetGroups(
        kinds=виды,
        groups={г: scores.subset(м) for г, м in маски.items()},
        halves={
            STREET: split_by_group(street["listing_id"].to_numpy()[маски[STREET]], seed),
            NONCAR: split_halves(виды[маски[NONCAR]], seed),
        },
    )


def measure_model(
    label: str,
    models: Path,
    expected_sha: str,
    known: pd.DataFrame,
    street: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    crops_root: Path,
    seed: int,
    batch_size: int,
) -> ModelResult:
    активы = load_model_assets(label, models, expected_sha)
    check_holdout_slots(list(known["model_id"]), list(known["class_idx"]), активы.meta)
    истина = known["class_idx"].to_numpy(dtype=np.int64)
    знакомые = model_scores(активы, known, истина, crops_root=crops_root, batch_size=batch_size)
    улица = model_scores(активы, street, None, crops_root=crops_root, batch_size=batch_size)
    справка = model_scores(активы, reference, None, crops_root=crops_root, batch_size=batch_size)
    уличные = street_groups(street, улица, seed)
    стороны = cross_fit_open_set(
        знакомые, уличные.groups, split_halves(истина, seed), уличные.halves
    )
    sim_min = float(активы.thresholds["sim_min"])
    margin_min = float(активы.thresholds["margin_min"])
    по_видам = {вид: улица.subset(уличные.kinds == вид) for вид in ("truck_bus", "detector_fp")}
    разделимость = {
        имя: (auroc(знакомые.sim1, s.sim1), fpr_at_tpr(знакомые.sim1, s.sim1))
        for имя, s in {**уличные.groups, **по_видам, REFERENCE_KIND: справка}.items()
        if len(s.sim1)
    }
    return ModelResult(
        label=label,
        classifier_sha256=активы.classifier_sha256,
        folds=стороны,
        final=select_open_set_pair(знакомые, уличные.groups),
        phone=open_set_point(знакомые, уличные.groups, sim_min, margin_min),
        reference_fpr_at_phone=operating_point(знакомые, справка, sim_min, margin_min).fpr,
        separability=разделимость,
    )


def relevant_known(holdout: Path, relevance: Path) -> pd.DataFrame:
    """Знакомые — релевантный набор `holdout_ru` без многомашинных кадров (правило §5.2)."""
    эталон = read_split(holdout)
    эталон = эталон[~эталон["is_multi_car"].astype(bool)].reset_index(drop=True)
    маска = subsets(эталон["photo_id"].tolist(), load_relevance(relevance))
    return эталон[маска[SET_RELEVANT]].reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Замер открытого множества на уличных негативах")
    парсер.add_argument(
        "--model",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "DIR", "SHA256"),
        help="метка, каталог ассетов, sha256 классификатора",
    )
    парсер.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    парсер.add_argument("--relevance", type=Path, default=Path("catalog/holdout_relevance.csv"))
    парсер.add_argument("--street", type=Path, default=splits_dir() / "negatives-street.parquet")
    парсер.add_argument("--reference", type=Path, default=splits_dir() / "negatives.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--batch-size", type=int, default=32)
    парсер.add_argument("--report", type=Path, default=Path("bench/open-set-street.md"))
    аргументы = парсер.parse_args(argv)

    эталон = relevant_known(аргументы.holdout, аргументы.relevance)
    улица = read_split(аргументы.street)
    справка = read_split(аргументы.reference)
    справка = справка[
        [neg_kind_of(str(л)) == REFERENCE_KIND for л in справка["listing_id"]]
    ].reset_index(drop=True)

    результаты = [
        measure_model(
            метка,
            Path(каталог),
            sha,
            эталон,
            улица,
            справка,
            crops_root=аргументы.crops_root,
            seed=аргументы.seed,
            batch_size=аргументы.batch_size,
        )
        for метка, каталог, sha in аргументы.model
    ]
    виды = [neg_kind_of(str(л)) for л in улица["listing_id"]]
    счётчики = {"знакомые (релевантный набор)": len(эталон)}
    счётчики.update({вид: виды.count(вид) for вид in sorted(set(виды))})
    счётчики["справочный off_catalog"] = len(справка)
    происхождение = {
        "holdout_sha256": file_sha256(аргументы.holdout),
        "relevance_sha256": file_sha256(аргументы.relevance),
        "negatives_street_sha256": file_sha256(аргументы.street),
        "negatives_reference_sha256": file_sha256(аргументы.reference),
        "seed": str(аргументы.seed),
        "max_error / max_fpr": f"{MAX_ERROR} / {MAX_FPR}",
        "sim_grid / margin_grid": f"{SIM_GRID[0]}…{SIM_GRID[-1]} / {MARGIN_GRID[0]}…"
        f"{MARGIN_GRID[-1]}",
    }
    write_generated_section(
        аргументы.report, render_open_set_report(результаты, счётчики, происхождение)
    )
    for р in результаты:
        итог = "нет допустимой пары" if р.final is None else f"покрытие {р.final.coverage:.3f}"
        print(f"{р.label}: {итог}; телефон FPR {р.phone.fpr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
