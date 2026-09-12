"""Frozen-unit few-shot guardrail для выбранного профиля редакции."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pipeline.catalog import Catalog, class_index, member_index
from pipeline.harvest.manifest import HarvestRow
from pipeline.plate_leak import clopper_pearson
from pipeline.redaction_calibration.baseline import FrozenUnit, Prediction
from pipeline.schema import COLUMNS

GUARDRAIL_SUCCESS_PHRASE = (
    "guardrail не обнаружил обвала; это не доказательство отсутствия ухудшения"
)


@dataclass(frozen=True, slots=True)
class GuardrailMetric:
    """Одна ось guardrail: точный базис, штрафной итог и диагностика выживших."""

    baseline_k: int
    candidate_k_all: int
    survivors: int
    destroyed: int
    candidate_k_survivors: int
    lower_bound: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    """Обе оси guardrail и хеши реально измеренных артефактов."""

    top1: GuardrailMetric
    top3: GuardrailMetric
    pool_sha256: str
    split_sha256: str
    index_sha256: str


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _frozen_by_photo(frozen_map: Sequence[FrozenUnit]) -> dict[str, FrozenUnit]:
    photo_ids = [unit.photo_id for unit in frozen_map]
    duplicates = _duplicates(photo_ids)
    if duplicates:
        raise ValueError(f"дубликат photo_id во frozen_map: {duplicates[:5]}")
    return {unit.photo_id: unit for unit in frozen_map}


def _manifest_to_split(
    new_manifest: Sequence[HarvestRow],
    frozen_map: Sequence[FrozenUnit],
    catalog: Catalog,
    *,
    split: Literal["pool", "holdout_ru"],
    excluded_listing_ids: frozenset[str] | set[str] = frozenset(),
) -> pd.DataFrame:
    """Провалидировать новый манифест и восстановить §13.1 из frozen authority."""
    frozen = _frozen_by_photo(frozen_map)
    photo_ids = [row.sha256 for row in new_manifest]
    duplicates = _duplicates(photo_ids)
    if duplicates:
        raise ValueError(f"дубликат photo_id в новом манифесте: {duplicates[:5]}")
    unknown = sorted(set(photo_ids) - set(frozen))
    if unknown:
        raise ValueError(f"неизвестные photo_id в новом манифесте: {unknown[:5]}")

    class_ids = class_index(catalog)
    model_to_class = member_index(catalog)
    models = {model.model_id: model for model in catalog.models}
    classes = {class_.class_id: class_ for class_ in catalog.classes}
    records: list[dict[str, object]] = []
    for row in new_manifest:
        unit = frozen[row.sha256]
        if unit.split != split or unit.listing_id in excluded_listing_ids:
            continue
        if row.class_id not in class_ids:
            raise ValueError(f"{row.sha256}: неизвестный class_id {row.class_id!r}")
        model_id = classes[row.class_id].members[0]
        if model_to_class[model_id] != row.class_id:
            raise ValueError(f"{row.sha256}: class_id не соответствует каталогу")
        records.append(
            {
                "photo_id": row.sha256,
                "listing_id": unit.listing_id,
                "source": row.source,
                "crop_path": f"{row.class_id}/{row.file}",
                "brand": models[model_id].brand,
                "model_id": model_id,
                "class_idx": class_ids[row.class_id],
                "body_style": None,
                "year": None,
                "bbox": [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h],
                "bbox_area_frac": row.bbox_area_frac,
                "det_conf": row.det_conf,
                "phash": row.phash,
                "raw_model_string": None,
                "is_multi_car": False,
            }
        )
    return pd.DataFrame.from_records(records, columns=COLUMNS)


def _frozen_holdout_listings(
    frozen_map: Sequence[FrozenUnit], frozen_holdout_ids: set[str]
) -> set[str]:
    frozen = _frozen_by_photo(frozen_map)
    unknown = sorted(frozen_holdout_ids - set(frozen))
    if unknown:
        raise ValueError(f"неизвестные frozen holdout photo_id: {unknown[:5]}")
    wrong_split = sorted(
        photo_id for photo_id in frozen_holdout_ids if frozen[photo_id].split != "holdout_ru"
    )
    if wrong_split:
        raise ValueError(f"frozen holdout ID не из holdout_ru: {wrong_split[:5]}")
    return {frozen[photo_id].listing_id for photo_id in frozen_holdout_ids}


def build_guardrail_pool(
    new_manifest: Sequence[HarvestRow],
    frozen_map: Sequence[FrozenUnit],
    frozen_holdout_ids: set[str],
    catalog: Catalog,
) -> pd.DataFrame:
    """Построить новый pool, исключив исходные единицы frozen holdout целиком."""
    excluded = _frozen_holdout_listings(frozen_map, frozen_holdout_ids)
    return _manifest_to_split(
        new_manifest,
        frozen_map,
        catalog,
        split="pool",
        excluded_listing_ids=excluded,
    )


def build_guardrail_holdout(
    new_manifest: Sequence[HarvestRow],
    frozen_map: Sequence[FrozenUnit],
    frozen_holdout_ids: set[str],
    catalog: Catalog,
) -> pd.DataFrame:
    """Построить выжившую часть frozen holdout по новым кропам."""
    _frozen_holdout_listings(frozen_map, frozen_holdout_ids)
    return _manifest_to_split(new_manifest, frozen_map, catalog, split="holdout_ru")


def assert_no_frozen_units(pool: pd.DataFrame, frozen_units: Sequence[FrozenUnit]) -> None:
    """На финальном parquet не должно быть кадра или sibling frozen holdout."""
    holdout = [unit for unit in frozen_units if unit.split == "holdout_ru"]
    holdout_ids = {unit.photo_id for unit in holdout}
    holdout_listings = {unit.listing_id for unit in holdout}
    bad_ids = sorted(set(str(value) for value in pool["photo_id"]) & holdout_ids)
    bad_listings = sorted(set(str(value) for value in pool["listing_id"]) & holdout_listings)
    if bad_ids or bad_listings:
        raise ValueError(
            "финальный pool пересекается с замороженными holdout-единицами: "
            f"photo_id={bad_ids[:5]}, listing_id={bad_listings[:5]}"
        )


def _prediction_map(predictions: Sequence[Prediction], *, label: str) -> dict[str, Prediction]:
    ids = [prediction.photo_id for prediction in predictions]
    duplicates = _duplicates(ids)
    if duplicates:
        raise ValueError(f"дубликат photo_id в {label}: {duplicates[:5]}")
    return {prediction.photo_id: prediction for prediction in predictions}


def _hit(prediction: Prediction, top_k: int) -> bool:
    return prediction.target in prediction.top_indices[:top_k]


def _guardrail_metric(
    baseline: Sequence[Prediction], candidate_by_id: Mapping[str, Prediction], top_k: int
) -> GuardrailMetric:
    baseline_k = sum(_hit(prediction, top_k) for prediction in baseline)
    candidate_k = sum(
        _hit(candidate_by_id[prediction.photo_id], top_k)
        for prediction in baseline
        if prediction.photo_id in candidate_by_id
    )
    n = len(baseline)
    lower_bound = clopper_pearson(baseline_k, n)[0]
    survivors = len(candidate_by_id)
    return GuardrailMetric(
        baseline_k=baseline_k,
        candidate_k_all=candidate_k,
        survivors=survivors,
        destroyed=n - survivors,
        candidate_k_survivors=candidate_k,
        lower_bound=lower_bound,
        passed=candidate_k / n >= lower_bound,
    )


def calculate_guardrail_report(
    baseline: Sequence[Prediction],
    candidate: Sequence[Prediction],
    *,
    pool_sha256: str,
    split_sha256: str,
    index_sha256: str,
) -> GuardrailReport:
    """Считать штрафные top-1/top-3 против exact baseline и survivor-диагностику."""
    if not baseline:
        raise ValueError("baseline predictions пусты")
    baseline_by_id = _prediction_map(baseline, label="baseline")
    candidate_by_id = _prediction_map(candidate, label="candidate")
    baseline_ids = set(baseline_by_id)
    foreign = sorted(set(candidate_by_id) - baseline_ids)
    if foreign:
        raise ValueError(f"неизвестные candidate photo_id: {foreign[:5]}")
    for photo_id, prediction in candidate_by_id.items():
        if prediction.target != baseline_by_id[photo_id].target:
            raise ValueError(f"target кандидата не совпал с frozen baseline для {photo_id}")
        if len(prediction.top_indices) < 3:
            raise ValueError(f"{photo_id}: candidate top_indices короче трёх")
    for prediction in baseline:
        if len(prediction.top_indices) < 3:
            raise ValueError(f"{prediction.photo_id}: baseline top_indices короче трёх")
    return GuardrailReport(
        top1=_guardrail_metric(baseline, candidate_by_id, 1),
        top3=_guardrail_metric(baseline, candidate_by_id, 3),
        pool_sha256=pool_sha256,
        split_sha256=split_sha256,
        index_sha256=index_sha256,
    )


def _metric_row(name: str, metric: GuardrailMetric) -> str:
    n = metric.survivors + metric.destroyed
    survivor_value = metric.candidate_k_survivors / metric.survivors if metric.survivors else 0.0
    result = "ПРОЙДЕН" if metric.passed else "НЕ ПРОЙДЕН"
    return (
        f"| {name} | {metric.baseline_k}/{n} | {metric.lower_bound:.6f} "
        f"| {metric.candidate_k_all}/{n} | {metric.candidate_k_survivors}/"
        f"{metric.survivors} ({survivor_value:.4f}) | {result} |"
    )


def render_guardrail_report(report: GuardrailReport) -> str:
    """Собрать отчёт, не приписывая guardrail доказательство not_worse."""
    passed = report.top1.passed and report.top3.passed
    verdict = GUARDRAIL_SUCCESS_PHRASE if passed else "guardrail обнаружил обвал"
    destroyed = report.top1.destroyed
    text = "\n".join(
        [
            "# эффект профиля при постоянной линейке из 1523 кадров",
            "",
            "Guardrail проверяет только отсутствие обвала при постоянной линейке. Он включает",
            "новые пиксели, новый индексный пул, отсев и гигиену H3; причинность одной причине",
            "не приписывается.",
            "",
            f"- sha256 guardrail pool: `{report.pool_sha256}`",
            f"- sha256 survivor split: `{report.split_sha256}`",
            f"- sha256 index: `{report.index_sha256}`",
            f"- уничтожено/не принято кадров линейки: {destroyed}",
            "",
            "| ось | exact baseline | нижняя 95% граница | все 1523 | выжившие | итог |",
            "|---|---:|---:|---:|---:|---|",
            _metric_row("top-1", report.top1),
            _metric_row("top-3", report.top3),
            "",
            f"**Вердикт: {verdict}.**",
            "",
        ]
    )
    return text + "\n"
