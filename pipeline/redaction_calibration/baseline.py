"""Неизменяемый базис кампании: точные предсказания, единицы и партии."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import pandas as pd

from pipeline.redaction_calibration.state import BatchRecord, BatchState, CampaignBudgetState
from pipeline.redaction_calibration.state import (
    campaign_budget_state as state_campaign_budget_state,
)

PersistedBatchState: TypeAlias = Literal[
    "sealed", "opened", "review_in_progress", "completed_pass", "completed_fail"
]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Точное ранжирование одного кадра постоянной few-shot линейки."""

    photo_id: str
    target: int
    top_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FrozenUnit:
    """Исходная неделимая единица кадра до любой пересборки профиля."""

    photo_id: str
    listing_id: str
    split: Literal["pool", "holdout_ru"]


@dataclass(frozen=True, slots=True)
class IndependentBatch:
    """Запечатанная независимая партия с полным списком и его хешем."""

    number: int
    state: PersistedBatchState
    photo_ids: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ExactMetrics:
    """Целочисленные успехи top-1/top-3 с общим точным знаменателем."""

    top1_k: int
    top3_k: int
    n: int


@dataclass(frozen=True, slots=True)
class BaselinePassport:
    """Машинный авторитет всех входов, свидетелей и точных результатов базиса."""

    schema_version: int
    created_at: str
    inputs: dict[str, dict[str, str | int]]
    pool_frames: int
    holdout_frames: int
    calibration_frames: int
    profile: dict[str, Any]
    git: dict[str, str | bool]
    detector_stage_sha256: str
    code_sha256: dict[str, str]
    weights_sha256: dict[str, str]
    jpeg: dict[str, str]
    runtime: dict[str, str]
    exact_metrics: ExactMetrics
    frozen_units: tuple[FrozenUnit, ...]
    calibration_ids: tuple[str, ...]
    independent_batches: tuple[IndependentBatch, ...]
    independent_remainder: tuple[str, ...]
    independent_seed: int
    independent_batch_size: int


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _ids_sha256(*, number: int | str, photo_ids: Sequence[str]) -> str:
    document = {"number": number, "photo_ids": list(photo_ids)}
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{payload}\n".encode()).hexdigest()


def partition_independent_ids(
    pool_ids: Sequence[str],
    calibration_ids: Sequence[str],
    *,
    seed: int = 1337,
    batch_size: int = 300,
    batch_count: int = 8,
) -> tuple[tuple[IndependentBatch, ...], tuple[str, ...]]:
    """До профилей разложить pool на полные партии и нерасходуемый остаток."""
    pool_duplicates = _duplicates(pool_ids)
    if pool_duplicates:
        raise ValueError(f"дубликат в pool_ids: {pool_duplicates[:5]}")
    calibration_duplicates = _duplicates(calibration_ids)
    if calibration_duplicates:
        raise ValueError(f"дубликат в calibration_ids: {calibration_duplicates[:5]}")
    if batch_size < 1 or batch_count < 1:
        raise ValueError("размер и число партий должны быть положительными")
    pool_set = set(pool_ids)
    unknown = sorted(set(calibration_ids) - pool_set)
    if unknown:
        raise ValueError(f"неизвестные calibration_ids: {unknown[:5]}")

    eligible = sorted(pool_set - set(calibration_ids))
    shuffled = list(eligible)
    random.Random(seed).shuffle(shuffled)
    needed = batch_size * batch_count
    if len(shuffled) < needed:
        raise ValueError(
            f"для {batch_count} полных партий нужно {needed} ID, доступно {len(shuffled)}"
        )
    batches = tuple(
        IndependentBatch(
            number=index + 1,
            state="sealed",
            photo_ids=tuple(shuffled[index * batch_size : (index + 1) * batch_size]),
            sha256=_ids_sha256(
                number=index + 1,
                photo_ids=shuffled[index * batch_size : (index + 1) * batch_size],
            ),
        )
        for index in range(batch_count)
    )
    return batches, tuple(shuffled[needed:])


def freeze_units(pool: pd.DataFrame, holdout: pd.DataFrame) -> tuple[FrozenUnit, ...]:
    """Заморозить конфликт-свободное `photo_id → listing_id` из обоих parquet."""
    collected: list[FrozenUnit] = []
    mappings: dict[str, str] = {}
    for split, frame in (("pool", pool), ("holdout_ru", holdout)):
        missing = {"photo_id", "listing_id"} - set(frame.columns)
        if missing:
            raise ValueError(f"нет колонок frozen-unit: {sorted(missing)}")
        ids = [str(value) for value in frame["photo_id"]]
        duplicates = _duplicates(ids)
        if duplicates:
            raise ValueError(f"дубликат photo_id в {split}: {duplicates[:5]}")
        for photo_id, listing_id_value in zip(frame["photo_id"], frame["listing_id"], strict=True):
            key = str(photo_id)
            listing_id = str(listing_id_value)
            prior = mappings.get(key)
            if prior is not None and prior != listing_id:
                raise ValueError(f"конфликт photo_id {key!r}: {prior!r} != {listing_id!r}")
            if prior is not None:
                raise ValueError(f"photo_id {key!r} одновременно в pool и holdout_ru")
            mappings[key] = listing_id
            collected.append(FrozenUnit(key, listing_id, cast(Any, split)))
    return tuple(sorted(collected, key=lambda unit: unit.photo_id))


def exact_prediction_counts(
    predictions: Sequence[Prediction], *, expected_ids: set[str]
) -> dict[str, dict[str, int]]:
    """Проверить полный экспорт и вывести точные целые успехи, не округлённые доли."""
    ids = [prediction.photo_id for prediction in predictions]
    duplicates = _duplicates(ids)
    if duplicates:
        raise ValueError(f"дубликат prediction photo_id: {duplicates[:5]}")
    missing = sorted(expected_ids - set(ids))
    if missing:
        raise ValueError(f"нет предсказаний для photo_id: {missing[:5]}")
    foreign = sorted(set(ids) - expected_ids)
    if foreign:
        raise ValueError(f"неизвестные предсказания photo_id: {foreign[:5]}")
    for prediction in predictions:
        if len(prediction.top_indices) < 3:
            raise ValueError(f"{prediction.photo_id}: top_indices короче трёх")
    n = len(predictions)
    return {
        "top1": {
            "k": sum(prediction.top_indices[0] == prediction.target for prediction in predictions),
            "n": n,
        },
        "top3": {
            "k": sum(prediction.target in prediction.top_indices[:3] for prediction in predictions),
            "n": n,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path, rows: int) -> dict[str, str | int]:
    return {"name": path.name, "rows": rows, "sha256": _sha256(path)}


def _validate_hashes(label: str, values: Mapping[str, str]) -> None:
    invalid = sorted(
        name for name, value in values.items() if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
    )
    if invalid:
        raise ValueError(f"{label}: SHA256 должен быть полным шестнадцатеричным: {invalid}")


def create_baseline_passport(
    *,
    manifest_path: Path,
    pool_path: Path,
    holdout_path: Path,
    pool: pd.DataFrame,
    holdout: pd.DataFrame,
    calibration_ids: Sequence[str],
    predictions: Sequence[Prediction],
    profile: Mapping[str, Any],
    git_sha: str,
    git_clean: bool,
    detector_stage_sha256: str,
    code_sha256: Mapping[str, str],
    weights_sha256: Mapping[str, str],
    jpeg: Mapping[str, str],
    runtime: Mapping[str, str],
    created_at: str,
    seed: int = 1337,
    batch_size: int = 300,
    batch_count: int = 8,
) -> BaselinePassport:
    """Собрать паспорт только из полных первичных свидетелей и точных предсказаний."""
    if not isinstance(predictions, Sequence) or isinstance(predictions, Mapping):
        raise TypeError("predictions должны быть последовательностью Prediction, не метриками")
    if not git_clean:
        raise ValueError("Git-дерево должно быть чистым до предрегистрации")
    if re.fullmatch(r"[0-9a-fA-F]{40}", git_sha) is None:
        raise ValueError("нужен полный Git SHA из 40 шестнадцатеричных символов")
    _validate_hashes("code_sha256", code_sha256)
    _validate_hashes("weights_sha256", weights_sha256)
    if re.fullmatch(r"[0-9a-fA-F]{64}", detector_stage_sha256) is None:
        raise ValueError("detector_stage_sha256 должен быть полным шестнадцатеричным")

    units = freeze_units(pool, holdout)
    pool_ids = [str(value) for value in pool["photo_id"]]
    holdout_ids = {str(value) for value in holdout["photo_id"]}
    if set(pool_ids) & holdout_ids:
        raise ValueError("pool и holdout_ru пересекаются по photo_id")
    calibration = tuple(sorted(str(value) for value in calibration_ids))
    batches, remainder = partition_independent_ids(
        pool_ids,
        calibration,
        seed=seed,
        batch_size=batch_size,
        batch_count=batch_count,
    )
    counts = exact_prediction_counts(predictions, expected_ids=holdout_ids)
    return BaselinePassport(
        schema_version=1,
        created_at=created_at,
        inputs={
            "manifest": _input_record(manifest_path, _csv_rows(manifest_path)),
            "pool": _input_record(pool_path, len(pool)),
            "holdout_ru": _input_record(holdout_path, len(holdout)),
        },
        pool_frames=len(pool),
        holdout_frames=len(holdout),
        calibration_frames=len(calibration),
        profile=dict(profile),
        git={"sha": git_sha, "clean": git_clean},
        detector_stage_sha256=detector_stage_sha256,
        code_sha256=dict(sorted(code_sha256.items())),
        weights_sha256=dict(sorted(weights_sha256.items())),
        jpeg=dict(sorted(jpeg.items())),
        runtime=dict(sorted(runtime.items())),
        exact_metrics=ExactMetrics(
            top1_k=counts["top1"]["k"],
            top3_k=counts["top3"]["k"],
            n=counts["top1"]["n"],
        ),
        frozen_units=units,
        calibration_ids=calibration,
        independent_batches=batches,
        independent_remainder=remainder,
        independent_seed=seed,
        independent_batch_size=batch_size,
    )


def _csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def campaign_budget_state(batches: Sequence[IndependentBatch]) -> CampaignBudgetState:
    """Вернуть границу бюджета для персистентных партий по правилу из `state`.

    Переходы партий принадлежат `state.transition_batch` (решение В-1 плана рекалибровки от
    2026-09-12); здесь остаётся только persistence, поэтому правило не дублируется, а
    вызывается через перевод записей в `BatchRecord`.
    """
    return state_campaign_budget_state(
        [
            BatchRecord(
                number=batch.number,
                state=BatchState(batch.state),
                batch_sha256=batch.sha256,
            )
            for batch in batches
        ]
    )


def _json_document(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_document(item) for item in value]
    if isinstance(value, list):
        return [_json_document(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_document(item) for key, item in value.items()}
    return value


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_document(document), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_campaign_artifacts(
    root: Path,
    passport: BaselinePassport,
    predictions: Sequence[Prediction],
    preregistration: Mapping[str, Any],
) -> tuple[Path, ...]:
    baseline_root = root / "baseline"
    baseline_root.mkdir(parents=True)
    _write_json(
        root / "campaign.json",
        {
            "schema_version": 1,
            "state": "preregistered",
            "created_at": passport.created_at,
            "training_authorized": False,
            "training_authorization_state": "full_pool_audited",
            "candidate_results_present": False,
        },
    )
    _write_json(root / "preregistration.json", dict(preregistration))
    _write_json(baseline_root / "passport.json", asdict(passport))
    units = [asdict(unit) for unit in passport.frozen_units]
    _write_csv(baseline_root / "frozen-unit-map.csv", ("photo_id", "listing_id", "split"), units)
    _write_csv(
        baseline_root / "frozen-holdout.csv",
        ("photo_id", "listing_id", "split"),
        [unit for unit in units if unit["split"] == "holdout_ru"],
    )
    _write_csv(
        baseline_root / "baseline-predictions.csv",
        ("photo_id", "target", "top_indices"),
        [
            {
                "photo_id": prediction.photo_id,
                "target": prediction.target,
                "top_indices": " ".join(map(str, prediction.top_indices)),
            }
            for prediction in predictions
        ],
    )
    _write_json(
        baseline_root / "independent-batches.json",
        {
            "seed": passport.independent_seed,
            "batch_size": passport.independent_batch_size,
            "budget_state": campaign_budget_state(passport.independent_batches),
            "batches": [asdict(batch) for batch in passport.independent_batches],
            "remainder": {
                "photo_ids": passport.independent_remainder,
                "sha256": _ids_sha256(number="remainder", photo_ids=passport.independent_remainder),
                "usable_as_batch": False,
            },
        },
    )
    return tuple(sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()))


def _publish_checkpoint(campaign_root: Path, staging_root: Path, files: Sequence[Path]) -> None:
    if campaign_root.exists():
        existing = tuple(
            sorted(
                path.relative_to(campaign_root)
                for path in campaign_root.rglob("*")
                if path.is_file()
            )
        )
        if existing != tuple(files) or any(
            (campaign_root / relative).read_bytes() != (staging_root / relative).read_bytes()
            for relative in files
        ):
            raise ValueError("неизменяемый checkpoint уже существует и отличается")
        return
    campaign_root.mkdir(parents=True)
    try:
        for relative in files:
            target = campaign_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging_root / relative).replace(target)
    except OSError:
        shutil.rmtree(campaign_root, ignore_errors=True)
        raise


def write_campaign_artifacts(
    campaign_root: Path,
    passport: BaselinePassport,
    predictions: Sequence[Prediction],
    preregistration: Mapping[str, Any],
) -> None:
    """Сверить свидетель, целиком отрендерить staging и неизменяемо опубликовать checkpoint."""
    counts = exact_prediction_counts(
        predictions,
        expected_ids={
            unit.photo_id for unit in passport.frozen_units if unit.split == "holdout_ru"
        },
    )
    derived = ExactMetrics(
        top1_k=counts["top1"]["k"],
        top3_k=counts["top3"]["k"],
        n=counts["top1"]["n"],
    )
    if derived != passport.exact_metrics:
        raise ValueError(
            f"passport exact_metrics {passport.exact_metrics!r} != predictions {derived!r}"
        )
    campaign_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{campaign_root.name}.staging-", dir=campaign_root.parent)
    )
    try:
        files = _render_campaign_artifacts(staging_root, passport, predictions, preregistration)
        _publish_checkpoint(campaign_root, staging_root, files)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
