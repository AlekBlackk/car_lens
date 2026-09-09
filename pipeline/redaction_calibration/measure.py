"""Чистые пиксельные меры и строгий CSV-контракт профильных результатов."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from pipeline.harvest.run import BLACK_PIXEL_THRESHOLD, MAX_BLACK_PIXEL_FRAC
from pipeline.redaction_calibration.model import (
    FrameOutcome,
    FrameProfileResult,
    ProfileMetrics,
)


def black_fraction(image: np.ndarray, threshold: int = BLACK_PIXEL_THRESHOLD) -> float:
    """Доля пикселей с яркостью строго ниже порога боевого правила редакции."""
    if image.size == 0:
        raise ValueError("нельзя измерить пустое изображение")
    серый = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float((серый < threshold).mean())


def summarize_profile_results(results: Sequence[FrameProfileResult]) -> ProfileMetrics:
    """Свести один результат на входную строку в детерминированные метрики профиля.

    P95 — nearest-rank: для N отсортированных значений берётся элемент с индексом
    `ceil(0.95 * N) - 1`. Дельта учитывает только прирост чёрной доли.
    """
    if not results:
        raise ValueError("нельзя свести пустой набор профильных результатов")
    дельты = sorted(result.delta_black for result in results if result.delta_black is not None)
    if not дельты:
        дельты = [0.0]
    середина = len(дельты) // 2
    медиана = дельты[середина] if len(дельты) % 2 else (дельты[середина - 1] + дельты[середина]) / 2
    p95 = дельты[math.ceil(0.95 * len(дельты)) - 1]
    отказы = Counter(
        result.reject_reason
        for result in results
        if result.outcome == "rejected" and result.reject_reason is not None
    )
    return ProfileMetrics(
        median_delta_black=медиана,
        p95_delta_black=p95,
        max_delta_black=дельты[-1],
        above_quarter=sum(delta > 0.25 for delta in дельты),
        new_over_redacted=sum(
            result.new_black_frac is not None and result.new_black_frac > MAX_BLACK_PIXEL_FRAC
            for result in results
        ),
        changed_crops=sum(result.pixels_equal is False for result in results),
        rejects=tuple(sorted(отказы.items())),
    )


_PROFILE_FIELDS = tuple(field.name for field in fields(FrameProfileResult))


def write_frame_profile_results(path: Path, results: Sequence[FrameProfileResult]) -> None:
    """Записать профильные результаты с колонками из полей доменной записи."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_PROFILE_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _optional_bool(value: str) -> bool | None:
    if value == "":
        return None
    if value not in {"True", "False"}:
        raise ValueError(f"недопустимое логическое значение: {value!r}")
    return value == "True"


def _required_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"недопустимое обязательное логическое значение: {value!r}")
    return value == "True"


def load_frame_profile_results(path: Path) -> list[FrameProfileResult]:
    """Прочитать CSV, отвергая отсутствующие или неизвестные колонки контракта."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        actual = tuple(reader.fieldnames or ())
        if actual != _PROFILE_FIELDS:
            raise ValueError(
                f"колонки профильных результатов не совпали: {actual!r} != {_PROFILE_FIELDS!r}"
            )
        loaded: list[FrameProfileResult] = []
        for row in reader:
            outcome = row["outcome"]
            if outcome not in {"accepted", "rejected"}:
                raise ValueError(f"неизвестный исход профильной строки: {outcome!r}")
            loaded.append(
                FrameProfileResult(
                    photo_id=row["photo_id"],
                    class_id=row["class_id"],
                    outcome=cast(FrameOutcome, outcome),
                    reject_reason=row["reject_reason"] or None,
                    old_black_frac=_optional_float(row["old_black_frac"]),
                    new_black_frac=_optional_float(row["new_black_frac"]),
                    delta_black=_optional_float(row["delta_black"]),
                    pixels_equal=_optional_bool(row["pixels_equal"]),
                    audit_preserved=_required_bool(row["audit_preserved"]),
                )
            )
    return loaded
