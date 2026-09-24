"""Оркестрация подбора: сверка слотов, половины, вердикт и отчёт (спека calibration-ru §5–6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.calibration import Operating, Scores
from training.tune_thresholds import (
    FoldReport,
    check_holdout_slots,
    cross_fit,
    release_verdict,
    scores_from,
    write_generated_section,
)

МЕТА = {
    "classes": [
        {"class_id": "lada__vesta", "members": ["lada__vesta"], "brand": "Lada"},
        {"class_id": "lada__granta", "members": ["lada__granta"], "brand": "Lada"},
        {"class_id": "grp__coolray", "members": ["geely__coolray", "belgee__x50"], "brand": None},
    ]
}


def test_слоты_эталона_сверяются_с_метой() -> None:
    check_holdout_slots(["lada__vesta", "belgee__x50"], [0, 2], МЕТА)
    with pytest.raises(ValueError, match="lada__granta"):
        check_holdout_slots(["lada__granta"], [0], МЕТА)


def test_марка_и_класс_попадания() -> None:
    бренды = ["Lada", "Lada", None]
    оценки = scores_from(
        top1=np.array([0, 1, 2, 2]),
        sim1=np.array([0.9, 0.8, 0.7, 0.6]),
        top2=np.array([1, 0, 0, 1]),
        sim2=np.array([0.1, 0.7, 0.6, 0.5]),
        truth=np.array([0, 0, 2, 1]),
        brands=бренды,
    )

    assert оценки.class_hit.tolist() == [True, False, True, False]
    # Гранта вместо Весты — марка верна; группа без марки верна только классом.
    assert оценки.brand_hit.tolist() == [True, True, True, False]
    # Марки топ-1/топ-2 общие только у пары Лад; у группы марки нет — как brandsMatch.
    assert оценки.same_brand.tolist() == [True, True, False, False]


def test_у_негативов_попаданий_нет() -> None:
    оценки = scores_from(
        np.array([0]), np.array([0.9]), np.array([1]), np.array([0.1]), None, ["Lada", "Lada"]
    )
    assert not оценки.class_hit.any() and not оценки.brand_hit.any()


def _знакомые(n: int, seed: int) -> Scores:
    генератор = np.random.default_rng(seed)
    sim1 = генератор.uniform(0.4, 1.0, n)
    верно = генератор.uniform(size=n) < np.clip(1.6 * sim1 - 0.6, 0.0, 1.0)
    return Scores(sim1, sim1 - 0.2, np.zeros(n, bool), верно, верно)


def _чужие(n: int, seed: int) -> Scores:
    sim1 = np.random.default_rng(seed).uniform(0.2, 0.7, n)
    return Scores(sim1, sim1 - 0.2, np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))


def test_перекрёстная_проверка_даёт_две_стороны() -> None:
    знакомые, чужие = _знакомые(400, 1), _чужие(200, 2)
    половина_з = np.arange(400) % 2 == 0
    половина_ч = np.arange(200) % 2 == 0

    стороны = cross_fit(
        знакомые, чужие, половина_з, половина_ч, max_error=0.2, max_fpr=0.2, max_knots=32
    )

    assert [с.tuned_on for с in стороны] == ["A", "B"]
    assert all(с.chosen is not None and с.held_out is not None for с in стороны)
    assert all(с.ece is not None and 0.0 <= с.ece <= 1.0 for с in стороны)


def _точка(покрытие: float) -> Operating:
    return Operating(0.5, 0.02, coverage=покрытие, error=0.1, fpr=0.1, shown=10)


def test_вердикт_стоп_при_отказе_больше_половины() -> None:
    стороны = [
        FoldReport("A", _точка(0.6), _точка(0.55), 0.03),
        FoldReport("B", _точка(0.6), _точка(0.45), 0.04),
    ]
    выпуск, причина = release_verdict(стороны, _точка(0.6))
    assert выпуск is False
    assert "B" in причина


def test_вердикт_стоп_без_допустимой_пары() -> None:
    стороны = [FoldReport("A", None, None, None), FoldReport("B", _точка(0.6), _точка(0.6), 0.0)]
    assert release_verdict(стороны, _точка(0.6))[0] is False
    assert release_verdict([], None)[0] is False


def test_вердикт_выпуск() -> None:
    стороны = [
        FoldReport("A", _точка(0.6), _точка(0.55), 0.03),
        FoldReport("B", _точка(0.6), _точка(0.52), 0.04),
    ]
    assert release_verdict(стороны, _точка(0.6))[0] is True


def test_генерируемый_раздел_не_трогает_рукописное(tmp_path: Path) -> None:
    отчёт = tmp_path / "calibration-ru.md"
    отчёт.write_text(
        "# Шапка\n\nрукой\n\n<!-- generated:start -->\nстарое\n<!-- generated:end -->\n\nвыводы\n",
        encoding="utf-8",
    )

    write_generated_section(отчёт, "новое\n")

    текст = отчёт.read_text(encoding="utf-8")
    assert "рукой" in текст and "выводы" in текст and "новое" in текст
    assert "старое" not in текст


def test_генерируемый_раздел_создаёт_файл(tmp_path: Path) -> None:
    отчёт = tmp_path / "нет.md"
    write_generated_section(отчёт, "тело\n")
    assert "<!-- generated:start -->\nтело\n<!-- generated:end -->" in отчёт.read_text(
        encoding="utf-8"
    )
