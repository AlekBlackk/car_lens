"""Пороги гейта Few-Shot: правило 3 Р-11."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from pipeline.fewshot.gate import (
    G_FS1_TOP1,
    G_FS2_CORE_CLASSES,
    FewShotGateError,
    check_gate,
    render_report,
)
from pipeline.fewshot.metrics import FewShotMetrics

КАТАЛОГ = 150


def _метрики(
    *,
    top1: float = 0.90,
    top3: float = 0.95,
    поклассово: Mapping[str, float] | None = None,
    без_опоры: Sequence[str] = (),
) -> FewShotMetrics:
    """150 классов, у всех top-3 равен единице, кроме перечисленных."""
    имена = [*G_FS2_CORE_CLASSES, *(f"класс{n:03d}" for n in range(КАТАЛОГ - 4))]
    поклассный = dict.fromkeys(имена, 1.0)
    поклассный.update(поклассово or {})
    опора = dict.fromkeys(имена, 10)
    for имя in без_опоры:
        поклассный.pop(имя, None)
        опора.pop(имя, None)
    return FewShotMetrics(top1=top1, top3=top3, per_class_top3=поклассный, per_class_support=опора)


def test_все_пороги_взяты() -> None:
    итог = check_gate(_метрики(), catalog_classes=КАТАЛОГ)
    assert итог.passed
    assert итог.failures == ()
    assert итог.deficient_classes == ()


def test_провал_g_fs1_по_top1() -> None:
    итог = check_gate(_метрики(top1=0.5), catalog_classes=КАТАЛОГ)
    assert not итог.passed
    assert any("G_FS1" in п and "top-1" in п for п in итог.failures)


def test_провал_g_fs1_по_top3() -> None:
    итог = check_gate(_метрики(top3=0.5), catalog_classes=КАТАЛОГ)
    assert any("G_FS1" in п and "top-3" in п for п in итог.failures)


def test_провал_g_fs2_называет_ядровой_класс() -> None:
    итог = check_gate(_метрики(поклассово={"lada__granta": 0.3}), catalog_classes=КАТАЛОГ)
    assert any("G_FS2" in п and "lada__granta" in п for п in итог.failures)


def test_провал_g_fs3_по_покрытию() -> None:
    слабые = {f"класс{n:03d}": 0.1 for n in range(20)}
    итог = check_gate(_метрики(поклассово=слабые), catalog_classes=КАТАЛОГ)
    assert any("G_FS3" in п for п in итог.failures)
    assert len(итог.deficient_classes) == 20


def test_ядровой_класс_без_эталона_делает_гейт_неизмеримым() -> None:
    with pytest.raises(FewShotGateError, match="lada__granta"):
        check_gate(_метрики(без_опоры=["lada__granta"]), catalog_classes=КАТАЛОГ)


def test_класс_без_эталона_не_зачитывается_в_покрытие() -> None:
    """Убрать 20 неядровых классов из измерения — это провал G_FS3, а не 130 из 130."""
    убранные = [f"класс{n:03d}" for n in range(20)]
    итог = check_gate(_метрики(без_опоры=убранные), catalog_classes=КАТАЛОГ)
    assert any("130" in п for п in итог.failures)


def test_отчёт_содержит_вердикт_пороги_и_провенанс() -> None:
    метрики = _метрики(top1=0.5)
    итог = check_gate(метрики, catalog_classes=КАТАЛОГ)
    отчёт = render_report(
        метрики,
        итог,
        git_sha="abc1234",
        run_date="2026-08-30",
        encoder="openclip_b32",
        index_sha256="deadbeef",
        holdout_frames=1442,
        pool_frames=2232,
        catalog_classes=КАТАЛОГ,
        missing_holdout=("lifan__x60",),
    )
    assert "НЕ ПРОЙДЕН" in отчёт
    assert str(G_FS1_TOP1) in отчёт
    assert "abc1234" in отчёт and "deadbeef" in отчёт
    assert "lifan__x60" in отчёт


def test_отчёт_содержит_блок_цена_решения_и_дроблёные_классы() -> None:
    метрики = _метрики(top1=0.5)
    итог = check_gate(метрики, catalog_classes=КАТАЛОГ)
    дроблёные = ("bmw__x5", "lada__granta")
    отчёт = render_report(
        метрики,
        итог,
        git_sha="abc1234",
        run_date="2026-08-30",
        encoder="openclip_b32",
        index_sha256="deadbeef",
        holdout_frames=1442,
        pool_frames=2232,
        catalog_classes=КАТАЛОГ,
        missing_holdout=(),
        refined_classes=дроблёные,
    )
    assert "Цена решения" in отчёт
    assert "утечк" in отчёт.lower()
    assert "2" in отчёт
    assert "`bmw__x5`" in отчёт
    assert "`lada__granta`" in отчёт
