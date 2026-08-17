"""Тесты расписаний обучения (M-1.2, §4.3 дизайна)."""

import pytest

from training.schedule import backbone_mult, lr_at, margin_at

БАЗА = 3.0e-4
ВСЕГО = 1000
РАЗОГРЕВ = 100


def test_lr_растёт_линейно_на_разогреве() -> None:
    первый = lr_at(0, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=РАЗОГРЕВ)
    середина = lr_at(49, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=РАЗОГРЕВ)
    assert первый == pytest.approx(БАЗА / РАЗОГРЕВ)
    assert середина == pytest.approx(БАЗА * 50 / РАЗОГРЕВ)


def test_lr_достигает_базы_в_конце_разогрева() -> None:
    последний_разогрева = lr_at(
        РАЗОГРЕВ - 1, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=РАЗОГРЕВ
    )
    первый_после = lr_at(РАЗОГРЕВ, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=РАЗОГРЕВ)
    assert последний_разогрева == pytest.approx(БАЗА)
    assert первый_после == pytest.approx(БАЗА)


def test_lr_убывает_после_разогрева() -> None:
    значения = [
        lr_at(шаг, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=РАЗОГРЕВ)
        for шаг in range(РАЗОГРЕВ, ВСЕГО, 50)
    ]
    assert значения == sorted(значения, reverse=True)


def test_lr_около_нуля_на_последнем_шаге() -> None:
    последний = lr_at(ВСЕГО - 1, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=РАЗОГРЕВ)
    assert 0.0 <= последний < БАЗА * 0.01


def test_lr_без_разогрева_начинается_с_базы() -> None:
    """`warmup_steps == 0`: разогрева нет, косинус идёт от `base_lr` с нулевого шага.

    Граница опасна делением на ноль в множителе разогрева — его обходит условие
    `warmup_steps > 0`. Поведение разумное (полный косинус вместо разогрева),
    поэтому закрепляется тестом, а не чинится.
    """
    assert lr_at(0, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=0) == pytest.approx(БАЗА)
    значения = [
        lr_at(шаг, base_lr=БАЗА, total_steps=ВСЕГО, warmup_steps=0) for шаг in range(0, ВСЕГО, 50)
    ]
    assert значения == sorted(значения, reverse=True)
    # Отрицательного lr быть не может ни на одном шаге: косинус не ниже -1.
    assert min(значения) >= 0.0


def test_маржа_без_разогрева_сразу_конечная() -> None:
    """`warmup_epochs == 0`: маржа равна `end` с нулевой эпохи, а не делится на ноль."""
    assert margin_at(0, start=0.2, end=0.5, warmup_epochs=0) == pytest.approx(0.5)
    assert margin_at(59, start=0.2, end=0.5, warmup_epochs=0) == pytest.approx(0.5)


def test_множитель_бэкбона_при_нулевых_эпохах_не_применяется() -> None:
    """`epochs == 0`: пониженного темпа у бэкбона нет ни на одной эпохе, включая нулевую."""
    assert backbone_mult(0, mult=0.1, epochs=0) == 1.0
    assert backbone_mult(59, mult=0.1, epochs=0) == 1.0


def test_маржа_стартовая_на_нулевой_эпохе() -> None:
    assert margin_at(0, start=0.2, end=0.5, warmup_epochs=10) == pytest.approx(0.2)


def test_маржа_конечная_после_разогрева() -> None:
    assert margin_at(10, start=0.2, end=0.5, warmup_epochs=10) == pytest.approx(0.5)
    assert margin_at(59, start=0.2, end=0.5, warmup_epochs=10) == pytest.approx(0.5)


def test_маржа_монотонна() -> None:
    значения = [margin_at(эпоха, start=0.2, end=0.5, warmup_epochs=10) for эпоха in range(12)]
    assert значения == sorted(значения)


def test_множитель_бэкбона_первые_пять_эпох() -> None:
    первые = [backbone_mult(эпоха, mult=0.1, epochs=5) for эпоха in range(5)]
    assert первые == [0.1] * 5
    assert backbone_mult(5, mult=0.1, epochs=5) == 1.0
    assert backbone_mult(59, mult=0.1, epochs=5) == 1.0
