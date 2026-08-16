"""Расписания темпа обучения, маржи ArcFace и множителя бэкбона (§4.2).

Отдельный модуль без torch: расписание — арифметика, и проверять её удобнее
таблицей значений, чем прогоном обучения. Шаги и эпохи считаются с нуля.
"""

from __future__ import annotations

import math


def lr_at(step: int, *, base_lr: float, total_steps: int, warmup_steps: int) -> float:
    """Темп обучения на шаге: линейный разогрев, затем косинус до нуля.

    Разогрев заканчивается ровно на `base_lr`, поэтому на шаге
    `warmup_steps - 1` и на шаге `warmup_steps` значение одно и то же —
    склейка расписаний без ступеньки.
    """
    if step < 0:
        raise ValueError(f"шаг {step} отрицательный")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    остаток = max(1, total_steps - warmup_steps)
    прогресс = min(1.0, max(0.0, (step - warmup_steps) / остаток))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * прогресс))


def margin_at(epoch: int, *, start: float, end: float, warmup_epochs: int) -> float:
    """Маржа ArcFace: линейный рост от `start` до `end` за `warmup_epochs`."""
    if epoch < 0:
        raise ValueError(f"эпоха {epoch} отрицательная")
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return end
    return start + (end - start) * epoch / warmup_epochs


def backbone_mult(epoch: int, *, mult: float, epochs: int) -> float:
    """Множитель темпа для бэкбона: `mult` первые `epochs` эпох, дальше 1.0."""
    if epoch < 0:
        raise ValueError(f"эпоха {epoch} отрицательная")
    return mult if epoch < epochs else 1.0
