"""Пороги гейта Few-Shot (правило 3 Р-11) и отчёт прогона.

Пороги назначены до прогона и скопированы из спеки дословно. Критерий, выбранный после
того, как увидены числа, — подгонка критерия под результат, а не решение.

Деления провалов на блокирующие и измерительные здесь нет, в отличие от гейта харвеста.
Р-11 говорит про все три порога одно: «Провал порогов отправляет конкретные классы в
контур дообучения / синтетики», то есть провал — измеренный результат, а не поломка
конвейера. Разделить их значило бы решить за спеку.

Блокирует другое — **неизмеримость**. Ядровой класс G_FS2 без единого эталонного кадра
нельзя ни зачесть, ни провалить; ноль вместо порога — та же ошибка, на которой прогон
харвеста 2026-08-29 отрапортовал H3 «пройден» по двум кадрам из двадцати восьми. Такой
случай — `FewShotGateError`, а не провал.

G_FS3 считает классы с top-3 не ниже порога среди **измеренных**, а сравнивает с абсолютным
числом 135 из спеки. Класс без эталона в числитель не попадает: зачесть класс, которого не
мерили, — то же самое завышение. Если каталог когда-нибудь вырастет, менять надо порог в
спеке, а не арифметику здесь.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.fewshot.metrics import FewShotMetrics

G_FS1_TOP1 = 0.70
G_FS1_TOP3 = 0.85
G_FS2_CORE_TOP3 = 0.60
G_FS2_CORE_CLASSES = ("lada__2107", "lada__granta", "lada__niva_2121", "uaz__patriot")
G_FS3_CLASS_TOP3 = 0.50
G_FS3_MIN_CLASSES = 135


class FewShotGateError(ValueError):
    """Гейт нельзя посчитать: у ядрового порога нет измерения."""


@dataclass(frozen=True, slots=True)
class FewShotGateResult:
    """Вердикт замера: пройден ли, чем провален и какие классы уходят в синтетику."""

    passed: bool
    failures: tuple[str, ...]
    deficient_classes: tuple[str, ...]


def check_gate(
    metrics: FewShotMetrics,
    *,
    catalog_classes: int,
    core_classes: Sequence[str] = G_FS2_CORE_CLASSES,
) -> FewShotGateResult:
    """Проверить три порога Р-11.

    `catalog_classes` идёт только в текст провала G_FS3: порог там абсолютный, а знаменатель
    нужен читателю отчёта, чтобы «130» не выглядело как «130 из 130».
    """
    без_эталона = [
        class_id for class_id in core_classes if metrics.per_class_support.get(class_id, 0) == 0
    ]
    if без_эталона:
        raise FewShotGateError(
            f"G_FS2 неизмерим: у ядровых классов нет эталона — {', '.join(без_эталона)}; "
            "чинить отбор `holdout_ru`, а не порог"
        )

    провалы: list[str] = []
    if metrics.top1 < G_FS1_TOP1:
        провалы.append(f"G_FS1: top-1 {metrics.top1:.4f} < {G_FS1_TOP1}")
    if metrics.top3 < G_FS1_TOP3:
        провалы.append(f"G_FS1: top-3 {metrics.top3:.4f} < {G_FS1_TOP3}")

    for class_id in core_classes:
        значение = metrics.per_class_top3[class_id]
        if значение < G_FS2_CORE_TOP3:
            провалы.append(f"G_FS2: ядровой {class_id} top-3 {значение:.4f} < {G_FS2_CORE_TOP3}")

    покрыто = _покрытие(metrics)
    if покрыто < G_FS3_MIN_CLASSES:
        провалы.append(
            f"G_FS3: классов с top-3 не ниже {G_FS3_CLASS_TOP3} — {покрыто} "
            f"из {catalog_classes}, нужно {G_FS3_MIN_CLASSES}"
        )

    дефицитные = tuple(
        sorted(
            class_id for class_id, доля in metrics.per_class_top3.items() if доля < G_FS3_CLASS_TOP3
        )
    )
    return FewShotGateResult(not провалы, tuple(провалы), дефицитные)


def _покрытие(metrics: FewShotMetrics) -> int:
    """Сколько измеренных классов взяли порог G_FS3."""
    return sum(1 for доля in metrics.per_class_top3.values() if доля >= G_FS3_CLASS_TOP3)


def _знак(пройдено: bool) -> str:
    return "✅" if пройдено else "❌"


def render_report(
    metrics: FewShotMetrics,
    gate: FewShotGateResult,
    *,
    git_sha: str,
    run_date: str,
    encoder: str,
    index_sha256: str,
    holdout_frames: int,
    pool_frames: int,
    catalog_classes: int,
    missing_holdout: Sequence[str],
) -> str:
    """Собрать `bench/fewshot-m2.md`.

    Отчёт отслеживается git, потому что `docs/` — нет (`.gitignore:65`): без него итог
    замера существовал бы только на одной машине.
    """
    покрыто = _покрытие(metrics)
    строки = [
        "# Few-shot baseline M2 (правило 3 Р-11)",
        "",
        f"- **Дата прогона**: {run_date}, `git_sha` — `{git_sha}`",
        f"- **Энкодер**: `{encoder}`",
        f"- **Индекс**: `catalog_index.bin`, sha256 `{index_sha256}`",
        f"- **Кадров в `holdout_ru`**: {holdout_frames} "
        f"по {len(metrics.per_class_support)} классам",
        f"- **Кадров в обучающем пуле**: {pool_frames}",
        f"- **Классов в каталоге**: {catalog_classes}",
        "",
        "## Пороги",
        "",
        "| Гейт | Порог | Факт | Итог |",
        "|---|---|---|---|",
        f"| G_FS1 top-1 | ≥ {G_FS1_TOP1} | {metrics.top1:.4f} "
        f"| {_знак(metrics.top1 >= G_FS1_TOP1)} |",
        f"| G_FS1 top-3 | ≥ {G_FS1_TOP3} | {metrics.top3:.4f} "
        f"| {_знак(metrics.top3 >= G_FS1_TOP3)} |",
    ]
    for class_id in G_FS2_CORE_CLASSES:
        значение = metrics.per_class_top3[class_id]
        строки.append(
            f"| G_FS2 `{class_id}` | ≥ {G_FS2_CORE_TOP3} | {значение:.4f} "
            f"| {_знак(значение >= G_FS2_CORE_TOP3)} |"
        )
    строки += [
        f"| G_FS3 покрытие | ≥ {G_FS3_MIN_CLASSES} классов с top-3 ≥ {G_FS3_CLASS_TOP3} "
        f"| {покрыто} из {catalog_classes} | {_знак(покрыто >= G_FS3_MIN_CLASSES)} |",
        "",
        f"**Вердикт гейта: {'ПРОЙДЕН' if gate.passed else 'НЕ ПРОЙДЕН'}.**",
        "",
    ]
    if gate.failures:
        строки += ["Провалы:", "", *(f"- {провал}" for провал in gate.failures), ""]
    if missing_holdout:
        строки += [
            f"## Классов без эталона: {len(missing_holdout)}",
            "",
            "Порог по ним не измерен, в покрытие G_FS3 они не зачтены; "
            "уходят в синтетический слой Р-11.",
            "",
            *(f"- `{class_id}`" for class_id in missing_holdout),
            "",
        ]
    if gate.deficient_classes:
        строки += [
            f"## Классов ниже {G_FS3_CLASS_TOP3} по top-3: {len(gate.deficient_classes)}",
            "",
            "| Класс | top-3 | кадров в эталоне |",
            "|---|---|---|",
            *(
                f"| `{c}` | {metrics.per_class_top3[c]:.4f} | {metrics.per_class_support[c]} |"
                for c in gate.deficient_classes
            ),
            "",
        ]
    return "\n".join(строки)
