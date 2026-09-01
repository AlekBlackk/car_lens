"""Отчёт `bench/harvest-c1.md` и гейт `G_C1`.

Блокирует ровно одно число — классов без эталона, — и это число есть цель задачи: 16
классов появились измерением задачи B, и C1 существует, чтобы вернуть их к дочистковым
шести честным провенансом, а не ослаблением порогов.

Остальные числа наблюдаемые. Доля категорийного базиса зависит от содержимого чужого
хранилища: требовать от кода 60 % значит требовать от Commons разложить машины так, как
удобно нам.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pipeline.harvest.audit import load_audit
from pipeline.harvest.expand_log import ExpandRow, expand_totals, load_expand_rows
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.harvest.rejects import Reject, load_rejects, reason_counts
from pipeline.harvest.sources import CATEGORY_BASIS_PREFIX
from pipeline.paths import harvest_dir

REPORT_PATH = Path(__file__).resolve().parent.parent / "bench" / "harvest-c1.md"
DEDUP_VERDICTS_NAME = "verdicts.dedup.csv"
EXPAND_LOG_NAME = "expand_log.csv"

G_C1_MAX_MISSING = 6


@dataclass(frozen=True, slots=True)
class C1Numbers:
    """Числа отчёта C1: одно блокирующее и наблюдаемые, включая потери обхода."""

    rows: int
    category_rows: int
    category_share: float
    classes_with_category: int
    classes_without_holdout: int
    conflicts: int
    supersedes: int
    reasons: dict[str, int]
    # Из журнала обхода категорий (задача 12): без него предикат допуска и стоп-слова срезали
    # подкатегории незаметно для отчёта — см. `pipeline/harvest/expand_log.py`.
    expand_requests: int
    expand_rejected_stopword: int
    expand_rejected_exclude: int
    expand_rejected_not_own: int
    expand_budget_exhausted: int


def collect(
    rows: Sequence[HarvestRow],
    rejects: Sequence[Reject],
    verdicts: Mapping[str, str],
    classes_without_holdout: int,
    expand_rows: Sequence[ExpandRow] = (),
) -> C1Numbers:
    """Числа отчёта. `verdicts` — содержимое `verdicts.dedup.csv`.

    `expand_rows` по умолчанию пуст: старый прогон без журнала обхода даёт нули, а не
    падение — отчёт обязан собираться и на данных до задачи 12.
    """
    категорийных = sum(1 for с in rows if с.basis.startswith(CATEGORY_BASIS_PREFIX))
    классов = {с.class_id for с in rows if с.basis.startswith(CATEGORY_BASIS_PREFIX)}
    обход = expand_totals(expand_rows)
    return C1Numbers(
        rows=len(rows),
        category_rows=категорийных,
        category_share=категорийных / len(rows) if rows else 0.0,
        classes_with_category=len(классов),
        classes_without_holdout=classes_without_holdout,
        conflicts=sum(1 for в in verdicts.values() if в == "conflict"),
        supersedes=sum(1 for в in verdicts.values() if в == "drop:superseded"),
        reasons=reason_counts(rejects),
        expand_requests=обход["requests"],
        expand_rejected_stopword=обход["rejected_stopword"],
        expand_rejected_exclude=обход["rejected_exclude"],
        expand_rejected_not_own=обход["rejected_not_own"],
        expand_budget_exhausted=обход["budget_exhausted"],
    )


def check_c1(numbers: C1Numbers) -> tuple[bool, list[str]]:
    """Единственный блокирующий порог задачи."""
    if numbers.classes_without_holdout > G_C1_MAX_MISSING:
        return False, [
            f"G_C1: классов без эталона — {numbers.classes_without_holdout} > {G_C1_MAX_MISSING}"
        ]
    return True, []


def render(
    numbers: C1Numbers,
    passed: bool,
    failures: Sequence[str],
    git_sha: str,
    ran_at: str,
) -> str:
    """Собрать `bench/harvest-c1.md`: числа задачи в отслеживаемом git месте."""
    строки = [
        "# Категорийный сбор с обходом подкатегорий, C1 — прогон",
        "",
        f"Прогон от {ran_at}, `git_sha` — `{git_sha}`.",
        "",
        f"- строк в манифесте: {numbers.rows}",
        f"- вердикт гейта: {'пройден' if passed else 'НЕ ПРОЙДЕН'}",
        "",
        "## Блокирующие провалы",
        "",
    ]
    if failures:
        строки.extend(f"- {провал}" for провал in failures)
    else:
        строки.append("Нет.")
    строки += [
        "",
        "## Наблюдаемые числа (не блокируют)",
        "",
        "| Что | До C1 | После C1 |",
        "|---|---|---|",
        f"| доля строк базиса `category:` | 23.8% | {numbers.category_share:.1%} |",
        f"| классов с ≥1 строкой `category:` | 68 из 150 | "
        f"{numbers.classes_with_category} из 150 |",
        f"| классов без эталона | 16 | {numbers.classes_without_holdout} |",
        f"| вытеснений `supersede` | 0 | {numbers.supersedes} |",
        f"| споров `conflict` | 0 | {numbers.conflicts} |",
        "",
        "## Причины отказа",
        "",
        "| Причина | Кандидатов |",
        "|---|---|",
    ]
    строки.extend(
        f"| `{причина}` | {число} |"
        for причина, число in sorted(numbers.reasons.items(), key=lambda пара: -пара[1])
    )
    строки += [
        "",
        "## Обход категорий: запросы и потери",
        "",
        "| Что | Число |",
        "|---|---|",
        f"| запросов к Commons | {numbers.expand_requests} |",
        f"| срезано стоп-словом | {numbers.expand_rejected_stopword} |",
        f"| срезано `exclude` | {numbers.expand_rejected_exclude} |",
        f"| отвергнуто не своей полкой | {numbers.expand_rejected_not_own} |",
        f"| категорий с исчерпанным бюджетом | {numbers.expand_budget_exhausted} |",
    ]
    return "\n".join(строки) + "\n"


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="отчёт C1 и гейт G_C1")
    парсер.add_argument(
        "--classes-without-holdout",
        type=int,
        required=True,
        help="число из вывода scripts/build_holdout_ru.py",
    )
    парсер.add_argument("--report", type=Path, default=REPORT_PATH)
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument(
        "--reject-log", type=Path, default=None, help="по умолчанию в каталоге данных"
    )
    парсер.add_argument(
        "--dedup-verdicts", type=Path, default=None, help="по умолчанию в каталоге данных"
    )
    парсер.add_argument(
        "--expand-log", type=Path, default=None, help="по умолчанию в каталоге данных"
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать отчёт и вернуть код по вердикту гейта."""
    аргументы = build_parser().parse_args(argv)
    # Корень данных спрашивается здесь, а не в `build_parser` и не на импорте: он приходит
    # из окружения (`CARLENS_DATA_ROOT`), и путь, вычисленный на импорте модуля, замер бы
    # раньше, чем окружение прогона будет собрано.
    корень = harvest_dir()
    журнал = аргументы.reject_log or (корень / "reject_log.csv")
    файл_вердиктов = аргументы.dedup_verdicts or (корень / DEDUP_VERDICTS_NAME)
    журнал_обхода = аргументы.expand_log or (корень / EXPAND_LOG_NAME)
    числа = collect(
        rows=load_manifest(аргументы.manifest),
        rejects=load_rejects(журнал),
        verdicts=load_audit(файл_вердиктов) if файл_вердиктов.exists() else {},
        classes_without_holdout=аргументы.classes_without_holdout,
        expand_rows=load_expand_rows(журнал_обхода),
    )
    прошёл, провалы = check_c1(числа)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    аргументы.report.parent.mkdir(parents=True, exist_ok=True)
    аргументы.report.write_text(
        render(числа, прошёл, провалы, sha, date.today().isoformat()), encoding="utf-8"
    )
    print(f"отчёт записан в {аргументы.report}; гейт {'пройден' if прошёл else 'НЕ ПРОЙДЕН'}")
    for провал in провалы:
        print(f"БЛОКИРУЕТ {провал}")
    return 0 if прошёл else 1


if __name__ == "__main__":
    raise SystemExit(main())
