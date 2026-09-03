"""Отчёт `bench/harvest-c2.md` и гейт `G_C2`.

Отдельный скрипт, а не расширение `report_c1.py`, по цене, которую проект уже платил:
`report_c1.main` пишет свой файл целиком через `write_text`, и чужая задача внутри него
исчезала бы при следующем запуске — ровно находка I2.

Блокирует одно число, и это число — цель задачи (§12 дизайна C2). Протокол разведения
Lanos считается и печатается, но не блокирует: правильность вердикта `ok` внутри C2
не измерима по построению, и выдуманный порог здесь передал бы дальше цифру вместо
знания.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from pipeline.holdout import classes_short_of_holdout, select_holdout

G_C2_MAX_MISSING = 0
BEFORE_C2_CLASSES_WITHOUT_HOLDOUT = 2

ОСНОВАНИЯ = ("badge", "context", "uncertain")


@dataclass(frozen=True, slots=True)
class DiscriminationRow:
    """Одна строка протокола разведения Lanos: чем человек развёл шильдик."""

    file: str
    basis: str
    note: str


def load_discrimination(path: Path) -> tuple[DiscriminationRow, ...]:
    """Протокол разведения или пустой кортеж, если ручного прохода ещё не было.

    Основание проверяется по закрытому словарю: свободная строка прошла бы молча и
    сделала бы наблюдаемое число непосчитанным — то есть уничтожила бы единственный
    смысл файла.
    """
    if not path.exists():
        return ()
    строки: list[DiscriminationRow] = []
    with path.open(encoding="utf-8", newline="") as поток:
        for запись in csv.DictReader(поток):
            основание = (запись.get("basis") or "").strip()
            if основание not in ОСНОВАНИЯ:
                raise ValueError(f"{path}: неизвестное основание {основание!r}")
            примечание = (запись.get("note") or "").strip()
            if остаток := запись.get(None):
                хвост = ",".join(остаток)
                примечание = f"{примечание},{хвост}".strip() if примечание else хвост.strip()
            строки.append(
                DiscriminationRow(
                    file=(запись.get("file") or "").strip(),
                    basis=основание,
                    note=примечание,
                )
            )
    return tuple(строки)


def discrimination_counts(rows: Sequence[DiscriminationRow]) -> dict[str, int]:
    """Сколько кадров развёл человек по каждому основанию."""
    return {основание: sum(1 for с in rows if с.basis == основание) for основание in ОСНОВАНИЯ}


def gate_failures(classes_without_holdout: int) -> list[str]:
    """Единственный блокирующий порог задачи."""
    if classes_without_holdout > G_C2_MAX_MISSING:
        return [f"G_C2: классов без эталона — {classes_without_holdout} > {G_C2_MAX_MISSING}"]
    return []


def render(
    classes_without_holdout: Sequence[str],
    holdout_frames: int,
    manifest_rows: int,
    discrimination: Sequence[DiscriminationRow],
    git_sha: str,
    ran_at: str,
) -> str:
    """Собрать текст отчёта. Числа считает вызывающий, здесь только форма."""
    провалы = gate_failures(len(classes_without_holdout))
    вердикт = "НЕ ПРОЙДЕН" if провалы else "пройден"
    свод = discrimination_counts(discrimination)
    части = [
        "# Внешний донор Flickr, C2 — прогон",
        "",
        f"Прогон от {ran_at}, `git_sha` — `{git_sha}`.",
        "",
        f"- строк в манифесте: {manifest_rows}",
        f"- кадров в `holdout_ru`: {holdout_frames}",
        f"- классов без эталона: {len(classes_without_holdout)} "
        f"(было {BEFORE_C2_CLASSES_WITHOUT_HOLDOUT})",
        f"- вердикт гейта: {вердикт}",
        "",
        "## Блокирующие провалы",
        "",
    ]
    части += [f"- {строка}" for строка in провалы] if провалы else ["Нет."]
    части += [
        "",
        "## Протокол разведения Lanos (не блокирует)",
        "",
        "Чем человек развёл Chevrolet и Daewoo Lanos при визуальном аудите. Число"
        " наблюдаемое: правильность вердикта `ok` внутри C2 не измерима, и замер M3"
        " обязан знать, насколько человек сам справлялся с той же задачей.",
        "",
        "| Основание | Кадров |",
        "|---|---|",
        f"| `badge` — виден шильдик или решётка | {свод['badge']} |",
        f"| `context` — номер, год, география | {свод['context']} |",
        f"| `uncertain` — принят без разведения | {свод['uncertain']} |",
        "",
    ]
    if classes_without_holdout:
        части += [
            "## Классы, оставшиеся без эталона",
            "",
            "Правило §5.2 каталога по паре `daewoo__lanos` / `chevrolet__lanos` остаётся"
            " неприменимым, если среди них `chevrolet__lanos`: пара держится раздельно"
            " по умолчанию, а не по замеру.",
            "",
        ]
        части += [f"- `{класс}`" for класс in classes_without_holdout]
        части.append("")
    return "\n".join(части)


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="отчёт C2 и гейт G_C2")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument(
        "--discrimination", type=Path, default=Path("provenance/lanos_discrimination.csv")
    )
    парсер.add_argument("--out", type=Path, default=Path("bench/harvest-c2.md"))
    парсер.add_argument("--git-sha", required=True)
    парсер.add_argument("--ran-at", required=True)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать отчёт и вернуть код по вердикту гейта."""
    аргументы = build_parser().parse_args(argv)
    строки = load_manifest(аргументы.manifest)
    классы = sorted({с.class_id for с in строки})
    отбор = select_holdout(строки)
    дефицитные = sorted(classes_short_of_holdout(отбор, строки, классы))
    текст = render(
        classes_without_holdout=дефицитные,
        holdout_frames=len(отбор.files),
        manifest_rows=len(строки),
        discrimination=load_discrimination(аргументы.discrimination),
        git_sha=аргументы.git_sha,
        ran_at=аргументы.ran_at,
    )
    аргументы.out.parent.mkdir(parents=True, exist_ok=True)
    аргументы.out.write_text(текст, encoding="utf-8")
    return 1 if gate_failures(len(дефицитные)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
