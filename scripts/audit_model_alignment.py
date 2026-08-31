"""Триаж выборки по именам файлов: автовердикты, очередь человеку, отчёт и гейт.

Скрипт офлайновый: сети нет ни в одном пути, кроме явной проверки категорий (§12
дизайна), которая включается флагом и по умолчанию выключена.

Порядок работы двухпроходный и иначе не выйдет. Первый прогон — без вердиктов человека:
он выдаёт автовердикты и очередь, и гейт при этом красный по `G_AL2`, потому что охват
ещё не закрыт. Второй прогон — с вердиктами всех проходов: он и даёт вердикт задачи.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.harvest.alignment import (
    auto_drops,
    check_alias_gate,
    render_alignment_report,
    review_queue,
    triage,
)
from pipeline.harvest.audit import load_audit, merge_verdicts
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.harvest.queries import DEFAULT_QUERIES_PATH
from pipeline.paths import harvest_dir

_КОРЕНЬ = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = _КОРЕНЬ / "bench" / "model-alignment.md"
DEFAULT_OUT_VERDICTS_PATH = _КОРЕНЬ / "verdicts.alias.csv"


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Триаж выборки по именам файлов (задача B)")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES_PATH)
    парсер.add_argument("--out-verdicts", type=Path, default=DEFAULT_OUT_VERDICTS_PATH)
    парсер.add_argument("--queue", type=Path, default=None, help="по умолчанию в каталоге данных")
    парсер.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    парсер.add_argument(
        "--verdicts",
        type=Path,
        nargs="*",
        default=[],
        help="CSV вердиктов всех проходов; поздний файл побеждает ранний",
    )
    парсер.add_argument(
        "--alias-verdicts",
        type=Path,
        nargs="*",
        default=[],
        help="CSV вердиктов алиасных проходов для G_AL2",
    )
    парсер.add_argument(
        "--check-categories",
        action="store_true",
        help="проверить существование категорий Commons у классов без строк из категорий",
    )
    парсер.add_argument("--contact", default=None, help="контакт для User-Agent (§3.2)")
    return парсер


def проверить_категории(
    rows: Sequence[HarvestRow],
    contact: str,
    queries_path: Path = DEFAULT_QUERIES_PATH,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, bool]:
    """Классы без принятых строк из категорий: жива ли хоть одна заявленная категория.

    Пауза между запросами — та же, что у харвеста (§3.2 SPEC): чужое хранилище не должно
    отличать нашу диагностику от вежливого прогона.
    """
    from pipeline.harvest.backends import UrllibTransport
    from pipeline.harvest.commons import MIN_INTERVAL_SECONDS, category_is_populated
    from pipeline.harvest.queries import load_queries

    из_категорий = {с.class_id for с in rows if с.basis.startswith("category:")}
    запросы = load_queries(queries_path)
    транспорт = UrllibTransport(contact)
    итог: dict[str, bool] = {}
    for class_id, запрос in sorted(запросы.items()):
        if class_id in из_категорий:
            continue
        итог[class_id] = any(
            category_is_populated(транспорт, категория, sleep=sleep)
            for категория in запрос.categories
        )
        sleep(MIN_INTERVAL_SECONDS)
    return итог


def main(argv: Sequence[str] | None = None) -> int:
    """Прогнать триаж, записать артефакты и вернуть код по вердикту гейта."""
    парсер = build_parser()
    аргументы = парсер.parse_args(argv)
    # Негодная пара флагов отвергается раньше любой записи: пути по умолчанию ведут в
    # рабочее дерево и в каталог данных, и падение на середине затирало бы очередь,
    # которую человек уже набрал глазами.
    if аргументы.check_categories and not аргументы.contact:
        парсер.error("--check-categories требует --contact (§3.2)")
    очередь_путь = аргументы.queue or (harvest_dir() / "alias_queue.txt")

    таблица = load_alias_table(аргументы.aliases)
    строки = load_manifest(аргументы.manifest)
    вердикты_строк = triage(строки, таблица)

    автоматом = auto_drops(вердикты_строк, таблица)
    аргументы.out_verdicts.parent.mkdir(parents=True, exist_ok=True)
    with аргументы.out_verdicts.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.writer(файл)
        писатель.writerow(["file", "verdict"])
        for вердикт in автоматом:
            писатель.writerow([вердикт.file, "drop:wrong_model"])

    очередь = review_queue(вердикты_строк, таблица)
    очередь_путь.parent.mkdir(parents=True, exist_ok=True)
    очередь_путь.write_text("".join(f"{вердикт.file}\n" for вердикт in очередь), encoding="utf-8")

    # Первый прогон считает гейт по одним автовердиктам: файлов человека ещё нет.
    # Второй получает всю цепочку файлов в хронологическом порядке и складывает её как
    # есть — подмешивать автовердикты сюда нельзя, иначе 1611 автоматических `ok` из
    # `verdicts.auto.csv` встанут позже дропов и воскресят выкинутые кадры.
    все_вердикты = (
        merge_verdicts(*(load_audit(путь) for путь in аргументы.verdicts))
        if аргументы.verdicts
        else {вердикт.file: "drop:wrong_model" for вердикт in автоматом}
    )
    # Охват `G_AL2` закрывают только вердикты алиасного прохода: `verdicts.csv` прохода H3
    # смотрел номера и лица, а не принадлежность модели, и его `ok` не свидетельство о
    # классе. Отсюда второй вход и отдельный флаг `--alias-verdicts` на два файла человека.
    алиасные_вердикты = (
        merge_verdicts(
            {вердикт.file: "drop:wrong_model" for вердикт in автоматом},
            *(load_audit(путь) for путь in аргументы.alias_verdicts),
        )
        if аргументы.alias_verdicts
        else {вердикт.file: "drop:wrong_model" for вердикт in автоматом}
    )
    классы = tuple(класс.class_id for класс in load_catalog(DEFAULT_CATALOG_PATH).classes)
    гейт = check_alias_gate(строки, таблица, все_вердикты, алиасные_вердикты, классы)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    категории = (
        проверить_категории(строки, аргументы.contact) if аргументы.check_categories else None
    )
    аргументы.report.parent.mkdir(parents=True, exist_ok=True)
    аргументы.report.write_text(
        render_alignment_report(
            строки,
            таблица,
            все_вердикты,
            гейт,
            sha,
            date.today().isoformat(),
            categories=категории,
        ),
        encoding="utf-8",
    )

    print(f"поисковых строк: {len(вердикты_строк)}, помечено: {len(автоматом) + len(очередь)}")
    print(f"автовердикты: {аргументы.out_verdicts} ({len(автоматом)})")
    print(f"очередь человеку: {очередь_путь} ({len(очередь)})")
    for причина in гейт.blocking_failures:
        print(f"БЛОКИРУЕТ {причина}")
    return 0 if гейт.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
