"""Отчёт `bench/harvest-c1.md` и гейт `G_C1`.

Блокирует ровно одно число — классов без эталона, — и это число есть цель задачи: 16
классов появились измерением задачи B, и C1 существует, чтобы вернуть их к дочистковым
шести честным провенансом, а не ослаблением порогов. Число считается здесь, по каталогу и
манифесту, а не переносится руками из вывода другого скрипта: гейт, который нельзя
пересчитать по репозиторию, проверяет внимательность оператора, а не данные.

Остальные числа наблюдаемые. Доля категорийного базиса зависит от содержимого чужого
хранилища: требовать от кода 60 % значит требовать от Commons разложить машины так, как
удобно нам.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.audit import DEFAULT_APPLIED_PATH, load_applied, load_audit_events
from pipeline.harvest.expand_log import ExpandRow, expand_totals, load_expand_rows
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.harvest.rejects import Reject, load_rejects, reason_counts
from pipeline.harvest.sources import CATEGORY_BASIS_PREFIX
from pipeline.holdout import HOLDOUT_MIN, classes_short_of_holdout, select_holdout
from pipeline.paths import harvest_dir

REPORT_PATH = Path(__file__).resolve().parent.parent / "bench" / "harvest-c1.md"
DEDUP_VERDICTS_NAME = "verdicts.dedup.csv"
EXPAND_LOG_NAME = "expand_log.csv"

G_C1_MAX_MISSING = 6

# Базис «до C1» — замер задачи B по манифесту BASE `14a70bf`. Именованные константы, а не
# литералы внутри `render`: иначе любой будущий прогон продолжит называть эти числа «до C1»,
# когда «до» станет другим, а дата в заголовке колонки говорит, к какому именно «до» они
# относятся. Знаменатель каталога у них свой и замороженный: «68 из 150» верно про каталог
# того замера, и подставлять сюда сегодняшний размер каталога значило бы править историю.
BEFORE_C1_MEASURED = "2026-08-31"
BEFORE_C1_CATEGORY_SHARE = 0.238
BEFORE_C1_CLASSES_WITH_CATEGORY = 68
BEFORE_C1_CATALOG_CLASSES = 150
BEFORE_C1_CLASSES_WITHOUT_HOLDOUT = 16


@dataclass(frozen=True, slots=True)
class C2ScopeRow:
    """Класс без эталона и чем он набран сегодня — строка предложения задаче C2."""

    class_id: str
    accepted: int
    category_rows: int

    @property
    def category_share(self) -> float:
        """Доля категорийного базиса. У класса без строк — ноль, а не деление на ноль."""
        return self.category_rows / self.accepted if self.accepted else 0.0


@dataclass(frozen=True, slots=True)
class C1Numbers:
    """Числа отчёта C1: одно блокирующее и наблюдаемые, включая потери обхода."""

    rows: int
    category_rows: int
    category_share: float
    classes_with_category: int
    classes_without_holdout: int
    catalog_classes: int
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
    # Поимённый дефицит: та же величина, что блокирует гейт, только разложенная по классам.
    # `classes_without_holdout == len(c2_scope)`, поэтому таблица и вердикт не могут
    # разойтись — до этой правки таблица дописывалась в отчёт руками, а число приходило
    # флагом, и сойтись они могли только вниманием оператора.
    c2_scope: tuple[C2ScopeRow, ...]


def collect(
    rows: Sequence[HarvestRow],
    rejects: Sequence[Reject],
    verdict_events: Sequence[str],
    catalog_classes: Sequence[str],
    expand_rows: Sequence[ExpandRow] = (),
    classes_without_holdout: int | None = None,
) -> C1Numbers:
    """Числа отчёта. `verdict_events` — по одной записи на событие дедупа.

    Именно события, а не итоговые вердикты по файлам: §9 просит `supersede` и `conflict`
    поштучно, а словарь `file -> verdict` считает файлы — два вытеснения одного файла дали бы
    в отчёте одно. Собирает список вызывающий, из двух источников (журнал применённых и
    очередь ещё не применённых), потому что после правки C1 очередь усекается.

    Блокирующее число считается здесь, а не приходит флагом: гейт, который нельзя
    пересчитать по репозиторию, держится на том, что оператор перенёс число из вывода
    другого скрипта без опечатки. Считает `pipeline.holdout.classes_short_of_holdout` —
    по классам **каталога**, поэтому класс, потерявший провенанс целиком, попадает в
    дефицит, а не исчезает из знаменателя (обоснование записано там же). Место правки —
    здесь: отбор эталона выбирает из того, что есть, и каталог ему для этого не нужен, а
    гейт мерит именно дефицит каталога, значит знаменатель — его забота.

    `classes_without_holdout` оставлен как перекрытие для сверки со старым выводом
    `build_holdout_ru`; в обычном прогоне он пуст и число берётся из `c2_scope`, то есть
    вердикт гейта и таблица скоупа C2 — одна и та же величина в двух видах.

    `expand_rows` по умолчанию пуст: старый прогон без журнала обхода даёт нули, а не
    падение — отчёт обязан собираться и на данных до задачи 12.
    """
    принято: Counter[str] = Counter(с.class_id for с in rows)
    из_категорий: Counter[str] = Counter(
        с.class_id for с in rows if с.basis.startswith(CATEGORY_BASIS_PREFIX)
    )
    категорийных = sum(из_категорий.values())
    дефицитные = classes_short_of_holdout(select_holdout(rows), rows, catalog_classes)
    скоуп = tuple(
        C2ScopeRow(class_id=класс, accepted=принято[класс], category_rows=из_категорий[класс])
        for класс in дефицитные
    )
    обход = expand_totals(expand_rows)
    return C1Numbers(
        rows=len(rows),
        category_rows=категорийных,
        category_share=категорийных / len(rows) if rows else 0.0,
        classes_with_category=len(из_категорий),
        catalog_classes=len(catalog_classes),
        classes_without_holdout=(
            len(скоуп) if classes_without_holdout is None else classes_without_holdout
        ),
        conflicts=sum(1 for в in verdict_events if в == "conflict"),
        supersedes=sum(1 for в in verdict_events if в == "drop:superseded"),
        reasons=reason_counts(rejects),
        expand_requests=обход["requests"],
        expand_rejected_stopword=обход["rejected_stopword"],
        expand_rejected_exclude=обход["rejected_exclude"],
        expand_rejected_not_own=обход["rejected_not_own"],
        expand_budget_exhausted=обход["budget_exhausted"],
        c2_scope=скоуп,
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
    """Собрать `bench/harvest-c1.md`: числа задачи в отслеживаемом git месте.

    Обе таблицы отчёта пожизненные, и отчёт об этом говорит сам: `reject_log.csv` и
    `expand_log.csv` копятся между прогонами, а `bench/harvest-m0.md` считает те же причины
    отказа дельтой прогона. Два git-отслеживаемых отчёта одного `git_sha` дают разные числа,
    и без этой оговорки расхождение читается как ошибка одного из них.
    """
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
        f"| Что | До C1 (замер {BEFORE_C1_MEASURED}) | После C1 |",
        "|---|---|---|",
        f"| доля строк базиса `category:` | {BEFORE_C1_CATEGORY_SHARE:.1%} |"
        f" {numbers.category_share:.1%} |",
        f"| классов с ≥1 строкой `category:` |"
        f" {BEFORE_C1_CLASSES_WITH_CATEGORY} из {BEFORE_C1_CATALOG_CLASSES} | "
        f"{numbers.classes_with_category} из {numbers.catalog_classes} |",
        f"| классов без эталона | {BEFORE_C1_CLASSES_WITHOUT_HOLDOUT} |"
        f" {numbers.classes_without_holdout} |",
        f"| вытеснений `supersede` | 0 | {numbers.supersedes} |",
        f"| споров `conflict` | 0 | {numbers.conflicts} |",
        "",
        "## Причины отказа",
        "",
        "Числа пожизненные: `reject_log.csv` копится между прогонами, поэтому таблица суммирует"
        " все отказы задачи, а не дельту последнего прогона. Тот же прогон в"
        " `bench/harvest-m0.md` показан дельтой из `ClassOutcome` — расхождение двух отчётов"
        " на одном `git_sha` этим и объясняется.",
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
        "Журнал обхода тоже пожизненный: `expand_log.csv` дописывается каждым прогоном, и кеш"
        " дерева в `state.json` означает, что повторный прогон новых строк в него не добавит."
        " «Отвергнуто не своей полкой» считает **отказы**, а не потерянные категории: одна"
        " подкатегория под двумя родителями даёт два отказа.",
        "",
        "| Что | Число |",
        "|---|---|",
        f"| запросов к Commons | {numbers.expand_requests} |",
        f"| срезано стоп-словом | {numbers.expand_rejected_stopword} |",
        f"| срезано `exclude` | {numbers.expand_rejected_exclude} |",
        f"| отвергнуто не своей полкой | {numbers.expand_rejected_not_own} |",
        f"| категорий с исчерпанным бюджетом | {numbers.expand_budget_exhausted} |",
        "",
        "## Классы без эталона (скоуп задачи C2)",
        "",
        f"Классы, у которых в `holdout_ru` меньше {HOLDOUT_MIN} кадров (недостаточно"
        " независимых серий авторов для честного эталона). Переходят в скоуп задачи C2"
        " (синтетический слой Р-11 и внешние источники).",
        "",
        "Второе предложение задаче C2 — строка «отвергнуто не своей полкой» выше: подавляющая"
        " часть этих отказов — молчание игл на поколенческих деревьях (E-класс, Cayenne, 3-я и"
        " 5-я серии BMW, `Lada 111/112`), то есть закрывается дописыванием игл в"
        " `catalog/model_aliases.yaml`, а не новыми источниками (§5 дизайна C1).",
        "",
    ]
    # Раздел считается здесь, а не дописывается в файл руками: `main` пишет отчёт целиком
    # через `write_text`, поэтому ручная таблица — единственная долговечная выгода
    # провалившегося гейта — молча исчезала бы при следующем запуске скрипта.
    if numbers.c2_scope:
        строки += [
            "| Класс | Принято всего | Из категорий | Доля `category:` |",
            "|---|---|---|---|",
        ]
        строки.extend(
            f"| `{класс.class_id}` | {класс.accepted} | {класс.category_rows} |"
            f" {класс.category_share:.1%} |"
            for класс in numbers.c2_scope
        )
    else:
        строки.append("Нет.")
    return "\n".join(строки) + "\n"


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="отчёт C1 и гейт G_C1")
    парсер.add_argument(
        "--classes-without-holdout",
        type=int,
        default=None,
        help="перекрыть вычисленное число (по умолчанию считается по каталогу и манифесту)",
    )
    парсер.add_argument("--report", type=Path, default=REPORT_PATH)
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    парсер.add_argument(
        "--reject-log", type=Path, default=None, help="по умолчанию в каталоге данных"
    )
    парсер.add_argument(
        "--dedup-verdicts", type=Path, default=None, help="по умолчанию в каталоге данных"
    )
    парсер.add_argument(
        "--applied-verdicts",
        type=Path,
        default=DEFAULT_APPLIED_PATH,
        help="журнал применённых автовердиктов под provenance/",
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
    каталог = load_catalog(аргументы.catalog)
    # События дедупа лежат в двух файлах: git-отслеживаемый журнал применённых
    # (`provenance/dedup_verdicts.csv`) и очередь ещё не применённых в каталоге данных, которую
    # `scripts/harvest.py` усекает сразу после применения. Считаются строки, а не ключи: §9
    # просит вердикты поштучно, и два вытеснения одного файла обязаны дать два. Пара
    # «файл + вердикт», уже лежащая в журнале, из очереди не досчитывается — это одно и то же
    # событие, чью очередь просто не усекли.
    применённые = load_applied(аргументы.applied_verdicts)
    ожидающие = load_audit_events(файл_вердиктов) if файл_вердиктов.exists() else []
    в_журнале = {(з.file, з.verdict) for з in применённые}
    события = [з.verdict for з in применённые] + [
        вердикт for файл, вердикт in ожидающие if (файл, вердикт) not in в_журнале
    ]
    числа = collect(
        rows=load_manifest(аргументы.manifest),
        rejects=load_rejects(журнал),
        verdict_events=события,
        catalog_classes=[к.class_id for к in каталог.classes],
        expand_rows=load_expand_rows(журнал_обхода),
        classes_without_holdout=аргументы.classes_without_holdout,
    )
    прошёл, провалы = check_c1(числа)
    if аргументы.classes_without_holdout is not None and аргументы.classes_without_holdout != len(
        числа.c2_scope
    ):
        print(
            f"ВНИМАНИЕ: перекрытие {аргументы.classes_without_holdout} расходится с"
            f" вычисленными {len(числа.c2_scope)} — гейт судит по перекрытию,"
            " а таблица скоупа C2 по вычисленному"
        )
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
