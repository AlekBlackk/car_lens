"""Переприём кадров, отбитых прежней редакцией: манифест добирает то, что правило уже приняло.

Сеть нужна один раз и только на метаданные: провенанс страницы (автор, лицензия, URL) в
журнале отказов не лежит, а пиксели лежат в кэше оригиналов. Отсюда устройство: один проход
`CommonsSource.lookup` по `pageid`, дальше офлайн через `CachedSource`.

Решения принимает `run.process_candidate` — тот же код, что боевой прогон. Своей копии
допуска, редакции или дедупа здесь нет: копия разошлась бы с боевой и приняла бы кадры не тем
конвейером, каким их обещает манифест.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.harvest.audit import load_audit
from pipeline.harvest.backends import UrllibTransport
from pipeline.harvest.categories import ExpandResult
from pipeline.harvest.commons import CommonsSource
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, append_row, load_manifest
from pipeline.harvest.queries import DEFAULT_LATIN_NAMES_PATH, load_latin_names
from pipeline.harvest.readmit import (
    CachedSource,
    accepted_delta,
    accepted_regressions,
    merge_verdicts_file,
    patch_state,
    plan_readmission,
)
from pipeline.harvest.rejects import load_rejects, unique_by_reason
from pipeline.harvest.run import HarvestDeps, process_candidate
from pipeline.harvest.sources import Candidate
from pipeline.harvest.state import HarvestState, load_state, save_state
from pipeline.paths import harvest_dir
from scripts.harvest import DEDUP_VERDICTS_FILENAME, build_deps

BACKUP_SUFFIX = ".json.bak.readmit"
# Вердикты дедупа уводятся из живого `verdicts.dedup.csv`: там их ждёт `--audit`, и дописанная
# до фиксации строка приговорила бы чужую строку манифеста заранее. Перелив — шаг фиксации.
VERDICTS_FILENAME = "verdicts.dedup.readmit.csv"


@dataclass(frozen=True, slots=True)
class ReadmitOutcome:
    """Итог переприёма до фиксации: новые строки и свод причин."""

    rows: tuple[HarvestRow, ...]
    reasons: dict[str, int]


def readmit(
    candidates: Sequence[Candidate], deps: HarvestDeps, dedup: DedupIndex
) -> ReadmitOutcome:
    """Провести кандидатов по цепочке решений, ничего не фиксируя.

    Строки копятся в памяти, а не дописываются в манифест по одной: обрыв на середине оставил
    бы манифест изменённым, а состояние прежним — рассинхронизация, которую потом нечем
    заметить. Кропы при этом пишет `process_candidate`, и это осознанная асимметрия: лишний
    файл кропа безвреден и перезапишется тем же именем, лишняя строка провенанса — нет.

    Журнал отказов не пополняется. Переприём — переderivация, а не сбор: `rebuild_redaction.py`
    в том же положении печатает свод, и 452 повторных `redaction_failed` раздули бы пожизненные
    суммы `bench/harvest-c1.md` на ровном месте.
    """
    строки: list[HarvestRow] = []
    причины: Counter[str] = Counter()
    for кандидат in candidates:
        причина, строка = process_candidate(кандидат, deps, dedup)
        причины[причина] += 1
        if строка is not None:
            строки.append(строка)
    return ReadmitOutcome(tuple(строки), dict(причины))


def commit(
    rows: Sequence[HarvestRow],
    state: HarvestState,
    *,
    manifest_path: Path,
    state_path: Path,
    backup_path: Path,
    verdicts_tmp: Path,
    verdicts_path: Path,
) -> None:
    """Зафиксировать переприём: копия состояния, манифест, вердикты дедупа, состояние.

    Порядок именно такой, и он весь здесь, в одном месте, — иначе его пришлось бы держать в
    голове по двум функциям.

    Манифест копируется байтами и дописывается `append_row`, а не пересобирается через
    `load_manifest`/`write_manifest`: круговой путь через разборщик мог бы переписать
    форматирование чужих строк, и вместо `+N` в git пришёл бы diff на весь файл провенанса.
    Подмена — атомарный `replace`, как в `rebuild_redaction.py`.

    Состояние пишется последним. Обрыв между шагами оставляет манифест дописанным, а состояние
    прежним, и это выбранная сторона: `patch_state` выводит `accepted` из манифеста, поэтому
    повторный прогон досчитает состояние сам, а кандидаты придут в дедуп как `duplicate`.
    Обратный порядок оставил бы состояние, обещающее строки, которых в манифесте нет, — а это
    уже не лечится повтором.

    Копия снимается только если её ещё нет: переприём повторяем, и второй прогон затёр бы
    точку возврата уже пропатченным состоянием — ровно то, за что `prune_state.py` ругает
    второй `--write`. Отказывать при существующей копии здесь нельзя: у прополки операция
    разовая, а повторный `--write` — штатный способ довести незавершённую фиксацию.
    """
    if state_path.exists() and not backup_path.exists():
        backup_path.write_bytes(state_path.read_bytes())
    if rows:
        временный = manifest_path.with_suffix(".readmit.csv")
        временный.unlink(missing_ok=True)
        if manifest_path.exists():
            временный.write_bytes(manifest_path.read_bytes())
        for строка in rows:
            append_row(временный, строка)
        временный.replace(manifest_path)
    merge_verdicts_file(verdicts_tmp, verdicts_path)
    save_state(state_path, state)


def sha256_by_page(path: Path) -> dict[str, str]:
    """`pageid -> sha256` из `reject_index.csv`. Пустой `sha256` в карту не попадает.

    `IndexRow` из `scripts/index_rejects.py` сюда не тащится: читаются три колонки, а импорт
    скрипта скриптом ради дата-класса связал бы два привода без нужды.
    """
    карта: dict[str, str] = {}
    for запись in csv.DictReader(path.read_text(encoding="utf-8").splitlines()):
        if запись.get("sha256"):
            карта[str(запись["source_item_id"])] = str(запись["sha256"])
    return карта


def _не_обходим(
    class_id: str, категория: str, глубина: int, include: tuple[str, ...], exclude: tuple[str, ...]
) -> ExpandResult:
    """Переприём не разворачивает категории: кандидаты известны по журналу отказов."""
    raise NotImplementedError("переприём не обходит категории, он перечитывает известные pageid")


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="переприём кадров, отбитых прежней редакцией")
    парсер.add_argument("--contact", required=True, help="контакт для User-Agent (§3.2)")
    парсер.add_argument("--reason", default="redaction_failed", help="причина отказа в журнале")
    парсер.add_argument("--source", default="commons", help="источник кандидатов")
    парсер.add_argument("--data-root", type=Path, default=None, help="по умолчанию каталог данных")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument("--allow-sa", action="store_true", help="разрешить Tier 3 CC-BY-SA")
    парсер.add_argument("--clip-model", default="ViT-B-32", help="архитектура open_clip")
    парсер.add_argument("--clip-weights", default="laion2b_s34b_b79k", help="веса open_clip")
    парсер.add_argument("--limit", type=int, default=0, help="0 — все кандидаты журнала")
    парсер.add_argument("--write", action="store_true", help="записать, а не показать свод")
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Переприём: свод по умолчанию, запись по `--write`."""
    аргументы = build_parser().parse_args(argv)
    # Умолчание считается здесь, а не в `build_parser`: иначе каталог данных понадобился бы
    # уже на импорте модуля, в том числе тестам, которые в него не ходят.
    корень = аргументы.data_root if аргументы.data_root is not None else harvest_dir()

    отказы = unique_by_reason(load_rejects(корень / "reject_log.csv"), аргументы.reason)
    готовые, отброшенные = plan_readmission(
        отказы, sha256_by_page(корень / "reject_index.csv"), корень, аргументы.source
    )
    print(f"кандидатов с причиной {аргументы.reason}: {len(отказы)}, с оригиналом: {len(готовые)}")
    for строка in отброшенные:
        print(f"  пропуск {строка}")
    if аргументы.limit:
        готовые = готовые[: аргументы.limit]
    if not готовые:
        print("переприёму нечего вести. Останов.")
        return 0

    заявки = {приём.source_item_id: (приём.class_id, приём.basis) for приём in готовые}
    кандидаты = CommonsSource(UrllibTransport(аргументы.contact)).lookup(заявки)
    print(f"метаданные получены: {len(кандидаты)} из {len(заявки)}")
    for pageid in sorted(заявки.keys() - {кандидат.source_item_id for кандидат in кандидаты}):
        print(f"  без метаданных {pageid}: страницы нет или лицензия перестала быть свободной")

    вердикты = корень / VERDICTS_FILENAME
    вердикты.unlink(missing_ok=True)
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки_манифеста = load_manifest(аргументы.manifest) if аргументы.manifest.exists() else []
    зависимости = replace(
        build_deps(
            CachedSource(
                аргументы.source,
                корень,
                {приём.source_item_id: приём.sha256 for приём in готовые},
            ),
            _не_обходим,
            root=корень,
            class_order=tuple(класс.class_id for класс in каталог.classes),
            latin_names=load_latin_names(DEFAULT_LATIN_NAMES_PATH),
            alias_table=load_alias_table(DEFAULT_ALIASES_PATH),
            manifest_path=аргументы.manifest,
            clip_model=аргументы.clip_model,
            clip_weights=аргументы.clip_weights,
            allow_sa=аргументы.allow_sa,
            checked=date.today().isoformat(),
        ),
        # Вердикты дедупа уводятся в отдельный файл: в живом `verdicts.dedup.csv` их ждёт
        # `--audit`, и дописать туда строку до того, как переприём вообще решил фиксироваться,
        # значило бы приговорить чужую строку манифеста заранее.
        dedup_verdicts_path=вердикты,
    )
    # Засев манифестом обязателен: без него переприём принял бы кадр, уже лежащий в манифесте
    # под другим классом (§8 дизайна C1).
    итог = readmit(кандидаты, зависимости, DedupIndex.from_manifest(строки_манифеста))
    print(f"исходы: {dict(sorted(итог.reasons.items()))}")
    print(f"новых строк: {len(итог.rows)}")
    return _зафиксировать(аргументы, корень, строки_манифеста, итог, вердикты)


def _зафиксировать(
    аргументы: argparse.Namespace,
    корень: Path,
    строки_манифеста: Sequence[HarvestRow],
    итог: ReadmitOutcome,
    вердикты: Path,
) -> int:
    """Предохранитель, затем фиксация. Предохранитель считается до первой записи на диск.

    Новое состояние выводится в памяти из «старые строки плюс новые» и проверяется, и только
    после этого начинается запись. Проверка на шаге записи состояния застала бы манифест уже
    переписанным, а состояние прежним — то есть развалила бы ровно то, ради чего порядок
    фиксации и выбран.
    """
    путь_состояния = корень / "state.json"
    состояние = load_state(путь_состояния)
    после = [*строки_манифеста, *итог.rows]
    дельта = accepted_delta(состояние, после)
    for класс, (было, станет) in дельта.items():
        print(f"  accepted {класс}: {было} -> {станет}")
    # Вердикты дедупа печатаются и в сухом прогоне: `supersede` означает, что новая строка
    # вытеснит живую строку манифеста, и человек обязан узнать об этом до записи, а не из
    # результата следующего `--audit`.
    вердиктов = len(load_audit(вердикты)) if вердикты.exists() else 0
    if вердиктов:
        print(f"вердиктов дедупа: {вердиктов} (файл {вердикты})")
    if регресс := accepted_regressions(дельта):
        print(f"accepted уменьшился у классов: {', '.join(регресс)}")
        print("Это расхождение манифеста со счётчиком, предмет прополки, а не переприёма.")
        print("Останов без записи.")
        return 2
    if not аргументы.write:
        print("сухой прогон; для записи повторить с --write")
        return 0
    commit(
        итог.rows,
        patch_state(состояние, после),
        manifest_path=аргументы.manifest,
        state_path=путь_состояния,
        backup_path=путь_состояния.with_suffix(BACKUP_SUFFIX),
        verdicts_tmp=вердикты,
        verdicts_path=корень / DEDUP_VERDICTS_FILENAME,
    )
    копия = путь_состояния.with_suffix(BACKUP_SUFFIX)
    print(f"записано: {аргументы.manifest} +{len(итог.rows)} строк")
    print(f"состояние: {путь_состояния}, точка возврата {копия}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
