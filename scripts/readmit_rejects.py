"""Переприём кадров, отбитых прежней редакцией: манифест добирает то, что правило уже приняло.

Сеть нужна один раз и только на метаданные: провенанс страницы (автор, лицензия, URL) в
журнале отказов не лежит, а пиксели лежат в кэше оригиналов. Отсюда устройство: один проход
`CommonsSource.lookup` по `pageid`, дальше офлайн через `CachedSource`.

Решения принимает `run.process_candidate` — тот же код, что боевой прогон. Своей копии
допуска, редакции или дедупа здесь нет: копия разошлась бы с боевой и приняла бы кадры не тем
конвейером, каким их обещает манифест.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.manifest import HarvestRow, append_row
from pipeline.harvest.readmit import merge_verdicts_file
from pipeline.harvest.run import HarvestDeps, process_candidate
from pipeline.harvest.sources import Candidate
from pipeline.harvest.state import HarvestState, save_state


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
