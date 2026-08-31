"""Однократная прополка `done_items` перед прогоном C1b.

Метка кандидата — `source:pageid`, и подкатегорийный обход вернёт те же самые `pageid`,
что полнотекстовый поиск уже перебрал. Метка уходит в `done_items` безусловно и раньше
цепочки решений, поэтому без прополки категорийный обход отработает вхолостую: новые
категории, старые файлы, ноль новых строк (§2.6 дизайна C1).

Прополка не выборочная, и это не лень: причин отказа за прошлые прогоны не существует
(§2.5), выбирать не из чего. `accepted` остаётся — это счётчик набранного, а не журнал
просмотра; обнулив его, мы попросили бы добрать поверх уже добранного.

Операция разовая, поэтому скрипт, а не флаг харвеста.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.harvest.state import HarvestState, load_state, save_state
from pipeline.paths import harvest_dir


def prune(state: HarvestState, rows: Sequence[HarvestRow]) -> HarvestState:
    """Пересобрать состояние: в `done_items` остаются только принятые строки."""
    return HarvestState(
        seen={},
        accepted=dict(state.accepted),
        tokens={},
        done_items={f"{строка.source}:{строка.source_item_id}" for строка in rows},
        categories={},
    )


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="прополка done_items перед прогоном C1b")
    парсер.add_argument("--write", action="store_true", help="записать, а не показать дельту")
    парсер.add_argument("--state", type=Path, default=None, help="по умолчанию в каталоге данных")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Напечатать дельту прополки и, по `--write`, записать её, сохранив прежнее в `.bak`."""
    парсер = build_parser()
    аргументы = парсер.parse_args(argv)
    # Умолчание считается здесь, а не в `build_parser`: иначе корень данных понадобился
    # бы уже на импорте модуля, в том числе тестам, которые в него не ходят.
    путь: Path = аргументы.state or harvest_dir() / "state.json"

    состояние = load_state(путь)
    строки = load_manifest(аргументы.manifest)
    новое = prune(состояние, строки)

    print(f"done_items: было {len(состояние.done_items)}, станет {len(новое.done_items)}")
    print(f"удаляется меток: {len(состояние.done_items - новое.done_items)}")
    print(f"seen обнуляется у {len(состояние.seen)} классов, tokens — {len(состояние.tokens)}")
    if not аргументы.write:
        print("сухой прогон; для записи повторить с --write")
        return 0

    # `load_state` считает отсутствие файла пустым состоянием, а не ошибкой, поэтому
    # копировать в этом случае нечего — и падать тоже не на чем. Флаг снимается до
    # записи: после `save_state` файл существует всегда.
    было_на_диске = путь.exists()
    бэкап = путь.with_suffix(".json.bak")
    if было_на_диске:
        бэкап.write_text(путь.read_text(encoding="utf-8"), encoding="utf-8")
    save_state(путь, новое)

    print(f"записано в {путь}")
    print(f"прежнее состояние в {бэкап}" if было_на_диске else "резервной копии нет: файла не было")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
