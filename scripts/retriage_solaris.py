"""Пересмотр принятых строк Accent и Solaris после смены границ классов.

Границы классов задаются теперь поколениями Commons (`Accent (RB)`, `(HC)` — это Solaris),
а 14 из 17 поисковых строк `grp__solaris_hs` помечены иглами `hyundai__accent`. Оставить
их как есть значит сохранить в эталоне ровно ту путаницу, которую поколенческие границы и
должны снять.

Сети нет: оригиналы берутся из кэша `raw/` по `sha256` строки манифеста, поэтому шаг
тестируем целиком. Полный пересчёт всех строк манифеста намеренно не делается — он менял бы
метрику одновременно с обходом и делал бы замер C1b нечитаемым.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pipeline.harvest.cache import cache_path
from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, HarvestRow, load_manifest
from pipeline.paths import harvest_dir

TARGET_CLASSES = ("hyundai__accent", "grp__solaris_hs")
QUEUE_NAME = "solaris_queue.txt"


def to_review(rows: Sequence[HarvestRow], class_ids: Sequence[str]) -> list[HarvestRow]:
    """Строки затронутых классов — вход очереди человека."""
    нужные = set(class_ids)
    return [строка for строка in rows if строка.class_id in нужные]


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="очередь пересмотра Accent и Solaris по кэшу")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument("--root", type=Path, default=None, help="по умолчанию каталог данных")
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать очередь пересмотра по кэшу и напечатать команду галереи."""
    аргументы = build_parser().parse_args(argv)
    # Умолчание считается здесь, а не в `build_parser`: иначе корень данных понадобился
    # бы уже на импорте модуля, в том числе тестам, которые в него не ходят.
    корень: Path = аргументы.root or harvest_dir()

    строки = to_review(load_manifest(аргументы.manifest), TARGET_CLASSES)
    в_очередь: list[str] = []
    без_кэша = 0
    for строка in строки:
        # Оригинал нужен потому, что человек в галерее смотрит кадр целиком, а не кроп:
        # поколение Accent различимо по кузову, а не по фрагменту.
        if cache_path(корень, строка.source, строка.sha256, "jpg").exists():
            в_очередь.append(строка.file)
        else:
            без_кэша += 1

    путь = корень / QUEUE_NAME
    путь.parent.mkdir(parents=True, exist_ok=True)
    # Пустая очередь — пустой файл: `"\n".join` дал бы файл из одного перевода строки,
    # то есть непустую очередь из одного безымянного кадра.
    путь.write_text("".join(f"{имя}\n" for имя in в_очередь), encoding="utf-8")
    print(f"в очередь: {len(в_очередь)}; нет в кэше: {без_кэша}; файл {путь}")
    # Подсказка печатается полной: у галереи `--manifest` и `--data-root` обязательные,
    # и команда без них падает на разборе аргументов, а не показывает кадры.
    print(
        "смотреть: uv run python -m scripts.audit_gallery "
        f"--manifest {аргументы.manifest} --data-root {корень} --only-files {путь}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
