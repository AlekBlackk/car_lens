"""Одноразовый генератор ЧЕРНОВИКА таблицы игл.

Скрипт не участвует ни в тестах, ни в шлюзе, ни в триаже: источник истины — файл
`catalog/model_aliases.yaml`, и он правится руками (§5 дизайна задачи B). Генератор
существует ровно затем, чтобы не набивать 150 классов с нуля, и его результат **без
ручной правки негоден**: он не знает ни кузовных кодов, ни экспортных имён, ни того, что
голая игла «niva» накрывает сразу три класса каталога.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import (
    DEFAULT_ALIASES_PATH,
    normalize_filename,
    render_alias_table,
)
from pipeline.harvest.queries import (
    DEFAULT_LATIN_NAMES_PATH,
    DEFAULT_QUERIES_PATH,
    load_latin_names,
    load_queries,
)


def черновые_иглы(brand: str | None, terms: Sequence[str], latin_name: str) -> list[str]:
    """Термины и латинское имя минус токены бренда. Группы режутся по слову or."""
    заготовки = [*terms, *latin_name.split(" or ")]
    бренд = set(normalize_filename(brand).split()) if brand else set()
    иглы: set[str] = set()
    for заготовка in заготовки:
        токены = normalize_filename(заготовка).split()
        while токены and токены[0] in бренд:
            токены.pop(0)
        if токены:
            иглы.add(" ".join(токены))
    return sorted(иглы)


def main(argv: Sequence[str] | None = None) -> int:
    """Записать черновик таблицы. Существующий файл не перезаписывается без --force."""
    парсер = argparse.ArgumentParser(description="Черновик таблицы игл (правится руками)")
    парсер.add_argument("--out", type=Path, default=DEFAULT_ALIASES_PATH)
    парсер.add_argument("--force", action="store_true", help="перезаписать существующий файл")
    аргументы = парсер.parse_args(argv)

    if аргументы.out.exists() and not аргументы.force:
        print(f"{аргументы.out} уже есть: ручные правки дороже черновика. Нужен --force.")
        return 2

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    запросы = load_queries(DEFAULT_QUERIES_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES_PATH)

    таблица: dict[str, tuple[str, ...]] = {}
    пусто: list[str] = []
    for класс in каталог.classes:
        иглы = черновые_иглы(класс.brand, запросы[класс.class_id].terms, имена[класс.class_id])
        if not иглы:
            пусто.append(класс.class_id)
            иглы = ["ЗАПОЛНИТЬ РУКАМИ"]
        таблица[класс.class_id] = tuple(иглы)
    аргументы.out.write_text(render_alias_table(таблица), encoding="utf-8")

    список_пустых = ", ".join(пусто)
    print(f"черновик записан: {аргументы.out}, классов {len(каталог.classes)}")
    if пусто:
        print(f"без игл, заполнить руками: {список_пустых}")
    print("Дальше: pytest tests/test_model_aliases_table.py и разведение конфликтов.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
