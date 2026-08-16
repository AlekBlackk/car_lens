"""Разовый генератор расходного каталога скелета `catalog/stanford.yaml`.

Запускается один раз, результат коммитится и дальше живёт как обычный
YAML. Продакшн-каталог `canonical.yaml` не трогается вовсе: Stanford —
снимок американского рынка по 2012 год с марками, которых больше не
существует (Plymouth, Eagle, Geo, Scion), и подмешивать его в описание
того, что ездит по улицам сегодня, значит потом вычищать по памяти
(решение Р-7).

Имена классов режутся публичной `normalize.model_core`, а не собственной
регуляркой. Иначе каталог и нормализатор разойдутся, и часть классов
перестанет находиться при ingest — молча, без единой ошибки.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from pipeline.catalog import STANFORD_CATALOG_PATH, brand_slug
from pipeline.normalize import model_core
from scripts.fetch_stanford import dataset_dir

# Марки из двух слов: первое слово именем марки не является.
СОСТАВНЫЕ_МАРКИ = ("AM General", "Aston Martin", "Land Rover", "Rolls-Royce")

# Слова, которыми Stanford заканчивает имя класса перед годом.
КУЗОВНЫЕ_СЛОВА = frozenset(
    {"sedan", "coupe", "convertible", "suv", "van", "minivan", "wagon", "hatchback", "cab"}
)

# Список синхронизирован с `_КАБИНА` в pipeline/normalize.py: там же есть
# «double cab» (Ram 1500 Double Cab), и `display_of` обязан снимать то же
# определение, что снимает нормализатор, — иначе `display` разойдётся с
# ядром модели.
ОПРЕДЕЛЕНИЯ_КАБИНЫ = frozenset(
    {"extended", "regular", "crew", "quad", "king", "club", "super", "double"}
)


def split_brand(class_name: str) -> tuple[str, str]:
    """Имя класса → каноническая марка и остаток строки."""
    низкое = class_name.lower()
    for марка in СОСТАВНЫЕ_МАРКИ:
        if низкое.startswith(марка.lower() + " "):
            return марка, class_name[len(марка) :].strip()
    марка, _, остаток = class_name.partition(" ")
    return марка, остаток.strip()


def display_of(class_name: str) -> str:
    """Человекочитаемое имя: без модельного года и без кузовного слова.

    Регистр берётся из исходной строки, а не восстанавливается из слага:
    `mercedes_benz__sl_class` обратно в «Mercedes-Benz SL-Class» не
    разворачивается.
    """
    слова = class_name.split()
    if слова and len(слова[-1]) == 4 and слова[-1].isdigit():
        слова = слова[:-1]
    while слова and слова[-1].lower() in КУЗОВНЫЕ_СЛОВА:
        слова = слова[:-1]
    # «Extended Cab» → после снятия «Cab» остаётся определение, оно тоже лишнее.
    if len(слова) > 1 and слова[-1].lower() in ОПРЕДЕЛЕНИЯ_КАБИНЫ:
        слова = слова[:-1]
    return " ".join(слова)


def build_catalog(class_names: Iterable[str]) -> dict[str, Any]:
    """Список имён классов → структура `canonical.yaml`.

    Кузовные варианты схлопываются: «A5 Coupe» и «A5 Cabriolet» дают один
    ключ `a5` с разным `body_style` при ingest (§3.3). Порядок входа на
    результат не влияет — при столкновении `display` побеждает лексически
    меньший, чтобы генератор был детерминирован.

    Ядро отвергается не только пустым, но и состоящим ровно из одного
    четырёхзначного модельного года: «Audi 2012» после отсечения марки
    даёт остаток «2012», а `model_core("2012")` возвращает непустое ядро
    `("2012", None)` — правило Task 4 снимает хвостовой год лишь когда в
    ядре есть что-то ещё, иначе оно съело бы «Peugeot 2008». Здесь марка
    уже отрезана `split_brand`, поэтому единственный оставшийся токен —
    сам год, и сопоставлять его не с чем.
    """
    марки: dict[str, dict[str, str]] = {}
    for имя in sorted(set(class_names)):
        марка, остаток = split_brand(имя)
        ядро, _ = model_core(остаток)
        if not ядро or (ядро.isdigit() and len(ядро) == 4):
            raise ValueError(f"пустое ядро модели у класса {имя!r}")
        ключ = brand_slug(ядро)
        показ = display_of(имя)
        модели = марки.setdefault(марка, {})
        прежний = модели.get(ключ)
        модели[ключ] = показ if прежний is None else min(прежний, показ)

    return {
        "version": 1,
        "brands": {
            марка: {"models": {ключ: {"display": модели[ключ]} for ключ in sorted(модели)}}
            for марка, модели in sorted(марки.items())
        },
    }


def _прочитать_имена(разметка: Path) -> list[str]:
    with разметка.open(encoding="utf-8", newline="") as ф:
        return [строка["class_name"] for строка in csv.DictReader(ф)]


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа. Возвращает код выхода процесса."""
    парсер = argparse.ArgumentParser(description="Сгенерировать catalog/stanford.yaml")
    парсер.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Путь к annotations.csv (по умолчанию — из CARLENS_DATA_ROOT)",
    )
    парсер.add_argument("--out", type=Path, default=STANFORD_CATALOG_PATH)
    аргументы = парсер.parse_args(argv)

    разметка = аргументы.annotations or (dataset_dir() / "annotations.csv")
    имена = _прочитать_имена(разметка)
    каталог = build_catalog(имена)

    аргументы.out.parent.mkdir(parents=True, exist_ok=True)
    аргументы.out.write_text(
        "# Расходный каталог скелета M-1, сгенерирован scripts/gen_stanford_catalog.py.\n"
        "# Руками не править: правка потеряется при следующем прогоне генератора.\n"
        "# Продакшн-каталог — canonical.yaml, этот файл к нему отношения не имеет (Р-7).\n"
        + yaml.safe_dump(каталог, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    классов = sum(len(т["models"]) for т in каталог["brands"].values())
    print(f"{len(set(имена))} имён классов → {классов} моделей, {len(каталог['brands'])} марок")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
