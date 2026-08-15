"""Добыча Stanford Cars и приведение к контракту на диске.

Оригинальные ссылки `ai.stanford.edu` мертвы примерно с 2021 года, а все
живые зеркала неофициальные и отдают данные каждое по-своему. Поэтому
скрипт делает ровно одно: приводит **что угодно** к раскладке, которую
ждёт `pipeline.ingest.stanford`:

    $CARLENS_DATA_ROOT/raw/stanford_cars/
    ├── images/*.jpg
    └── annotations.csv

Всё, что выше этой черты, ломается и меняется. Всё, что ниже, — чистая
функция от каталога с файлами. Ни один тест не вызывает этот скрипт: он
единственное место в конвейере, которому нужна сеть.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.paths import raw_dir

STANFORD_DIRNAME = "stanford_cars"

ANNOTATION_COLUMNS: tuple[str, ...] = (
    "image",
    "x1",
    "y1",
    "x2",
    "y2",
    "class_name",
    "official_split",
)

# Зеркало проверяется руками на шаге разведки; значение меняется одной строкой.
ЗЕРКАЛО = "tanganke/stanford_cars"

ИНСТРУКЦИЯ = """
Автоматическая загрузка не удалась. Разложить датасет руками:

  $CARLENS_DATA_ROOT/raw/stanford_cars/
  ├── images/          — все .jpg одним плоским каталогом
  └── annotations.csv  — колонки: image,x1,y1,x2,y2,class_name,official_split

Колонки x1,y1,x2,y2 — ручной bbox из devkit в координатах исходного фото.
Если разметки нет, оставить их пустыми: кропы тогда режет только YOLO,
а сверка IoU из отчёта выпадает.

Колонка official_split — train или test из оригинального деления. Она
читается для отчёта и на раскладку по сплитам не влияет (решение Р-10).
"""


def dataset_dir() -> Path:
    """Каталог датасета внутри `raw/`."""
    return raw_dir() / STANFORD_DIRNAME


def write_annotations(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    """Записать разметку в CSV строго по контракту.

    Лишнее поле — ошибка, а не игнорируемый мусор: контракт закрытый, и
    молча проглоченная колонка означала бы, что источник отдал что-то, о
    чём мы не знаем.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as ф:
        писатель = csv.DictWriter(ф, fieldnames=list(ANNOTATION_COLUMNS))
        писатель.writeheader()
        for строка in rows:
            лишние = sorted(set(строка) - set(ANNOTATION_COLUMNS))
            if лишние:
                raise ValueError(f"лишние поля в разметке: {', '.join(лишние)}")
            писатель.writerow(
                {к: "" if строка.get(к) is None else строка[к] for к in ANNOTATION_COLUMNS}
            )


def _скачать_с_зеркала(куда: Path) -> int:
    """Скачать зеркало и разложить по контракту. Вернуть число фотографий.

    Импорт `datasets` внутри функции намеренно: пакет живёт в группе
    `train`, которой в CI нет, а модуль обязан импортироваться и без неё.
    """
    from datasets import load_dataset

    изображения = куда / "images"
    изображения.mkdir(parents=True, exist_ok=True)

    имена_классов: list[str] | None = None
    строки: list[dict[str, Any]] = []
    номер = 0

    for деление in ("train", "test"):
        набор = load_dataset(ЗЕРКАЛО, split=деление)
        if имена_классов is None:
            имена_классов = list(набор.features["label"].names)
        for запись in набор:
            номер += 1
            имя = f"{номер:05d}.jpg"
            запись["image"].convert("RGB").save(изображения / имя, quality=95)
            строки.append(
                {
                    "image": имя,
                    "x1": запись.get("x1"),
                    "y1": запись.get("y1"),
                    "x2": запись.get("x2"),
                    "y2": запись.get("y2"),
                    "class_name": имена_классов[int(запись["label"])],
                    "official_split": деление,
                }
            )

    write_annotations(строки, куда / "annotations.csv")
    return номер


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа. Возвращает код выхода процесса."""
    парсер = argparse.ArgumentParser(description="Скачать Stanford Cars по контракту M-1b")
    парсер.add_argument(
        "--force",
        action="store_true",
        help="Скачать заново, даже если каталог уже заполнен",
    )
    аргументы = парсер.parse_args(argv)

    куда = dataset_dir()
    разметка = куда / "annotations.csv"
    if разметка.exists() and not аргументы.force:
        print(f"Датасет уже на месте: {куда}. Перекачать: --force")
        return 0

    try:
        сколько = _скачать_с_зеркала(куда)
    except Exception as ошибка:  # причин отказа много, ответ один
        print(f"Зеркало {ЗЕРКАЛО} недоступно: {ошибка}", file=sys.stderr)
        print(ИНСТРУКЦИЯ, file=sys.stderr)
        return 1

    print(f"Готово: {сколько} фотографий в {куда}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
