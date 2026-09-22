"""Листы-превью кропов по классам с номерами, впаянными в картинку.

Номер впаивается, а не пишется подписью рядом: параллельное чтение картинок возвращает их
не в порядке вызова, и номер на самом изображении — единственная надёжная привязка.
Рядом с листами пишется `<class_id>.txt`: номер → `photo_id`, `listing_id`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.paths import harvest_dir, splits_dir
from pipeline.schema import read_split

CELL = (240, 180)
COLUMNS_PER_SHEET = 8
PER_SHEET = 48


def render_sheet(paths: Sequence[Path], first_number: int = 0) -> Image.Image:
    """Сетка миниатюр по `COLUMNS_PER_SHEET` в ряд, у каждой — жёлтая плашка с номером."""
    рядов = (len(paths) + COLUMNS_PER_SHEET - 1) // COLUMNS_PER_SHEET
    лист = Image.new("RGB", (CELL[0] * COLUMNS_PER_SHEET, CELL[1] * рядов), "black")
    for позиция, путь in enumerate(paths):
        миниатюра = Image.open(путь).convert("RGB")
        миниатюра.thumbnail(CELL)
        x = (позиция % COLUMNS_PER_SHEET) * CELL[0]
        y = (позиция // COLUMNS_PER_SHEET) * CELL[1]
        лист.paste(
            миниатюра,
            (x + (CELL[0] - миниатюра.width) // 2, y + (CELL[1] - миниатюра.height) // 2),
        )
        рисунок = ImageDraw.Draw(лист)
        рисунок.rectangle((x, y, x + 40, y + 18), fill=(255, 255, 0))
        рисунок.text((x + 4, y + 3), str(first_number + позиция), fill=(0, 0, 0))
    return лист


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Листы-превью кропов по классам")
    парсер.add_argument("--split", type=Path, default=splits_dir() / "pool-ext.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--out", type=Path, required=True)
    парсер.add_argument("--classes", nargs="*", default=None, help="class_id; умолчание — все")
    аргументы = парсер.parse_args(argv)

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    кадр = read_split(аргументы.split).sort_values("photo_id")
    аргументы.out.mkdir(parents=True, exist_ok=True)
    for номер_класса, класс in enumerate(каталог.classes):
        if аргументы.classes is not None and класс.class_id not in аргументы.classes:
            continue
        строки = кадр[кадр["class_idx"] == номер_класса]
        if строки.empty:
            continue
        пути = [аргументы.crops_root / p for p in строки["crop_path"]]
        for начало in range(0, len(пути), PER_SHEET):
            лист = render_sheet(пути[начало : начало + PER_SHEET], first_number=начало)
            лист.save(аргументы.out / f"{класс.class_id}_{начало // PER_SHEET}.jpg", quality=85)
        # `listing_id` несёт имя исходного файла (`manual`) или галерею (`dir_path`).
        опись = [
            f"{i}\t{row.photo_id}\t{row.listing_id}" for i, row in enumerate(строки.itertuples())
        ]
        (аргументы.out / f"{класс.class_id}.txt").write_text("\n".join(опись), "utf-8")
    print(f"листы: {аргументы.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
