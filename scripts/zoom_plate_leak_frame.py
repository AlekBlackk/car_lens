"""Зум спорного места кадра выборки — из ОРИГИНАЛА кропа, без уменьшения (§4 отчёта).

Просмотровый вид (`stage_plate_leak_views.py`) уже уменьшен до `--long-side`: спорный
участок (сама табличка номера) на нём может быть мельче полного разрешения кропа.
Этот скрипт вырезает область вокруг спорного места из исходного файла и подписывает
номером выборки — по той же причине, что и виды: привязка «номер = картинка» не должна
зависеть от порядка, в котором файл вернётся инструменту чтения.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pipeline.paths import data_root, harvest_dir

MARGIN = 0.25
MAX_SIDE = 1500
MIN_SIDE = 700
CAPTION_HEIGHT = 40
CAPTION_FONT_SIZE = 26
DEFAULT_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")


def expand_box(
    box: tuple[float, float, float, float], size: tuple[int, int], margin: float
) -> tuple[int, int, int, int]:
    """Расширить относительный бокс (доли ширины/высоты, `0..1`) на поле `margin`.

    Поле — не декорация: спорное место у самого края бокса без запаса читается хуже,
    а граница таблички часто и есть край размытого пятна.
    """
    x0, y0, x1, y1 = box
    ш, в = size
    бш, бв = (x1 - x0) * ш, (y1 - y0) * в
    return (
        max(0, round(x0 * ш - margin * бш)),
        max(0, round(y0 * в - margin * бв)),
        min(ш, round(x1 * ш + margin * бш)),
        min(в, round(y1 * в + margin * бв)),
    )


def fit_scale(size: tuple[int, int], *, max_side: int, min_side: int) -> float:
    """Множитель: сжать выше `max_side`, увеличить (не более 3x) ниже `min_side`."""
    сторона = max(size)
    if сторона > max_side:
        return max_side / сторона
    if сторона < min_side:
        return min(3.0, min_side / сторона)
    return 1.0


def render_zoom(
    image: Image.Image,
    box: tuple[int, int, int, int],
    caption: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Вырезать `box`, отмасштабировать по `fit_scale`, подписать полосой сверху."""
    вырезка = image.crop(box)
    к = fit_scale(вырезка.size, max_side=MAX_SIDE, min_side=MIN_SIDE)
    if к != 1.0:
        вырезка = вырезка.resize(
            (round(вырезка.size[0] * к), round(вырезка.size[1] * к)), Image.Resampling.LANCZOS
        )
    полотно = Image.new("RGB", (вырезка.size[0], вырезка.size[1] + CAPTION_HEIGHT), "white")
    полотно.paste(вырезка, (0, CAPTION_HEIGHT))
    ImageDraw.Draw(полотно).text((8, 6), caption, fill="black", font=font)
    return полотно


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Зум спорного места кадра выборки")
    парсер.add_argument(
        "--sample", type=Path, required=True, help="черновик выборки с колонкой crop_path"
    )
    парсер.add_argument("--number", type=int, required=True, help="номер кадра в выборке")
    парсер.add_argument(
        "--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), required=True
    )
    парсер.add_argument("--label", default="zoom")
    парсер.add_argument("--crops-root", type=Path, default=None)
    парсер.add_argument("--out-dir", type=Path, default=None)
    парсер.add_argument("--font", type=Path, default=DEFAULT_FONT)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Найти кадр по номеру, вырезать спорное место, сохранить зум."""
    аргументы = build_parser().parse_args(argv)

    выборка = pd.read_csv(аргументы.sample).set_index("номер")
    if аргументы.number not in выборка.index:
        raise ValueError(f"номера {аргументы.number} нет в {аргументы.sample}")
    строка = выборка.loc[аргументы.number]

    crops_root = аргументы.crops_root or harvest_dir() / "crops"
    источник = crops_root / str(строка["crop_path"])
    if not источник.is_file():
        raise FileNotFoundError(f"нет файла кропа: {источник}")

    картинка = Image.open(источник).convert("RGB")
    box = expand_box(tuple(аргументы.box), картинка.size, MARGIN)
    ш, в = картинка.size
    подпись = (
        f"#{аргументы.number:03d} {аргументы.label} "
        f"вырезка {box[2] - box[0]}x{box[3] - box[1]} из {ш}x{в}"
    )
    шрифт = ImageFont.truetype(str(аргументы.font), CAPTION_FONT_SIZE)
    зум = render_zoom(картинка, box, подпись, font=шрифт)

    out_dir = аргументы.out_dir or (data_root() / "audit" / "plate_leak_views" / "zooms")
    out_dir.mkdir(parents=True, exist_ok=True)
    цель = out_dir / f"{аргументы.number:03d}_{аргументы.label}.jpg"
    зум.save(цель, quality=95, subsampling=0)
    print(цель)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
