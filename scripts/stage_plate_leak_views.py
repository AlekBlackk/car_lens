"""Просмотровые виды выборки утечки номера — номер впаян в пиксели (§4 отчёта).

Почему впаян, а не HTML-подписью `audit_gallery.py`: ту галерею смотрит человек в браузере,
где подпись и картинка на экране естественно связаны местом. Здесь просмотр идёт через
инструмент Read, а он при нескольких вызовах за один ход возвращает картинки не в порядке
вызова — на замере 2026-09-05 виды 005–010 пришли как 007, 005, 006, 008, 010, 009. Подпись
в самом файле делает привязку «номер = картинка» верной независимо от порядка получения:
без неё вердикт может приклеиться к чужому кадру, и это не обнаруживается вообще — оба
кадра выглядят правдоподобно (память `image-audit-needs-burned-in-numbers`).

Источник — черновик выборки (`sample_plate_leak.py`, несёт `crop_path`), а не итоговый
`bench/plate-leak-sample.csv`: у него `crop_path` нет вовсе (§11 отчёта).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pipeline.paths import data_root, harvest_dir

DEFAULT_LONG_SIDE = 1600
CAPTION_HEIGHT = 46
CAPTION_FONT_SIZE = 30
DEFAULT_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")


def caption_text(
    number: int, class_id: str, plates: int, faces: int, width: int, height: int
) -> str:
    """Подпись, впаиваемая в кадр целиком: номер, класс, числа детектора, размер."""
    return f"#{number:03d}  {class_id}  plates={plates} faces={faces}  {width}x{height}"


def render_view(
    image: Image.Image,
    caption: str,
    *,
    long_side: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Уменьшить (если нужно) до `long_side`, дописать белую полосу с подписью сверху.

    Полоса — не наложение поверх кадра: номер обязан читаться, даже если фон машины
    белый или подпись легла на светлый участок кузова.
    """
    ш, в = image.size
    к = long_side / max(ш, в)
    вид = (
        image if к >= 1.0 else image.resize((round(ш * к), round(в * к)), Image.Resampling.LANCZOS)
    )
    полотно = Image.new("RGB", (вид.size[0], вид.size[1] + CAPTION_HEIGHT), "white")
    полотно.paste(вид, (0, CAPTION_HEIGHT))
    ImageDraw.Draw(полотно).text((10, 8), caption, fill="black", font=font)
    return полотно


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(
        description="Просмотровые виды выборки утечки номера с впаянным номером"
    )
    парсер.add_argument(
        "--sample", type=Path, required=True, help="черновик выборки с колонкой crop_path"
    )
    парсер.add_argument("--crops-root", type=Path, default=None)
    парсер.add_argument("--out-dir", type=Path, default=None)
    парсер.add_argument("--long-side", type=int, default=DEFAULT_LONG_SIDE)
    парсер.add_argument("--font", type=Path, default=DEFAULT_FONT)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать вид на каждый кадр выборки, записать сводный `views.csv`."""
    аргументы = build_parser().parse_args(argv)

    выборка = pd.read_csv(аргументы.sample)
    if "crop_path" not in выборка.columns:
        raise ValueError(
            f"{аргументы.sample}: нет колонки crop_path — это черновик sample_plate_leak.py?"
        )

    crops_root = аргументы.crops_root or harvest_dir() / "crops"
    out_dir = аргументы.out_dir or (data_root() / "audit" / "plate_leak_views")
    out_dir.mkdir(parents=True, exist_ok=True)
    шрифт = ImageFont.truetype(str(аргументы.font), CAPTION_FONT_SIZE)

    записи: list[dict[str, object]] = []
    пропущено = 0
    for номер, sha256, класс, plates, faces, crop_path in zip(
        выборка["номер"],
        выборка["sha256"],
        выборка["класс"],
        выборка["plates"],
        выборка["faces"],
        выборка["crop_path"],
        strict=True,
    ):
        источник = crops_root / str(crop_path)
        if not источник.is_file():
            print(f"пропущен #{int(номер):03d}: нет файла {источник}")
            пропущено += 1
            continue
        картинка = Image.open(источник).convert("RGB")
        ш, в = картинка.size
        подпись = caption_text(int(номер), str(класс), int(plates), int(faces), ш, в)
        вид = render_view(картинка, подпись, long_side=аргументы.long_side, font=шрифт)
        имя_файла = f"{int(номер):03d}.jpg"
        вид.save(out_dir / имя_файла, quality=92, subsampling=1)
        записи.append({"номер": номер, "sha256": sha256, "файл": имя_файла})

    pd.DataFrame(записи).to_csv(out_dir / "views.csv", index=False, encoding="utf-8")
    print(f"подготовлено видов: {len(записи)} (пропущено {пропущено}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
