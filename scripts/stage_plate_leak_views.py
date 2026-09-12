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
STATUS_TILE_SIZE = (1600, 900)
DEFAULT_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")


def caption_text(
    number: int, class_id: str, plates: int, faces: int, width: int, height: int
) -> str:
    """Подпись, впаиваемая в кадр целиком: номер, класс, числа детектора, размер."""
    return f"#{number:03d}  {class_id}  plates={plates} faces={faces}  {width}x{height}"


def party_caption_text(
    *,
    party_number: int,
    frame_number: int,
    photo_id: str,
    class_id: str,
    pipeline_outcome: str,
    plates: int,
    faces: int,
    width: int,
    height: int,
) -> str:
    """Связать party-вид с immutable ID, порядком, исходом и детекторными числами."""
    return (
        f"party={party_number:03d} frame={frame_number:03d} photo={photo_id[:12]} "
        f"{class_id} outcome={pipeline_outcome} plates={plates} faces={faces} "
        f"{width}x{height}"
    )


def render_view(
    image: Image.Image,
    caption: str,
    *,
    long_side: int | None = DEFAULT_LONG_SIDE,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Уменьшить (если нужно) до `long_side`, дописать белую полосу с подписью сверху.

    Полоса — не наложение поверх кадра: номер обязан читаться, даже если фон машины
    белый или подпись легла на светлый участок кузова.
    """
    ш, в = image.size
    if long_side is None:
        вид = image
    else:
        к = long_side / max(ш, в)
        вид = (
            image
            if к >= 1.0
            else image.resize((round(ш * к), round(в * к)), Image.Resampling.LANCZOS)
        )
    полотно = Image.new("RGB", (вид.size[0], вид.size[1] + CAPTION_HEIGHT), "white")
    полотно.paste(вид, (0, CAPTION_HEIGHT))
    ImageDraw.Draw(полотно).text((10, 8), caption, fill="black", font=font)
    return полотно


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(
        description="Просмотровые виды выборки утечки номера с впаянным номером"
    )
    источник = парсер.add_mutually_exclusive_group(required=True)
    источник.add_argument("--sample", type=Path, help="черновик выборки с колонкой crop_path")
    источник.add_argument("--party-ledger", type=Path, help="ledger полной независимой партии")
    парсер.add_argument("--party-number", type=int, default=None)
    парсер.add_argument("--crops-root", type=Path, default=None)
    парсер.add_argument("--out-dir", type=Path, default=None)
    парсер.add_argument("--long-side", type=int, default=DEFAULT_LONG_SIDE)
    парсер.add_argument("--font", type=Path, default=DEFAULT_FONT)
    return парсер


def _load_font(path: Path) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if str(path) == "default":
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), CAPTION_FONT_SIZE)


def _cell(row: object, name: str, default: str = "") -> str:
    value = getattr(row, name, default)
    return default if pd.isna(value) else str(value)


def _number(row: object, name: str) -> int:
    value = getattr(row, name, 0)
    return 0 if pd.isna(value) else int(value)


def _status_tile(detail: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    image = Image.new("RGB", STATUS_TILE_SIZE, "#eeeeee")
    draw = ImageDraw.Draw(image)
    draw.text((40, 100), detail, fill="#8b0000", font=font)
    return image


def _party_file_name(party_number: int, frame_number: int, photo_id: str) -> str:
    safe_prefix = "".join(character for character in photo_id[:12] if character.isalnum())
    return f"{party_number:03d}-{frame_number:03d}-{safe_prefix}.jpg"


def _stage_party(arguments: argparse.Namespace) -> int:
    if arguments.party_number is None or arguments.party_number < 1:
        raise ValueError("party mode требует положительный --party-number")
    ledger = pd.read_csv(arguments.party_ledger, keep_default_na=False)
    required = {"photo_id", "pipeline_outcome", "reject_reason"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"в party ledger нет колонок: {missing}")
    duplicates = ledger[ledger["photo_id"].duplicated()]["photo_id"].tolist()
    if duplicates:
        raise ValueError(f"дубликат photo_id в party ledger: {duplicates[:5]}")
    crops_root = arguments.crops_root or harvest_dir() / "crops"
    out_dir = arguments.out_dir or data_root() / "audit" / "independent_plate_leak_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(arguments.font)
    records: list[dict[str, object]] = []
    for frame_number, row in enumerate(ledger.itertuples(index=False), start=1):
        photo_id = _cell(row, "photo_id")
        outcome = _cell(row, "pipeline_outcome")
        reason = _cell(row, "reject_reason")
        crop_path = _cell(row, "crop_path")
        class_id = _cell(row, "class_id")
        plates = _number(row, "plates")
        faces = _number(row, "faces")
        source = crops_root / crop_path
        if outcome == "accepted" and crop_path and source.is_file():
            image = Image.open(source).convert("RGB")
            width, height = image.size
            view_status = "image"
            detail = ""
        elif outcome == "rejected":
            image = Image.new("RGB", STATUS_TILE_SIZE, "#eeeeee")
            width, height = image.size
            view_status = "rejected_tile"
            detail = f"REJECTED: {reason}"
        else:
            outcome = "operational_error"
            reason = "missing_crop"
            image = Image.new("RGB", STATUS_TILE_SIZE, "#eeeeee")
            width, height = image.size
            view_status = "operational_error"
            detail = f"OPERATIONAL ERROR: missing accepted crop {source}"
        caption = party_caption_text(
            party_number=arguments.party_number,
            frame_number=frame_number,
            photo_id=photo_id,
            class_id=class_id,
            pipeline_outcome=outcome,
            plates=plates,
            faces=faces,
            width=width,
            height=height,
        )
        if view_status == "image":
            view = render_view(image, caption, long_side=None, font=font)
        else:
            tile = _status_tile(detail, font)
            view = render_view(tile, caption, long_side=None, font=font)
        file_name = _party_file_name(arguments.party_number, frame_number, photo_id)
        view.save(out_dir / file_name, quality=92, subsampling=1)
        records.append(
            {
                "party_number": arguments.party_number,
                "frame_number": frame_number,
                "photo_id": photo_id,
                "class_id": class_id,
                "pipeline_outcome": outcome,
                "reject_reason": reason,
                "plates": plates,
                "faces": faces,
                "view_status": view_status,
                "caption": caption,
                "file": file_name,
            }
        )
    pd.DataFrame(records).to_csv(out_dir / "views.csv", index=False, encoding="utf-8")
    print(f"подготовлено party-видов: {len(records)} -> {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать вид на каждый кадр выборки, записать сводный `views.csv`."""
    аргументы = build_parser().parse_args(argv)
    if аргументы.party_ledger is not None:
        return _stage_party(аргументы)

    выборка = pd.read_csv(аргументы.sample)
    if "crop_path" not in выборка.columns:
        raise ValueError(
            f"{аргументы.sample}: нет колонки crop_path — это черновик sample_plate_leak.py?"
        )

    crops_root = аргументы.crops_root or harvest_dir() / "crops"
    out_dir = аргументы.out_dir or (data_root() / "audit" / "plate_leak_views")
    out_dir.mkdir(parents=True, exist_ok=True)
    шрифт = _load_font(аргументы.font)

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
