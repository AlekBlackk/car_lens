"""Кампания MAD-Cars: выборка, скачивание, листы (`python -m scripts.mad_cars <команда>`)."""

from __future__ import annotations

import argparse
import http.client
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.ingest.mad_cars import fetch_frames, read_excluded, read_mad_mapping, sample_frames
from pipeline.paths import data_root

DEFAULT_ROOT = data_root() / "external" / "mad_cars"
DEFAULT_MAPPING = Path("catalog/external/mad_cars.csv")
DEFAULT_EXCLUDED = Path("catalog/external/mad_cars_excluded.csv")
CELL = (250, 150)
CARS_PER_SHEET = 16


def _sample(аргументы: argparse.Namespace) -> int:
    таблица = read_mad_mapping(аргументы.mapping, load_catalog(DEFAULT_CATALOG_PATH))
    манифест = pd.read_csv(
        аргументы.manifest, dtype=str, usecols=["brand", "model", "car_id", "view_id", "url"]
    )
    выборка = sample_frames(
        манифест, таблица, cars_per_class=аргументы.cars_per_class, seed=аргументы.seed
    )
    выборка.to_csv(аргументы.out, index=False)
    print(
        f"выборка: {выборка['model_id'].nunique()} классов, {выборка['car_id'].nunique()} "
        f"машин, {len(выборка)} кадров → {аргументы.out}"
    )
    return 0


def download_url(url: str, путь: Path) -> None:
    for попытка in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as ответ:
                путь.write_bytes(ответ.read())
            return
        except (OSError, http.client.HTTPException):
            if попытка == 2:
                raise


def _fetch(аргументы: argparse.Namespace) -> int:
    выборка = pd.read_csv(аргументы.sample, dtype=str)
    отчёт = fetch_frames(выборка, аргументы.root, read_excluded(аргументы.excluded), download_url)
    print(отчёт)
    return 1 if отчёт.failed else 0


def render_car_sheet(
    машины: Sequence[tuple[str, Sequence[Path]]], first_number: int
) -> Image.Image:
    """Две колонки машин, по 3 кадра машины в строку; номер и `car_id` впаяны в кадр."""
    половина = (len(машины) + 1) // 2
    лист = Image.new("RGB", (CELL[0] * 6, CELL[1] * половина), (30, 30, 30))
    рисовать = ImageDraw.Draw(лист)
    for i, (машина, кадры) in enumerate(машины):
        x0 = 0 if i < половина else CELL[0] * 3
        y0 = CELL[1] * (i % половина)
        for j, путь in enumerate(кадры[:3]):
            with Image.open(путь) as исходник:
                миниатюра = исходник.convert("RGB")
            миниатюра.thumbnail((CELL[0], CELL[1] - 16))
            лист.paste(миниатюра, (x0 + CELL[0] * j, y0))
            рисовать.text(
                (x0 + CELL[0] * j + 4, y0 + CELL[1] - 14),
                f"{first_number + i} car {машина}",
                fill=(255, 255, 0),
            )
    return лист


def _по_машинам(папка: Path) -> list[tuple[str, list[Path]]]:
    """Кадры класса по машинам: `car_id` и `view_id` сортируются числом."""
    машины: dict[str, list[Path]] = {}
    for путь in папка.glob("*.jpg"):
        машины.setdefault(путь.stem.split("_")[0], []).append(путь)
    return [
        (машина, sorted(кадры, key=lambda п: int(п.stem.split("_")[1])))
        for машина, кадры in sorted(машины.items(), key=lambda пара: int(пара[0]))
    ]


def _sheets(аргументы: argparse.Namespace) -> int:
    аргументы.out.mkdir(parents=True, exist_ok=True)
    классы = аргументы.classes or sorted(п.name for п in аргументы.root.iterdir() if п.is_dir())
    for класс in классы:
        машины = _по_машинам(аргументы.root / класс)
        for начало in range(0, len(машины), CARS_PER_SHEET):
            лист = render_car_sheet(машины[начало : начало + CARS_PER_SHEET], first_number=начало)
            лист.save(аргументы.out / f"{класс}_{начало // CARS_PER_SHEET}.jpg", quality=85)
        for _, кадры in машины:
            for путь in кадры:
                with Image.open(путь) as картинка:
                    if картинка.height > картинка.width:
                        print(f"повёрнут: {путь}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Кампания MAD-Cars")
    команды = парсер.add_subparsers(dest="command", required=True)
    выборка = команды.add_parser("sample", help="выборка кадров из mad.csv")
    выборка.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "mad.csv")
    выборка.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    выборка.add_argument("--cars-per-class", type=int, default=100)
    выборка.add_argument("--seed", type=int, required=True)
    выборка.add_argument("--out", type=Path, required=True)
    выборка.set_defaults(run=_sample)
    скачать = команды.add_parser("fetch", help="докачать выборку, удалить выброшенные машины")
    скачать.add_argument("--sample", type=Path, required=True)
    скачать.add_argument("--root", type=Path, default=DEFAULT_ROOT / "full")
    скачать.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    скачать.set_defaults(run=_fetch)
    листы = команды.add_parser("sheets", help="листы по машинам для просмотра")
    листы.add_argument("--root", type=Path, default=DEFAULT_ROOT / "full")
    листы.add_argument("--out", type=Path, required=True)
    листы.add_argument("--classes", nargs="*", default=None)
    листы.set_defaults(run=_sheets)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    аргументы = build_parser().parse_args(argv)
    return int(аргументы.run(аргументы))


if __name__ == "__main__":
    raise SystemExit(main())
