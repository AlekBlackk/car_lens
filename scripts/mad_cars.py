"""Кампания MAD-Cars: выборка, скачивание, листы (`python -m scripts.mad_cars <команда>`)."""

from __future__ import annotations

import argparse
import http.client
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.ingest.mad_cars import (
    fetch_frames,
    read_car_ids,
    read_excluded,
    read_mad_disputed,
    read_mad_mapping,
    sample_eval_frames,
    sample_frames,
    sample_negative_frames,
    sample_reject_pool_frames,
    select_off_catalog_pairs,
)
from pipeline.paths import data_root

DEFAULT_ROOT = data_root() / "external" / "mad_cars"
DEFAULT_MAPPING = Path("catalog/external/mad_cars.csv")
DEFAULT_EXCLUDED = Path("catalog/external/mad_cars_excluded.csv")
DEFAULT_DISPUTED = Path("catalog/external/mad_cars_disputed.csv")
DEFAULT_MEASURED = Path("catalog/external/mad_cars_negatives_sample.csv")
DEFAULT_SAMPLE_05 = Path("catalog/external/mad_cars_sample.csv")
DEFAULT_REJECT_POOL = Path("catalog/external/mad_cars_reject_pool.csv")
DEFAULT_NEW_NAMES = Path("catalog/external/mad_cars_cat192_names.csv")
CELL = (250, 150)
CARS_PER_SHEET = 16
CARS_PER_SINGLE_SHEET = 48
SINGLE_SHEET_COLUMNS = 6


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


def _cat192_sample(аргументы: argparse.Namespace) -> int:
    """Выборки каталога 192 (`bench/catalog-192-preregistration.md`): обучение и две проверки.

    Ни одна машина не выполняет двух ролей: проверка и обучение не пересекаются, машины
    замера отказа (негативы, пул прототипов) не берутся ни туда, ни туда, а проверка старых
    классов не трогает выборку 0.5, на которой они учились.
    """
    таблица = read_mad_mapping(аргументы.mapping, load_catalog(DEFAULT_CATALOG_PATH))
    новые = {
        (строка["mad_brand"], строка["mad_model"])
        for строка in pd.read_csv(аргументы.new_names, dtype=str).to_dict("records")
    }
    нет = sorted(новые - set(таблица))
    if нет:
        raise SystemExit(f"названий нет в таблице соответствия: {нет}")
    манифест = pd.read_csv(
        аргументы.manifest, dtype=str, usecols=["brand", "model", "car_id", "view_id", "url"]
    ).dropna(subset=["brand", "model"])

    замер = read_car_ids(аргументы.negatives, аргументы.reject_pool)
    прежние = read_car_ids(аргументы.sample_05)
    новая_таблица = {пара: класс for пара, класс in таблица.items() if пара in новые}
    старая_таблица = {пара: класс for пара, класс in таблица.items() if пара not in новые}

    проверка_новых = sample_eval_frames(
        манифест,
        новая_таблица,
        cars_per_group=аргументы.eval_cars,
        by_name=True,
        seed=аргументы.seed,
        exclude_cars=замер | прежние,
    )
    # `--only-new`: проверка старых классов уже собрана прежним запуском и не пересобирается.
    проверка_старых = (
        None
        if аргументы.only_new
        else sample_eval_frames(
            манифест,
            старая_таблица,
            cars_per_group=аргументы.eval_cars,
            by_name=False,
            seed=аргументы.seed,
            exclude_cars=замер | прежние,
        )
    )
    занятые = (
        замер
        | set(проверка_новых["car_id"])
        | (set() if проверка_старых is None else set(проверка_старых["car_id"]))
    )
    # Классы обучения — классы новых названий: новые и четыре преемника (их кадры пересобираются).
    классы = set(новая_таблица.values())
    обучение = sample_frames(
        манифест,
        {пара: класс for пара, класс in таблица.items() if класс in классы},
        cars_per_class=аргументы.train_cars,
        seed=аргументы.seed,
        exclude_cars=занятые,
    )
    аргументы.out_dir.mkdir(parents=True, exist_ok=True)
    выборки = [("train", обучение), ("eval_new", проверка_новых)]
    if проверка_старых is not None:
        выборки.append(("eval_old", проверка_старых))
    for имя, кадр in выборки:
        путь = аргументы.out_dir / f"mad_cars_{аргументы.prefix}_{имя}.csv"
        кадр.to_csv(путь, index=False)
        print(
            f"{имя}: {кадр['model_id'].nunique()} классов, {кадр['car_id'].nunique()} машин, "
            f"{len(кадр)} кадров → {путь}"
        )
    return 0


def _чужие_пары(
    аргументы: argparse.Namespace,
) -> tuple[pd.DataFrame, list[tuple[str, str]], dict[tuple[str, str], str]]:
    """Манифест MAD-Cars и чужие пары: одни фильтры у выборки замера и у пула прототипов."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    точные = read_mad_mapping(аргументы.mapping, каталог)
    спорные = read_mad_disputed(аргументы.disputed, каталог)
    # Машина без марки и модели может оказаться машиной каталога, чужой её не назовёшь.
    манифест = pd.read_csv(
        аргументы.manifest, dtype=str, usecols=["brand", "model", "car_id", "view_id", "url"]
    ).dropna(subset=["brand", "model"])
    чужие, отброшены = select_off_catalog_pairs(
        zip(манифест["brand"], манифест["model"], strict=True),
        точные.keys(),
        спорные,
        load_alias_table(DEFAULT_ALIASES_PATH),
    )
    return манифест, чужие, отброшены


def _negatives_sample(аргументы: argparse.Namespace) -> int:
    манифест, чужие, отброшены = _чужие_пары(аргументы)
    выборка = sample_negative_frames(
        манифест,
        чужие,
        cars_per_model=аргументы.cars_per_model,
        total=аргументы.total,
        seed=аргументы.seed,
    )
    выборка.to_csv(аргументы.out, index=False)
    причины = Counter(причина.split(":")[0] for причина in отброшены.values())
    print(
        f"чужих моделей {len(чужие)}, отброшено {dict(sorted(причины.items()))}; выборка: "
        f"{выборка['model_id'].nunique()} моделей, {len(выборка)} машин → {аргументы.out}"
    )
    return 0


def _reject_pool_sample(аргументы: argparse.Namespace) -> int:
    манифест, чужие, _ = _чужие_пары(аргументы)
    замер = set(pd.read_csv(аргументы.measured, dtype=str)["model_id"])
    пул = sample_reject_pool_frames(
        манифест, чужие, замер, cars_per_model=аргументы.cars_per_model, seed=аргументы.seed
    )
    пул.to_csv(аргументы.out, index=False)
    print(
        f"чужих моделей {len(чужие)}, в замере {len(замер)}; пул: "
        f"{пул['model_id'].nunique()} моделей, {пул['car_id'].nunique()} машин, "
        f"{len(пул)} кадров → {аргументы.out}"
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
    машины: Sequence[tuple[str, Sequence[Path]]],
    first_number: int,
    *,
    frames_per_car: int = 3,
    columns: int = 2,
) -> Image.Image:
    """Колонки машин, по `frames_per_car` кадров машины в строку; номер и `car_id` впаяны в кадр."""
    строк = (len(машины) + columns - 1) // columns
    лист = Image.new("RGB", (CELL[0] * frames_per_car * columns, CELL[1] * строк), (30, 30, 30))
    рисовать = ImageDraw.Draw(лист)
    for i, (машина, кадры) in enumerate(машины):
        x0 = CELL[0] * frames_per_car * (i // строк)
        y0 = CELL[1] * (i % строк)
        for j, путь in enumerate(кадры[:frames_per_car]):
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
        if аргументы.single:
            # Обзорный лист: средний кадр каждой машины, 48 машин на лист.
            обзор = [(машина, [кадры[len(кадры) // 2]]) for машина, кадры in машины]
            for начало in range(0, len(обзор), CARS_PER_SINGLE_SHEET):
                лист = render_car_sheet(
                    обзор[начало : начало + CARS_PER_SINGLE_SHEET],
                    first_number=начало,
                    frames_per_car=1,
                    columns=SINGLE_SHEET_COLUMNS,
                )
                номер = начало // CARS_PER_SINGLE_SHEET
                лист.save(аргументы.out / f"{класс}_{номер}.jpg", quality=85)
        else:
            for начало in range(0, len(машины), CARS_PER_SHEET):
                лист = render_car_sheet(
                    машины[начало : начало + CARS_PER_SHEET], first_number=начало
                )
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
    каталог192 = команды.add_parser(
        "cat192-sample", help="выборки каталога 192: обучение и проверки"
    )
    каталог192.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "mad.csv")
    каталог192.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    каталог192.add_argument("--new-names", type=Path, default=DEFAULT_NEW_NAMES)
    каталог192.add_argument("--sample-05", type=Path, default=DEFAULT_SAMPLE_05)
    каталог192.add_argument("--negatives", type=Path, default=DEFAULT_MEASURED)
    каталог192.add_argument("--reject-pool", type=Path, default=DEFAULT_REJECT_POOL)
    каталог192.add_argument("--eval-cars", type=int, default=20)
    каталог192.add_argument("--train-cars", type=int, default=100)
    каталог192.add_argument("--seed", type=int, required=True)
    каталог192.add_argument("--out-dir", type=Path, default=Path("catalog/external"))
    каталог192.add_argument("--prefix", default="cat192", help="имена файлов: mad_cars_<prefix>_*")
    каталог192.add_argument(
        "--only-new", action="store_true", help="только новые названия: без проверки старых классов"
    )
    каталог192.set_defaults(run=_cat192_sample)
    негативы = команды.add_parser("negatives-sample", help="чужие машины для замера отказа")
    негативы.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "mad.csv")
    негативы.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    негативы.add_argument("--disputed", type=Path, default=DEFAULT_DISPUTED)
    негативы.add_argument("--cars-per-model", type=int, default=10)
    негативы.add_argument("--total", type=int, default=2000)
    негативы.add_argument("--seed", type=int, required=True)
    негативы.add_argument("--out", type=Path, required=True)
    негативы.set_defaults(run=_negatives_sample)
    пул = команды.add_parser("reject-pool-sample", help="пул отказных прототипов вне замера")
    пул.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "mad.csv")
    пул.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    пул.add_argument("--disputed", type=Path, default=DEFAULT_DISPUTED)
    пул.add_argument("--measured", type=Path, default=DEFAULT_MEASURED)
    пул.add_argument("--cars-per-model", type=int, default=10)
    пул.add_argument("--seed", type=int, required=True)
    пул.add_argument("--out", type=Path, required=True)
    пул.set_defaults(run=_reject_pool_sample)
    скачать = команды.add_parser("fetch", help="докачать выборку, удалить выброшенные машины")
    скачать.add_argument("--sample", type=Path, required=True)
    скачать.add_argument("--root", type=Path, default=DEFAULT_ROOT / "full")
    скачать.add_argument("--excluded", type=Path, default=DEFAULT_EXCLUDED)
    скачать.set_defaults(run=_fetch)
    листы = команды.add_parser("sheets", help="листы по машинам для просмотра")
    листы.add_argument("--root", type=Path, default=DEFAULT_ROOT / "full")
    листы.add_argument("--out", type=Path, required=True)
    листы.add_argument("--classes", nargs="*", default=None)
    листы.add_argument(
        "--single", action="store_true", help="обзорные листы: по одному кадру, 48 машин на лист"
    )
    листы.set_defaults(run=_sheets)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    аргументы = build_parser().parse_args(argv)
    return int(аргументы.run(аргументы))


if __name__ == "__main__":
    raise SystemExit(main())
