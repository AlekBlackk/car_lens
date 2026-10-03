"""Сплиты каталога 192 (`bench/catalog-192-preregistration.md`): `base` и `compose`.

`base` — до приёма новых кадров: `holdout_ru` и `pool-clean` с пересчитанной нумерацией,
негативы без машин, ставших своими. Приёму (`scripts.ingest_external`) нужны именно они, поэтому
он идёт после `base` и с `CARLENS_SPLITS_DIR` на выходной каталог.

`compose` — после приёма: внешний пул 0.5 с пересобранными классами, `pool-plus` и две проверки
(`eval-new`, `eval-old`) по одному кадру на машину, без утечки в обучение.

`groups` и `studio` — модель 0.6.1 (`bench/catalog-192-studio-preregistration.md`): разбиение
групп autoevolution новых классов на обучающие и отложенные и сборка сплитов с этими кадрами.

Старые сплиты не правятся: всё пишется в `--out`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Collection, Sequence
from pathlib import Path

import pandas as pd

from pipeline.cat192 import (
    build_external_pool,
    cap_per_class,
    leaking_cars,
    pick_one_frame_per_car,
    remap_class_idx,
    split_groups,
    without_close_frames,
    without_close_twins,
)
from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, AliasTable, load_alias_table
from pipeline.ingest.external import DEFAULT_MAX_PER_CLASS
from pipeline.ingest.mad_cars import off_catalog_reason, read_mad_disputed, read_mad_mapping
from pipeline.negatives import catalog_needle_hits
from pipeline.paths import data_root
from pipeline.schema import read_split, write_split

DEFAULT_SRC = data_root() / "splits"
DEFAULT_POOL_05 = DEFAULT_SRC / "mad_full"
DEFAULT_EXCLUDE = Path("catalog/external/negatives_exclude_cat192.csv")
DEFAULT_MAPPING = Path("catalog/external/mad_cars.csv")
DEFAULT_DISPUTED = Path("catalog/external/mad_cars_disputed.csv")
NEGATIVE_SETS = ("negatives", "negatives-street")
POOL_SEED = 1337
EVAL_SEED = 20260930
MAD_OWN_CAP = 150
DEFAULT_AE_MAPPING = Path("catalog/external/autoevolution.csv")
DEFAULT_NEW_CLASSES = Path("catalog/external/cat192_new_classes.txt")
STUDIO_SALT = "20261002"
STUDIO_EVAL_SEED = 20261002
STUDIO_EVAL_CAP = 40
HELD_SEED = 20261003
HELD_TWIN_DISTANCE = 10
UNCHANGED_SPLITS = ("holdout_ru", "negatives", "negatives-street", "eval-new", "eval-old")


def own_car_reason(
    raw_name: str,
    *,
    mad_folder: bool,
    exact: Collection[tuple[str, str]],
    disputed: frozenset[tuple[str, str]],
    needles: AliasTable,
) -> str | None:
    """Причина, по которой негатив после расширения каталога — своя машина, или None.

    Кадры уличных негативов MAD-Cars названы папкой `<brand>__<model>`, остальные — именем
    группы autoevolution/Stanford; обе ветки — тот же фильтр, которым негативы отбирались.
    """
    if mad_folder:
        марка, _, модель = raw_name.partition("__")
        return off_catalog_reason((марка, модель), exact, disputed, needles)
    совпали = catalog_needle_hits(raw_name, needles)
    return "needle:" + "|".join(совпали) if совпали else None


def _raw_names(негативы: pd.DataFrame) -> list[str]:
    return [str(имя) if isinstance(имя, str) else "" for имя in негативы["raw_model_string"]]


def _base(аргументы: argparse.Namespace) -> int:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    иглы = load_alias_table(DEFAULT_ALIASES_PATH)
    точные = read_mad_mapping(аргументы.mapping, каталог).keys()
    спорные = read_mad_disputed(аргументы.disputed, каталог)
    аргументы.out.mkdir(parents=True, exist_ok=True)

    for имя in ("holdout_ru", "pool-clean"):
        перенумерованный = remap_class_idx(read_split(аргументы.src / f"{имя}.parquet"), каталог)
        write_split(перенумерованный, аргументы.out / f"{имя}.parquet")

    исключения: list[tuple[str, str]] = []
    убрано: dict[str, int] = {}
    for набор in NEGATIVE_SETS:
        негативы = read_split(аргументы.src / f"{набор}.parquet")
        причины = [
            own_car_reason(
                сырое,
                mad_folder=набор == "negatives-street" and "__" in сырое and ".jpg" not in сырое,
                exact=точные,
                disputed=спорные,
                needles=иглы,
            )
            for сырое in _raw_names(негативы)
        ]
        убрать = pd.Series([причина is not None for причина in причины], index=негативы.index)
        write_split(негативы[~убрать], аргументы.out / f"{набор}.parquet")
        убрано[набор] = int(убрать.sum())
        исключения += [
            (str(photo_id), f"{набор}: {причина}")
            for photo_id, причина in zip(негативы["photo_id"], причины, strict=True)
            if причина is not None
        ]
    with аргументы.exclude_out.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.writer(файл, lineterminator="\n")
        писатель.writerow(["kind", "value", "reason"])
        писатель.writerows(("photo", photo_id, причина) for photo_id, причина in исключения)
    print(f"негативов убрано: {убрано}; список → {аргументы.exclude_out}")
    return 0


def _compose(аргументы: argparse.Namespace) -> int:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    чистый = read_split(аргументы.out / "pool-clean.parquet")
    итог = build_external_pool(
        read_split(аргументы.pool_05 / "pool-ext.parquet"),
        read_split(аргументы.train_ext),
        каталог,
        max_per_class=DEFAULT_MAX_PER_CLASS,
        own_caps={"mad_cars": MAD_OWN_CAP},
        seed=POOL_SEED,
    )
    write_split(итог.external, аргументы.out / "pool-ext.parquet")
    пул = pd.concat([чистый, итог.external], ignore_index=True)
    write_split(пул, аргументы.out / "pool-plus.parquet")
    хеши = [int(хеш) for хеш in пул["phash"]]

    сводка: dict[str, object] = {
        "pool_clean": len(чистый),
        "pool_external": len(итог.external),
        "pool_plus": len(пул),
        "merge_rejected": итог.rejected,
        "changed_classes": sorted(итог.changed_classes),
    }
    for имя, путь in (("eval-new", аргументы.eval_new_ext), ("eval-old", аргументы.eval_old_ext)):
        кадры = remap_class_idx(read_split(путь), каталог)
        утекли = leaking_cars(кадры, хеши)
        выбор = pick_one_frame_per_car(кадры[~кадры["listing_id"].isin(утекли)], seed=EVAL_SEED)
        write_split(выбор, аргументы.out / f"{имя}.parquet")
        сводка[имя] = {
            "ingested_frames": len(кадры),
            "cars_ingested": int(кадры["listing_id"].nunique()),
            "cars_leaked": len(утекли),
            "cars": len(выбор),
        }
    сводка_путь = аргументы.out / "cat192.json"
    сводка_путь.write_text(json.dumps(сводка, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(сводка, ensure_ascii=False))
    return 0


def _groups(аргументы: argparse.Namespace) -> int:
    новые = {
        строка.strip()
        for строка in аргументы.new_classes.read_text(encoding="utf-8").splitlines()
        if строка.strip() and not строка.startswith("#")
    }
    with аргументы.mapping.open(encoding="utf-8", newline="") as файл:
        читатель = csv.DictReader(файл)
        колонки = list(читатель.fieldnames or ())
        строки = [строка for строка in читатель if строка["model_id"] in новые]
    отложены = split_groups(
        {строка["dir_path"]: строка["model_id"] for строка in строки}, salt=STUDIO_SALT
    )
    for путь, откладываем in ((аргументы.train_map, False), (аргументы.eval_map, True)):
        with путь.open("w", encoding="utf-8", newline="") as файл:
            писатель = csv.DictWriter(файл, fieldnames=колонки, lineterminator="\n")
            писатель.writeheader()
            писатель.writerows(
                строка for строка in строки if (строка["dir_path"] in отложены) == откладываем
            )
    классы_студии = {строка["model_id"] for строка in строки if строка["dir_path"] in отложены}
    print(
        f"групп новых классов {len(строки)}: в обучение {len(строки) - len(отложены)}, "
        f"отложено {len(отложены)} ({len(классы_студии)} классов имеют студийную проверку)"
    )
    return 0


def _studio(аргументы: argparse.Namespace) -> int:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    база = аргументы.base
    # Без `--eval-ext` студийная проверка базы переходит как есть и тоже опорная: кадры обучения
    # не должны быть близки ни к одной проверке.
    своя_студия = аргументы.eval_ext is not None
    опорные_сплиты = ("eval-new", "eval-old", "holdout_ru") + (
        () if своя_студия else ("eval-studio",)
    )
    опорные = [
        int(хеш) for имя in опорные_сплиты for хеш in read_split(база / f"{имя}.parquet")["phash"]
    ] + [int(хеш) for путь in аргументы.also_avoid for хеш in read_split(путь)["phash"]]
    принятые = remap_class_idx(read_split(аргументы.train_ext), каталог)
    обучение = without_close_frames(принятые, опорные)
    итог = build_external_pool(
        read_split(база / "pool-ext.parquet"),
        обучение,
        каталог,
        max_per_class=DEFAULT_MAX_PER_CLASS,
        own_caps={"mad_cars": MAD_OWN_CAP},
        seed=POOL_SEED,
        source=аргументы.source,
    )
    аргументы.out.mkdir(parents=True, exist_ok=True)
    write_split(итог.external, аргументы.out / "pool-ext.parquet")
    пул = pd.concat([read_split(база / "pool-clean.parquet"), итог.external], ignore_index=True)
    write_split(пул, аргументы.out / "pool-plus.parquet")
    write_split(read_split(база / "pool-clean.parquet"), аргументы.out / "pool-clean.parquet")

    сводка: dict[str, object] = {
        "train_ingested": len(принятые),
        "train_close_to_eval": len(принятые) - len(обучение),
        "pool_external": len(итог.external),
        "pool_plus": len(пул),
        "merge_rejected": итог.rejected,
        "changed_classes": len(итог.changed_classes),
    }
    if своя_студия:
        приняты_студия = remap_class_idx(read_split(аргументы.eval_ext), каталог)
        без_утечки = without_close_frames(приняты_студия, [int(хеш) for хеш in пул["phash"]])
        студия = cap_per_class(без_утечки, STUDIO_EVAL_CAP, seed=STUDIO_EVAL_SEED)
        write_split(студия, аргументы.out / "eval-studio.parquet")
        сводка |= {
            "studio_ingested": len(приняты_студия),
            "studio_leaked": len(приняты_студия) - len(без_утечки),
            "studio_frames": len(студия),
            "studio_classes": int(студия["class_idx"].nunique()),
        }
    else:
        shutil.copy2(база / "eval-studio.parquet", аргументы.out / "eval-studio.parquet")
    for имя in UNCHANGED_SPLITS:
        shutil.copy2(база / f"{имя}.parquet", аргументы.out / f"{имя}.parquet")

    (аргументы.out / "studio.json").write_text(
        json.dumps(сводка, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(сводка, ensure_ascii=False))
    return 0


def _heldout(аргументы: argparse.Namespace) -> int:
    """`T_held`: кадры проверки без совпадающих со смоуком и без близнецов одной машины."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    принятые = remap_class_idx(read_split(аргументы.eval_ext), каталог)
    смоук = [int(хеш) for хеш in read_split(аргументы.fresh9)["phash"]]
    без_смоука = without_close_frames(принятые, смоук)
    итог = without_close_twins(без_смоука, max_distance=HELD_TWIN_DISTANCE, seed=HELD_SEED)
    write_split(итог, аргументы.out)
    print(
        f"кадров проверки {len(принятые)}: общих со смоуком {len(принятые) - len(без_смоука)}, "
        f"близнецов {len(без_смоука) - len(итог)}, в `T_held` {len(итог)}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Сплиты каталога 192")
    команды = парсер.add_subparsers(dest="command", required=True)
    база = команды.add_parser("base", help="перенумеровать сплиты, почистить негативы")
    база.add_argument("--src", type=Path, default=DEFAULT_SRC)
    база.add_argument("--out", type=Path, required=True)
    база.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    база.add_argument("--disputed", type=Path, default=DEFAULT_DISPUTED)
    база.add_argument("--exclude-out", type=Path, default=DEFAULT_EXCLUDE)
    база.set_defaults(run=_base)
    сборка = команды.add_parser("compose", help="пул и проверки после приёма новых кадров")
    сборка.add_argument("--out", type=Path, required=True)
    сборка.add_argument("--pool-05", type=Path, default=DEFAULT_POOL_05)
    сборка.add_argument("--train-ext", type=Path, required=True)
    сборка.add_argument("--eval-new-ext", type=Path, required=True)
    сборка.add_argument("--eval-old-ext", type=Path, required=True)
    сборка.set_defaults(run=_compose)
    группы = команды.add_parser(
        "groups", help="разбить группы autoevolution на обучение и проверку"
    )
    группы.add_argument("--mapping", type=Path, default=DEFAULT_AE_MAPPING)
    группы.add_argument("--new-classes", type=Path, default=DEFAULT_NEW_CLASSES)
    группы.add_argument("--train-map", type=Path, required=True)
    группы.add_argument("--eval-map", type=Path, required=True)
    группы.set_defaults(run=_groups)
    студия = команды.add_parser("studio", help="сплиты 0.6.1/0.6.2: пул с новым источником")
    студия.add_argument(
        "--base", type=Path, required=True, help="сплиты-основа (`compose`/`studio`)"
    )
    студия.add_argument("--train-ext", type=Path, required=True)
    студия.add_argument("--source", default="autoevolution", help="источник кадров обучения")
    студия.add_argument(
        "--eval-ext",
        type=Path,
        default=None,
        help="кадры студийной проверки; без него `eval-studio` базы остаётся как есть",
    )
    студия.add_argument(
        "--also-avoid",
        type=Path,
        nargs="*",
        default=[],
        help="ещё проверки, к чьим кадрам обучение не должно быть близким (`T_held`, `T_fresh9`)",
    )
    студия.add_argument("--out", type=Path, required=True)
    студия.set_defaults(run=_studio)
    свежая = команды.add_parser("heldout", help="сплит 0.6.3: проверка `T_held` после гигиены")
    свежая.add_argument("--eval-ext", type=Path, required=True, help="принятые кадры проверки")
    свежая.add_argument(
        "--fresh9", type=Path, required=True, help="`T_fresh9`: его кадры не берутся"
    )
    свежая.add_argument("--out", type=Path, required=True, help="куда писать `T_held` (parquet)")
    свежая.set_defaults(run=_heldout)
    аргументы = парсер.parse_args(argv)
    return int(аргументы.run(аргументы))


if __name__ == "__main__":
    raise SystemExit(main())
