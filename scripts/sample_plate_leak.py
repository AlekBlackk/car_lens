"""Простая случайная выборка сплита для замера утечки номера (§4 `bench/plate-leak-bound.md`).

Отбирает `--size` кадров сплита без возвращения, сид фиксирован. Сортировка по `photo_id`
перед `sample` — отбор не зависит от порядка строк parquet, который не специфицирован и
может поменяться при пересборке сплита.

Черновик выборки (без вердикта) идёт человеку на просмотр — полным разрешением через
`scripts/audit_gallery.py --only-files <файл с sha256.jpg> --thumb-px <шире стороны кадра>`
(галерея уже открывает миниатюру по клику на полный размер; `--thumb-px` больше стороны
кадра означает «не уменьшать» — см. `render_gallery`). Человек пишет `номер,вердикт[,заметка]`
в свой CSV; `scripts/measure_plate_leak.py` сводит его с этим черновиком и считает интервал.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH
from pipeline.paths import splits_dir
from pipeline.schema import read_split

DEFAULT_SIZE = 300
DEFAULT_SEED = 1337
DEFAULT_OUT = Path("bench") / "plate-leak-sample-draft.csv"

MANIFEST_COLUMNS = ("sha256", "class_id", "source_item_id", "plates", "faces", "audit")

# `crop_path` держит просмотр (staged views ищут файл на диске) и не входит в
# `pipeline.plate_leak.SAMPLE_COLUMNS`: закоммиченный итог не несёт путей до чужой машины.
DRAFT_COLUMNS = ("номер", "sha256", "pageid", "класс", "plates", "faces", "штамп", "crop_path")


def sha256_of_file(path: Path) -> str:
    """sha256 файла блоками по 1 МБ — для паспорта воспроизводимости."""
    хеш = hashlib.sha256()
    with path.open("rb") as поток:
        for блок in iter(lambda: поток.read(1 << 20), b""):
            хеш.update(блок)
    return хеш.hexdigest()


def join_manifest(split: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Свести сплит с манифестом по `photo_id == sha256`, проверить полноту пары.

    Строка сплита без пары в манифесте — рассинхрон данных, а не редкий случай: сплит
    собирается из манифеста, и разойтись они могут только если один из двух файлов устарел.
    """
    объединено = split.merge(
        manifest[list(MANIFEST_COLUMNS)], left_on="photo_id", right_on="sha256", how="left"
    )
    без_пары = объединено[объединено["sha256"].isna()]
    if not без_пары.empty:
        примеры = без_пары["photo_id"].tolist()[:5]
        raise ValueError(f"{len(без_пары)} строк сплита без пары в манифесте, например {примеры}")
    return объединено


def draw_sample(joined: pd.DataFrame, *, size: int, seed: int) -> pd.DataFrame:
    """Черновик выборки: номер, sha256, pageid, класс, plates, faces, штамп."""
    if size > len(joined):
        raise ValueError(f"размер выборки {size} больше населения {len(joined)}")
    выборка = (
        joined.sort_values("photo_id", kind="mergesort")
        .sample(n=size, random_state=seed)
        .reset_index(drop=True)
    )
    выборка.insert(0, "номер", range(1, len(выборка) + 1))
    return выборка.rename(
        columns={"source_item_id": "pageid", "class_id": "класс", "audit": "штамп"}
    )[list(DRAFT_COLUMNS)]


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(
        description="Стратифицированная выборка сплита для замера утечки номера"
    )
    парсер.add_argument("--split", default="pool", help="сплит-популяция (по умолчанию пул)")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument("--size", type=int, default=DEFAULT_SIZE)
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать черновик выборки и паспорт прогона рядом с ним."""
    аргументы = build_parser().parse_args(argv)

    путь_сплита = splits_dir() / f"{аргументы.split}.parquet"
    if not путь_сплита.is_file():
        raise FileNotFoundError(f"сплит не найден: {путь_сплита}")
    if not аргументы.manifest.is_file():
        raise FileNotFoundError(f"манифест не найден: {аргументы.manifest}")

    сплит = read_split(путь_сплита)
    манифест = pd.read_csv(аргументы.manifest, dtype={"source_item_id": str})
    объединено = join_manifest(сплит, манифест)
    выборка = draw_sample(объединено, size=аргументы.size, seed=аргументы.seed)

    аргументы.out.parent.mkdir(parents=True, exist_ok=True)
    выборка.to_csv(аргументы.out, index=False, encoding="utf-8")

    паспорт = {
        "сид": аргументы.seed,
        "метод": "sort_values('photo_id') -> DataFrame.sample(n, random_state=сид)",
        "сплит": {
            "имя": аргументы.split,
            "файл": str(путь_сплита),
            "строк": len(сплит),
            "sha256": sha256_of_file(путь_сплита),
        },
        "манифест": {
            "файл": str(аргументы.manifest),
            "строк": len(манифест),
            "sha256": sha256_of_file(аргументы.manifest),
        },
        "выборка": {
            "размер": len(выборка),
            "страта_plates_0": int((выборка["plates"] == 0).sum()),
            "страта_plates_ge1": int((выборка["plates"] > 0).sum()),
        },
    }
    паспорт_путь = аргументы.out.with_name(f"{аргументы.out.stem}-meta.json")
    паспорт_путь.write_text(json.dumps(паспорт, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"черновик выборки: {аргументы.out} ({len(выборка)} строк)")
    print(f"паспорт: {паспорт_путь}")
    print(
        "дальше: просмотр в полном разрешении (audit_gallery.py --only-files --thumb-px), "
        "вердикт по номеру -> scripts.measure_plate_leak"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
