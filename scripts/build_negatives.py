"""Сборка `splits/negatives.parquet` из данных на диске (спека calibration-ru §4).

Ничего не скачивает. Три источника: папки autoevolution вне таблицы соответствия и классы
Stanford — машины вне каталога; отбраковки харвеста `no_vehicle`/`not_a_car_photo` —
ложные срабатывания детектора, грузовики и автобусы.

`--list-only` пишет только `negatives-groups.csv` без детектора: список имён оставленных
групп просматривается глазами до нарезки (гейт по именам, Task 8 плана).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.ingest.external import IMAGE_SUFFIXES
from pipeline.ingest.stanford import read_annotations
from pipeline.negatives import (
    NegativeCandidate,
    OffCatalogGroup,
    build_negative_frame,
    read_exclusions,
    reject_candidates,
    sample_candidates,
    select_off_catalog,
)
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import write_split

_КОРЕНЬ = Path(__file__).resolve().parent.parent
DEFAULT_MAPPING = _КОРЕНЬ / "catalog" / "external" / "autoevolution.csv"
DEFAULT_EXCLUSIONS = _КОРЕНЬ / "catalog" / "external" / "negatives_exclude.csv"
DEFAULT_SEED = 20260923


def _картинки(папка: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(ф for ф in папка.iterdir() if ф.is_file() and ф.suffix.lower() in IMAGE_SUFFIXES)
    )


def autoevolution_groups(root: Path, mapping_csv: Path) -> list[OffCatalogGroup]:
    """Папки набора, которых нет в `dir_path` таблицы соответствия."""
    строки = csv.DictReader(mapping_csv.read_text(encoding="utf-8").splitlines())
    присоединённые = {строка["dir_path"] for строка in строки}
    группы: list[OffCatalogGroup] = []
    for папка in sorted(п for п in root.iterdir() if п.is_dir()):
        if папка.name in присоединённые:
            continue
        файлы = _картинки(папка)
        if файлы:
            группы.append(OffCatalogGroup(f"autoevolution/{папка.name}", папка.name, файлы))
    return группы


def stanford_groups(root: Path) -> list[OffCatalogGroup]:
    """Классы Stanford: имя класса — то, по чему идёт фильтр игл."""
    по_классам: dict[str, list[Path]] = defaultdict(list)
    for фото in read_annotations(root):
        по_классам[фото.class_name].append(фото.path)
    return [
        OffCatalogGroup(f"stanford/{имя}", имя, tuple(sorted(пути)))
        for имя, пути in sorted(по_классам.items())
    ]


def _записать_группы(
    путь: Path,
    оставлены: Sequence[OffCatalogGroup],
    отброшены: Sequence[tuple[OffCatalogGroup, str]],
) -> None:
    with путь.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.writer(файл)
        писатель.writerow(["origin", "name", "photos", "decision"])
        for группа in оставлены:
            писатель.writerow([группа.origin, группа.name, len(группа.paths), "kept"])
        for группа, причина in отброшены:
            писатель.writerow([группа.origin, группа.name, len(группа.paths), причина])


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Собрать негативы для подбора порогов")
    парсер.add_argument(
        "--autoevolution", type=Path, default=data_root() / "external" / "autoevolution"
    )
    парсер.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    парсер.add_argument("--stanford", type=Path, default=data_root() / "raw" / "stanford_cars")
    парсер.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    парсер.add_argument("--per-group", type=int, default=3)
    парсер.add_argument("--off-catalog-cap", type=int, default=2000)
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--weights", default="yolo11n.pt")
    парсер.add_argument("--list-only", action="store_true")
    аргументы = парсер.parse_args(argv)

    таблица = load_alias_table(DEFAULT_ALIASES_PATH)
    исключения = read_exclusions(аргументы.exclusions)
    группы = autoevolution_groups(аргументы.autoevolution, аргументы.mapping) + stanford_groups(
        аргументы.stanford
    )
    оставлены, отброшены = select_off_catalog(группы, таблица, исключения)
    _записать_группы(splits_dir() / "negatives-groups.csv", оставлены, отброшены)
    print(f"групп: оставлено {len(оставлены)}, отброшено {len(отброшены)}")
    if аргументы.list_only:
        return 0

    чужие = [
        NegativeCandidate(путь, True, группа.origin, группа.name)
        for группа, путь in sample_candidates(оставлены, аргументы.per_group, аргументы.seed)
    ]
    харвест = harvest_dir()
    отбраковки, без_файла = reject_candidates(
        харвест / "reject_log.csv", харвест / "reject_index.csv", харвест
    )

    # Импорт здесь: `ultralytics` живёт в группе `train`, которой в CI нет. Порог 0.35 —
    # штатный у адаптера и §13.5: ложные срабатывания ищутся ниже порога выбора бокса.
    from pipeline.detectors.yolo import YoloDetector

    итог = build_negative_frame(
        [*отбраковки, *чужие],
        YoloDetector(weights=аргументы.weights),
        харвест / "crops",
        off_catalog_cap=аргументы.off_catalog_cap,
        excluded_photos=исключения.photos,
    )
    write_split(итог.frame, splits_dir() / "negatives.parquet")
    счётчики = {
        "kept": итог.kept,
        "rejected": итог.rejected,
        "reject_log_without_file": без_файла,
        "groups_kept": len(оставлены),
        "groups_dropped": len(отброшены),
        "seed": аргументы.seed,
    }
    (splits_dir() / "negatives.json").write_text(
        json.dumps(счётчики, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(счётчики, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
