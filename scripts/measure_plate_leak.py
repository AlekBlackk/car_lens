"""Замер частоты утечки номера по выборке с вердиктами человека (правило 1 Р-11, §7).

Читает выборку с вердиктами — `bench/plate-leak-sample.csv` (номер, sha256, pageid, класс,
plates, faces, штамп, вердикт, заметка), которую даёт `scripts/sample_plate_leak.py` плюс
прополка человеком через `scripts/audit_gallery.py --only-files --thumb-px` в полном
разрешении, — считает доверительные интервалы `pipeline.plate_leak.compute_bound` и пишет:

- отчёт замера рядом с выборкой (`--out-report`) — числа для `bench/plate-leak-bound.md`,
  машиночитаемо;
- строгие утечки в формате `file,verdict` манифеста (`--out-verdicts`), готовые к
  `python -m scripts.harvest --audit` — прополка, а не текст отчёта, применяет решение.

Размер сплита и манифеста читаются из файлов при запуске, а не зашиты литералом: зашитое
число — та самая мина, из-за которой прежний замер молча путал `pool` до и после прополки
(`bench/plate-leak-bound.md`, §11).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from pipeline.paths import splits_dir
from pipeline.plate_leak import (
    BoundReport,
    PopulationSizes,
    compute_bound,
    merge_sample_verdicts,
    project_to_sample_schema,
    strict_leak_verdicts,
)
from pipeline.schema import read_split

DEFAULT_SAMPLE = Path("bench") / "plate-leak-sample.csv"
DEFAULT_REPORT_OUT = Path("bench") / "plate-leak-bound-report.json"
DEFAULT_VERDICTS_OUT = Path("verdicts.plate-leak.csv")


def report_to_dict(report: BoundReport) -> dict[str, Any]:
    """Свести `BoundReport` в JSON-совместимый словарь (проценты округлены до сотых)."""
    словарь: dict[str, Any] = {
        "выборка": report.sample_size,
        "распределение": report.verdict_counts,
    }
    for границы in report.bounds:
        словарь[границы.definition] = {
            "утечек": границы.leaks,
            "доля_%": round(границы.share * 100, 2),
            "клоппер_пирсон_95_%": [
                round(границы.ci[0] * 100, 2),
                round(границы.ci[1] * 100, 2),
            ],
            "кадров_в_пуле_оценка": границы.pool_estimate,
            "кадров_в_пуле_интервал": list(границы.pool_interval),
            "кадров_в_принятом_интервал": list(границы.accepted_interval),
        }
    словарь["страты"] = {
        страта.stratum: {
            "просмотрено": страта.seen,
            "утечек": страта.leaks,
            "доля_%": round(страта.share * 100, 2),
            "клоппер_пирсон_95_%": [round(страта.ci[0] * 100, 2), round(страта.ci[1] * 100, 2)],
        }
        for страта in report.strata
    }
    словарь["кадры_утечек"] = [asdict(кадр) for кадр in report.leak_frames]
    return словарь


def write_manifest_verdicts(merged: pd.DataFrame, out_path: Path) -> int:
    """Записать строгие утечки в формате `file,verdict` манифеста. Отдаёт число строк."""
    строгие = strict_leak_verdicts(merged).sort_values("sha256")
    with out_path.open("w", newline="", encoding="utf-8") as поток:
        писатель = csv.writer(поток)
        писатель.writerow(["file", "verdict"])
        for sha256 in строгие["sha256"]:
            писатель.writerow([f"{sha256}.jpg", "drop:plate"])
    return len(строгие)


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(
        description="Замер частоты утечки номера по выборке с вердиктами (правило 1 Р-11)"
    )
    парсер.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="CSV выборки; без --verdicts — уже с вердиктами, с --verdicts — черновик",
    )
    парсер.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="CSV суждений человека (номер, вердикт[, заметка]) для свода с черновиком --sample",
    )
    парсер.add_argument(
        "--out-sample",
        type=Path,
        default=None,
        help="куда записать смерженный итог при --verdicts (по умолчанию — поверх --sample)",
    )
    парсер.add_argument(
        "--split",
        default="pool",
        help="сплит-популяция для переноса доли на кадры (по умолчанию — обучающий пул)",
    )
    парсер.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="манифест харвеста — источник размера принятой популяции",
    )
    парсер.add_argument("--out-report", type=Path, default=DEFAULT_REPORT_OUT)
    парсер.add_argument("--out-verdicts", type=Path, default=DEFAULT_VERDICTS_OUT)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Посчитать интервалы, записать отчёт и вердикты манифеста. Не применяет их."""
    аргументы = build_parser().parse_args(argv)

    if not аргументы.sample.is_file():
        raise FileNotFoundError(f"выборка не найдена: {аргументы.sample}")
    if not аргументы.manifest.is_file():
        raise FileNotFoundError(f"манифест не найден: {аргументы.manifest}")
    if аргументы.verdicts is not None and not аргументы.verdicts.is_file():
        raise FileNotFoundError(f"вердикты не найдены: {аргументы.verdicts}")

    выборка = pd.read_csv(аргументы.sample)
    if аргументы.verdicts is not None:
        # --sample был черновиком (нёс crop_path, не нёс вердикт): свод с суждениями
        # человека и урезка до девяти колонок закоммиченного артефакта.
        вердикты = pd.read_csv(аргументы.verdicts)
        выборка = project_to_sample_schema(merge_sample_verdicts(выборка, вердикты))
        путь_итога = аргументы.out_sample or аргументы.sample
        путь_итога.parent.mkdir(parents=True, exist_ok=True)
        выборка.to_csv(путь_итога, index=False, encoding="utf-8")
        print(f"смерженная выборка: {путь_итога}")

    сплит = read_split(splits_dir() / f"{аргументы.split}.parquet")
    манифест = load_manifest(аргументы.manifest)
    население = PopulationSizes(pool_frames=len(сплит), accepted_frames=len(манифест))

    отчёт = compute_bound(выборка, население)

    аргументы.out_report.parent.mkdir(parents=True, exist_ok=True)
    аргументы.out_report.write_text(
        json.dumps(report_to_dict(отчёт), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    строк_вердиктов = write_manifest_verdicts(выборка, аргументы.out_verdicts)

    строго = next(б for б in отчёт.bounds if б.definition == "строго")
    print(f"выборка {отчёт.sample_size} из популяции {население.pool_frames} ({аргументы.split})")
    print(
        f"строгих утечек: {строго.leaks} ({строго.share * 100:.2f}%), "
        f"95% ДИ [{строго.ci[0] * 100:.2f}, {строго.ci[1] * 100:.2f}]%"
    )
    print(
        f"оценка на популяции: {строго.pool_estimate} "
        f"(интервал {строго.pool_interval[0]}…{строго.pool_interval[1]})"
    )
    print(f"отчёт: {аргументы.out_report}")
    print(f"вердиктов манифеста записано: {строк_вердиктов} -> {аргументы.out_verdicts}")
    print("вердикты не применены: python -m scripts.harvest --audit <файл> применяет решение")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
