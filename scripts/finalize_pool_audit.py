"""Финализация сплошного аудита пула: реестр, приёмка, `pool-audited`, свидетель.

Спека: `docs/superpowers/specs/2026-09-18-pool-privacy-audit-design.md`, раздел «Предикат
приёмки». `full_pool_audited` истинен, когда реестр покрывает ровно пул (1), без `pending`
(2), без протухших `crop_sha256` (3), и `pool-audited.parquet` равен пулу минус утечки (4).

`pool.parquet` не меняется: его sha держат паспорт baseline, разбиение запечатанных
партий и `promotion.py`. Свидетель несёт `training_authorized: false` — флаг снимает только
письменное решение человека (`technical_states_do_not_authorize_training`).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.pool_audit import (
    RegistryRow,
    audited_mismatch,
    audited_pool,
    check_registry,
    file_sha256,
    index_registry,
    read_registry,
    write_registry,
)
from pipeline.schema import read_split, write_split
from scripts.stage_plate_leak_views import _resolve_crop

DEFAULT_WITNESS = Path("bench") / "pool-privacy-audit.json"
DEFAULT_REGISTRY_COPY = Path("bench") / "pool-audit-registry.csv"


def current_crop_hashes(pool: pd.DataFrame, crops_root: Path) -> dict[str, str | None]:
    """sha256 байтов кропа на диске сейчас; `None` — кропа нет."""
    hashes: dict[str, str | None] = {}
    for photo_id, crop_path in zip(pool["photo_id"], pool["crop_path"], strict=True):
        crop = _resolve_crop(crops_root, str(crop_path))
        hashes[str(photo_id)] = file_sha256(crop) if crop.is_file() else None
    return hashes


def witness_document(
    pool: pd.DataFrame,
    rows: Sequence[RegistryRow],
    *,
    pool_sha256: str,
    registry_sha256: str,
    audited_sha256: str,
    audited_frames: int,
) -> dict[str, Any]:
    """JSON-свидетель приёмки. Распределение по классам — к риску дефицитных классов."""
    excluded = [row for row in rows if row.excluded]
    class_of = dict(zip(pool["photo_id"], pool["model_id"], strict=True))
    pool_by_class = Counter(str(value) for value in pool["model_id"])
    excluded_by_class = Counter(str(class_of[row.photo_id]) for row in excluded)
    return {
        "schema_version": 1,
        "full_pool_audited": True,
        "training_authorized": False,
        "training_authorization_note": "флаг снимает только письменное решение человека",
        "pool": {"sha256": pool_sha256, "frames": len(pool)},
        "registry_sha256": registry_sha256,
        "pool_audited": {"sha256": audited_sha256, "frames": audited_frames},
        "excluded": {
            "total": len(excluded),
            "plate_leak": sum(row.plate_verdict == "leak" for row in excluded),
            "plate_leak_hist": sum(row.plate_verdict == "leak_hist" for row in excluded),
            "face_leak": sum(row.face_verdict == "leak" for row in excluded),
            "by_class": {
                name: {"excluded": count, "pool": pool_by_class[name]}
                for name, count in sorted(excluded_by_class.items())
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Финализация аудита приватности пула")
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--split", type=Path, default=None, help="умолчание: pool.parquet")
    parser.add_argument("--crops-root", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--audited-out", type=Path, default=None)
    parser.add_argument("--witness", type=Path, default=DEFAULT_WITNESS)
    parser.add_argument("--registry-copy", type=Path, default=DEFAULT_REGISTRY_COPY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    split: Path = arguments.split or splits_dir() / "pool.parquet"
    crops_root: Path = arguments.crops_root or harvest_dir() / "crops"
    registry: Path = arguments.registry or data_root() / "pool-audit" / "main" / "registry.csv"
    audited_out: Path = arguments.audited_out or splits_dir() / "pool-audited.parquet"
    if audited_out.resolve() == split.resolve():
        raise ValueError("pool-audited не может перезаписать исходный пул")

    rows = [row for path in arguments.exports for row in read_registry(path)]
    index_registry(rows)
    pool = read_split(split)
    check = check_registry(
        rows,
        pool_ids=pool["photo_id"],
        current_crop_sha256=current_crop_hashes(pool, crops_root),
    )
    registry.parent.mkdir(parents=True, exist_ok=True)
    write_registry(registry, sorted(rows, key=lambda row: row.photo_id))
    print(
        f"реестр: {len(rows)} строк; нет в реестре: {len(check.missing_ids)}, "
        f"лишних: {len(check.extra_ids)}, pending: {len(check.pending_ids)}, "
        f"протухших: {len(check.stale_ids)}"
    )
    if not check.passed:
        print("приёмка не пройдена")
        return 1

    write_split(audited_pool(pool, rows), audited_out)
    audited_ids = read_split(audited_out)["photo_id"]
    mismatch = audited_mismatch(rows, pool_ids=pool["photo_id"], audited_ids=audited_ids)
    if mismatch:
        print(f"pool-audited расходится с «пул минус утечки»: {len(mismatch)} кадров")
        return 1

    witness = witness_document(
        pool,
        rows,
        pool_sha256=file_sha256(split),
        registry_sha256=file_sha256(registry),
        audited_sha256=file_sha256(audited_out),
        audited_frames=len(audited_ids),
    )
    arguments.witness.write_text(
        json.dumps(witness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copyfile(registry, arguments.registry_copy)
    print(f"full_pool_audited: исключено {witness['excluded']['total']} из {len(pool)}")
    print(f"pool-audited: {audited_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
