"""Сборка отказных прототипов по пулу `reject-pool.parquet` (спека 2026-09-28 §2).

Эмбеддинги считает тот самый классификатор из каталога ассетов, которым будут меряться и
решать прототипы. Запуск — в venv экспорта (LiteRT, sklearn).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pipeline.negatives import neg_kind_of
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.pool_audit import file_sha256
from pipeline.reject_index import REJECT_KINDS, build_reject_centroids, write_reject_index
from pipeline.schema import read_split
from training.build_index import kmeans_factory
from training.tune_thresholds import embed_crops

DEFAULT_SEED = 20260928


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Собрать отказные прототипы")
    парсер.add_argument("--models", type=Path, default=Path("app/android/models"))
    парсер.add_argument("--expected-sha", required=True, help="sha256 классификатора")
    парсер.add_argument("--pool", type=Path, default=splits_dir() / "reject-pool.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--batch-size", type=int, default=32)
    парсер.add_argument("--out", type=Path, default=None)
    аргументы = парсер.parse_args(argv)

    манифест = json.loads((аргументы.models / "models_manifest.json").read_text("utf-8"))
    классификатор = аргументы.models / манифест["files"]["classifier"]["name"]
    sha = file_sha256(классификатор)
    if sha != аргументы.expected_sha:
        raise SystemExit(f"sha256 классификатора {sha}, ожидался {аргументы.expected_sha}")
    пул = read_split(аргументы.pool)
    виды = np.array([neg_kind_of(str(л)) for л in пул["listing_id"]])
    эмбеддинги = embed_crops(
        классификатор,
        [аргументы.crops_root / п for п in пул["crop_path"]],
        манифест,
        аргументы.batch_size,
    )
    центроиды, классы = build_reject_centroids(
        {вид: эмбеддинги[виды == вид] for вид in REJECT_KINDS},
        kmeans=kmeans_factory(аргументы.seed),
    )
    выход = аргументы.out or (data_root() / "reject" / f"reject_{datetime.now():%Y%m%d_%H%M%S}")
    write_reject_index(
        выход,
        центроиды,
        классы,
        classifier_sha256=sha,
        pool_sha256=file_sha256(аргументы.pool),
        frames={вид: int((виды == вид).sum()) for вид in REJECT_KINDS},
        generated_at=datetime.now(UTC).isoformat(),
    )
    print(f"прототипы: {центроиды.shape[0]} классов × {центроиды.shape[1]} → {выход}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
