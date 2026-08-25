"""Оценка индекса эталонов: top-1/top-5 по косинусу к центроидам (приёмка M-1.3).

Арифметика метрик не дублируется — берётся `training.metrics.aggregate`, тот же код,
которым мерилась голова ArcFace на M-1.2. Иначе сравнение «индекс против головы»
сравнивало бы заодно две реализации метрики.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from pipeline.catalog import STANFORD_CATALOG_PATH
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import top5_by_centroids
from pipeline.paths import crops_dir, splits_dir
from training.build_index import collect_embeddings
from training.dataset import build_transforms, make_loader
from training.metrics import aggregate
from training.model import create_model
from training.samples import read_samples


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Оценка индекса эталонов kNN")
    парсер.add_argument("--index", type=Path, required=True, help="каталог с catalog_index.bin")
    парсер.add_argument("--checkpoint", type=Path, required=True)
    парсер.add_argument("--catalog", type=Path, default=STANFORD_CATALOG_PATH)
    парсер.add_argument("--splits", nargs="+", default=["val", "test"])
    парсер.add_argument("--device", default="cuda")
    return парсер.parse_args()


def main() -> None:
    аргументы = _parse_args()
    заголовок, центроиды = read_index(аргументы.index / "catalog_index.bin")
    verify_meta(заголовок, (аргументы.index / "catalog_meta.json").read_bytes())

    чекпоинт = torch.load(аргументы.checkpoint, map_location="cpu", weights_only=True)
    индекс: dict[str, int] = чекпоинт["class_index"]
    класс_по_индексу = {значение: ключ for ключ, значение in индекс.items()}
    конфиг = чекпоинт["config"]
    модель, _ = create_model(
        конфиг["backbone"],
        num_classes=len(индекс),
        embedding_dim=конфиг["embedding_dim"],
        pretrained=False,
    )
    модель.load_state_dict(чекпоинт["model_state"])
    устройство = torch.device(аргументы.device)
    модель.to(устройство)

    итоги = {}
    for имя in аргументы.splits:
        загрузчик = make_loader(
            read_samples(
                splits_dir() / f"{имя}.parquet",
                crops_dir() / "stanford",
                num_classes=len(индекс),
                skip_multi_car=False,
            ),
            build_transforms(
                чекпоинт["img_size"],
                train=False,
                mean=tuple(чекпоинт["mean"]),
                std=tuple(чекпоинт["std"]),
            ),
            batch_size=конфиг["batch_size"],
            shuffle=False,
            num_workers=конфиг["num_workers"],
        )
        эмбеддинги, метки = collect_embeddings(модель, загрузчик, устройство)
        метрики = aggregate(
            top5_by_centroids(эмбеддинги, центроиды),
            [int(метка) for метка in метки],
            idx_to_class_id=класс_по_индексу,
        )
        итоги[имя] = asdict(метрики)
        print(
            f"{имя}: top1_model {метрики.top1_model:.4f}, "
            f"top5_model {метрики.top5_model:.4f}, top1_brand {метрики.top1_brand:.4f}"
        )
    (аргументы.index / "index_metrics.json").write_text(
        json.dumps(итоги, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
