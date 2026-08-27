"""Фаза 1, часть 1: zero-shot на тех же сплитах, что мерил обученный чекпоинт.

Сравнение прямое: M-1.2 даёт 94.3% top-1 val и 95.3% top-1 test. Ни то, ни другое не
уличная метрика — но домен у обоих замеров один и тот же, и это единственное честное
численное сравнение, которое у спайка есть.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pipeline.catalog import STANFORD_CATALOG_PATH, Catalog, class_index, load_catalog
from pipeline.paths import crops_dir, splits_dir
from spike.clip_zeroshot.encoders import load_encoder
from spike.clip_zeroshot.evaluate import topk_hits
from spike.clip_zeroshot.manifest import DEFAULT_MODELS_MANIFEST, load_models_manifest
from spike.clip_zeroshot.text_index import build_text_index

КОРЕНЬ_ИНДЕКСОВ = Path(__file__).resolve().parent / "index"


def stanford_names(catalog: Catalog) -> dict[str, str]:
    """`class_id` → английское имя. У Stanford это ровно `display`."""
    return {к.class_id: к.display for к in catalog.classes}


def main() -> None:
    from training.samples import read_samples

    парсер = argparse.ArgumentParser(description="Zero-shot на Stanford, фаза 1")
    парсер.add_argument("--device", default="cuda")
    парсер.add_argument("--splits", nargs="+", default=["val", "test"])
    аргументы = парсер.parse_args()

    каталог = load_catalog(STANFORD_CATALOG_PATH)
    индекс_классов = class_index(каталог)
    имена = stanford_names(каталог)
    кэш = Path.home() / "carlens-data" / "spike_models"

    итоги: dict[str, dict[str, float]] = {}
    for пин in load_models_manifest(DEFAULT_MODELS_MANIFEST):
        кодировщик = load_encoder(пин, кэш, аргументы.device)
        каталог_вывода = КОРЕНЬ_ИНДЕКСОВ / пин.name / "stanford"
        путь = build_text_index(
            каталог,
            имена,
            кодировщик.encode_text,
            пин,
            кодировщик.tokenizer_name,
            каталог_вывода,
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        from pipeline.index_format import read_index

        _, центроиды = read_index(путь)

        for сплит in аргументы.splits:
            образцы = read_samples(
                splits_dir() / f"{сплит}.parquet",
                crops_dir() / "stanford",
                num_classes=len(индекс_классов),
                skip_multi_car=True,
            )
            пути = [о.path for о in образцы]
            метки = np.array([о.class_idx for о in образцы], dtype=np.int64)
            вектора = кодировщик.encode_images(пути)
            нормы = np.linalg.norm(вектора, axis=1, keepdims=True)
            вектора = (вектора / нормы).astype(np.float32)
            итоги[f"{пин.name}/{сплит}"] = {
                "top1": float(topk_hits(вектора, центроиды, метки, k=1).mean()),
                "top3": float(topk_hits(вектора, центроиды, метки, k=3).mean()),
                "n": float(len(образцы)),
            }
            print(f"{пин.name} {сплит}: {итоги[f'{пин.name}/{сплит}']}")

    (КОРЕНЬ_ИНДЕКСОВ / "stanford_results.json").write_text(
        json.dumps(итоги, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
