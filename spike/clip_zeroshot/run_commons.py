"""Фаза 1, часть 2: smoke-test российского домена при индексе из всех 150 классов.

Индекс намеренно полный, а не из 25 измеряемых классов: иначе задача искусственно легче
продуктовой, и случайный уровень для top-3 был бы 3/25 вместо 3/150.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pipeline.catalog import DEFAULT_CATALOG_PATH, class_index, load_catalog
from pipeline.index_format import read_index
from spike.clip_zeroshot.commons import DEFAULT_COMMONS_MANIFEST, load_commons_manifest
from spike.clip_zeroshot.commons_classes import CONTROL, CORE_G4, RU_CN
from spike.clip_zeroshot.encoders import load_encoder
from spike.clip_zeroshot.evaluate import accuracy_by_class, topk_hits
from spike.clip_zeroshot.gate import PhaseOneResult, check_phase_one
from spike.clip_zeroshot.labels import DEFAULT_LATIN_NAMES, load_latin_names
from spike.clip_zeroshot.manifest import DEFAULT_MODELS_MANIFEST, load_models_manifest
from spike.clip_zeroshot.text_index import build_text_index

КОРЕНЬ_ИНДЕКСОВ = Path(__file__).resolve().parent / "index"
КОРЕНЬ_ФОТО = Path.home() / "carlens-data" / "spike_commons"


def собрать_итог(
    попадания3: np.ndarray,
    метки: np.ndarray,
    индекс_классов: dict[str, int],
    stanford: dict[str, float],
) -> PhaseOneResult:
    """Свести попадания top-3 в срезы G2, G3 и G4."""
    по_классам = accuracy_by_class(попадания3, метки, классов=len(индекс_классов))
    по_id = {
        class_id: по_классам[индекс]
        for class_id, индекс in индекс_классов.items()
        if индекс in по_классам
    }

    def доля(классы: tuple[str, ...]) -> float:
        индексы = [индекс_классов[c] for c in классы if c in индекс_классов]
        маска = np.isin(метки, индексы)
        return float(попадания3[маска].mean()) if bool(маска.any()) else 0.0

    return PhaseOneResult(
        stanford_top1=stanford["top1"],
        stanford_top3=stanford["top3"],
        control_top3=доля(CONTROL),
        ru_cn_top3=доля(RU_CN),
        core_top3={c: по_id[c] for c in CORE_G4 if c in по_id},
    )


def main() -> None:
    парсер = argparse.ArgumentParser(description="Zero-shot на Commons, фаза 1")
    парсер.add_argument("--device", default="cuda")
    аргументы = парсер.parse_args()

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    индекс_классов = class_index(каталог)
    имена = load_latin_names(DEFAULT_LATIN_NAMES)
    фото = load_commons_manifest(DEFAULT_COMMONS_MANIFEST)
    пути = [КОРЕНЬ_ФОТО / ф.class_id / ф.file for ф in фото]
    метки = np.array([индекс_классов[ф.class_id] for ф in фото], dtype=np.int64)
    stanford_итоги = json.loads(
        (КОРЕНЬ_ИНДЕКСОВ / "stanford_results.json").read_text(encoding="utf-8")
    )
    кэш = Path.home() / "carlens-data" / "spike_models"

    отчёт: dict[str, dict[str, object]] = {}
    for пин in load_models_manifest(DEFAULT_MODELS_MANIFEST):
        кодировщик = load_encoder(пин, кэш, аргументы.device)
        путь = build_text_index(
            каталог,
            имена,
            кодировщик.encode_text,
            пин,
            кодировщик.tokenizer_name,
            КОРЕНЬ_ИНДЕКСОВ / пин.name / "canonical",
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _, центроиды = read_index(путь)
        вектора = кодировщик.encode_images(пути)
        нормы = np.linalg.norm(вектора, axis=1, keepdims=True)
        вектора = (вектора / нормы).astype(np.float32)
        попадания3 = topk_hits(вектора, центроиды, метки, k=3)

        # Stanford берётся по сплиту `val`: он и сравнивается с 94.3% обученного чекпоинта.
        итог = собрать_итог(попадания3, метки, индекс_классов, stanford_итоги[f"{пин.name}/val"])
        прошёл, причины = check_phase_one(итог)
        отчёт[пин.name] = {
            "stanford_top1": итог.stanford_top1,
            "stanford_top3": итог.stanford_top3,
            "control_top3": итог.control_top3,
            "ru_cn_top3": итог.ru_cn_top3,
            "core_top3": итог.core_top3,
            "phase_one_passed": прошёл,
            "reasons": причины,
        }
        print(f"{пин.name}: прошёл={прошёл} причины={причины}")

    (КОРЕНЬ_ИНДЕКСОВ / "phase_one_results.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
