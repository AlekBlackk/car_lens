"""Приёмка §5.2 на полном и релевантных наборах `holdout_ru` (спека holdout-relevance §7).

Путь `training.tune_thresholds`: ассеты с телефона, эталон без многомашинных кадров, запуск
в venv экспорта (LiteRT импортируется внутри `embed_crops`). Полный набор сверяется с
паритетом модели до записи отчёта: расхождение — ошибка кода, а не результат.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline.holdout_relevance import (
    SET_FULL,
    SET_TITLES,
    VERDICTS,
    load_relevance,
    subsets,
)
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import ranked_by_centroids, top2_with_sims
from pipeline.paths import harvest_dir, splits_dir
from pipeline.pool_audit import file_sha256
from pipeline.schema import read_split
from training.acceptance import SetMetrics, member_brands, set_metrics
from training.tune_thresholds import (
    check_holdout_slots,
    embed_crops,
    scores_from,
    write_generated_section,
)

TARGETS = {
    "top1_model": 0.75,
    "top5_model": 0.92,
    "top1_brand": 0.93,
    "auroc": 0.88,
    "fpr_at_tpr95": 0.15,
}
UNSURE_WARN_SHARE = 0.05
# Строки таблицы: поле SetMetrics, подпись, цель в тексте.
_СТРОКИ = (
    ("top1_model", "top-1 по модели", f"≥ {TARGETS['top1_model']}"),
    ("top5_model", "top-5 по модели", f"≥ {TARGETS['top5_model']}"),
    ("top1_brand_strict", "марка, строго", f"≥ {TARGETS['top1_brand']}"),
    ("top1_brand_grouped", "марка с учётом групп", f"≥ {TARGETS['top1_brand']}"),
    ("auroc", "AUROC (sim1)", f"≥ {TARGETS['auroc']}"),
    ("fpr_at_tpr95", "FPR@TPR95", f"≤ {TARGETS['fpr_at_tpr95']}"),
    ("coverage", "покрытие при порогах 0.4.1", "—"),
    ("error", "неверных среди показанных", "—"),
)


def read_parity(path: Path) -> dict[str, float]:
    """Числа паритета того же fp32-классификатора, что стоит на телефоне."""
    fp32 = json.loads(path.read_text(encoding="utf-8"))["runtimes"]["holdout_ru"]["tflite_fp32"]
    return {ключ: float(fp32[ключ]) for ключ in ("top1_model", "top5_model", "top1_brand")}


def check_reference(full: SetMetrics, expected: Mapping[str, tuple[float, float]]) -> None:
    """Полный набор обязан повторить прошлые замеры; иначе прогон падает до отчёта."""
    поля = asdict(full)
    расхождения = [
        f"{имя}: {поля[имя]:.6f} против {ожидание:.6f} (допуск {допуск})"
        for имя, (ожидание, допуск) in expected.items()
        if abs(поля[имя] - ожидание) > допуск
    ]
    if расхождения:
        raise ValueError("полный набор не сходится с прошлыми замерами: " + "; ".join(расхождения))


def render_report(
    *,
    results: Mapping[str, SetMetrics],
    verdict_counts: Mapping[str, int],
    top_down: int,
    provenance: Mapping[str, str],
) -> str:
    всего = sum(verdict_counts.values())
    наборы = list(results)
    строки = ["## Происхождение", ""]
    строки += [f"- {ключ}: `{значение}`" for ключ, значение in provenance.items()]
    строки += ["", "## Разметка", ""]
    строки += [f"- {вердикт}: {verdict_counts.get(вердикт, 0)}" for вердикт in VERDICTS]
    строки.append(f"- флаг «сверху»: {top_down}")
    неуверенных = verdict_counts.get("unsure", 0)
    if всего and неуверенных / всего > UNSURE_WARN_SHARE:
        строки += [
            "",
            f"**unsure больше 5% эталона ({неуверенных} из {всего})** — релевантный набор "
            "может быть смещён: часть сомнительных кадров в нём не участвует.",
        ]
    строки += [
        "",
        "## Приёмка §5.2",
        "",
        "| метрика | цель | " + " | ".join(SET_TITLES[н] for н in наборы) + " |",
        "|---|---|" + "---|" * len(наборы),
        "| кадров | — | " + " | ".join(str(results[н].n) for н in наборы) + " |",
        "| классов | — | " + " | ".join(str(results[н].classes) for н in наборы) + " |",
    ]
    for поле, подпись, цель in _СТРОКИ:
        значения = " | ".join(f"{asdict(results[н])[поле]:.4f}" for н in наборы)
        строки.append(f"| {подпись} | {цель} | {значения} |")
    строки.append("")
    return "\n".join(строки)


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Приёмка §5.2 на наборах holdout_ru")
    парсер.add_argument("--models", type=Path, default=Path("app/android/models"))
    парсер.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    парсер.add_argument("--negatives", type=Path, default=splits_dir() / "negatives.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--relevance", type=Path, default=Path("catalog/holdout_relevance.csv"))
    парсер.add_argument("--parity", type=Path, required=True)
    парсер.add_argument("--expect-auroc", type=float, default=None)
    парсер.add_argument("--expect-fpr95", type=float, default=None)
    парсер.add_argument("--report", type=Path, default=Path("bench/acceptance-relevance.md"))
    парсер.add_argument("--batch-size", type=int, default=32)
    аргументы = парсер.parse_args(argv)

    манифест = json.loads((аргументы.models / "models_manifest.json").read_text("utf-8"))
    байты_меты = (аргументы.models / "catalog_meta.json").read_bytes()
    заголовок, центроиды = read_index(аргументы.models / "catalog_index.bin")
    verify_meta(заголовок, байты_меты)
    мета = json.loads(байты_меты)
    классы = мета["classes"]
    бренды = [класс["brand"] for класс in классы]
    пороги = json.loads((аргументы.models / "thresholds.json").read_text("utf-8"))
    классификатор = аргументы.models / манифест["files"]["classifier"]["name"]

    эталон = read_split(аргументы.holdout)
    эталон = эталон[~эталон["is_multi_car"].astype(bool)].reset_index(drop=True)
    check_holdout_slots(list(эталон["model_id"]), list(эталон["class_idx"]), мета)
    разметка = load_relevance(аргументы.relevance)
    маски = subsets(эталон["photo_id"].tolist(), разметка)
    негативы = read_split(аргументы.negatives)

    корень = аргументы.crops_root
    э_эт = embed_crops(
        классификатор, [корень / п for п in эталон["crop_path"]], манифест, аргументы.batch_size
    )
    э_нег = embed_crops(
        классификатор, [корень / п for п in негативы["crop_path"]], манифест, аргументы.batch_size
    )
    истина = эталон["class_idx"].to_numpy(dtype=np.int64)
    знакомые = scores_from(*top2_with_sims(э_эт, центроиды), истина, бренды)
    чужие = scores_from(*top2_with_sims(э_нег, центроиды), None, бренды)
    топ5 = np.array(ranked_by_centroids(э_эт, центроиды, 5), dtype=np.int64)

    результаты = {
        имя: set_metrics(
            top5=топ5,
            truth=истина,
            known=знакомые,
            negatives=чужие,
            mask=маска,
            idx_to_class_id={i: к["class_id"] for i, к in enumerate(классы)},
            idx_to_brand={i: к["brand"] for i, к in enumerate(классы)},
            brands=member_brands([к["members"] for к in классы]),
            sim_min=float(пороги["sim_min"]),
            margin_min=float(пороги["margin_min"]),
        )
        for имя, маска in маски.items()
    }

    паритет = read_parity(аргументы.parity)
    ожидания: dict[str, tuple[float, float]] = {
        "top1_model": (паритет["top1_model"], 1e-9),
        "top5_model": (паритет["top5_model"], 1e-9),
        "top1_brand_strict": (паритет["top1_brand"], 1e-9),
    }
    if аргументы.expect_auroc is not None:
        ожидания["auroc"] = (аргументы.expect_auroc, 5e-4)
    if аргументы.expect_fpr95 is not None:
        ожидания["fpr_at_tpr95"] = (аргументы.expect_fpr95, 5e-4)
    check_reference(результаты[SET_FULL], ожидания)

    по_id = разметка.set_index("photo_id").loc[эталон["photo_id"].tolist()]
    # value_counts даёт dict с ключом Hashable; вердикты — строки из VERDICTS.
    счётчики = {str(в): int(n) for в, n in по_id["verdict"].value_counts().items()}
    write_generated_section(
        аргументы.report,
        render_report(
            results=результаты,
            verdict_counts=счётчики,
            top_down=int(по_id["top_down"].sum()),
            provenance={
                "checkpoint_sha256": манифест["provenance"]["checkpoint_sha256"],
                "classifier_sha256": file_sha256(классификатор),
                "holdout_sha256": file_sha256(аргументы.holdout),
                "negatives_sha256": file_sha256(аргументы.negatives),
                "relevance_sha256": file_sha256(аргументы.relevance),
            },
        ),
    )
    for имя, м in результаты.items():
        print(f"{SET_TITLES[имя]}: n={м.n} top1={м.top1_model:.4f} top5={м.top5_model:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
