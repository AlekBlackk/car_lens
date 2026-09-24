"""Подбор `thresholds.json` и `calibration.json` для выпущенной модели (спека calibration-ru §5).

Эмбеддинги считает тот самый `classifier_fp32.tflite` из каталога ассетов, по его индексу и
мете: подбирается то, что стоит на телефоне, а не соседний чекпоинт. Запуск — в venv
экспорта (LiteRT); torch, albumentations и LiteRT импортируются внутри функций (Р-15),
остальное тестируется в `dev`.

Знакомые кадры — `holdout_ru`, поделённый пополам по классам: пара порогов и шкала,
подобранные на одной половине, меряются на другой, в отчёт идут обе стороны. Ассеты —
подбор на всём наборе, и пишутся они только при положительном вердикте §6 спеки.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.index_format import read_index, verify_meta
from pipeline.knn import top2_with_sims
from pipeline.negatives import neg_kind_of
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import read_split
from training.calibration import (
    CLIP,
    MODEL,
    Operating,
    Scores,
    auroc,
    calibration_doc,
    calibration_knots,
    confidence_from_sim,
    decide,
    ece,
    fpr_at_tpr,
    operating_point,
    select_pair,
    split_halves,
    thresholds_doc,
)

НАЧАЛО = "<!-- generated:start -->"
КОНЕЦ = "<!-- generated:end -->"
MAX_HELD_OUT_REJECTION = 0.5
DEFAULT_SEED = 20260923


def check_holdout_slots(
    model_ids: Sequence[str], class_idx: Sequence[int], meta: Mapping[str, Any]
) -> None:
    """`class_idx` эталона обязан указывать на слот меты, где эта модель — участник."""
    классы = meta["classes"]
    for модель, слот in zip(model_ids, class_idx, strict=True):
        if модель not in классы[int(слот)]["members"]:
            raise ValueError(
                f"эталон: {модель} в слоте {слот}, а там {классы[int(слот)]['class_id']}"
            )


def scores_from(
    top1: np.ndarray,
    sim1: np.ndarray,
    top2: np.ndarray,
    sim2: np.ndarray,
    truth: np.ndarray | None,
    brands: Sequence[str | None],
) -> Scores:
    """Собрать `Scores`; `truth=None` — негативы. Марки сравниваются как `brandsMatch`."""
    марка = np.array([b if b is not None else "" for b in brands], dtype=object)
    есть_марка = np.array([b is not None for b in brands])
    общая = есть_марка[top1] & есть_марка[top2] & (марка[top1] == марка[top2])
    if truth is None:
        нет = np.zeros(len(top1), dtype=bool)
        return Scores(sim1, sim2, общая, нет, нет)
    класс = top1 == truth
    по_марке = класс | (есть_марка[top1] & есть_марка[truth] & (марка[top1] == марка[truth]))
    return Scores(sim1, sim2, общая, класс, по_марке)


@dataclass(frozen=True, slots=True)
class FoldReport:
    tuned_on: str
    chosen: Operating | None
    held_out: Operating | None
    ece: float | None


def _шкала(known: Scores, point: Operating, max_knots: int) -> list[tuple[float, float]]:
    """Шкала учится только на ответах моделью: процент пользователь видит только у них."""
    модель = decide(known, point.sim_min, point.margin_min) == MODEL
    return calibration_knots(known.sim1[модель], known.class_hit[модель], point.sim_min, max_knots)


def cross_fit(
    known: Scores,
    negatives: Scores,
    known_a: np.ndarray,
    neg_a: np.ndarray,
    *,
    max_error: float,
    max_fpr: float,
    max_knots: int,
) -> list[FoldReport]:
    стороны: list[FoldReport] = []
    for имя, маска_з, маска_ч in (("A", known_a, neg_a), ("B", ~known_a, ~neg_a)):
        учёба_з, учёба_ч = known.subset(маска_з), negatives.subset(маска_ч)
        проверка_з, проверка_ч = known.subset(~маска_з), negatives.subset(~маска_ч)
        точка = select_pair(учёба_з, учёба_ч, max_error=max_error, max_fpr=max_fpr)
        if точка is None:
            стороны.append(FoldReport(имя, None, None, None))
            continue
        узлы = _шкала(учёба_з, точка, max_knots)
        отложенная = operating_point(проверка_з, проверка_ч, точка.sim_min, точка.margin_min)
        модель = decide(проверка_з, точка.sim_min, точка.margin_min) == MODEL
        уверенность = confidence_from_sim(проверка_з.sim1[модель], узлы)
        стороны.append(
            FoldReport(имя, точка, отложенная, ece(уверенность, проверка_з.class_hit[модель]))
        )
    return стороны


def release_verdict(folds: Sequence[FoldReport], final: Operating | None) -> tuple[bool, str]:
    """§6 спеки: стоп без допустимой пары или при отказе > 50% на отложенной половине."""
    if final is None or not folds or any(с.held_out is None for с in folds):
        return False, "допустимой пары нет хотя бы на одной стороне"
    хуже = [
        с.tuned_on
        for с in folds
        if с.held_out is not None and 1.0 - с.held_out.coverage > MAX_HELD_OUT_REJECTION
    ]
    if хуже:
        return False, f"отказ на знакомых > 50% на отложенной половине: {', '.join(хуже)}"
    return True, "ограничения выполнены на обеих отложенных половинах"


def write_generated_section(path: Path, body: str) -> None:
    """Заменить только текст между маркерами; рукописная шапка и выводы остаются."""
    раздел = f"{НАЧАЛО}\n{body}{КОНЕЦ}"
    if not path.exists():
        path.write_text(раздел + "\n", encoding="utf-8")
        return
    текст = path.read_text(encoding="utf-8")
    if НАЧАЛО not in текст or КОНЕЦ not in текст:
        raise ValueError(f"{path}: нет маркеров генерируемого раздела")
    до, _, остаток = текст.partition(НАЧАЛО)
    _, _, после = остаток.partition(КОНЕЦ)
    path.write_text(до + раздел + после, encoding="utf-8")


def _sha256(путь: Path) -> str:
    хеш = hashlib.sha256()
    with путь.open("rb") as файл:
        for кусок in iter(lambda: файл.read(1 << 20), b""):
            хеш.update(кусок)
    return хеш.hexdigest()


def _точка_md(точка: Operating | None) -> str:
    if точка is None:
        return "— | — | — | — | —"
    return (
        f"{точка.sim_min:.2f} | {точка.margin_min:.3f} | {точка.coverage:.3f} | "
        f"{точка.error:.3f} | {точка.fpr:.3f}"
    )


def render_report(
    *,
    folds: Sequence[FoldReport],
    final: Operating | None,
    verdict: tuple[bool, str],
    provenance: Mapping[str, str],
    counts: Mapping[str, int],
    auroc_value: float,
    fpr95: float,
    knots_n: int | None,
) -> str:
    строки = ["## Происхождение", ""]
    строки += [f"- {ключ}: `{значение}`" for ключ, значение in provenance.items()]
    строки += ["", "## Выборки", ""]
    строки += [f"- {ключ}: {значение}" for ключ, значение in counts.items()]
    строки += [
        "",
        "## Перекрёстная проверка",
        "",
        "Покрытие — доля знакомых с любым ответом; ошибка — доля неверных среди показанных;",
        "FPR — доля принятых негативов. «Подбор» — на своей половине, «проверка» — та же",
        "пара на другой.",
        "",
        "| сторона | где | sim_min | margin_min | покрытие | ошибка | FPR | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for с in folds:
        ece_md = "—" if с.ece is None else f"{с.ece:.3f}"
        строки.append(f"| {с.tuned_on} | подбор | {_точка_md(с.chosen)} | — |")
        строки.append(f"| {с.tuned_on} | проверка | {_точка_md(с.held_out)} | {ece_md} |")
    строки += [
        "",
        "## Итог на всём наборе",
        "",
        "| sim_min | margin_min | покрытие | ошибка | FPR |",
        "|---|---|---|---|---|",
        f"| {_точка_md(final)} |",
        "",
        f"Узлов шкалы: {knots_n if knots_n is not None else '—'}. "
        f"AUROC (sim1): {auroc_value:.3f}. FPR@TPR95: {fpr95:.3f} "
        "(справочно, §5.2 SPEC; выпуск от них не зависит).",
        "",
        f"**Вердикт:** {'выпуск' if verdict[0] else 'СТОП'} — {verdict[1]}.",
        "",
    ]
    return "\n".join(строки)


def _эмбеддинги(
    модель: Path, пути: Sequence[Path], манифест: Mapping[str, Any], батч: int
) -> np.ndarray:
    from training.build_index import collect_embeddings_tflite
    from training.dataset import build_transforms, make_loader
    from training.samples import Sample

    классификатор = манифест["classifier"]
    среднее, отклонение = классификатор["mean"], классификатор["std"]
    трансформы = build_transforms(
        int(классификатор["img_size"]),
        train=False,
        mean=(float(среднее[0]), float(среднее[1]), float(среднее[2])),
        std=(float(отклонение[0]), float(отклонение[1]), float(отклонение[2])),
    )
    # Метка образцу не нужна: считаются только эмбеддинги.
    образцы = [Sample(path=п, class_idx=0, is_multi_car=False) for п in пути]
    загрузчик = make_loader(образцы, трансформы, batch_size=батч, shuffle=False, num_workers=0)
    эмбеддинги, _ = collect_embeddings_tflite(модель, загрузчик)
    return эмбеддинги


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Подбор порогов отказа и шкалы уверенности")
    парсер.add_argument("--models", type=Path, default=Path("app/android/models"))
    парсер.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    парсер.add_argument("--negatives", type=Path, default=splits_dir() / "negatives.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--max-error", type=float, default=0.15)
    парсер.add_argument("--max-fpr", type=float, default=0.15)
    парсер.add_argument("--max-knots", type=int, default=32)
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--batch-size", type=int, default=32)
    парсер.add_argument("--report", type=Path, default=Path("bench/calibration-ru.md"))
    парсер.add_argument("--out-dir", type=Path, default=None)
    аргументы = парсер.parse_args(argv)

    манифест = json.loads((аргументы.models / "models_manifest.json").read_text("utf-8"))
    байты_меты = (аргументы.models / "catalog_meta.json").read_bytes()
    заголовок, центроиды = read_index(аргументы.models / "catalog_index.bin")
    verify_meta(заголовок, байты_меты)
    мета = json.loads(байты_меты)
    бренды = [класс["brand"] for класс in мета["classes"]]
    классификатор = аргументы.models / манифест["files"]["classifier"]["name"]

    # Как в `eval_finetune_ru`: многомашинные кадры эталона не меряются.
    эталон = read_split(аргументы.holdout)
    эталон = эталон[~эталон["is_multi_car"].astype(bool)].reset_index(drop=True)
    check_holdout_slots(list(эталон["model_id"]), list(эталон["class_idx"]), мета)
    негативы = read_split(аргументы.negatives)

    корень = аргументы.crops_root
    э_эт = _эмбеддинги(
        классификатор, [корень / п for п in эталон["crop_path"]], манифест, аргументы.batch_size
    )
    э_нег = _эмбеддинги(
        классификатор, [корень / п for п in негативы["crop_path"]], манифест, аргументы.batch_size
    )
    истина = эталон["class_idx"].to_numpy(dtype=np.int64)
    знакомые = scores_from(*top2_with_sims(э_эт, центроиды), истина, бренды)
    чужие = scores_from(*top2_with_sims(э_нег, центроиды), None, бренды)

    виды = np.array([neg_kind_of(str(л)) for л in негативы["listing_id"]])
    половина_з = split_halves(истина, аргументы.seed)
    половина_ч = split_halves(виды, аргументы.seed)
    стороны = cross_fit(
        знакомые,
        чужие,
        половина_з,
        половина_ч,
        max_error=аргументы.max_error,
        max_fpr=аргументы.max_fpr,
        max_knots=аргументы.max_knots,
    )
    итог = select_pair(знакомые, чужие, max_error=аргументы.max_error, max_fpr=аргументы.max_fpr)
    вердикт = release_verdict(стороны, итог)
    узлы = _шкала(знакомые, итог, аргументы.max_knots) if итог is not None else None

    происхождение = {
        "checkpoint_sha256": манифест["provenance"]["checkpoint_sha256"],
        "classifier_sha256": _sha256(классификатор),
        "holdout_sha256": _sha256(аргументы.holdout),
        "negatives_sha256": _sha256(аргументы.negatives),
        "seed": str(аргументы.seed),
        "max_error / max_fpr": f"{аргументы.max_error} / {аргументы.max_fpr}",
    }
    счётчики = {"знакомые (holdout_ru без многомашинных)": len(эталон)}
    for вид in sorted(set(виды)):
        счётчики[f"негативы {вид}"] = int((виды == вид).sum())
    write_generated_section(
        аргументы.report,
        render_report(
            folds=стороны,
            final=итог,
            verdict=вердикт,
            provenance=происхождение,
            counts=счётчики,
            auroc_value=auroc(знакомые.sim1, чужие.sim1),
            fpr95=fpr_at_tpr(знакомые.sim1, чужие.sim1),
            knots_n=None if узлы is None else len(узлы),
        ),
    )
    print(f"вердикт: {вердикт}")
    if not вердикт[0] or итог is None or узлы is None:
        return 0

    выход = аргументы.out_dir or (
        data_root() / "calibration" / f"calib_{datetime.now():%Y%m%d_%H%M%S}"
    )
    выход.mkdir(parents=True, exist_ok=True)
    for имя, документ in (
        ("thresholds.json", thresholds_doc(итог, len(эталон), len(негативы))),
        ("calibration.json", calibration_doc(узлы, CLIP)),
    ):
        (выход / имя).write_text(
            json.dumps(документ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"ассеты: {выход}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
