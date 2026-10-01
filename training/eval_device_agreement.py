"""Согласие GPU и CPU классификатора на телефоне на настоящих кропах (M4, латентность §5.4).

GPU-делегат считает активации в fp16, поэтому тензорный паритет с питоном для него не
требуется; важно, меняются ли решения. Здесь считается ровно это: одни и те же кропы прошли
через классификатор на CPU и на GPU на устройстве (`GpuAgreementTest`), эмбеддинги легли в файлы,
а этот модуль ставит их против индекса, порогов и негативов тем же правилом, что приёмка и
телефон (`training.calibration.decide`).

Две подкоманды. `pack` собирает кропы знакомых (релевантный набор `holdout_ru` и, если нужно,
целые доп. сплиты) и уличных негативов в один несжатый zip, который кладётся на телефон через
adb. `report` читает файлы эмбеддингов, снятые тестом, и пишет отчёт с гейтом.

Пороги гейта записаны в `bench/gpu-classifier-preregistration.md` до замера и продублированы
константами: менять их после замера нельзя.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.index_format import read_index, verify_meta
from pipeline.knn import top2_with_sims
from pipeline.negatives import neg_kind_of
from pipeline.paths import harvest_dir, splits_dir
from pipeline.schema import read_split
from training.calibration import REJECT_OFF, Scores, decide
from training.eval_open_set import GROUP_KINDS, NONCAR, STREET, open_set_point, relevant_known
from training.tune_thresholds import scores_from

# Гейт: `bench/gpu-classifier-preregistration.md`.
GATE_TOP1_CHURN = 0.005
GATE_COVERAGE_DROP = 0.005
GATE_FPR = 0.15
GATE_FLIPS = 0.01

PART_KNOWN = "known"
PART_STREET = "street"


@dataclass(frozen=True, slots=True)
class Embeddings:
    """Эмбеддинги двух путей и разметка строк: кто знакомый, а кто негатив какого вида."""

    cpu: np.ndarray
    gpu: np.ndarray
    is_known: np.ndarray
    truth: np.ndarray
    kinds: np.ndarray


def read_embeddings(path: Path, rows: int, dim: int) -> np.ndarray:
    """float32 little-endian `[rows, dim]`; чужая длина файла — отказ, а не тихий обрез."""
    числа = np.fromfile(path, dtype="<f4")
    if числа.size != rows * dim:
        raise ValueError(f"{path}: {числа.size} чисел, ожидалось {rows}×{dim}")
    return числа.reshape(rows, dim).astype(np.float32)


def _scores(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    brands: Sequence[str | None],
    truth: np.ndarray | None,
) -> Scores:
    return scores_from(*top2_with_sims(embeddings, centroids), truth, brands)


def _path_result(
    embeddings: np.ndarray,
    data: Embeddings,
    centroids: np.ndarray,
    brands: Sequence[str | None],
    sim_min: float,
    margin_min: float,
) -> dict[str, Any]:
    """Оценки одного пути: точность на знакомых, точка телефона и решения по всем строкам."""
    знакомые = _scores(embeddings[data.is_known], centroids, brands, data.truth[data.is_known])
    улица = _scores(embeddings[~data.is_known], centroids, brands, None)
    виды = data.kinds[~data.is_known]
    группы = {имя: улица.subset(np.isin(виды, GROUP_KINDS[имя])) for имя in (STREET, NONCAR)}
    точка = open_set_point(знакомые, группы, sim_min, margin_min)
    top1, *_ = top2_with_sims(embeddings[data.is_known], centroids)
    return {
        "top1": top1,
        "accuracy": float(знакомые.class_hit.mean()),
        "coverage": точка.coverage,
        "error": точка.error,
        "fpr": dict(точка.fpr),
        "decisions": np.concatenate(
            [
                decide(знакомые, sim_min, margin_min, REJECT_OFF),
                decide(улица, sim_min, margin_min, REJECT_OFF),
            ]
        ),
    }


def measure_agreement(
    data: Embeddings,
    centroids: np.ndarray,
    brands: Sequence[str | None],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Метрики согласия GPU с CPU в точке телефона (`thresholds.json` ассетов)."""
    sim_min, margin_min = float(thresholds["sim_min"]), float(thresholds["margin_min"])
    cpu = _path_result(data.cpu, data, centroids, brands, sim_min, margin_min)
    gpu = _path_result(data.gpu, data, centroids, brands, sim_min, margin_min)
    косинусы = np.sum(data.cpu * data.gpu, axis=1) / np.maximum(
        np.linalg.norm(data.cpu, axis=1) * np.linalg.norm(data.gpu, axis=1), 1e-12
    )
    известных = int(data.is_known.sum())
    return {
        "known": известных,
        "negatives": int((~data.is_known).sum()),
        "phone": {"sim_min": sim_min, "margin_min": margin_min},
        "cosine": {
            "min": float(косинусы.min()),
            "p1": float(np.percentile(косинусы, 1)),
            "median": float(np.median(косинусы)),
        },
        "top1_churn": int((cpu["top1"] != gpu["top1"]).sum()),
        "top1_churn_share": float((cpu["top1"] != gpu["top1"]).sum()) / известных,
        "flips": int((cpu["decisions"] != gpu["decisions"]).sum()),
        "flips_share": float((cpu["decisions"] != gpu["decisions"]).mean()),
        "cpu": {k: v for k, v in cpu.items() if k not in ("top1", "decisions")},
        "gpu": {k: v for k, v in gpu.items() if k not in ("top1", "decisions")},
    }


def verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Гейт предрегистрации (условия а–в): значение, предел, признак прохождения."""
    cpu, gpu = result["cpu"], result["gpu"]
    падение = cpu["coverage"] - gpu["coverage"]
    fpr_gpu = max(gpu["fpr"].values())
    условия = {
        "a_top1_churn": {
            "value": result["top1_churn_share"],
            "limit": GATE_TOP1_CHURN,
            "ok": result["top1_churn_share"] <= GATE_TOP1_CHURN,
        },
        "b_coverage_drop": {
            "value": падение,
            "limit": GATE_COVERAGE_DROP,
            "ok": падение <= GATE_COVERAGE_DROP,
        },
        "b_fpr_gpu_max": {"value": fpr_gpu, "limit": GATE_FPR, "ok": fpr_gpu <= GATE_FPR},
        "c_decision_flips": {
            "value": result["flips_share"],
            "limit": GATE_FLIPS,
            "ok": result["flips_share"] <= GATE_FLIPS,
        },
    }
    return {"conditions": условия, "passed": all(у["ok"] for у in условия.values())}


def render_report(
    result: Mapping[str, Any], итог: Mapping[str, Any], origin: Mapping[str, str]
) -> str:
    cpu, gpu = result["cpu"], result["gpu"]
    строки = [
        "# Классификатор на GPU против CPU — согласие решений на устройстве",
        "",
        "Предрегистрация: `bench/gpu-classifier-preregistration.md`. Замер: "
        "`GpuAgreementTest` на S23 и `python -m training.eval_device_agreement report`.",
        "",
        *[f"- {ключ}: `{значение}`" for ключ, значение in origin.items()],
        "",
        f"Знакомых кропов {result['known']}, негативов {result['negatives']}. Точка телефона: "
        f"sim_min {result['phone']['sim_min']}, margin_min {result['phone']['margin_min']}.",
        "",
        f"Косинус эмбеддинга GPU к CPU: min {result['cosine']['min']:.5f}, "
        f"p1 {result['cosine']['p1']:.5f}, медиана {result['cosine']['median']:.6f}.",
        "",
        "| | top-1 по модели | покрытие | ошибка | FPR чужие машины | FPR не-машины |",
        "|---|---|---|---|---|---|",
    ]
    for имя, путь in (("CPU", cpu), ("GPU", gpu)):
        строки.append(
            f"| {имя} | {путь['accuracy']:.4f} | {путь['coverage']:.4f} | {путь['error']:.4f} "
            f"| {путь['fpr'][STREET]:.4f} | {путь['fpr'][NONCAR]:.4f} |"
        )
    строки += [
        "",
        f"Расхождений top-1 между GPU и CPU: {result['top1_churn']} из {result['known']}. "
        f"Решений принять/отказать поменялось: {result['flips']} из "
        f"{result['known'] + result['negatives']}.",
        "",
        "## Гейт",
        "",
        "| условие | значение | предел | |",
        "|---|---|---|---|",
    ]
    for имя, у in итог["conditions"].items():
        строки.append(
            f"| {имя} | {у['value']:.5f} | ≤ {у['limit']} | {'да' if у['ok'] else 'НЕТ'} |"
        )
    строки += ["", f"**Вердикт: {'гейт пройден' if итог['passed'] else 'гейт НЕ пройден'}.**", ""]
    return "\n".join(строки)


def _rows(
    holdout: Path, relevance: Path, street: Path, extra_known: Sequence[Path] = ()
) -> list[dict[str, Any]]:
    """Порядок строк: знакомые, потом негативы — как в `order.csv` и в файлах эмбеддингов.

    `extra_known` — сплиты, чьи кадры целиком знакомые (проверка новых классов: у них нет
    разметки релевантности).
    """
    знакомые = pd.concat(
        [relevant_known(holdout, relevance), *(read_split(путь) for путь in extra_known)],
        ignore_index=True,
    )
    улица = read_split(street)
    строки: list[dict[str, Any]] = [
        {"part": PART_KNOWN, "crop_path": str(путь), "key": str(ключ), "class_idx": int(класс)}
        for путь, ключ, класс in zip(
            знакомые["crop_path"], знакомые["photo_id"], знакомые["class_idx"], strict=True
        )
    ]
    строки += [
        {"part": PART_STREET, "crop_path": str(путь), "key": str(ключ), "class_idx": -1}
        for путь, ключ in zip(улица["crop_path"], улица["listing_id"], strict=True)
    ]
    return строки


def pack(
    holdout: Path,
    relevance: Path,
    street: Path,
    crops_root: Path,
    out_dir: Path,
    extra_known: Sequence[Path] = (),
) -> int:
    """Кропы одним несжатым zip и `order.csv`; имя записи — номер строки, порядок сортируемый."""
    out_dir.mkdir(parents=True, exist_ok=True)
    строки = _rows(holdout, relevance, street, extra_known)
    with zipfile.ZipFile(out_dir / "crops.zip", "w", compression=zipfile.ZIP_STORED) as архив:
        for номер, строка in enumerate(строки):
            путь = crops_root / строка["crop_path"]
            архив.write(путь, f"{номер:05d}{путь.suffix.lower()}")
    with (out_dir / "order.csv").open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=["idx", "part", "crop_path", "key", "class_idx"])
        писатель.writeheader()
        for номер, строка in enumerate(строки):
            писатель.writerow({"idx": номер, **строка})
    return len(строки)


def _load_order(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as файл:
        return list(csv.DictReader(файл))


def report(agree_dir: Path, assets: Path, out: Path) -> dict[str, Any]:
    порядок = _load_order(agree_dir / "order.csv")
    манифест = json.loads((assets / "models_manifest.json").read_text("utf-8"))
    байты_меты = (assets / "catalog_meta.json").read_bytes()
    заголовок, центроиды = read_index(assets / "catalog_index.bin")
    verify_meta(заголовок, байты_меты)
    бренды = [класс["brand"] for класс in json.loads(байты_меты)["classes"]]
    измерение = int(манифест["classifier"]["embedding_dim"])
    данные = Embeddings(
        cpu=read_embeddings(agree_dir / "emb_cpu.f32", len(порядок), измерение),
        gpu=read_embeddings(agree_dir / "emb_gpu.f32", len(порядок), измерение),
        is_known=np.array([р["part"] == PART_KNOWN for р in порядок]),
        truth=np.array([int(р["class_idx"]) for р in порядок], dtype=np.int64),
        kinds=np.array(
            [neg_kind_of(р["key"]) if р["part"] == PART_STREET else "" for р in порядок]
        ),
    )
    результат = measure_agreement(
        данные, центроиды, бренды, json.loads((assets / "thresholds.json").read_text("utf-8"))
    )
    итог = verdict(результат)
    происхождение = {
        "classifier": манифест["files"]["classifier"]["name"],
        "classifier_sha256": манифест["files"]["classifier"]["sha256"],
        "order_rows": str(len(порядок)),
    }
    out.write_text(render_report(результат, итог, происхождение), encoding="utf-8")
    return итог


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Согласие GPU и CPU классификатора на устройстве")
    подкоманды = парсер.add_subparsers(dest="command", required=True)
    упаковать = подкоманды.add_parser("pack", help="собрать crops.zip и order.csv")
    упаковать.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    упаковать.add_argument("--relevance", type=Path, default=Path("catalog/holdout_relevance.csv"))
    упаковать.add_argument("--street", type=Path, default=splits_dir() / "negatives-street.parquet")
    упаковать.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    упаковать.add_argument(
        "--extra-known",
        type=Path,
        nargs="*",
        default=[],
        help="сплиты, целиком идущие в знакомые (например, eval-new.parquet)",
    )
    упаковать.add_argument(
        "--out", type=Path, required=True, help="каталог для crops.zip и order.csv"
    )
    отчёт = подкоманды.add_parser("report", help="посчитать гейт по файлам эмбеддингов")
    отчёт.add_argument("--agree-dir", type=Path, required=True, help="order.csv и emb_*.f32")
    отчёт.add_argument("--assets", type=Path, required=True, help="ассеты модели с thresholds.json")
    отчёт.add_argument("--report", type=Path, required=True)
    аргументы = парсер.parse_args(argv)

    if аргументы.command == "pack":
        число = pack(
            аргументы.holdout,
            аргументы.relevance,
            аргументы.street,
            аргументы.crops_root,
            аргументы.out,
            аргументы.extra_known,
        )
        print(f"упаковано {число} кропов в {аргументы.out}")
        return 0
    итог = report(аргументы.agree_dir, аргументы.assets, аргументы.report)
    print("гейт пройден" if итог["passed"] else "гейт НЕ пройден", аргументы.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
