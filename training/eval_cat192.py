"""Проверки каталога 192 на `eval-new` и `eval-old` (`bench/catalog-192-preregistration.md`).

По одному кадру на машину MAD-Cars, отложенную до обучения. Новые классы `holdout_ru` не
покрывает, поэтому мерятся здесь; старые классы меряются тем же набором, чтобы сравнить новые с
ними одной линейкой. Запуск в venv экспорта (LiteRT импортируется внутри `embed_crops`).

Ассеты 0.5 на `eval-old` — справочная строка: классы набора там те же, а `eval-new` для них
не определён (классов нет в индексе), и набор пропускается.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog, member_index
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import ranked_by_centroids, top2_with_sims
from pipeline.paths import harvest_dir, splits_dir
from pipeline.pool_audit import file_sha256
from pipeline.schema import read_split
from training.acceptance import SetMetrics, member_brands, set_metrics
from training.calibration import MODEL, decide
from training.tune_thresholds import embed_crops, scores_from, write_generated_section

GATE_NEW_TOP1 = 0.75
GATE_NEW_BELOW_OLD = 0.10
GATE_NEW_ERROR = 0.15
TOP_CONFUSIONS = 15


def truth_slots(
    frame: pd.DataFrame, catalog_members: Mapping[str, str], meta_ids: Sequence[str]
) -> np.ndarray | None:
    """Индекс истинного класса в мете ассетов или None, если в индексе такого класса нет."""
    слот = {class_id: i for i, class_id in enumerate(meta_ids)}
    классы = [catalog_members[str(модель)] for модель in frame["model_id"]]
    if any(класс not in слот for класс in классы):
        return None
    return np.array([слот[класс] for класс in классы], dtype=np.int64)


def confusions(
    truth_ids: Sequence[str], predicted_ids: Sequence[str]
) -> list[tuple[str, str, int]]:
    """Самые частые ошибки «истина → ответ»."""
    счёт = Counter((т, п) for т, п in zip(truth_ids, predicted_ids, strict=True) if т != п)
    return [(т, п, n) for (т, п), n in счёт.most_common(TOP_CONFUSIONS)]


def per_group(truth_hits: np.ndarray, groups: Sequence[str]) -> pd.DataFrame:
    """Доля верного top-1 по группам (название MAD-Cars или класс), от худшей к лучшей."""
    таблица = pd.DataFrame({"group": list(groups), "hit": truth_hits.astype(float)})
    итог = таблица.groupby("group")["hit"].agg(["mean", "count"]).sort_values(["mean", "count"])
    return итог.rename(columns={"mean": "top1", "count": "n"})


def verdict(new: SetMetrics, old: SetMetrics) -> tuple[bool, str]:
    """Условие (в) и часть (д) гейта: новые классы не слабее старых и не врут в показанном."""
    причины: list[str] = []
    if new.top1_model < GATE_NEW_TOP1:
        причины.append(f"top-1 новых {new.top1_model:.4f} < {GATE_NEW_TOP1}")
    if new.top1_model < old.top1_model - GATE_NEW_BELOW_OLD:
        причины.append(
            f"top-1 новых {new.top1_model:.4f} ниже старых {old.top1_model:.4f} больше чем на "
            f"{GATE_NEW_BELOW_OLD}"
        )
    if new.error > GATE_NEW_ERROR:
        причины.append(f"ошибка среди показанных на новых {new.error:.4f} > {GATE_NEW_ERROR}")
    return (not причины), "; ".join(причины) or "условия (в) и ошибка (д) выполнены"


def _строка_метрик(название: str, м: SetMetrics) -> str:
    поля = asdict(м)
    return (
        f"| {название} | {поля['n']} | {поля['classes']} | {поля['top1_model']:.4f} | "
        f"{поля['top5_model']:.4f} | {поля['top1_brand_strict']:.4f} | "
        f"{поля['top1_brand_grouped']:.4f} | {поля['coverage']:.4f} | {поля['error']:.4f} |"
    )


def render_report(
    *,
    metrics: Mapping[str, SetMetrics],
    by_name: pd.DataFrame | None,
    worst_classes: Mapping[str, pd.DataFrame],
    confusion_lists: Mapping[str, list[tuple[str, str, int]]],
    provenance: Mapping[str, str],
    gate: tuple[bool, str] | None,
) -> str:
    строки = ["## Происхождение", ""]
    строки += [f"- {ключ}: `{значение}`" for ключ, значение in provenance.items()]
    строки += [
        "",
        "## Метрики",
        "",
        "| набор | машин | классов | top-1 | top-5 | марка строго | марка с группами | "
        "покрытие | ошибка среди показанных |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    строки += [_строка_метрик(название, м) for название, м in metrics.items()]
    if gate is not None:
        строки += ["", f"**Условие (в) и ошибка (д):** {'да' if gate[0] else 'нет'} — {gate[1]}."]
    if by_name is not None:
        строки += ["", "## Новые классы по названиям MAD-Cars (худшие сверху)", ""]
        строки += ["| название | машин | top-1 |", "|---|---|---|"]
        строки += [f"| {имя} | {int(р.n)} | {р.top1:.3f} |" for имя, р in by_name.iterrows()]
    for имя, таблица in worst_classes.items():
        строки += ["", f"## {имя}: худшие классы по top-1", "", "| класс | машин | top-1 |"]
        строки.append("|---|---|---|")
        строки += [f"| `{к}` | {int(р.n)} | {р.top1:.3f} |" for к, р in таблица.head(15).iterrows()]
    for имя, список in confusion_lists.items():
        строки += ["", f"## {имя}: частые ошибки", "", "| истина | ответ | раз |", "|---|---|---|"]
        строки += [f"| `{т}` | `{п}` | {n} |" for т, п, n in список]
    строки.append("")
    return "\n".join(строки)


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Проверки каталога 192")
    парсер.add_argument("--models", type=Path, default=Path("app/android/models"))
    парсер.add_argument("--eval-new", type=Path, default=splits_dir() / "eval-new.parquet")
    парсер.add_argument("--eval-old", type=Path, default=splits_dir() / "eval-old.parquet")
    парсер.add_argument(
        "--eval-studio",
        type=Path,
        default=None,
        help="студийная проверка новых классов (`bench/catalog-192-studio-preregistration.md`)",
    )
    парсер.add_argument(
        "--names", type=Path, default=Path("catalog/external/mad_cars_cat192_eval_new.csv")
    )
    парсер.add_argument("--negatives", type=Path, default=splits_dir() / "negatives.parquet")
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--report", type=Path, required=True)
    парсер.add_argument("--predictions", type=Path, default=None)
    парсер.add_argument("--batch-size", type=int, default=32)
    аргументы = парсер.parse_args(argv)

    манифест = json.loads((аргументы.models / "models_manifest.json").read_text("utf-8"))
    байты_меты = (аргументы.models / "catalog_meta.json").read_bytes()
    заголовок, центроиды = read_index(аргументы.models / "catalog_index.bin")
    verify_meta(заголовок, байты_меты)
    классы = json.loads(байты_меты)["classes"]
    пороги = json.loads((аргументы.models / "thresholds.json").read_text("utf-8"))
    классификатор = аргументы.models / манифест["files"]["classifier"]["name"]
    члены = member_index(load_catalog(DEFAULT_CATALOG_PATH))
    бренды = [класс["brand"] for класс in классы]
    class_ids = [класс["class_id"] for класс in классы]

    негативы = read_split(аргументы.negatives)
    э_нег = embed_crops(
        классификатор,
        [аргументы.crops_root / п for п in негативы["crop_path"]],
        манифест,
        аргументы.batch_size,
    )
    чужие = scores_from(*top2_with_sims(э_нег, центроиды), None, бренды)

    метрики: dict[str, SetMetrics] = {}
    худшие: dict[str, pd.DataFrame] = {}
    ошибки: dict[str, list[tuple[str, str, int]]] = {}
    по_названиям: pd.DataFrame | None = None
    ответы: list[pd.DataFrame] = []
    наборы = [("eval-new", аргументы.eval_new), ("eval-old", аргументы.eval_old)]
    if аргументы.eval_studio is not None:
        наборы.append(("eval-studio", аргументы.eval_studio))
    for имя, путь in наборы:
        кадры = read_split(путь).reset_index(drop=True)
        истина = truth_slots(кадры, члены, class_ids)
        if истина is None:
            print(f"{имя}: классов набора нет в индексе ассетов — пропуск")
            continue
        эмбеддинги = embed_crops(
            классификатор,
            [аргументы.crops_root / п for п in кадры["crop_path"]],
            манифест,
            аргументы.batch_size,
        )
        пары = top2_with_sims(эмбеддинги, центроиды)
        топ5 = np.array(ranked_by_centroids(эмбеддинги, центроиды, 5), dtype=np.int64)
        известные = scores_from(*пары, истина, бренды)
        метрики[имя] = set_metrics(
            top5=топ5,
            truth=истина,
            known=известные,
            negatives=чужие,
            mask=np.ones(len(кадры), dtype=bool),
            idx_to_class_id=dict(enumerate(class_ids)),
            idx_to_brand=dict(enumerate(бренды)),
            brands=member_brands([класс["members"] for класс in классы]),
            sim_min=float(пороги["sim_min"]),
            margin_min=float(пороги["margin_min"]),
        )
        попал = пары[0] == истина
        истинные = [class_ids[i] for i in истина]
        худшие[имя] = per_group(попал, истинные)
        ошибки[имя] = confusions(истинные, [class_ids[i] for i in пары[0]])
        if имя == "eval-new":
            машина_название = dict(
                pd.read_csv(аргументы.names, dtype=str)[["car_id", "mad_name"]]
                .drop_duplicates()
                .values
            )
            названия = [машина_название[str(л).rsplit(":", 1)[1]] for л in кадры["listing_id"]]
            по_названиям = per_group(попал, названия)
        решения = decide(известные, float(пороги["sim_min"]), float(пороги["margin_min"]))
        ответы.append(
            pd.DataFrame(
                {
                    "set": имя,
                    "photo_id": кадры["photo_id"],
                    "class_id": истинные,
                    "top1": [class_ids[i] for i in пары[0]],
                    "sim1": пары[1],
                    # Что покажет телефон в точке порогов: модель названа верно.
                    "shown_correct": (решения == MODEL) & известные.class_hit,
                }
            )
        )
    вердикт = (
        verdict(метрики["eval-new"], метрики["eval-old"])
        if {"eval-new", "eval-old"} <= метрики.keys()
        else None
    )
    write_generated_section(
        аргументы.report,
        render_report(
            metrics=метрики,
            by_name=по_названиям,
            worst_classes=худшие,
            confusion_lists=ошибки,
            provenance={
                "classifier_sha256": file_sha256(классификатор),
                "eval_new_sha256": file_sha256(аргументы.eval_new),
                "eval_old_sha256": file_sha256(аргументы.eval_old),
                **(
                    {"eval_studio_sha256": file_sha256(аргументы.eval_studio)}
                    if аргументы.eval_studio is not None
                    else {}
                ),
                "negatives_sha256": file_sha256(аргументы.negatives),
                "thresholds": f"{пороги['sim_min']} / {пороги['margin_min']}",
            },
            gate=вердикт,
        ),
    )
    if аргументы.predictions is not None:
        pd.concat(ответы).to_csv(аргументы.predictions, index=False)
    for имя, м in метрики.items():
        print(f"{имя}: n={м.n} top1={м.top1_model:.4f} top5={м.top5_model:.4f}")
    if вердикт is not None:
        print(f"условие (в)/(д): {вердикт}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
