"""Отказные прототипы 0.8.2: банк, прототипы, перекрёстная оценка по моделям, точка, гейт П1–П3.

Банк — три домена (студийные и уличные чужие машины, не-машины COCO).

Правила, пределы и гейт записаны до замера в `bench/reject-v2-preregistration.md`. Классификатор и
индекс каталога — 0.8.1, не переобучаются. Негатив оценивается прототипами, построенными без его
фолда моделей; знакомые — прототипами по всему банку (консервативно). Ассеты (прототипы, пороги,
шкала) пишутся только при пройденных П1–П3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.knn import top2_with_sims
from pipeline.paths import harvest_dir
from pipeline.pool_audit import file_sha256
from pipeline.reject_bank import FOREIGN, NON_CAR, bank_kind, fold_of, listing_kind, model_key
from pipeline.reject_index import (
    CENTROIDS_PER_KIND,
    REJECT_KINDS,
    build_reject_centroids,
    write_reject_index,
)
from pipeline.schema import read_split
from training.build_index import kmeans_factory
from training.calibration import (
    CLIP,
    MODEL,
    REJECT_GRID,
    Operating,
    Scores,
    auroc,
    calibration_doc,
    calibration_knots,
    decide,
    split_halves,
    thresholds_doc,
)
from training.eval_open_set import (
    NO_REJECT,
    OpenSetFold,
    OpenSetPoint,
    binding_constraint,
    cross_fit_open_set,
    load_model_assets,
    relevant_known,
    select_open_set_pair,
    split_by_group,
)
from training.tune_thresholds import (
    check_holdout_slots,
    embed_crops,
    scores_from,
    write_generated_section,
)

DEFAULT_SEED = 20261008
FOLDS = 5
POINT_MAX_FPR = 0.13  # предел точки на каждую группу: запас под шум подбора
GATE_MAX_FPR = 0.15  # предел П2 на отложенных половинах
GATE_GAIN = 0.02  # П1
EXPLORED_COVERAGE = 0.807  # разведка предрегистрации, П3
EXPLORED_PAIR = (0.68, 0.0, 0.04)
GROUPS = ("AE", "MAD", "NC")
CATALOG_NAME_FILES = ("mad_cars", "mad_cars_disputed", "mad_cars_cat236_names")


def catalog_mad_names(catalog_dir: Path) -> set[str]:
    """`бренд__модель` MAD-Cars, которые теперь классы каталога (точные, спорные, новые 28)."""
    имена: set[str] = set()
    for файл in CATALOG_NAME_FILES:
        таблица = pd.read_csv(catalog_dir / f"{файл}.csv")
        имена |= {
            f"{б}__{м}" for б, м in zip(таблица["mad_brand"], таблица["mad_model"], strict=True)
        }
    return имена


def drop_catalog_models(pool: pd.DataFrame, names: set[str]) -> pd.DataFrame:
    """Пул v1 собран до 236 классов: ставшие классами модели MAD отказывали бы своему классу."""
    улица = pool["listing_id"].str.startswith("off_catalog_street/")
    имя = pool["listing_id"].str.rsplit("/", n=1).str[-1]
    return pool[~(улица & имя.isin(names))].reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class Bank:
    """Банк прототипов: эмбеддинги кадров, их вид, ключ модели и фолд."""

    embeddings: np.ndarray
    kinds: np.ndarray
    keys: list[str]
    folds: np.ndarray
    sources: np.ndarray  # `ae`, `street`, `pool`

    def mask(self, kind: str, exclude_fold: int | None = None) -> np.ndarray:
        выбор = np.asarray(self.kinds == kind, dtype=bool)
        if exclude_fold is None:
            return выбор
        return np.asarray(выбор & (self.folds != exclude_fold), dtype=bool)


def make_bank(
    parts: Mapping[str, tuple[np.ndarray, pd.DataFrame]], seed: int, folds: int = FOLDS
) -> Bank:
    эмбеддинги, виды, ключи, источники = [], [], [], []
    for источник, (e, кадры) in parts.items():
        ids = list(кадры["listing_id"])
        эмбеддинги.append(e)
        виды += [bank_kind(i) for i in ids]
        ключи += [model_key(i) for i in ids]
        источники += [источник] * len(ids)
    return Bank(
        embeddings=np.concatenate(эмбеддинги),
        kinds=np.array(виды),
        keys=ключи,
        folds=np.array([fold_of(k, seed, folds) for k in ключи]),
        sources=np.array(источники),
    )


def build_prototypes(
    bank: Bank,
    *,
    seed: int,
    exclude_fold: int | None = None,
    per_kind: Mapping[str, int] = CENTROIDS_PER_KIND,
) -> tuple[np.ndarray, list[str]]:
    """Центроиды `(классы, 4, dim)` и виды по банку без фолда (None — весь банк)."""
    return build_reject_centroids(
        {вид: bank.embeddings[bank.mask(вид, exclude_fold)] for вид in REJECT_KINDS},
        kmeans=kmeans_factory(seed, n_init=1),
        centroids_per_kind=per_kind,
    )


def max_sim_flat(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    плоские = centroids.reshape(-1, centroids.shape[-1])
    return np.asarray((embeddings @ плоские.T).max(axis=1), dtype=np.float64)


def cross_fit_r(
    bank: Bank, indices: np.ndarray, prototypes: Mapping[int, np.ndarray]
) -> np.ndarray:
    """`r` кадров банка по прототипам без их фолда: каждый фолд — свой набор центроидов."""
    r = np.empty(len(indices))
    for f, центроиды in prototypes.items():
        выбор = bank.folds[indices] == f
        if выбор.any():
            r[выбор] = max_sim_flat(bank.embeddings[indices[выбор]], центроиды)
    return r


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def evaluate_gate(
    base_folds: Sequence[OpenSetFold],
    new_folds: Sequence[OpenSetFold],
    final: OpenSetPoint | None,
) -> list[GateCheck]:
    """П1–П3 предрегистрации."""
    проверки: list[GateCheck] = []
    приросты: list[str] = []
    верно_п1 = True
    for база, новая in zip(base_folds, new_folds, strict=True):
        if база.held_out is None or новая.held_out is None:
            верно_п1 = False
            приросты.append(f"{новая.tuned_on}: точки нет")
            continue
        прирост = новая.held_out.coverage - база.held_out.coverage
        верно_п1 = верно_п1 and прирост >= GATE_GAIN - 1e-12
        приросты.append(f"{новая.tuned_on}: {прирост:+.3f}")
    проверки.append(
        GateCheck("П1 прирост покрытия отложенных половин ≥ +0.02", верно_п1, ", ".join(приросты))
    )
    верно_п2 = True
    превышения: list[str] = []
    for новая in new_folds:
        if новая.held_out is None:
            верно_п2 = False
            continue
        for группа, fpr in новая.held_out.fpr.items():
            if fpr > GATE_MAX_FPR:
                верно_п2 = False
                превышения.append(f"{новая.tuned_on}/{группа} {fpr:.3f}")
    проверки.append(
        GateCheck(
            "П2 FPR групп на отложенных половинах ≤ 0.15",
            верно_п2,
            ", ".join(превышения) or "в пределе",
        )
    )
    if final is None:
        проверки.append(GateCheck("П3 точка воспроизводит разведку", False, "пары нет"))
    else:
        пара = (round(final.sim_min, 2), round(final.margin_min, 3), round(final.reject_margin, 2))
        верно_п3 = abs(final.coverage - EXPLORED_COVERAGE) <= 0.001 and пара == EXPLORED_PAIR
        проверки.append(
            GateCheck(
                "П3 точка воспроизводит разведку (0.807; 0.68/0.000/0.04)",
                верно_п3,
                f"{final.coverage:.3f}; {пара}",
            )
        )
    return проверки


def _точка_md(точка: OpenSetPoint | None) -> str:
    if точка is None:
        return "—"
    fpr = " / ".join(f"{g} {v:.3f}" for g, v in точка.fpr.items())
    return (
        f"{точка.sim_min:.2f} / {точка.margin_min:.3f} / {точка.reject_margin} → "
        f"покрытие {точка.coverage:.3f}, "
        f"ошибка {точка.error:.3f}, FPR {fpr}"
    )


def render_report(
    *,
    final_base: OpenSetPoint | None,
    final_new: OpenSetPoint | None,
    base_folds: Sequence[OpenSetFold],
    new_folds: Sequence[OpenSetFold],
    separability: Mapping[str, tuple[float, float]],
    checks: Sequence[GateCheck],
    counts: Mapping[str, int],
    origin: Mapping[str, str],
) -> str:
    строки = ["## Происхождение", ""]
    строки += [f"- {k}: `{v}`" for k, v in origin.items()]
    строки += ["", "## Выборки", ""] + [f"- {k}: {v}" for k, v in counts.items()]
    строки += [
        "",
        "## Точка и отложенные половины (sim_min / margin_min / reject_margin; FPR групп ≤ 0.13)",
        "",
        "| вариант | где | точка и результат |",
        "|---|---|---|",
        f"| без прототипов | итог | {_точка_md(final_base)} |",
        f"| с прототипами | итог | {_точка_md(final_new)} |",
    ]
    for вариант, фолды in (("без прототипов", base_folds), ("с прототипами", new_folds)):
        for ф in фолды:
            строки.append(f"| {вариант} | сторона {ф.tuned_on}, подбор | {_точка_md(ф.chosen)} |")
            строки.append(
                f"| {вариант} | сторона {ф.tuned_on}, отложенная | {_точка_md(ф.held_out)} |"
            )
    строки += [
        "",
        "## Разделимость",
        "",
        "| группа | AUROC (sim1) | AUROC (sim1 − r) |",
        "|---|---|---|",
    ]
    строки += [f"| {g} | {a:.3f} | {b:.3f} |" for g, (a, b) in separability.items()]
    строки += ["", "## Гейт П1–П3", ""]
    строки += [f"- {'✓' if п.passed else '✗'} {п.name}: {п.detail}" for п in checks]
    строки += ["", f"**Вердикт:** {'пройден' if all(п.passed for п in checks) else 'СТОП'}."]
    return "\n".join(строки)


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(
        description="Прототипы 0.8.2: сборка, перекрёстная оценка, точка, гейт П1–П3"
    )
    парсер.add_argument("--models", type=Path, required=True, help="ассеты 0.8.1 (fp32)")
    парсер.add_argument("--expected-sha", required=True, help="sha256 классификатора")
    парсер.add_argument("--splits", type=Path, required=True, help="сплиты cat236v2")
    парсер.add_argument("--pool", type=Path, required=True, help="reject-pool.parquet (v1)")
    парсер.add_argument(
        "--holdout-relevance", type=Path, default=Path("catalog/holdout_relevance.csv")
    )
    парсер.add_argument("--catalog-dir", type=Path, default=Path("catalog/external"))
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--cache-dir", type=Path, required=True, help="кэш эмбеддингов банка")
    парсер.add_argument("--seed", type=int, default=DEFAULT_SEED)
    парсер.add_argument("--report", type=Path, required=True)
    парсер.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="reject_index.bin, reject_meta.json, thresholds.json, calibration.json",
    )
    парсер.add_argument("--max-knots", type=int, default=32)
    аргументы = парсер.parse_args(argv)

    активы = load_model_assets("0.8.1", аргументы.models, аргументы.expected_sha)
    эталон = relevant_known(аргументы.splits / "holdout_ru.parquet", аргументы.holdout_relevance)
    check_holdout_slots(list(эталон["model_id"]), list(эталон["class_idx"]), активы.meta)
    ae = read_split(аргументы.splits / "negatives.parquet")
    улица = read_split(аргументы.splits / "negatives-street.parquet")
    пул = drop_catalog_models(read_split(аргументы.pool), catalog_mad_names(аргументы.catalog_dir))

    аргументы.cache_dir.mkdir(parents=True, exist_ok=True)

    def эмбеддинги(имя: str, кадры: pd.DataFrame) -> np.ndarray:
        кэш = аргументы.cache_dir / f"emb_{имя}.npy"
        if кэш.exists():
            return np.asarray(np.load(кэш), dtype=np.float32)
        e = embed_crops(
            активы.classifier,
            [аргументы.crops_root / п for п in кадры["crop_path"]],
            активы.manifest,
            32,
        )
        np.save(кэш, e)
        return e

    e_known, e_ae, e_street, e_pool = (
        эмбеддинги("known", эталон),
        эмбеддинги("neg", ae),
        эмбеддинги("street", улица),
        эмбеддинги("pool", пул),
    )
    банк = make_bank(
        {"ae": (e_ae, ae), "street": (e_street, улица), "pool": (e_pool, пул)}, аргументы.seed
    )

    полные, виды = build_prototypes(банк, seed=аргументы.seed)
    по_фолдам = {
        f: build_prototypes(банк, seed=аргументы.seed, exclude_fold=f)[0] for f in range(FOLDS)
    }

    истина = эталон["class_idx"].to_numpy(dtype=np.int64)
    знакомые = replace(
        scores_from(*top2_with_sims(e_known, активы.centroids), истина, активы.brands),
        sim_reject=max_sim_flat(e_known, полные),
    )
    база_знакомые = replace(знакомые, sim_reject=None)

    идеи = {
        "AE": np.where(банк.sources == "ae")[0],
        "MAD": np.where((банк.sources == "street") & (банк.kinds == FOREIGN))[0],
        "NC": np.where((банк.sources == "street") & (банк.kinds == NON_CAR))[0],
    }
    оц_ae = scores_from(*top2_with_sims(e_ae, активы.centroids), None, активы.brands)
    оц_улицы = scores_from(*top2_with_sims(e_street, активы.centroids), None, активы.brands)
    вид_улицы = np.array([listing_kind(i) for i in улица["listing_id"]])
    базовые = {
        "AE": оц_ae,
        "MAD": оц_улицы.subset(вид_улицы == "off_catalog_street"),
        "NC": оц_улицы.subset(np.isin(вид_улицы, ("truck_bus", "detector_fp"))),
    }

    группы: dict[str, Scores] = {}
    группы_база: dict[str, Scores] = {}
    половины: dict[str, np.ndarray] = {}
    for имя in GROUPS:
        r = cross_fit_r(банк, идеи[имя], по_фолдам)
        группы[имя] = replace(базовые[имя], sim_reject=r)
        группы_база[имя] = базовые[имя]
        половины[имя] = split_by_group(np.array([банк.keys[i] for i in идеи[имя]]), аргументы.seed)
    половина_знакомых = split_halves(истина, аргументы.seed)

    итог_база = select_open_set_pair(
        база_знакомые, группы_база, max_fpr=POINT_MAX_FPR, reject_grid=NO_REJECT
    )
    итог_новая = select_open_set_pair(
        знакомые, группы, max_fpr=POINT_MAX_FPR, reject_grid=REJECT_GRID
    )
    стороны_база = cross_fit_open_set(
        база_знакомые,
        группы_база,
        половина_знакомых,
        половины,
        max_fpr=POINT_MAX_FPR,
        reject_grid=NO_REJECT,
    )
    стороны_новая = cross_fit_open_set(
        знакомые,
        группы,
        половина_знакомых,
        половины,
        max_fpr=POINT_MAX_FPR,
        reject_grid=REJECT_GRID,
    )
    разрыв = знакомые.sim1 - знакомые.sim_reject
    разделимость = {
        г: (auroc(знакомые.sim1, s.sim1), auroc(разрыв, s.sim1 - s.sim_reject))
        for г, s in группы.items()
    }
    проверки = evaluate_gate(стороны_база, стороны_новая, итог_новая)
    счётчики = {
        "знакомые (релевантный набор)": len(эталон),
        "банк, всего": len(банк.embeddings),
        "банк, чужие машины": int((банк.kinds == FOREIGN).sum()),
        "банк, не-машины": int((банк.kinds == NON_CAR).sum()),
        "ключей моделей чужих": len(
            {k for k, v in zip(банк.keys, банк.kinds, strict=True) if v == FOREIGN}
        ),
        **{f"группа {г}": len(идеи[г]) for г in GROUPS},
    }
    происхождение = {
        "classifier_sha256": активы.classifier_sha256,
        "negatives_sha256": file_sha256(аргументы.splits / "negatives.parquet"),
        "negatives_street_sha256": file_sha256(аргументы.splits / "negatives-street.parquet"),
        "reject_pool_sha256": file_sha256(аргументы.pool),
        "holdout_sha256": file_sha256(аргументы.splits / "holdout_ru.parquet"),
        "seed": str(аргументы.seed),
        "folds": str(FOLDS),
        "k_по_видам": json.dumps(dict(CENTROIDS_PER_KIND)),
        "point_max_fpr": str(POINT_MAX_FPR),
    }
    write_generated_section(
        аргументы.report,
        render_report(
            final_base=итог_база,
            final_new=итог_новая,
            base_folds=стороны_база,
            new_folds=стороны_новая,
            separability=разделимость,
            checks=проверки,
            counts=счётчики,
            origin=происхождение,
        ),
    )
    print("гейт:", "пройден" if all(п.passed for п in проверки) else "СТОП")
    for п in проверки:
        print(f"  {'✓' if п.passed else '✗'} {п.name}: {п.detail}")
    if итог_новая is not None:
        print("узкое ограничение итога:", binding_constraint(итог_новая, max_fpr=POINT_MAX_FPR))
    if not all(п.passed for п in проверки) or итог_новая is None:
        return 0

    рабочая = Operating(
        sim_min=итог_новая.sim_min,
        margin_min=итог_новая.margin_min,
        coverage=итог_новая.coverage,
        error=итог_новая.error,
        fpr=max(итог_новая.fpr.values()),
        shown=итог_новая.shown,
        reject_margin=итог_новая.reject_margin,
    )
    показаны = decide(знакомые, рабочая.sim_min, рабочая.margin_min, рабочая.reject_margin) == MODEL
    узлы = calibration_knots(
        знакомые.sim1[показаны], знакомые.class_hit[показаны], рабочая.sim_min, аргументы.max_knots
    )
    аргументы.out_dir.mkdir(parents=True, exist_ok=True)
    write_reject_index(
        аргументы.out_dir,
        полные,
        виды,
        classifier_sha256=активы.classifier_sha256,
        pool_sha256=hashlib.sha256(
            "".join(
                происхождение[k] for k in sorted(происхождение) if k.endswith("_sha256")
            ).encode()
        ).hexdigest(),
        frames={вид: int((банк.kinds == вид).sum()) for вид in REJECT_KINDS},
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    n_neg = sum(len(идеи[г]) for г in GROUPS)
    for имя, документ in (
        ("thresholds.json", thresholds_doc(рабочая, len(эталон), n_neg)),
        ("calibration.json", calibration_doc(узлы, CLIP)),
    ):
        (аргументы.out_dir / имя).write_text(
            json.dumps(документ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"ассеты: {аргументы.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
