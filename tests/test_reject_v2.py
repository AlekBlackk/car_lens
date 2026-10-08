"""Банк, перекрёстный `r`, раскладки половин и гейт П1′–П3 прототипов 0.8.2 на синтетике."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.calibration import Scores
from training.eval_open_set import OpenSetPoint
from training.reject_v2 import (
    Bank,
    StudyRow,
    cross_fit_r,
    drop_catalog_models,
    evaluate_gate,
    make_bank,
    max_sim_flat,
    split_study,
)


def _точка(cov: float, sim: float = 0.71, margin: float = 0.0, rej: float = 0.04) -> OpenSetPoint:
    return OpenSetPoint(sim, margin, cov, 0.06, 100, {"AE": 0.1, "MAD": 0.1, "NC": 0.1}, rej)


def _замер(
    seed: int, база: float = 0.72, новая: float = 0.78, fpr: float = 0.10, сторона: str = "A"
) -> StudyRow:
    return StudyRow(seed, сторона, база, новая, {"AE": fpr, "MAD": fpr, "NC": fpr})


def test_из_пула_уходят_модели_ставшие_классами_каталога() -> None:
    пул = pd.DataFrame(
        {
            "listing_id": [
                "off_catalog_street/mad_cars/renault__scenic",
                "off_catalog_street/mad_cars/acura__rdx",
                "truck_bus/coco/renault__scenic",
            ],
            "crop_path": ["a", "b", "c"],
        }
    )
    итог = drop_catalog_models(пул, {"renault__scenic"})
    assert list(итог["listing_id"]) == [
        "off_catalog_street/mad_cars/acura__rdx",
        "truck_bus/coco/renault__scenic",
    ]


def test_банк_делит_виды_и_одна_машина_двух_доменов_в_одном_фолде() -> None:
    e = np.eye(4, dtype=np.float32)
    ae = pd.DataFrame(
        {"listing_id": ["off_catalog/autoevolution/2003_Opel_Signum"], "crop_path": ["x"]}
    )
    улица = pd.DataFrame(
        {
            "listing_id": [
                "off_catalog_street/mad_cars/opel__signum",
                "truck_bus/coco/1.jpg",
                "truck_bus/coco/2.jpg",
            ],
            "crop_path": ["y", "z", "w"],
        }
    )
    банк = make_bank({"ae": (e[:1], ae), "street": (e[1:], улица)}, seed=7, folds=5)
    assert list(банк.kinds) == ["foreign_car", "foreign_car", "non_car", "non_car"]
    assert банк.folds[0] == банк.folds[1]
    assert list(банк.sources) == ["ae", "street", "street", "street"]


def test_cross_fit_r_не_видит_свой_фолд() -> None:
    # Кадр 0 лежит в фолде 0 и совпадает с центроидом фолда 1; в своём фолде совпадения нет.
    банк = Bank(
        embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32),
        kinds=np.array(["foreign_car", "foreign_car"]),
        keys=["a|a", "b|b"],
        folds=np.array([0, 1]),
        sources=np.array(["ae", "ae"]),
    )
    прототипы = {
        0: np.array([[[0, 1]]], dtype=np.float32),  # фолд 0 оценивается без себя: остался кадр 1
        1: np.array([[[1, 0]]], dtype=np.float32),  # фолд 1 оценивается без себя: остался кадр 0
    }
    r = cross_fit_r(банк, np.array([0, 1]), прототипы)
    assert r.tolist() == [0.0, 0.0]
    assert max_sim_flat(банк.embeddings[:1], прототипы[1]).tolist() == [1.0]


def _оценки(sim: np.ndarray, reject: float | None, hit: bool) -> Scores:
    n = len(sim)
    return Scores(
        sim1=sim,
        sim2=np.full(n, 0.1),
        same_brand=np.zeros(n, dtype=bool),
        class_hit=np.full(n, hit),
        brand_hit=np.full(n, hit),
        sim_reject=None if reject is None else np.full(n, reject),
    )


def test_split_study_даёт_две_стороны_на_раскладку_и_прирост_от_прототипа() -> None:
    n = 40
    # Половина знакомых с sim 0.9, половина с 0.75; негативы все с 0.75. Без прототипа порог
    # обязан быть выше 0.75 и теряет вторую половину; с прототипом негативы отделяет `r`.
    sim_знакомых = np.where(np.arange(n) % 2 == 0, 0.9, 0.75)
    known = _оценки(sim_знакомых, reject=0.2, hit=True)
    groups = {"AE": _оценки(np.full(n, 0.75), reject=0.9, hit=False)}
    ключи = {"AE": np.array([f"бренд|модель{i}" for i in range(n)])}
    истина = np.arange(n) % 8
    строки = split_study(known, groups, ключи, истина, seeds=(1, 2), max_fpr=0.1)
    assert [(r.seed, r.side) for r in строки] == [(1, "A"), (1, "B"), (2, "A"), (2, "B")]
    assert all(r.new_coverage > r.base_coverage for r in строки)
    assert all(r.new_fpr["AE"] <= 0.1 for r in строки)


def test_гейт_проходит_при_приросте_пределе_и_верной_точке() -> None:
    замеры = [_замер(s, сторона=сторона) for s in range(20) for сторона in ("A", "B")]
    проверки = evaluate_gate(замеры, _точка(0.7981))
    assert [п.passed for п in проверки] == [True, True, True]


def test_гейт_ловит_малый_прирост_и_нулевой_минимум() -> None:
    замеры = [_замер(s, новая=0.74) for s in range(10)]  # +0.02 в среднем
    assert not evaluate_gate(замеры, _точка(0.798))[0].passed
    замеры = [_замер(s) for s in range(10)] + [_замер(99, база=0.80, новая=0.80)]
    assert not evaluate_gate(замеры, _точка(0.798))[0].passed


def test_гейт_ловит_высокий_средний_fpr_и_выброс() -> None:
    assert not evaluate_gate([_замер(s, fpr=0.145) for s in range(10)], _точка(0.798))[1].passed
    замеры = [_замер(s) for s in range(10)] + [_замер(99, fpr=0.18)]
    assert not evaluate_gate(замеры, _точка(0.798))[1].passed


def test_гейт_п3_ловит_другую_точку_и_отсутствие_пары() -> None:
    замеры = [_замер(s) for s in range(4)]
    assert not evaluate_gate(замеры, _точка(0.82))[2].passed
    assert not evaluate_gate(замеры, _точка(0.798, sim=0.68))[2].passed
    assert not evaluate_gate(замеры, None)[2].passed
