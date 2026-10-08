"""Банк, перекрёстный `r` и гейт П1–П3 отказных прототипов 0.8.2 на синтетике."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.calibration import REJECT_OFF
from training.eval_open_set import OpenSetFold, OpenSetPoint
from training.reject_v2 import (
    Bank,
    cross_fit_r,
    drop_catalog_models,
    evaluate_gate,
    make_bank,
    max_sim_flat,
)


def _точка(
    cov: float, fpr: float = 0.1, sim: float = 0.68, margin: float = 0.0, rej: float = 0.04
) -> OpenSetPoint:
    return OpenSetPoint(sim, margin, cov, 0.06, 100, {"AE": fpr, "MAD": fpr, "NC": fpr}, rej)


def _сторона(имя: str, cov_подбор: float, cov_отложенная: float, fpr: float = 0.1) -> OpenSetFold:
    return OpenSetFold(имя, _точка(cov_подбор), _точка(cov_отложенная, fpr))


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
        0: np.array(
            [[[0, 1]]], dtype=np.float32
        ),  # фолд 0 оценивается без себя: остался только кадр 1
        1: np.array(
            [[[1, 0]]], dtype=np.float32
        ),  # фолд 1 оценивается без себя: остался только кадр 0
    }
    r = cross_fit_r(банк, np.array([0, 1]), прототипы)
    assert r.tolist() == [0.0, 0.0]
    assert max_sim_flat(банк.embeddings[:1], прототипы[1]).tolist() == [1.0]


def test_гейт_проходит_при_приросте_и_пределе() -> None:
    база = [_сторона("A", 0.74, 0.72), _сторона("B", 0.75, 0.74)]
    новая = [_сторона("A", 0.80, 0.78), _сторона("B", 0.81, 0.80)]
    final = _точка(0.8071)
    проверки = evaluate_gate(база, новая, final)
    assert [п.passed for п in проверки] == [True, True, True]


def test_гейт_ловит_малый_прирост_превышение_fpr_и_чужую_точку() -> None:
    база = [_сторона("A", 0.74, 0.72), _сторона("B", 0.75, 0.74)]
    новая = [_сторона("A", 0.80, 0.73), _сторона("B", 0.81, 0.80, fpr=0.16)]
    проверки = evaluate_gate(база, новая, _точка(0.82, sim=0.70))
    assert [п.passed for п in проверки] == [False, False, False]


def test_гейт_без_точки_останавливает() -> None:
    база = [_сторона("A", 0.74, 0.72), OpenSetFold("B", None, None)]
    новая = [_сторона("A", 0.80, 0.78), _сторона("B", 0.81, 0.80)]
    проверки = evaluate_gate(база, новая, None)
    assert not проверки[0].passed
    assert not проверки[2].passed
    assert REJECT_OFF < 0
