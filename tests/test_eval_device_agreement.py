"""Согласие GPU и CPU: метрики, гейт и чтение файла эмбеддингов (M4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.schema import COLUMNS, write_split
from training.eval_device_agreement import (
    Embeddings,
    _rows,
    measure_agreement,
    read_embeddings,
    verdict,
)

ПОРОГИ = {"sim_min": 0.5, "margin_min": 0.0}
БРЕНДЫ: list[str | None] = ["a", "b", "c", "d"]


def _нормировать(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / np.linalg.norm(x, axis=1, keepdims=True), dtype=np.float32)


def _данные(gpu_шум: float, *, перемешать: bool = False) -> tuple[Embeddings, np.ndarray]:
    """Четыре ортогональных центроида; знакомые лежат у своего, негативы далеко от всех."""
    генератор = np.random.default_rng(0)
    одиночные = np.eye(4, 8, dtype=np.float32)
    центроиды = одиночные[:, None, :]  # (классы, центроиды, измерения)
    знакомые = _нормировать(
        np.repeat(одиночные, 25, axis=0) + 0.05 * генератор.standard_normal((100, 8))
    )
    негативы = _нормировать(
        np.eye(8, dtype=np.float32)[4:].repeat(20, axis=0)
        + 0.02 * генератор.standard_normal((80, 8))
    )
    cpu = np.vstack([знакомые, негативы])
    gpu = _нормировать(cpu + gpu_шум * генератор.standard_normal(cpu.shape))
    if перемешать:
        gpu = gpu[генератор.permutation(len(gpu))]
    данные = Embeddings(
        cpu=cpu,
        gpu=gpu,
        is_known=np.array([True] * 100 + [False] * 80),
        truth=np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25 + [-1] * 80),
        kinds=np.array([""] * 100 + ["off_catalog_street"] * 40 + ["truck_bus"] * 40),
    )
    return данные, центроиды


def test_согласие_идентичных_путей_проходит_гейт() -> None:
    данные, центроиды = _данные(0.0)
    результат = measure_agreement(данные, центроиды, БРЕНДЫ, ПОРОГИ)
    assert результат["top1_churn"] == 0
    assert результат["flips"] == 0
    assert verdict(результат)["passed"]


def test_малый_шум_gpu_не_меняет_решений() -> None:
    данные, центроиды = _данные(0.005)
    результат = measure_agreement(данные, центроиды, БРЕНДЫ, ПОРОГИ)
    assert verdict(результат)["passed"]
    assert результат["cosine"]["min"] > 0.999


def test_перемешанные_эмбеддинги_gpu_роняют_гейт() -> None:
    данные, центроиды = _данные(0.0, перемешать=True)
    результат = measure_agreement(данные, центроиды, БРЕНДЫ, ПОРОГИ)
    итог = verdict(результат)
    assert not итог["passed"]
    assert not итог["conditions"]["a_top1_churn"]["ok"]


def test_чтение_эмбеддингов_ловит_чужую_длину(tmp_path: Path) -> None:
    путь = tmp_path / "emb.f32"
    np.arange(12, dtype="<f4").tofile(путь)
    assert read_embeddings(путь, 3, 4).shape == (3, 4)
    with pytest.raises(ValueError, match="ожидалось 4×4"):
        read_embeddings(путь, 4, 4)


def _сплит(путь: Path, строки: list[tuple[str, int]]) -> Path:
    кадры = pd.DataFrame(
        [
            {
                "photo_id": photo_id,
                "listing_id": photo_id,
                "source": "mad_cars",
                "crop_path": f"{photo_id}.jpg",
                "brand": "Audi",
                "model_id": "audi__a3",
                "class_idx": класс,
                "body_style": None,
                "year": None,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "bbox_area_frac": 0.5,
                "det_conf": 0.9,
                "phash": 0,
                "raw_model_string": None,
                "is_multi_car": False,
            }
            for photo_id, класс in строки
        ],
        columns=list(COLUMNS),
    )
    write_split(кадры, путь)
    return путь


def test_доп_знакомые_идут_между_релевантными_и_негативами(tmp_path: Path) -> None:
    holdout = _сплит(tmp_path / "holdout.parquet", [("h1", 1), ("h2", 2)])
    новые = _сплит(tmp_path / "eval-new.parquet", [("n1", 7)])
    улица = _сплит(tmp_path / "street.parquet", [("s1", 0)])
    релевантность = tmp_path / "relevance.csv"
    релевантность.write_text(
        "photo_id,verdict,top_down\nh1,relevant,false\nh2,unsure,false\n", encoding="utf-8"
    )

    строки = _rows(holdout, релевантность, улица, [новые])

    assert [(с["part"], с["key"], с["class_idx"]) for с in строки] == [
        ("known", "h1", 1),
        ("known", "n1", 7),
        ("street", "s1", -1),
    ]
