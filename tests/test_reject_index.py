"""Отказные прототипы: раскладка по видам, мета в схеме каталога, сверка классификатора."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.index_format import IndexFormatError
from pipeline.reject_index import (
    REJECT_INDEX_NAME,
    REJECT_META_NAME,
    RejectIndexError,
    build_reject_centroids,
    read_reject_index,
    write_reject_index,
)

РАЗМЕРНОСТЬ = 8
ПО_32 = {"foreign_car": 32, "non_car": 32}


def _эмбеддинги(n: int, seed: int) -> np.ndarray:
    сырые = np.random.default_rng(seed).normal(size=(n, РАЗМЕРНОСТЬ)).astype(np.float32)
    return np.asarray(сырые / np.linalg.norm(сырые, axis=1, keepdims=True), dtype=np.float32)


def _первые(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
    """Подставной k-means: первые `k` точек — детерминированно и без sklearn."""
    return эмбеддинги[:k]


def _собрать() -> tuple[np.ndarray, list[str]]:
    return build_reject_centroids(
        {"foreign_car": _эмбеддинги(40, 1), "non_car": _эмбеддинги(33, 2)},
        kmeans=_первые,
        centroids_per_kind=ПО_32,
    )


def test_по_32_центроида_на_вид_упакованы_по_четыре() -> None:
    центроиды, виды = _собрать()
    assert центроиды.shape == (16, 4, РАЗМЕРНОСТЬ)
    assert центроиды.dtype == np.float32
    assert виды == ["foreign_car"] * 8 + ["non_car"] * 8
    assert np.allclose(np.linalg.norm(центроиды, axis=2), 1.0)


def test_мало_кадров_вида_роняет_а_не_добивает() -> None:
    with pytest.raises(RejectIndexError, match="non_car"):
        build_reject_centroids(
            {"foreign_car": _эмбеддинги(40, 1), "non_car": _эмбеддинги(31, 2)},
            kmeans=_первые,
            centroids_per_kind=ПО_32,
        )


def test_пустой_вид_роняет() -> None:
    with pytest.raises(RejectIndexError, match="foreign_car"):
        build_reject_centroids(
            {"non_car": _эмбеддинги(40, 2)}, kmeans=_первые, centroids_per_kind=ПО_32
        )


def test_свой_k_на_каждый_вид() -> None:
    центроиды, виды = build_reject_centroids(
        {"foreign_car": _эмбеддинги(40, 1), "non_car": _эмбеддинги(12, 2)},
        kmeans=_первые,
        centroids_per_kind={"foreign_car": 16, "non_car": 8},
    )
    assert центроиды.shape == (6, 4, РАЗМЕРНОСТЬ)
    assert виды == ["foreign_car"] * 4 + ["non_car"] * 2


def test_k_не_кратное_четырём_роняет() -> None:
    with pytest.raises(RejectIndexError, match="не делится"):
        build_reject_centroids(
            {"foreign_car": _эмбеддинги(40, 1), "non_car": _эмбеддинги(40, 2)},
            kmeans=_первые,
            centroids_per_kind={"foreign_car": 10, "non_car": 8},
        )


def _записать(каталог: Path, классификатор: str = "c" * 64) -> np.ndarray:
    центроиды, виды = _собрать()
    write_reject_index(
        каталог,
        центроиды,
        виды,
        classifier_sha256=классификатор,
        pool_sha256="p" * 64,
        frames={"foreign_car": 40, "non_car": 33},
        generated_at="2026-09-28T00:00:00+00:00",
    )
    return центроиды


def test_запись_и_чтение_сходятся(tmp_path: Path) -> None:
    центроиды = _записать(tmp_path)
    прочитанные = read_reject_index(tmp_path, "c" * 64)
    assert np.array_equal(прочитанные, центроиды)


def test_мета_в_схеме_каталога(tmp_path: Path) -> None:
    _записать(tmp_path)
    мета = json.loads((tmp_path / REJECT_META_NAME).read_text(encoding="utf-8"))
    assert мета["classifier_sha256"] == "c" * 64
    assert мета["centroids_per_class"] == 4
    assert мета["frames"] == {"foreign_car": 40, "non_car": 33}
    первый = мета["classes"][0]
    assert первый == {
        "class_id": "reject__foreign_car__00",
        "members": [],
        "brand": None,
        "display": "",
        "kind": "foreign_car",
    }
    assert мета["classes"][8]["kind"] == "non_car"


def test_чужой_классификатор_роняет(tmp_path: Path) -> None:
    _записать(tmp_path, классификатор="a" * 64)
    with pytest.raises(RejectIndexError, match="классификатор"):
        read_reject_index(tmp_path, "b" * 64)


def test_подменённая_мета_роняет_по_sha(tmp_path: Path) -> None:
    _записать(tmp_path)
    путь = tmp_path / REJECT_META_NAME
    путь.write_text(путь.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(IndexFormatError, match="sha256"):
        read_reject_index(tmp_path, "c" * 64)
    assert (tmp_path / REJECT_INDEX_NAME).is_file()
