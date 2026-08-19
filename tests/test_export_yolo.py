"""Экспорт детектора и отчёт о его контракте (§7 дизайна M-1.5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from export.detector_contract import contract_from_graph
from scripts.export_yolo import build_report, graph_details, probe_graph

ВХОД: dict[str, Any] = {"index": 0, "dtype": "float32", "shape": [1, 416, 416, 3]}
# Предсказаний 3549 — столько якорей у v11 на кадре 416 (52²+26²+13²), как в
# `tests/test_detector_contract.py`. Число обязано быть больше 84: раскладку `CN`
# `contract_from_graph` отличает от `NC` тем, что каналов меньше, чем предсказаний.
ВЫХОД: dict[str, Any] = {"index": 1, "dtype": "float32", "shape": [1, 84, 3549]}


class ПодставнойИнтерпретатор:
    """Отдаёт координаты в долях и одну ненулевую уверенность."""

    def __init__(self, координаты: float) -> None:
        self._координаты = координаты

    def allocate_tensors(self) -> None: ...

    def get_input_details(self) -> list[dict[str, Any]]:
        return [ВХОД]

    def get_output_details(self) -> list[dict[str, Any]]:
        return [ВЫХОД]

    def set_tensor(self, index: int, value: np.ndarray) -> None: ...

    def invoke(self) -> None: ...

    def get_tensor(self, index: int) -> np.ndarray:
        выход = np.zeros((1, 84, 3549), dtype=np.float32)
        выход[0, :4, :] = self._координаты
        выход[0, 4 + 2, 3] = 0.9
        return выход


def test_зонд_снимает_диапазон_координат() -> None:
    зонд = probe_graph(ПодставнойИнтерпретатор(0.75), imgsz=416)
    assert зонд["max_coord"] == pytest.approx(0.75)
    assert зонд["min_coord"] == pytest.approx(0.75)


def test_зонд_роняет_на_неконечных_значениях() -> None:
    class Сломанный(ПодставнойИнтерпретатор):
        def get_tensor(self, index: int) -> np.ndarray:
            выход = super().get_tensor(index)
            выход[0, 0, 0] = np.nan
            return выход

    with pytest.raises(ValueError, match="nan"):
        probe_graph(Сломанный(0.5), imgsz=416)


def test_подробности_графа_приводят_dtype_к_имени() -> None:
    """Настоящий рантайм кладёт под `dtype` класс numpy, а не строку.

    Первый настоящий прогон экспорта (Шаг 5 брифа) на этом и остановился: контракт
    сверяет `dtype` со строкой `"float32"`, а `str(np.float32)` — это
    `<class 'numpy.float32'>`, то есть fp32-граф был отвергнут как квантованный.
    """

    class Настоящий(ПодставнойИнтерпретатор):
        def get_input_details(self) -> list[dict[str, Any]]:
            return [{**ВХОД, "dtype": np.float32}]

        def get_output_details(self) -> list[dict[str, Any]]:
            return [{**ВЫХОД, "dtype": np.float32}]

    вход, выход = graph_details(Настоящий(0.75))
    assert вход["dtype"] == "float32"
    assert вход["shape"] == [1, 416, 416, 3]
    контракт = contract_from_graph(
        вход, выход, imgsz=416, probe={"max_coord": 0.9, "min_coord": 0.1}
    )
    assert контракт.input_dtype == "float32"


def test_отчёт_несёт_хеши_и_контракт_целиком() -> None:
    контракт = contract_from_graph(
        ВХОД, ВЫХОД, imgsz=416, probe={"max_coord": 0.9, "min_coord": 0.1}
    )
    отчёт = build_report(
        weights_sha256="aa" * 32,
        graph_sha256="bb" * 32,
        ultralytics_version="8.4.120",
        contract=контракт,
    )
    assert отчёт["weights_sha256"] == "aa" * 32
    assert отчёт["graph_sha256"] == "bb" * 32
    assert отчёт["ultralytics_version"] == "8.4.120"
    assert отчёт["contract"]["coord_units"] == "normalized"
    assert отчёт["contract"]["letterbox_pad_value"] == 114
    assert "generated_at" in отчёт


def test_отчёт_содержит_только_известные_ключи() -> None:
    контракт = contract_from_graph(
        ВХОД, ВЫХОД, imgsz=416, probe={"max_coord": 0.9, "min_coord": 0.1}
    )
    отчёт = build_report(
        weights_sha256="aa" * 32,
        graph_sha256="bb" * 32,
        ultralytics_version="8.4.120",
        contract=контракт,
    )
    assert set(отчёт) == {
        "generated_at",
        "weights_sha256",
        "graph_sha256",
        "ultralytics_version",
        "contract",
    }
