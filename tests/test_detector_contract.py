"""Контракт экспортированного детектора (§7.1 дизайна M-1.5)."""

from __future__ import annotations

from typing import Any

import pytest

from export.detector_contract import ContractError, contract_from_graph

ВХОД: dict[str, Any] = {"dtype": "float32", "shape": [1, 416, 416, 3]}
ВЫХОД: dict[str, Any] = {"dtype": "float32", "shape": [1, 84, 3549]}
ЗОНД: dict[str, Any] = {"max_coord": 0.98, "min_coord": 0.004}


def test_контракт_собирается_из_форм_и_зонда() -> None:
    контракт = contract_from_graph(ВХОД, ВЫХОД, imgsz=416, probe=ЗОНД)
    assert контракт.input_layout == "NHWC"
    assert контракт.output_layout == "CN"
    assert контракт.num_classes == 80
    assert контракт.coord_units == "normalized"
    assert контракт.vehicle_classes == (2, 5, 7)
    assert контракт.score_layout == "classes_only"


def test_координаты_в_пикселях_распознаются_по_зонду() -> None:
    контракт = contract_from_graph(
        ВХОД, ВЫХОД, imgsz=416, probe={"max_coord": 402.0, "min_coord": 1.5}
    )
    assert контракт.coord_units == "pixels"


def test_неоднозначный_зонд_роняет_с_обоими_числами() -> None:
    with pytest.raises(ContractError) as отказ:
        contract_from_graph(ВХОД, ВЫХОД, imgsz=416, probe={"max_coord": 3.0, "min_coord": 0.5})
    assert "3.0" in str(отказ.value)
    assert "416" in str(отказ.value)


def test_чужое_число_классов_роняет() -> None:
    with pytest.raises(ContractError, match="76"):
        contract_from_graph(
            ВХОД, {"dtype": "float32", "shape": [1, 80, 3549]}, imgsz=416, probe=ЗОНД
        )


def test_квантованная_граница_роняет() -> None:
    with pytest.raises(ContractError, match="int8"):
        contract_from_graph(
            {"dtype": "int8", "shape": [1, 416, 416, 3]}, ВЫХОД, imgsz=416, probe=ЗОНД
        )


def test_форма_входа_не_квадрат_роняет() -> None:
    with pytest.raises(ContractError, match="640"):
        contract_from_graph(
            {"dtype": "float32", "shape": [1, 416, 640, 3]}, ВЫХОД, imgsz=416, probe=ЗОНД
        )


def test_to_json_содержит_все_поля_контракта() -> None:
    словарь = contract_from_graph(ВХОД, ВЫХОД, imgsz=416, probe=ЗОНД).to_json()
    assert set(словарь) == {
        "imgsz",
        "input_dtype",
        "input_layout",
        "channel_order",
        "input_range",
        "letterbox_pad_value",
        "output_layout",
        "box_format",
        "coord_units",
        "num_classes",
        "vehicle_classes",
        "score_layout",
        "max_det",
    }
    assert словарь["vehicle_classes"] == [2, 5, 7]
