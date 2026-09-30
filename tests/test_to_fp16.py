"""fp16-веса поверх fp32-графа: размер, границы и близость эмбеддингов (M4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.export
# В venv разработки этих пакетов нет: без пропуска сбор модуля роняет весь прогон.
pytest.importorskip("litert_torch")
pytest.importorskip("ai_edge_quantizer")

ВХОД = (1, 3, 16, 16)
ВЫХОД = 64


def _игрушечный_fp32(путь: Path) -> None:
    """Линейный слой на 768×64 весов: достаточно, чтобы выигрыш от fp16 был виден в байтах."""
    import litert_torch
    import torch

    torch.manual_seed(0)
    модель = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, ВЫХОД)).eval()
    litert_torch.convert(модель, (torch.zeros(*ВХОД),)).export(str(путь))


def test_fp16_вдвое_меньше_и_близок_к_fp32(tmp_path: Path) -> None:
    from export.to_fp16 import convert_fp16, self_check

    fp32 = tmp_path / "classifier_fp32.tflite"
    fp16 = tmp_path / "classifier_fp16.tflite"
    _игрушечный_fp32(fp32)

    convert_fp16(fp32, fp16)

    assert fp16.stat().st_size < 0.6 * fp32.stat().st_size
    проверка = self_check(fp32, fp16)
    assert проверка["cosine_min"] > 0.9999


def test_границы_fp16_графа_остаются_float32(tmp_path: Path) -> None:
    from ai_edge_litert.interpreter import Interpreter

    from export.to_fp16 import convert_fp16

    fp32 = tmp_path / "classifier_fp32.tflite"
    fp16 = tmp_path / "classifier_fp16.tflite"
    _игрушечный_fp32(fp32)
    convert_fp16(fp32, fp16)

    интерпретатор = Interpreter(model_path=str(fp16))
    интерпретатор.allocate_tensors()
    assert интерпретатор.get_input_details()[0]["dtype"] == np.float32
    assert интерпретатор.get_output_details()[0]["dtype"] == np.float32


def test_самопроверка_сравнивает_многомерный_выход(tmp_path: Path) -> None:
    """Выход детектора — карта `[каналы, якоря]`, а не вектор: сравнение идёт целиком."""
    import litert_torch
    import torch

    from export.to_fp16 import convert_fp16, self_check

    torch.manual_seed(0)
    модель = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 8 * 20), torch.nn.Unflatten(1, (8, 20))
    ).eval()
    fp32 = tmp_path / "detector_fp32.tflite"
    fp16 = tmp_path / "detector_fp16.tflite"
    litert_torch.convert(модель, (torch.zeros(*ВХОД),)).export(str(fp32))
    convert_fp16(fp32, fp16)

    assert self_check(fp32, fp16)["cosine_min"] > 0.9999


def _число_dequantize(путь: Path) -> int:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    модель = tfl_flatbuffer_utils.read_model(путь)
    return sum(
        1
        for подграф in модель.subgraphs
        for операция in подграф.operators
        if max(
            модель.operatorCodes[операция.opcodeIndex].builtinCode,
            модель.operatorCodes[операция.opcodeIndex].deprecatedBuiltinCode,
        )
        == schema.BuiltinOperator.DEQUANTIZE
    )


def test_свёртка_убирает_dequantize_и_держит_значения_fp16(tmp_path: Path) -> None:
    from export.to_fp16 import convert_fp16, fold_dequantize, self_check

    fp32 = tmp_path / "classifier_fp32.tflite"
    fp16 = tmp_path / "classifier_fp16.tflite"
    свёрнутый = tmp_path / "classifier_fp16w.tflite"
    _игрушечный_fp32(fp32)
    convert_fp16(fp32, fp16)
    assert _число_dequantize(fp16) > 0

    свёрнуто = fold_dequantize(fp16, свёрнутый)

    assert свёрнуто > 0
    assert _число_dequantize(свёрнутый) == 0
    # Веса вернулись в fp32: файл размером с исходный, а не с fp16.
    assert свёрнутый.stat().st_size > 0.9 * fp32.stat().st_size
    assert self_check(fp16, свёрнутый)["cosine_min"] > 0.99999


def test_свёртка_отказывает_графу_без_fp16_весов(tmp_path: Path) -> None:
    from export.to_fp16 import fold_dequantize
    from export.to_tflite import ExportError

    fp32 = tmp_path / "classifier_fp32.tflite"
    _игрушечный_fp32(fp32)

    with pytest.raises(ExportError, match="не fp16-граф"):
        fold_dequantize(fp32, tmp_path / "out.tflite")
