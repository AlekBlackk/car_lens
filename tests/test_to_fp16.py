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
