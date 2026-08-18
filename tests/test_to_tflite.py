"""Тест цепочки конвертации на игрушечной модели.

Проверяется обёртка и работоспособность тулчейна, а не MobileNet: на настоящую
модель нужны веса, GPU и минуты. Тест помечен маркером `export` — в CI такого
окружения нет (Р-31).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.export


def _игрушка() -> Any:
    """Свёртка и линейная голова: конвертеру хватает, а весов и минут не нужно.

    Модель собирается функцией, а не на уровне модуля: `nn` нельзя импортировать до
    `importorskip`, иначе файл не собрался бы там, где torch не установлен. По той же
    причине `torch` внутри тестов импортируется отдельной строкой, а не берётся
    возвращённым значением `importorskip` — по переменной mypy не разбирает
    аннотации `Tensor` (тот же приём в `tests/test_build_index.py:60-72`).
    """
    from torch import Tensor, nn

    class Игрушка(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.head = nn.Linear(4, 8)

        def forward(self, картинка: Tensor) -> Tensor:
            признаки = self.conv(картинка).mean(dim=(2, 3))
            return self.head(признаки)

    модель: Any = Игрушка().eval()
    return модель


def test_fp32_повторяет_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("litert_torch")

    import torch

    from export.to_tflite import _convert_fp32, _run_tflite, runtime_details

    torch.manual_seed(0)
    модель = _игрушка()
    пример = torch.zeros(1, 3, 16, 16)
    путь = tmp_path / "toy.tflite"
    _convert_fp32(модель, пример, путь)
    assert путь.stat().st_size > 0

    подробности = runtime_details(путь)
    assert подробности["input_details"][0]["shape"] in ([1, 3, 16, 16], [1, 16, 16, 3])
    assert подробности["output_details"][0]["shape"] == [1, 8]

    вход = torch.randn(1, 3, 16, 16)
    с_torch = модель(вход).detach().numpy()
    с_tflite = _run_tflite(путь, вход.numpy())
    assert float(np.abs(с_torch - с_tflite).max()) <= 1e-3


def test_int8_отдаёт_деквантованный_выход(tmp_path: Path) -> None:
    """Границы INT8-графа int8, и скрыть это обязан `_run_tflite`, а не вызывающий.

    Task 7 и Task 8 оба считают по выходу этой функции косинусы к центроидам. Без
    деквантования они мерили бы сырые коды int8 — то есть шум — и выдали бы зелёный
    вердикт на сломанной модели, поэтому место закрыто тестом, а не докстрингом.
    """
    pytest.importorskip("torch")
    pytest.importorskip("litert_torch")

    import torch

    from export.to_tflite import _convert_int8, _run_tflite, runtime_details

    torch.manual_seed(0)
    модель = _игрушка()
    пример = torch.zeros(1, 3, 16, 16)
    калибровка = [torch.randn(1, 3, 16, 16) for _ in range(8)]
    путь = tmp_path / "toy_int8.tflite"
    _convert_int8(модель, пример, калибровка, путь)
    assert путь.stat().st_size > 0

    подробности = runtime_details(путь)
    assert подробности["input_details"][0]["dtype"] == "int8"
    assert подробности["output_details"][0]["dtype"] == "int8"
    assert подробности["output_details"][0]["quantization"]["scale"] > 0.0

    вход = torch.randn(1, 3, 16, 16)
    с_torch = модель(вход).detach().numpy()
    с_tflite = _run_tflite(путь, вход.numpy())
    assert с_tflite.dtype == np.float32
    # Сырые коды int8 лежат в пределах ±127, то есть на два порядка больше выхода
    # головы: порядок величины и есть проверка того, что деквантование случилось.
    assert float(np.abs(с_tflite).max()) <= 4.0 * float(np.abs(с_torch).max())
    # Квантизация — потеря точности, а не поломка, поэтому порог свободнее fp32-го;
    # сырые коды в него не пролезли бы даже близко.
    assert float(np.abs(с_torch - с_tflite).max()) <= 0.05
