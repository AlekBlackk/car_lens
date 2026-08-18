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


def _подробности(dtype: Any, quantization: tuple[float, int]) -> dict[str, Any]:
    """Запись `get_input_details()`, собранная руками: восемь ключей из §6 разведки.

    Типы ключей повторяются дословно: `dtype` — класс numpy-типа, а не строка, `shape` —
    `ndarray`, а не список. Подделка с `dtype='int8'` проверяла бы не тот контракт, что
    живёт в рантайме, и молча разошлась бы с ним в первую же правку.
    """
    return {
        "name": "serving_default_args_0",
        "index": 0,
        "shape": np.array([1, 3, 16, 16], dtype=np.int32),
        "shape_signature": np.array([1, 3, 16, 16], dtype=np.int32),
        "dtype": dtype,
        "quantization": quantization,
        "quantization_parameters": {
            "scales": np.array([], dtype=np.float32),
            "zero_points": np.array([], dtype=np.int32),
            "quantized_dimension": 0,
            "block_size": 0,
        },
        "sparsity_parameters": {},
    }


def _рантайм(входы: list[list[int]], выходы: list[list[int]]) -> dict[str, Any]:
    """Ответ `runtime_details`, собранный руками: `_проверить_контракт` ест именно его.

    Форма записи здесь беднее, чем у `_подробности`: `runtime_details` уже привёл
    `ndarray` к списку и класс типа к имени, и проверять надо то, что доходит до отчёта.
    """

    def описать(формы: list[list[int]]) -> list[dict[str, Any]]:
        return [
            {
                "name": f"tensor_{номер}",
                "shape": форма,
                "dtype": "float32",
                "quantization": {"scale": 0.0, "zero_point": 0},
            }
            for номер, форма in enumerate(формы)
        ]

    return {"input_details": описать(входы), "output_details": описать(выходы), "layout": "NCHW"}


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


def test_страж_ловит_неконечные_значения() -> None:
    """`nan` и `inf` обязаны быть отказом, а не числом в отчёте.

    С любым порогом `nan` сравнивается как False, то есть быстрая сверка на сломанном
    графе была бы fail-open, а `json.dumps` дописал бы в `export_report.json` литерал
    `NaN`, которого строгий парсер на M-2 не примет. Роль обязана попадать в сообщение:
    без неё непонятно, какой из трёх матриц или косинусов испорчен.
    """
    pytest.importorskip("litert_torch")

    from export.to_tflite import ExportError, _проверить_конечность

    # Здоровый случай молчит. `assert ... is None` тут невозможен: mypy запрещает читать
    # значение функции, объявленной как `-> None`.
    _проверить_конечность(np.array([[1.0, -2.5], [0.0, 3.0]]), "эталон")
    for значение in (np.nan, np.inf, -np.inf):
        with pytest.raises(ExportError, match="косинусы"):
            _проверить_конечность(np.array([1.0, значение]), "косинусы")


def test_параметры_границы_отвергают_чужой_dtype(tmp_path: Path) -> None:
    """Границей может быть float32 или int8, всё остальное — отказ.

    Конвертер не запускается: проверяется разбор `get_input_details()`, а не тулчейн.
    Последствия у третьего dtype асимметричны — на входе `set_tensor` упал бы сам по
    несовпадению типов, а на выходе `uint8` или `int16` тихо доехал бы сырыми кодами до
    косинусов как эмбеддинг.
    """
    pytest.importorskip("litert_torch")

    from export.to_tflite import ExportError, _параметры_границы

    путь = tmp_path / "toy.tflite"
    assert _параметры_границы(_подробности(np.float32, (0.0, 0)), "вход", путь) is None
    assert _параметры_границы(_подробности(np.int8, (0.5, -3)), "вход", путь) == (0.5, -3)
    with pytest.raises(ExportError, match="нулевым scale"):
        _параметры_границы(_подробности(np.int8, (0.0, 0)), "выход", путь)
    with pytest.raises(ExportError) as отказ:
        _параметры_границы(_подробности(np.uint8, (0.5, 128)), "выход", путь)
    # Путь и фактическое имя типа — в сообщении: иначе по падению не понять, какой из
    # двух артефактов прогона испорчен и чем именно.
    assert "uint8" in str(отказ.value)
    assert str(путь) in str(отказ.value)


def test_контракт_форм_ловит_чужой_граф(tmp_path: Path) -> None:
    """Формы графа против §4.5, включая число тензоров.

    `embedding_dim` в манифесте берётся из конфига чекпоинта, поэтому `check_provenance`
    сверит конфиг с индексом и расхождение самого графа не заметит: без этой проверки
    конвертер, отдавший другую размерность, оставил бы зелёный отчёт.
    """
    pytest.importorskip("litert_torch")

    from export.to_tflite import ExportError, _проверить_контракт

    путь = tmp_path / "toy.tflite"
    # Обе раскладки законны: порядок осей выбирает конвертер, а не мы.
    _проверить_контракт(_рантайм([[1, 3, 16, 16]], [[1, 8]]), путь, 16, 8)
    _проверить_контракт(_рантайм([[1, 16, 16, 3]], [[1, 8]]), путь, 16, 8)
    with pytest.raises(ExportError, match="320"):
        _проверить_контракт(_рантайм([[1, 3, 16, 16]], [[1, 8]]), путь, 320, 8)
    with pytest.raises(ExportError, match="embedding_dim"):
        _проверить_контракт(_рантайм([[1, 3, 16, 16]], [[1, 8]]), путь, 16, 256)
    with pytest.raises(ExportError, match="раскладок"):
        _проверить_контракт(_рантайм([[1, 1, 16, 16]], [[1, 8]]), путь, 16, 8)
    # Лишний выход — не безвредная деталь: `_run_tflite` жёстко берёт нулевой индекс, то
    # есть за эмбеддинг ушёл бы тензор, случайно оказавшийся первым.
    with pytest.raises(ExportError, match="выходов 2"):
        _проверить_контракт(_рантайм([[1, 3, 16, 16]], [[1, 8], [1, 8]]), путь, 16, 8)
    with pytest.raises(ExportError, match="входов 2"):
        _проверить_контракт(_рантайм([[1, 3, 16, 16], [1, 4]], [[1, 8]]), путь, 16, 8)
