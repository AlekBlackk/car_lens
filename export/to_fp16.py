"""Веса классификатора и детектора в fp16 поверх готового fp32-графа (M4, бюджет размера §5.2).

Граф остаётся float32 на входе и выходе: веса лежат в файле половинной точностью и
разворачиваются в fp32 операцией `DEQUANTIZE` перед арифметикой. Контракт §4.5 не
меняется, индекс и пороги fp32-модели остаются в силе — поэтому и квантизатор
`FLOAT_CASTING`, а не INT8: у INT8 косинус эмбеддинга к fp32 держался на 0.976 в медиане
(`bench/int8-0.5-preregistration.md`), а веса, приведённые к fp16, расходятся с fp32 на
уровне округления.

Модуль работает с файлом, а не с чекпоинтом: PyTorch и калибровка не нужны, результат
детерминирован. Что fp16 не портит решения на настоящих кадрах, проверяет
`export.verify_parity` для классификатора и `export.detector_parity` для детектора, а не
самопроверка здесь: она на шумовых входах и ловит только сломанный граф.

Второй режим — `fold_dequantize`: из fp16-графа получается fp32-граф с теми же значениями весов.
Причина в скорости, а не в точности. `DEQUANTIZE` над fp16-весами XNNPACK на телефоне не
сворачивает и рвёт граф классификатора на 50 разделов (29 из 185 узлов у делегата): кадр стоит
198 мс против 105 у fp32 (`bench/latency.md`, 2026-09-30). Свёрнутый граф считается как fp32 и
даёт числа, которые уже прошли гейт fp16; выигрыш размера остаётся только в APK, где файл сжимается
(33.6 → 19.5 МБ), а не в самом файле.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

from export.to_tflite import ExportError, _проверить_контракт, runtime_details, sha256_of_file

FP32_NAME = "classifier_fp32.tflite"
FP16_NAME = "classifier_fp16.tflite"
DETECTOR_FP16_NAME = "detector_fp16.tflite"
FOLDED_NAME = "classifier_fp16w.tflite"

SELF_CHECK_SAMPLES = 16
SELF_CHECK_SEED = 1337
# Ниже этого косинуса fp16-граф к fp32 — уже не округление весов, а сломанная конвертация.
# Проба на классификаторе 0.5 (шумовые входы) дала минимум 0.9999999.
SELF_CHECK_COSINE_MIN = 0.9999

# Веса — fp16, вычисления — fp32, `DEQUANTIZE` явный: без него граф несёт fp16-тензоры,
# которых XNNPACK на устройстве не ждёт.
_РЕЦЕПТ: list[dict[str, Any]] = [
    {
        "regex": ".*",
        "operation": "*",
        "algorithm_key": "float_casting",
        "op_config": {
            "weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"},
            "compute_precision": "FLOAT",
            "explicit_dequantize": True,
        },
    }
]


def convert_fp16(src: Path, dst: Path) -> None:
    """Записать в `dst` копию графа `src` с весами в fp16."""
    средство = quantizer.Quantizer(str(src))
    средство.load_quantization_recipe(_РЕЦЕПТ)
    средство.quantize().export_model(str(dst), overwrite=True)


def fold_dequantize(src: Path, dst: Path) -> int:
    """Записать в `dst` копию fp16-графа `src`, где `DEQUANTIZE` заменён константами fp32.

    Значение каждой константы — fp16-вес, приведённый к fp32: арифметика графа не меняется,
    исчезают только узлы, мешающие делегату. Вернуть число свёрнутых узлов.
    """
    модель = tfl_flatbuffer_utils.read_model(src)
    свёрнуто = 0
    for подграф in модель.subgraphs:
        оставшиеся = []
        for операция in подграф.operators:
            код = модель.operatorCodes[операция.opcodeIndex]
            # Старые файлы кладут код операции только в устаревшее поле.
            if max(код.builtinCode, код.deprecatedBuiltinCode) != schema.BuiltinOperator.DEQUANTIZE:
                оставшиеся.append(операция)
                continue
            вход = подграф.tensors[операция.inputs[0]]
            выход = подграф.tensors[операция.outputs[0]]
            данные = модель.buffers[вход.buffer].data
            if (
                вход.type != schema.TensorType.FLOAT16
                or выход.type != schema.TensorType.FLOAT32
                or данные is None
            ):
                # Не константа fp16 → fp32: сворачивать нечего, узел остаётся как был.
                оставшиеся.append(операция)
                continue
            веса = np.frombuffer(bytes(данные), dtype=np.float16).astype(np.float32)
            новый = schema.BufferT()
            новый.data = np.frombuffer(веса.tobytes(), dtype=np.uint8)
            модель.buffers.append(новый)
            выход.buffer = len(модель.buffers) - 1
            модель.buffers[вход.buffer].data = None
            свёрнуто += 1
        подграф.operators = оставшиеся
    if свёрнуто == 0:
        raise ExportError(f"{src}: нет ни одного DEQUANTIZE над fp16-константой — это не fp16-граф")
    tfl_flatbuffer_utils.write_model(модель, dst)
    return свёрнуто


def _прогнать(path: Path, вход: np.ndarray) -> np.ndarray:
    """Выходы `.tflite` с float32-границами по одному примеру (batch зафиксирован)."""
    интерпретатор = Interpreter(model_path=str(path))
    интерпретатор.allocate_tensors()
    подробности_входа = интерпретатор.get_input_details()[0]
    подробности_выхода = интерпретатор.get_output_details()[0]
    результаты: list[np.ndarray] = []
    for пример in вход:
        интерпретатор.set_tensor(подробности_входа["index"], пример[None, ...])
        интерпретатор.invoke()
        результаты.append(интерпретатор.get_tensor(подробности_выхода["index"]).copy())
    собранные: np.ndarray = np.concatenate(результаты, axis=0).astype(np.float32)
    return собранные


def self_check(fp32_path: Path, fp16_path: Path) -> dict[str, float]:
    """Косинус fp16 к fp32 на шумовых входах: ловит сломанный граф, а не потерю точности.

    Выход любой формы сравнивается целиком, одним вектором на пример: эмбеддинг
    классификатора и карта `[каналы, якоря]` детектора проходят одним путём.
    """
    форма = [int(x) for x in Interpreter(model_path=str(fp32_path)).get_input_details()[0]["shape"]]
    вход = (
        np.random.default_rng(SELF_CHECK_SEED)
        .standard_normal((SELF_CHECK_SAMPLES, *форма[1:]))
        .astype(np.float32)
    )
    эталон = _прогнать(fp32_path, вход)
    полученное = _прогнать(fp16_path, вход)
    if эталон.shape != полученное.shape:
        raise ExportError(f"форма выхода fp16 {полученное.shape} не совпала с fp32 {эталон.shape}")
    эталон = эталон.reshape(len(эталон), -1)
    полученное = полученное.reshape(len(полученное), -1)
    if not (bool(np.isfinite(эталон).all()) and bool(np.isfinite(полученное).all())):
        raise ExportError("неконечные эмбеддинги: nan или inf в выходе графа")
    нормы = np.linalg.norm(эталон, axis=1) * np.linalg.norm(полученное, axis=1)
    косинусы = np.sum(эталон * полученное, axis=1) / np.maximum(нормы, 1e-6)
    минимум = float(косинусы.min())
    if минимум < SELF_CHECK_COSINE_MIN:
        raise ExportError(
            f"косинус fp16 к fp32 {минимум:.6f} ниже {SELF_CHECK_COSINE_MIN}: "
            "это не округление весов, конвертация сломана"
        )
    return {
        "samples": float(SELF_CHECK_SAMPLES),
        "cosine_min": минимум,
        "max_abs": float(np.abs(эталон - полученное).max()),
    }


def _versions() -> dict[str, str]:
    return {имя: importlib.metadata.version(имя) for имя in ("ai-edge-quantizer", "ai-edge-litert")}


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="fp16-веса поверх fp32-графа")
    режим = парсер.add_mutually_exclusive_group(required=True)
    режим.add_argument(
        "--export", type=Path, help="каталог экспорта классификатора с classifier_fp32.tflite"
    )
    режим.add_argument(
        "--detector", type=Path, help="detector_fp32.tflite с detector_report.json рядом"
    )
    режим.add_argument(
        "--fold",
        type=Path,
        help="каталог с classifier_fp16.tflite и fp16_report.json: свернуть DEQUANTIZE",
    )
    return парсер.parse_args()


def _записать(отчёт: dict[str, Any], куда: Path) -> None:
    куда.write_text(json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(отчёт, ensure_ascii=False, indent=2))


def _описание(путь: Path) -> dict[str, Any]:
    return {"name": путь.name, "sha256": sha256_of_file(путь), "bytes": путь.stat().st_size}


def _classifier(каталог: Path) -> None:
    исходный = каталог / FP32_NAME
    результат = каталог / FP16_NAME
    отчёт_экспорта = json.loads((каталог / "export_report.json").read_text("utf-8"))
    манифест = отчёт_экспорта["manifest"]
    if sha256_of_file(исходный) != отчёт_экспорта["models"][FP32_NAME]["sha256"]:
        raise ExportError(f"{исходный}: sha256 не совпал с export_report.json — файл подменён")

    convert_fp16(исходный, результат)
    подробности = runtime_details(результат)
    _проверить_контракт(
        подробности, результат, int(манифест["img_size"]), int(манифест["embedding_dim"])
    )
    _записать(
        {
            "source": _описание(исходный),
            "fp16": {**_описание(результат), "runtime": подробности},
            "self_check": self_check(исходный, результат),
            "versions": _versions(),
        },
        каталог / "fp16_report.json",
    )


def _fold(каталог: Path) -> None:
    исходный = каталог / FP16_NAME
    результат = каталог / FOLDED_NAME
    отчёт_fp16 = json.loads((каталог / "fp16_report.json").read_text("utf-8"))
    if sha256_of_file(исходный) != отчёт_fp16["fp16"]["sha256"]:
        raise ExportError(f"{исходный}: sha256 не совпал с fp16_report.json — файл подменён")
    отчёт_экспорта = json.loads((каталог / "export_report.json").read_text("utf-8"))
    манифест = отчёт_экспорта["manifest"]

    свёрнуто = fold_dequantize(исходный, результат)
    подробности = runtime_details(результат)
    _проверить_контракт(
        подробности, результат, int(манифест["img_size"]), int(манифест["embedding_dim"])
    )
    _записать(
        {
            "source": _описание(исходный),
            "folded": {**_описание(результат), "runtime": подробности, "folded_ops": свёрнуто},
            "self_check": self_check(исходный, результат),
            "versions": _versions(),
        },
        каталог / "fp16w_report.json",
    )


def _граница(интерпретатор: Interpreter, выход: bool) -> tuple[tuple[int, ...], str]:
    подробности = интерпретатор.get_output_details() if выход else интерпретатор.get_input_details()
    return tuple(int(x) for x in подробности[0]["shape"]), np.dtype(подробности[0]["dtype"]).name


def _detector(исходный: Path) -> None:
    """fp16-детектор рядом с fp32: та же форма входа и выхода, тот же контракт §7.1.

    Контракт детектора (`detector_report.json`) не пересчитывается: он описывает границы
    графа, а они у fp16-графа те же — это и проверяется ниже, а не предполагается.
    """
    отчёт_детектора = json.loads(исходный.with_name("detector_report.json").read_text("utf-8"))
    if sha256_of_file(исходный) != отчёт_детектора["graph_sha256"]:
        raise ExportError(f"{исходный}: sha256 не совпал с detector_report.json — файл подменён")
    результат = исходный.with_name(DETECTOR_FP16_NAME)

    convert_fp16(исходный, результат)
    эталон = Interpreter(model_path=str(исходный))
    полученный = Interpreter(model_path=str(результат))
    for выход in (False, True):
        if _граница(эталон, выход) != _граница(полученный, выход):
            raise ExportError(f"{результат}: границы графа разошлись с fp32")
    _записать(
        {
            "source": _описание(исходный),
            "fp16": _описание(результат),
            "self_check": self_check(исходный, результат),
            "versions": _versions(),
        },
        исходный.with_name("detector_fp16_report.json"),
    )


def main() -> None:
    аргументы = _parse_args()
    if аргументы.export is not None:
        _classifier(аргументы.export)
    elif аргументы.fold is not None:
        _fold(аргументы.fold)
    else:
        _detector(аргументы.detector)


if __name__ == "__main__":
    main()
