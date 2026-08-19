"""Экспорт `yolo11n.pt` в TFLite fp32 и выяснение контракта графа (§7 дизайна M-1.5).

Запускается в отдельной venv: ultralytics доустанавливает конвертер сам и прямо в то
окружение, где запущен, а venv обучения и venv экспорта собраны по `uv.lock` и держат
цепочку конвертации классификатора. Та же логика, по которой §13.4 вынес экспорт
классификатора в свою venv.

Замерено на 8.4.120 (`ultralytics/engine/exporter.py:571`): `format="tflite"` объявлен
устаревшим в 8.4.83 и переадресуется на `format="litert"`, то есть граф собирает
`litert_torch`, а не `tensorflow` с `onnx2tf`, как было в старых версиях. Значение
аргумента оставлено прежним: пин `ultralytics==8.4.120` держит и эту переадресацию, а
переход на `"litert"` стоит делать вместе со сдвигом пина, а не отдельно от него.

Лицензия ultralytics — AGPL-3.0, использование допустимо по Р-4, пока проект личный.
Результат — расходный артефакт прогона, в git не идёт (§15, Р-22).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from export.detector_contract import DetectorContract, contract_from_graph
from pipeline.paths import data_root

IMGSZ = 416


# Третья копия `sha256_of_file` в дереве, и именно копия, а не импорт соседней.
# `export/to_tflite.py:57` в этой venv не импортируется вовсе: он тянет `litert_torch`,
# timm и albumentations на уровне модуля, а группа `yolo` их не ставит (проверено импортом
# в собранной venv — `ModuleNotFoundError: No module named 'litert_torch'`).
# `training/build_index.py:140` импортировался бы: там импорты ленивые по Р-15. Но это
# модуль сборки индекса, хеш файла его контрактом не является, и экспорт детектора
# зависел бы от обучающего кода из-за семи строк. Общего дома у такого хеша в дереве нет
# (`pipeline/hashing.py` — про перцептивные хеши), а заводить его — не эта задача.
# Тот же критерий, по которому продублирован `_MIN_DENOMINATOR` в `export/to_tflite.py:44-49`.
def sha256_of_file(path: Path) -> str:
    хеш = hashlib.sha256()
    with path.open("rb") as файл:
        for кусок in iter(lambda: файл.read(1 << 20), b""):
            хеш.update(кусок)
    return хеш.hexdigest()


def probe_graph(interpreter: Any, imgsz: int) -> dict[str, Any]:
    """Прогнать серый кадр и снять диапазон координат бокса.

    Единицы координат из форм не выводятся: `[1, 84, N]` выглядит одинаково и для
    долей, и для пикселей. Ошибиться здесь — получить рамку размером в пиксель или
    в четыре экрана, причём без единого исключения.

    Кадр собирается по форме входа самого графа, а не по `imgsz`: раскладку выбирает
    конвертер, и `set_tensor` принимает только ту форму, что в графе. Аргумент остаётся
    в подписи ради симметрии с `contract_from_graph`, который сверяет с ним форму входа.
    """
    interpreter.allocate_tensors()
    вход = interpreter.get_input_details()[0]
    выход = interpreter.get_output_details()[0]
    форма = [int(значение) for значение in вход["shape"]]
    interpreter.set_tensor(вход["index"], np.full(форма, 0.5, dtype=np.float32))
    interpreter.invoke()
    предсказания = np.asarray(interpreter.get_tensor(выход["index"]))
    if not np.isfinite(предсказания).all():
        raise ValueError("в выходе детектора nan или inf: граф сконвертирован сломанным")
    координаты = (
        предсказания[0, :4, :]
        if предсказания.shape[1] < предсказания.shape[2]
        else предсказания[0, :, :4]
    )
    return {"max_coord": float(координаты.max()), "min_coord": float(координаты.min())}


def graph_details(interpreter: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Подробности входа и выхода в том виде, который ест `contract_from_graph`.

    Подробности не передаются в контракт как есть: под ключом `dtype` рантайм держит
    класс numpy-типа, а не строку, и `str(np.float32)` — это `<class 'numpy.float32'>`.
    Сверка такой строки с `"float32"` внутри контракта вечно ложна, то есть fp32-граф
    отвергается как квантованный — ровно на этом остановился первый настоящий прогон
    экспорта. Тем же `np.dtype(...).name` приводит границы `export/to_tflite.py:220`.
    """

    def описать(запись: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dtype": np.dtype(запись["dtype"]).name,
            "shape": [int(значение) for значение in запись["shape"]],
        }

    return (
        описать(interpreter.get_input_details()[0]),
        описать(interpreter.get_output_details()[0]),
    )


def build_report(
    *,
    weights_sha256: str,
    graph_sha256: str,
    ultralytics_version: str,
    contract: DetectorContract,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "weights_sha256": weights_sha256,
        "graph_sha256": graph_sha256,
        "ultralytics_version": ultralytics_version,
        "contract": contract.to_json(),
    }


def _экспортировать(weights: Path, imgsz: int, куда: Path) -> Path:
    """Сконвертировать веса и положить граф туда, где его ждут остальные шаги.

    ultralytics пишет результат рядом с весами и имя выбирает сам (`yolo11n.tflite` в
    корне репозитория), поэтому граф копируется в каталог данных: артефакт прогона
    обязан лежать там, а не в дереве репозитория (§15, Р-22).
    """
    from ultralytics import YOLO

    результат = YOLO(str(weights)).export(format="tflite", imgsz=imgsz)
    источник = Path(str(результат))
    куда.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(источник, куда)
    return куда


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Экспорт детектора в TFLite fp32")
    парсер.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    парсер.add_argument("--imgsz", type=int, default=IMGSZ)
    парсер.add_argument("--out", type=Path, default=None)
    return парсер.parse_args()


def main() -> None:
    аргументы = _parse_args()
    каталог = аргументы.out or (data_root() / "detector")
    граф = _экспортировать(аргументы.weights, аргументы.imgsz, каталог / "detector_fp32.tflite")

    import ultralytics
    from ai_edge_litert.interpreter import Interpreter

    интерпретатор = Interpreter(model_path=str(граф))
    зонд = probe_graph(интерпретатор, аргументы.imgsz)
    вход, выход = graph_details(интерпретатор)
    контракт = contract_from_graph(вход, выход, imgsz=аргументы.imgsz, probe=зонд)
    отчёт = build_report(
        weights_sha256=sha256_of_file(аргументы.weights),
        graph_sha256=sha256_of_file(граф),
        ultralytics_version=str(ultralytics.__version__),
        contract=контракт,
    )
    (каталог / "detector_report.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"детектор: {граф}\nконтракт: {json.dumps(контракт.to_json(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
