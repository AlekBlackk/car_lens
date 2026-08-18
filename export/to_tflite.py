"""Экспорт классификатора в LiteRT: fp32-эталон и INT8 для устройства (M-1.4).

Голова ArcFace не экспортируется: на устройстве работает kNN по `catalog_index.bin`.

Границы INT8-графа остались int8 — это отступление от Р-26 и от контракта §4.5,
которые требуют float32 на входе и выходе при int8 внутри. Причина: оба проверенных
пути конвертации (PT2E через `litert_torch` и `ai_edge_quantizer` поверх готового
fp32) отдают int8 на границах, а обёртки, которая вернула бы float32-границы, в
тулчейне нет. Отступление компенсируется, а не замалчивается: вся арифметика
квантования собрана в одной функции `_run_tflite`, а `scale` и `zero_point` каждой
границы попадают в `export_report.json` через `runtime_details`, то есть `Classifier.kt`
на M-2 возьмёт числа из манифеста, а не из константы в коде.

Модуль тонкий: отбор калибровки живёт в `export/calibration.py`, метрики — в
`export/parity.py`. Здесь — загрузка весов, вызовы конвертера и запись отчёта.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import litert_torch
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from torch import Tensor

from export.calibration import Calibration, select_calibration
from pipeline.paths import crops_dir, splits_dir
from training.dataset import PREPROCESS_VERSION, build_transforms
from training.model import create_model
from training.samples import read_samples

SELF_CHECK_SAMPLES = 32

# Знаменатель косинусов прижимается снизу тем же порогом, что `_MIN_DENOMINATOR` в
# `export/parity.py`: эмбеддинг выходит из `BatchNorm1d` и содержит значения около
# нуля, а `nan` от деления на ноль сравнивается с любым порогом как False — сломанный
# граф прошёл бы самопроверку молча. Константа продублирована, а не импортирована:
# приватное имя соседнего модуля контрактом не является, а значение обязано совпадать,
# иначе быстрая сверка и полный паритет ответят по-разному на один и тот же ноль.
_MIN_DENOMINATOR = 1e-6


class ExportError(RuntimeError):
    """Экспорт невозможен."""


def sha256_of_file(path: Path) -> str:
    """sha256 файла в hex."""
    хеш = hashlib.sha256()
    with path.open("rb") as поток:
        for блок in iter(lambda: поток.read(1 << 20), b""):
            хеш.update(блок)
    return хеш.hexdigest()


def calibration_tensors(calibration: Calibration, transform: Any, root: Path) -> list[Tensor]:
    """Кропы калибровки как тензоры `(1, 3, H, W)` — тем же препроцессингом, что инференс.

    Декодер и порядок каналов повторяют `CropDataset.__getitem__`: `cv2.imread` плюс
    `COLOR_BGR2RGB`. Выбор декодера JPEG в этом проекте — предмет пина
    (`pyproject.toml:8`), и калибровка обязана видеть те же пиксели, что видели
    обучение и сборка индекса, иначе шкалы квантования считаются не по тем данным.
    """
    тензоры: list[Tensor] = []
    for образец in calibration.samples:
        # `read_samples` отдаёт абсолютные пути, но ветка остаётся: относительный путь
        # раскрылся бы от рабочего каталога, а он у экспорта не тот же, что у обучения.
        путь = root / образец.path if not образец.path.is_absolute() else образец.path
        картинка = cv2.imread(str(путь), cv2.IMREAD_COLOR)
        if картинка is None:
            raise ExportError(f"не читается кроп {путь}")
        картинка = cv2.cvtColor(картинка, cv2.COLOR_BGR2RGB)
        тензор: Tensor = transform(image=картинка)["image"]
        тензоры.append(тензор.unsqueeze(0))
    if not тензоры:
        raise ExportError("калибровочная выборка пуста")
    return тензоры


def _convert_fp32(model: torch.nn.Module, example: Tensor, path: Path) -> None:
    """Конвертация без квантизации. Имена API — из фактов разведки Task 1 (§2, §3)."""
    преобразованная = litert_torch.convert(model.eval(), (example,))
    преобразованная.export(str(path))


def _convert_int8(
    model: torch.nn.Module, example: Tensor, tensors: list[Tensor], path: Path
) -> None:
    """PTQ по representative dataset: веса по каналам, активации по тензору.

    Калибровочная выборка передаётся не аргументом, а прогоном подготовленного
    модуля: наблюдатели PT2E собирают статистику сами, и без прогона scale вышел бы
    скаляром вместо вектора по каналам. Границы графа получаются int8 — см. докстринг
    модуля.
    """
    from litert_torch.quantize import pt2e_quantizer, quant_config
    from torchao.quantization.pt2e import quantize_pt2e

    if not tensors:
        raise ExportError("калибровочная выборка пуста: квантовать не по чему")
    квантайзер = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    # `torch.export.export`, а не `export_for_training`: последнего в torch 2.12 нет.
    # `.eval()` до захвата обязателен — иначе в граф попала бы train-ветка BatchNorm,
    # которая на калибровочном батче из одного примера падает.
    графовый = torch.export.export(model.eval(), (example,)).module()
    подготовленная = quantize_pt2e.prepare_pt2e(графовый, квантайзер)
    with torch.no_grad():
        for тензор in tensors:
            подготовленная(тензор)
    сконвертированная = quantize_pt2e.convert_pt2e(подготовленная, fold_quantize=False)
    преобразованная = litert_torch.convert(
        сконвертированная,
        (example,),
        quant_config=quant_config.QuantConfig(pt2e_quantizer=квантайзер),
    )
    преобразованная.export(str(path))


def _параметры_границы(
    подробности: dict[str, Any], роль: str, path: Path
) -> tuple[float, int] | None:
    """`(scale, zero_point)` границы графа, если она int8, и `None`, если float32.

    Сравнение идёт по `is np.int8`: под ключом `dtype` интерпретатор держит класс
    numpy-типа, а не строку, и сравнение со строкой было бы вечно ложным — то есть
    квантование молча не выполнялось бы вовсе.

    Любой третий dtype — отказ, а не `None`: последствия асимметричны. На входе
    `set_tensor` упал бы громко по несовпадению типов, а на выходе `uint8` или `int16`
    (другой квантайзер, следующий релиз конвертера) тихо доехал бы сырыми кодами до
    косинусов как эмбеддинг — ровно тот сценарий, от которого модуль защищается для
    int8. Незнакомый dtype означает, что арифметику границы здесь считать нечем.
    """
    if подробности["dtype"] is np.float32:
        return None
    if подробности["dtype"] is not np.int8:
        raise ExportError(
            f"{path}: {роль} имеет dtype {np.dtype(подробности['dtype']).name}, "
            "а поддерживаются только float32 и int8"
        )
    масштаб, ноль = подробности["quantization"]
    if not np.isfinite(масштаб) or float(масштаб) == 0.0:
        # Делить на этот ноль нельзя: `inf` на входе интерпретатора превратится в
        # случайные коды, и модель ответит числами, по которым поломки не видно.
        # Конечность прижата тем же условием, потому что `nan == 0.0` — это False:
        # неконечный scale прошёл бы дальше, деление дало бы `nan`, но `astype(np.int8)`
        # превращает его в мусорный код, а не в `nan`, — то есть страж конечности
        # промолчал бы, а `json.dumps` дописал бы в отчёт литерал `NaN` в
        # `runtime.*_details[0].quantization.scale`.
        raise ExportError(
            f"{path}: {роль} int8 со scale {масштаб}: параметры квантования не попали в файл"
        )
    return float(масштаб), int(ноль)


def _run_tflite(path: Path, batch: np.ndarray) -> np.ndarray:
    """Прогнать батч через `.tflite` по одному примеру: batch зафиксирован единицей.

    Контракт снаружи один и тот же для fp32 и INT8: на входе float32 NCHW, на выходе
    float32. Квантование входа и деквантование выхода — здесь, потому что границы
    INT8-графа int8 (см. докстринг модуля), а вызывающие считают по выходу косинусы и
    не заметили бы подмены эмбеддинга сырыми кодами.
    """
    интерпретатор = Interpreter(model_path=str(path))
    интерпретатор.allocate_tensors()
    вход = интерпретатор.get_input_details()[0]
    выход = интерпретатор.get_output_details()[0]
    квант_входа = _параметры_границы(вход, "вход", path)
    квант_выхода = _параметры_границы(выход, "выход", path)
    ожидаемая_форма = list(вход["shape"])
    результаты: list[np.ndarray] = []
    for пример in batch:
        подготовленный = пример[None, ...].astype(np.float32)
        # Конвертер сам решает раскладку: NCHW у torch, NHWC у большинства сборок
        # LiteRT. Ориентируемся на форму входа модели, а не на ожидание.
        if ожидаемая_форма[1] != подготовленный.shape[1]:
            подготовленный = np.transpose(подготовленный, (0, 2, 3, 1))
        if квант_входа is not None:
            масштаб, ноль = квант_входа
            подготовленный = np.clip(np.round(подготовленный / масштаб) + ноль, -128, 127).astype(
                np.int8
            )
        интерпретатор.set_tensor(вход["index"], подготовленный)
        интерпретатор.invoke()
        # Индекс выхода — из `get_output_details()`: у двух путей квантизации он разный.
        сырой = интерпретатор.get_tensor(выход["index"]).astype(np.float32)
        if квант_выхода is None:
            результаты.append(сырой)
            continue
        масштаб, ноль = квант_выхода
        результаты.append((сырой - ноль) * масштаб)
    return np.concatenate(результаты, axis=0)


def runtime_details(path: Path) -> dict[str, Any]:
    """Дословные `input_details`/`output_details` плюс выведенная раскладка."""
    интерпретатор = Interpreter(model_path=str(path))
    интерпретатор.allocate_tensors()

    def описать(подробности: list[dict[str, Any]]) -> list[dict[str, Any]]:
        описанные = []
        for запись in подробности:
            квантизация = запись.get("quantization", (0.0, 0))
            описанные.append(
                {
                    "name": str(запись["name"]),
                    "shape": [int(значение) for значение in запись["shape"]],
                    "dtype": np.dtype(запись["dtype"]).name,
                    "quantization": {
                        "scale": float(квантизация[0]),
                        "zero_point": int(квантизация[1]),
                    },
                }
            )
        return описанные

    входы = описать(интерпретатор.get_input_details())
    выходы = описать(интерпретатор.get_output_details())
    форма = входы[0]["shape"]
    раскладка = "NCHW" if len(форма) == 4 and форма[1] in (1, 3) else "NHWC"
    return {"input_details": входы, "output_details": выходы, "layout": раскладка}


def _проверить_контракт(runtime: dict[str, Any], path: Path, img_size: int, dim: int) -> None:
    """Формы графа против §4.5: один вход `1×3×img_size×img_size`, один выход `1×dim`.

    Записать формы в отчёт недостаточно: `embedding_dim` в манифесте берётся из конфига
    чекпоинта, а не из графа, поэтому `check_provenance` сверит конфиг с индексом и
    расхождение самого графа не заметит — конвертер, отдавший другую размерность выхода,
    оставил бы зелёный отчёт до самого устройства.

    Раскладку выбирает конвертер, поэтому проверяется состав осей, а не их порядок:
    NCHW даёт `[1, 3, S, S]`, NHWC — `[1, S, S, 3]`, и обе законны.

    Число тензоров проверяется отдельно от их форм: `_run_tflite` жёстко берёт нулевой
    индекс в обе стороны, и у многовыходного графа мы молча мерили бы тот тензор, который
    случайно оказался нулевым, — а §4.5 объявляет ровно один вход и ровно один выход.

    Здесь же дожимается dtype границ. Единственный другой его читатель — `_run_tflite`,
    но с `--skip-int8` `self_check` не вызывается вовсе, и fp32-граф получил бы сверку
    форм без сверки типов: граф с нефлотовой границей оставил бы зелёный отчёт без
    диагноза. Эта функция исполняется для fp32-пути в обоих режимах.
    """
    входы = runtime["input_details"]
    выходы = runtime["output_details"]
    if len(входы) != 1 or len(выходы) != 1:
        raise ExportError(
            f"{path}: входов {len(входы)}, выходов {len(выходы)}, а контракт §4.5 требует "
            "по одному: нулевой индекс в `_run_tflite` иначе означает случайный тензор"
        )
    форма_входа = [int(значение) for значение in входы[0]["shape"]]
    форма_выхода = [int(значение) for значение in выходы[0]["shape"]]
    допустимые = ([1, 3, img_size, img_size], [1, img_size, img_size, 3])
    if форма_входа not in допустимые:
        raise ExportError(
            f"{path}: вход {форма_входа}, а контракт §4.5 требует 1×3×{img_size}×{img_size} "
            "в любой из двух раскладок"
        )
    if форма_выхода != [1, dim]:
        raise ExportError(f"{path}: выход {форма_выхода}, а embedding_dim чекпоинта равен {dim}")
    for запись, роль in ((входы[0], "вход"), (выходы[0], "выход")):
        # Результат выбрасывается: нужен только отказ, а правило про допустимые dtype
        # остаётся в одной функции вместо двух копий условия. `runtime_details` уже привёл
        # класс типа к имени, а кортеж квантизации — к словарю, поэтому запись собирается
        # обратно: `np.dtype(имя).type` даёт тот самый класс, по которому идёт сравнение.
        квантизация = запись["quantization"]
        _параметры_границы(
            {
                "dtype": np.dtype(запись["dtype"]).type,
                "quantization": (квантизация["scale"], квантизация["zero_point"]),
            },
            роль,
            path,
        )


def _проверить_конечность(значения: np.ndarray, роль: str) -> None:
    """Страж `nan`/`inf` и пустоты. Тот же дефект чинили в Task 4 — `export/parity.py:57-64`.

    Прижатый знаменатель закрывает только 0/0; `nan`, пришедший из самого графа, дошёл
    бы до вывода как есть, а последствие двойное. С любым порогом `nan` сравнивается как
    False, то есть сломанный граф прошёл бы быструю сверку молча. И `json.dumps` пишет в
    `export_report.json` нестандартный литерал `NaN`: Python его прочтёт, а строгий
    парсер на M-2 (`kotlinx.serialization`) — нет, то есть отчёт станет нечитаемым ровно
    в том прогоне, где он нужнее всего.
    """
    # Страж парный, как у процитированного соседа, и по той же причине: `size`, а не число
    # строк (форма `(n, 0)` — тоже пустая выборка), потому что `np.isfinite(пустой).all()`
    # — это `True`, а `np.median([])` вернул бы в отчёт тот же `nan`. Сегодня недостижимо:
    # пустую выборку роняют и `calibration_tensors`, и `torch.cat`.
    if значения.size == 0:
        raise ExportError(f"{роль}: выборка пуста — сверять нечего")
    if not bool(np.isfinite(значения).all()):
        raise ExportError(f"{роль}: неконечные значения (nan или inf) — сверка неопределена")


def self_check(
    model: torch.nn.Module, fp32_path: Path, int8_path: Path, tensors: list[Tensor]
) -> dict[str, float]:
    """Быстрая сверка на 32 кропах прямо после конвертации.

    Полный паритет считает `verify_parity.py` на двух сплитах, но узнать о сломанном
    графе через сорок минут — дорого. Здесь тот же вопрос за секунды.

    Числа заведомо оптимистичны и вердиктом §4.6 не являются: они мерятся на тех же
    кропах, по которым считались шкалы квантования, то есть в самых выгодных для INT8
    условиях. Честный паритет на val и test — задача `verify_parity.py`.
    """
    # Равномерно по всей выборке, а не первые 32 подряд: `select_calibration` набирает её
    # классами по возрастанию индекса, по три кропа на класс, так что срез с начала накрыл
    # бы около одиннадцати классов из ста семидесяти одного. `linspace` берётся вместо
    # шага среза потому, что флор-деление отрезает хвост: при 512 кропах последним взятым
    # оказывался 496-й, то есть последние пять классов не попадали в сверку никогда.
    # Количество прижато длиной выборки — иначе на короткой выборке индексы задублируются.
    количество = min(SELF_CHECK_SAMPLES, len(tensors))
    индексы = np.linspace(0, len(tensors) - 1, количество).astype(int)
    выборка = torch.cat([tensors[int(номер)] for номер in индексы], dim=0)
    with torch.no_grad():
        эталон = model.eval()(выборка).float().cpu().numpy()
    fp32 = _run_tflite(fp32_path, выборка.numpy())
    int8 = _run_tflite(int8_path, выборка.numpy())
    _проверить_конечность(эталон, "эталон PyTorch")
    _проверить_конечность(fp32, "выход fp32-TFLite")
    _проверить_конечность(int8, "выход INT8-TFLite")
    нормы_эталона = np.linalg.norm(эталон, axis=1)
    нормы_int8 = np.linalg.norm(int8, axis=1)
    знаменатель = np.maximum(нормы_эталона * нормы_int8, _MIN_DENOMINATOR)
    косинусы = np.sum(эталон * int8, axis=1) / знаменатель
    # Отдельно от матриц: произведение конечных норм переполняет float32 в `inf`, и
    # медиана косинусов уехала бы в отчёт неконечной при трёх чистых матрицах.
    _проверить_конечность(косинусы, "косинусы INT8")
    return {
        "samples": float(len(выборка)),
        "fp32_max_abs": float(np.abs(эталон - fp32).max()),
        "int8_max_abs": float(np.abs(эталон - int8).max()),
        "int8_cos_median": float(np.median(косинусы)),
    }


def _versions() -> dict[str, str]:
    # `torchao` в списке потому, что PT2E-квантизация берётся оттуда напрямую: без его
    # версии INT8-числа отчёта не воспроизводятся.
    пакеты = ("torch", "litert-torch", "ai-edge-litert", "torchao", "numpy", "timm")
    версии = {"python": platform.python_version()}
    for имя in пакеты:
        try:
            версии[имя] = importlib.metadata.version(имя)
        except importlib.metadata.PackageNotFoundError:
            версии[имя] = "отсутствует"
    return версии


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Экспорт классификатора в TFLite (§4.5)")
    парсер.add_argument("--checkpoint", type=Path, required=True)
    парсер.add_argument("--out", type=Path, default=None)
    парсер.add_argument("--weights", choices=("raw", "ema"), default="raw")
    парсер.add_argument("--total", type=int, default=512)
    парсер.add_argument("--seed", type=int, default=1337)
    парсер.add_argument("--skip-int8", action="store_true")
    return парсер.parse_args()


def main() -> None:
    аргументы = _parse_args()
    чекпоинт: dict[str, Any] = torch.load(
        аргументы.checkpoint, map_location="cpu", weights_only=True
    )
    индекс: dict[str, int] = чекпоинт["class_index"]
    конфиг = чекпоинт["config"]
    модель, _ = create_model(
        конфиг["backbone"],
        num_classes=len(индекс),
        embedding_dim=конфиг["embedding_dim"],
        pretrained=False,
    )
    модель.load_state_dict(чекпоинт["model_state" if аргументы.weights == "raw" else "ema_state"])
    модель.eval()

    образцы = read_samples(
        splits_dir() / "train.parquet",
        crops_dir() / "stanford",
        num_classes=len(индекс),
        skip_multi_car=True,
    )
    калибровка = select_calibration(
        образцы, class_index=индекс, total=аргументы.total, seed=аргументы.seed
    )
    преобразование = build_transforms(
        чекпоинт["img_size"],
        train=False,
        mean=tuple(чекпоинт["mean"]),
        std=tuple(чекпоинт["std"]),
    )
    тензоры = calibration_tensors(калибровка, преобразование, crops_dir() / "stanford")
    пример = torch.zeros(1, 3, чекпоинт["img_size"], чекпоинт["img_size"])

    каталог = аргументы.out or аргументы.checkpoint.parent / "export"
    каталог.mkdir(parents=True, exist_ok=True)
    путь_fp32 = каталог / "classifier_fp32.tflite"
    начало = time.monotonic()
    _convert_fp32(модель, пример, путь_fp32)
    время_fp32 = time.monotonic() - начало

    время_int8 = 0.0
    путь_int8 = каталог / "classifier_int8.tflite"
    квантована = not аргументы.skip_int8
    if квантована:
        начало = time.monotonic()
        _convert_int8(модель, пример, тензоры, путь_int8)
        время_int8 = time.monotonic() - начало

    # Состав отчёта — по факту конвертации в этом прогоне, а не по наличию файла: с
    # `--skip-int8` рядом мог лежать INT8-артефакт прошлого прогона, и отчёт приписал бы
    # свежему прогону чужие хеш, размер и цифры самопроверки.
    модели: dict[str, Any] = {}
    for путь in (путь_fp32, путь_int8) if квантована else (путь_fp32,):
        подробности = runtime_details(путь)
        _проверить_контракт(
            подробности, путь, int(чекпоинт["img_size"]), int(конфиг["embedding_dim"])
        )
        модели[путь.name] = {
            "bytes": путь.stat().st_size,
            "sha256": sha256_of_file(путь),
            "runtime": подробности,
        }

    размеры = sorted(калибровка.per_class_counts.values())
    отчёт = {
        "manifest": {
            "checkpoint_path": str(аргументы.checkpoint),
            "checkpoint_sha256": sha256_of_file(аргументы.checkpoint),
            "weights": аргументы.weights,
            "embedding_dim": int(конфиг["embedding_dim"]),
            "img_size": int(чекпоинт["img_size"]),
            "mean": [float(значение) for значение in чекпоинт["mean"]],
            "std": [float(значение) for значение in чекпоинт["std"]],
            "preprocess_version": PREPROCESS_VERSION,
            "git_sha": чекпоинт.get("git_sha"),
        },
        "versions": _versions(),
        "calibration": {
            "requested_total": калибровка.requested_total,
            "total": len(калибровка.samples),
            "per_class_min": размеры[0],
            "per_class_median": размеры[len(размеры) // 2],
            "per_class_max": размеры[-1],
            "short_classes": sorted(
                имя for имя, число in калибровка.per_class_counts.items() if число < размеры[-1]
            )[:5],
            "seed": калибровка.seed,
        },
        "models": модели,
        "self_check": (
            self_check(модель, путь_fp32, путь_int8, тензоры) if квантована else {"samples": 0.0}
        ),
        "timing": {"convert_fp32_s": время_fp32, "convert_int8_s": время_int8},
    }
    (каталог / "export_report.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(отчёт, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
