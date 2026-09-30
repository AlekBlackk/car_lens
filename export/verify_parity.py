"""Паритет PyTorch против TFLite fp32 и INT8 (§4.6 дизайна M-1.4).

Метрика считается через индекс эталонов, а не через голову ArcFace: голова на
устройство не едет, и мерить ей значит измерять путь, которым продукт не
пользуется (Р-27).

Порог берётся от базы этого же прогона — PyTorch с текущим fp32-индексом на той же
выборке, — а не от чисел §8 дизайна M-1.3: там другая выборка и другой отбор.

Приёмочный прогон идёт с `--limit 0`, то есть сплитом целиком. Срез вердиктом §4.6
не является: `read_samples` сохраняет порядок сплита, а он сгруппирован по классам,
поэтому `--limit N` накрывает только первые классы — числа выйдут правдоподобными и
при этом ни о чём.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export.parity import cosine_deltas, metric_delta, raw_deltas, top1_churn
from export.provenance import check_provenance
from export.to_tflite import _run_tflite, sha256_of_file
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import top5_by_centroids
from pipeline.paths import splits_dir
from training.build_index import resolve_crops_root
from training.dataset import build_transforms, make_loader
from training.metrics import Metrics, aggregate
from training.model import create_model
from training.samples import read_samples

RAW_MAX_ABS = 1e-3
TOP1_DROP_LIMIT_PP = -1.5

# Норма, ниже которой нормализовать нечего. Значение то же, что у `_MIN_DENOMINATOR` в
# `export/parity.py` и `export/to_tflite.py`, но константа продублирована, а не
# импортирована: приватное имя соседнего модуля контрактом не является. Задача у неё и
# другая — не прижать знаменатель, а отказаться от прогона: здесь деление идёт на живой
# знаменатель, и подмена нуля порогом дала бы не диагноз, а мусорный эмбеддинг.
_MIN_NORM = 1e-6


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    результат: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return результат


def _required(документ: Mapping[str, Any], ключ: str, откуда: str) -> Any:
    """Значение по ключу или `SystemExit` с именем ключа и документа.

    Обращения по строке, до которых `check_provenance` не доходит, обязаны отказывать так
    же диагностируемо, как обёрнутые в перехват ниже. Ключ `weights` сверка в ветке
    INT8-индекса не читает вовсе (`export/provenance.py:66`), поэтому голый `KeyError`
    доехал бы до пользователя трейсбеком без имени файла — тем самым, от которого перехват
    и избавлялся.
    """
    if ключ not in документ:
        raise SystemExit(
            f"{откуда}: нет ключа {ключ!r} — документ неполон или собран чужой версией"
        )
    return документ[ключ]


def _torch_embeddings(
    model: torch.nn.Module, loader: Any, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Сырые, **ненормированные** эмбеддинги и метки.

    `collect_embeddings` из `training.build_index` здесь не годится: он нормализует
    внутри, а сравнение §4.4 требует сырых значений — нормализация скрыла бы ровно
    ту ошибку конвертации, ради которой сырой слой и заведён.
    """
    model.eval()
    куски: list[np.ndarray] = []
    метки: list[np.ndarray] = []
    with torch.no_grad():
        for картинки, цели in loader:
            эмбеддинг = model(картинки.to(device, non_blocking=True)).float()
            куски.append(эмбеддинг.cpu().numpy().astype(np.float32))
            метки.append(np.asarray(цели.cpu(), dtype=np.int64))
    if not куски:
        raise SystemExit("выборка пуста: сравнивать нечего")
    return np.concatenate(куски), np.concatenate(метки)


def _tflite_embeddings(path: Path, loader: Any) -> np.ndarray:
    """Тот же загрузчик через LiteRT.

    Загрузчик перебирается заново на каждый рантайм: при `shuffle=False` порядок
    повторяем, а держать все картинки сплита в памяти — это около двух гигабайт
    ради удобства записи.

    `_run_tflite` создаёт `Interpreter` на каждый батч, и это принято сознательно:
    время прогона определяется покадровым `invoke` внутри него, а кеш интерпретатора
    потребовал бы правки чужого модуля ради выигрыша, которого не видно.
    """
    куски = [_run_tflite(path, картинки.numpy()) for картинки, _ in loader]
    # Через явно типизированную переменную: стабы numpy перегружают `concatenate` по dtype,
    # на списке `ndarray` без параметров перегрузка неоднозначна, и результат вырождается в
    # `Any`, которого `.astype` уже не восстанавливает. Strict-mypy ловит такой возврат.
    собранные: np.ndarray = np.concatenate(куски, axis=0).astype(np.float32)
    return собранные


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Паритет PyTorch против TFLite")
    парсер.add_argument("--export", type=Path, required=True, help="каталог с .tflite")
    парсер.add_argument("--index", type=Path, required=True, help="каталог с catalog_index.bin")
    парсер.add_argument("--checkpoint", type=Path, required=True)
    парсер.add_argument("--splits", nargs="+", default=["val", "test"])
    парсер.add_argument("--limit", type=int, default=0, help="0 — сплит целиком")
    парсер.add_argument("--crops-root", type=Path, default=None, help="корень кропов сплитов")
    # По умолчанию CPU, а не CUDA. Torch включает TF32 для свёрток сам, мантисса TF32 —
    # десять бит, и погрешность самой базы выходит порядка 1e-3 при пороге 1e-3. Замерено
    # на этом проекте, сплит `val`, один и тот же экспорт, менялся только флаг:
    # `cpu` — 6.2e-06, `cuda` с NVIDIA_TF32_OVERRIDE=0 — 8.1e-04, `cuda` по умолчанию —
    # 3.8e-03. То есть на CUDA приёмка падает на исправном графе. `--device cuda` остаётся
    # доступным явным выбором того, кто про TF32 знает.
    парсер.add_argument("--device", default="cpu")
    # Точность, которая едет на устройство. При `int8` гейт судит просадку квантованной
    # модели (путь M-1). При `fp32` на телефон едет fp32-граф, совпадающий с PyTorch по
    # построению, и просадка INT8 считается ради отчёта, но приговором не является; порог
    # `TOP1_DROP_LIMIT_PP` при этом не трогается — меняется не он, а чью просадку блокировать.
    парсер.add_argument(
        "--precision",
        choices=["int8", "fp32", "fp16"],
        default="int8",
        help="точность, которая едет на устройство: её просадку судит гейт",
    )
    return парсер.parse_args()


def main() -> None:
    аргументы = _parse_args()
    отчёт_экспорта = _load_json(аргументы.export / "export_report.json")
    if отчёт_экспорта is None:
        raise SystemExit(f"{аргументы.export}: нет export_report.json — сначала to_tflite")
    манифест_экспорта = отчёт_экспорта["manifest"]
    # Хеш файла на диске против записанного в манифесте. `check_provenance` эту дыру не
    # закрывает: он сверяет записанные строки друг с другом (`export/provenance.py:91-92`),
    # а не то, что лежит по пути `--checkpoint`. Опечатка в пути или сосед с похожим именем
    # дали бы чужую базу PyTorch при зелёной сверке происхождения, дельты считались бы от
    # неверного эталона, а `parity.json` вышел бы с виду валидным — то есть вердикт §4.6
    # стал бы неправдой без единого сообщения об ошибке.
    ожидаемый_чекпоинт = str(
        _required(манифест_экспорта, "checkpoint_sha256", "export_report.json → manifest")
    )
    фактический_чекпоинт = sha256_of_file(аргументы.checkpoint)
    if фактический_чекпоинт != ожидаемый_чекпоинт:
        raise SystemExit(
            f"{аргументы.checkpoint}: sha256 {фактический_чекпоинт} против "
            f"{ожидаемый_чекпоинт} в манифесте экспорта — экспортирован другой чекпоинт, "
            "и база PyTorch считалась бы не той моделью"
        )
    путь_fp32 = аргументы.export / "classifier_fp32.tflite"
    путь_int8 = аргументы.export / "classifier_int8.tflite"
    # fp16 — необязательный четвёртый рантайм: файл появляется только после `export.to_fp16`.
    # Когда он и есть то, что едет на устройство, его отсутствие — ошибка, а не пропуск.
    путь_fp16 = аргументы.export / "classifier_fp16.tflite"
    if аргументы.precision == "fp16" and not путь_fp16.is_file():
        raise SystemExit(f"{путь_fp16}: файла нет — сначала `python -m export.to_fp16`")
    есть_fp16 = путь_fp16.is_file()
    # Без этой проверки первым об отсутствии файла сообщил бы `sha256_of_file` — сырым
    # `FileNotFoundError` из середины сверки происхождения, где про причину не сказано
    # ничего. А причина у пропавшего INT8 ровно одна и известная.
    for путь in (путь_fp32, путь_int8):
        if not путь.is_file():
            raise SystemExit(
                f"{путь}: файла нет — вероятно, экспорт шёл с --skip-int8; "
                "перезапустите `python -m export.to_tflite` без этого флага"
            )

    заголовок, центроиды = read_index(аргументы.index / "catalog_index.bin")
    байты_меты = (аргументы.index / "catalog_meta.json").read_bytes()
    verify_meta(заголовок, байты_меты)
    мета = json.loads(байты_меты)
    # Манифест читается первым: его наличие переключает `check_provenance` на ветку
    # INT8-индекса, где `report.json` не участвует вовсе. Передать туда прочитанный
    # отчёт значило бы намекать, что он на что-то влияет.
    манифест_индекса = _load_json(аргументы.index / "index_manifest.json")
    отчёт_индекса = (
        None if манифест_индекса is not None else _load_json(аргументы.index / "report.json")
    )
    # Все документы сверки читаются здесь, значит и входной контракт — обязанность этого
    # модуля. `check_provenance` берёт ключи по индексу, и отсутствующий ключ доехал бы до
    # пользователя трейсбеком, в котором не назван ни файл, ни то, что его надо пересобрать.
    try:
        check_provenance(
            export_manifest=манифест_экспорта,
            catalog_meta=мета,
            index_report=отчёт_индекса,
            index_manifest=манифест_индекса,
            int8_sha256=sha256_of_file(путь_int8),
        )
    except KeyError as ошибка:
        ключ = str(ошибка.args[0]) if ошибка.args else "?"
        источники: dict[str, Any] = {
            "export_report.json → manifest": манифест_экспорта,
            "catalog_meta.json": мета,
            "report.json": отчёт_индекса,
            "index_manifest.json": манифест_индекса,
        }
        где = ", ".join(
            имя
            for имя, документ in источники.items()
            if документ is not None and ключ not in документ
        )
        raise SystemExit(
            f"сверка происхождения: нет ключа {ключ!r} в {где or 'документах сверки'} — "
            "документ собран другой версией пайплайна, пересоберите его"
        ) from ошибка

    чекпоинт: dict[str, Any] = torch.load(
        аргументы.checkpoint, map_location="cpu", weights_only=True
    )
    индекс: dict[str, int] = чекпоинт["class_index"]
    класс_по_индексу = {значение: ключ for ключ, значение in индекс.items()}
    # Марка для `top1_brand` — из меты, а не из префикса `class_id`: её байты уже
    # сверены с sha256 в заголовке индекса, по которому считаются топы, а позиция
    # записи в `classes` и есть индекс класса §13.2.
    марка_по_индексу = {номер: класс["brand"] for номер, класс in enumerate(мета["classes"])}
    конфиг = чекпоинт["config"]
    модель, _ = create_model(
        конфиг["backbone"],
        num_classes=len(индекс),
        embedding_dim=конфиг["embedding_dim"],
        pretrained=False,
    )
    # Оба обращения через `_required`: `weights` сверка происхождения в ветке INT8-индекса
    # не читает, а имени `ema_state` она не проверяет ни в одной из ветвей — один файл
    # чекпоинта содержит двое весов, и отсутствие нужных обязано называться, а не падать
    # `KeyError` из середины загрузки.
    режим = _required(манифест_экспорта, "weights", "export_report.json → manifest")
    веса = "model_state" if режим == "raw" else "ema_state"
    модель.load_state_dict(_required(чекпоинт, веса, str(аргументы.checkpoint)))
    устройство = torch.device(аргументы.device)
    модель.to(устройство)

    итоги: dict[str, Any] = {"baseline": {}, "runtimes": {}, "deltas": {}}
    провалы: list[str] = []
    for имя in аргументы.splits:
        образцы = read_samples(
            splits_dir() / f"{имя}.parquet",
            resolve_crops_root(аргументы.crops_root),
            num_classes=len(индекс),
            skip_multi_car=False,
        )
        if аргументы.limit:
            образцы = образцы[: аргументы.limit]
        загрузчик = make_loader(
            образцы,
            build_transforms(
                чекпоинт["img_size"],
                train=False,
                mean=tuple(чекпоинт["mean"]),
                std=tuple(чекпоинт["std"]),
            ),
            batch_size=конфиг["batch_size"],
            shuffle=False,
            num_workers=конфиг["num_workers"],
        )
        сырой_эталон, метки = _torch_embeddings(модель, загрузчик, устройство)
        сырой_fp32 = _tflite_embeddings(путь_fp32, загрузчик)
        сырой_int8 = _tflite_embeddings(путь_int8, загрузчик)
        сырые_рантаймов = {
            "pytorch": сырой_эталон,
            "tflite_fp32": сырой_fp32,
            "tflite_int8": сырой_int8,
        }
        if есть_fp16:
            сырые_рантаймов["tflite_fp16"] = _tflite_embeddings(путь_fp16, загрузчик)

        по_рантаймам: dict[str, Metrics] = {}
        топы: dict[str, list[list[int]]] = {}
        for ключ, сырые in сырые_рантаймов.items():
            # Норма считается один раз и служит сразу приговором и делителем. Без стража
            # деление отдаёт `nan` (или ноль, если норма переполнилась в `inf`),
            # `top5_by_centroids` возвращает по нему мусор, а `aggregate` печатает
            # правдоподобные проценты. Роняет прогон в итоге `export/parity.py` на
            # дельтах, то есть на два шага позже места поломки и с чужим диагнозом.
            нормы = np.linalg.norm(сырые, axis=1, keepdims=True)
            if not bool(np.isfinite(сырые).all()) or not bool(np.isfinite(нормы).all()):
                raise SystemExit(
                    f"{имя}/{ключ}: неконечные эмбеддинги — nan, inf либо переполнение "
                    "нормы; нормализовать их нельзя, а метрики вышли бы правдоподобными"
                )
            if bool((нормы < _MIN_NORM).any()):
                raise SystemExit(
                    f"{имя}/{ключ}: норма эмбеддинга {float(нормы.min()):.3e} ниже "
                    f"{_MIN_NORM:.0e} — направления у вектора нет, косинус не определён"
                )
            нормированные = сырые / нормы
            топ5 = top5_by_centroids(нормированные.astype(np.float32), центроиды)
            топы[ключ] = топ5
            по_рантаймам[ключ] = aggregate(
                топ5,
                [int(метка) for метка in метки],
                idx_to_class_id=класс_по_индексу,
                idx_to_brand=марка_по_индексу,
            )

        итоги["baseline"][имя] = {
            **asdict(по_рантаймам["pytorch"]),
            "examples": len(метки),
        }
        итоги["runtimes"][имя] = {ключ: asdict(значение) for ключ, значение in по_рантаймам.items()}
        итоги["deltas"][имя] = {
            "fp32": {
                "raw": asdict(raw_deltas(сырой_эталон, сырой_fp32)),
                "cosine": asdict(cosine_deltas(сырой_эталон, сырой_fp32)),
                "top1_churn": top1_churn(топы["pytorch"], топы["tflite_fp32"]),
                "metrics_pp": metric_delta(по_рантаймам["pytorch"], по_рантаймам["tflite_fp32"]),
            },
            "int8": {
                "raw": asdict(raw_deltas(сырой_эталон, сырой_int8)),
                "cosine": asdict(cosine_deltas(сырой_эталон, сырой_int8)),
                "top1_churn": top1_churn(топы["pytorch"], топы["tflite_int8"]),
                "metrics_pp": metric_delta(по_рантаймам["pytorch"], по_рантаймам["tflite_int8"]),
            },
        }
        if есть_fp16:
            итоги["deltas"][имя]["fp16"] = {
                "raw": asdict(raw_deltas(сырой_эталон, сырые_рантаймов["tflite_fp16"])),
                "cosine": asdict(cosine_deltas(сырой_эталон, сырые_рантаймов["tflite_fp16"])),
                "top1_churn": top1_churn(топы["pytorch"], топы["tflite_fp16"]),
                "metrics_pp": metric_delta(по_рантаймам["pytorch"], по_рантаймам["tflite_fp16"]),
            }

        расхождение_fp32 = итоги["deltas"][имя]["fp32"]["raw"]["max_abs"]
        if расхождение_fp32 > RAW_MAX_ABS:
            провалы.append(
                f"{имя}: fp32-TFLite расходится с PyTorch на {расхождение_fp32:.2e} "
                f"при пороге {RAW_MAX_ABS:.0e} — сломан граф, а не квантизация"
            )
        просадка = итоги["deltas"][имя]["int8"]["metrics_pp"]["top1_model"]
        # Проверка графа (raw fp32 выше) безусловна: сломанный граф ломает и fp32. А вот
        # просадка квантованной точности блокирует гейт только когда она и едет на устройство.
        if аргументы.precision in ("int8", "fp16"):
            дельты_точности = итоги["deltas"][имя][аргументы.precision]
            просадка_точности = дельты_точности["metrics_pp"]["top1_model"]
            if просадка_точности < TOP1_DROP_LIMIT_PP:
                провалы.append(
                    f"{имя}: top1_model {аргументы.precision} ниже базы на "
                    f"{abs(просадка_точности):.2f} п.п. при допуске {abs(TOP1_DROP_LIMIT_PP)} п.п."
                )
        print(
            f"{имя}: база {по_рантаймам['pytorch'].top1_model:.4f}, "
            f"fp32 {по_рантаймам['tflite_fp32'].top1_model:.4f}, "
            f"int8 {по_рантаймам['tflite_int8'].top1_model:.4f} "
            f"({просадка:+.2f} п.п.), churn {итоги['deltas'][имя]['int8']['top1_churn']:.3f}"
        )
        if есть_fp16:
            print(
                f"{имя}: fp16 {по_рантаймам['tflite_fp16'].top1_model:.4f} "
                f"({итоги['deltas'][имя]['fp16']['metrics_pp']['top1_model']:+.2f} п.п.), "
                f"churn {итоги['deltas'][имя]['fp16']['top1_churn']:.3f}"
            )

    итоги["thresholds"] = {
        "raw_max_abs": RAW_MAX_ABS,
        "top1_drop_limit_pp": TOP1_DROP_LIMIT_PP,
        "precision": аргументы.precision,
        "failures": провалы,
        "passed": not провалы,
    }
    итоги["manifest"] = {
        "export_report": манифест_экспорта,
        "index_dir": str(аргументы.index),
        "index_sha256": sha256_of_file(аргументы.index / "catalog_index.bin"),
        "limit": аргументы.limit,
    }
    (аргументы.export / "parity.json").write_text(
        json.dumps(итоги, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if провалы:
        for строка in провалы:
            print(f"ПРОВАЛ {строка}", file=sys.stderr)
        raise SystemExit(1)
    print("паритет пройден")


if __name__ == "__main__":
    main()
