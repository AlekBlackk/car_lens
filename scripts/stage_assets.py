"""Раскладка моделей прогона в дерево сборки Android (§10 дизайна M-1.5).

Каталог назначения один — `app/android/models/`, а подключают его к сборке оба
модуля: `:demo` как ассеты приложения, `:vision` как ассеты инструментальных тестов.
Копировать модели в два дерева нельзя: тридцать мегабайт в двух экземплярах
разъезжаются на первом же прогоне, когда обновили один каталог, а второй забыли.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from export.provenance import check_provenance

# Поля контракта детектора (§7.1 дизайна). Пустое или отсутствующее — отказ, а не
# «наверное, как обычно»: устройство читает контракт отсюда и проверять ему больше
# нечем.
ОБЯЗАТЕЛЬНЫЕ_ПОЛЯ = (
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
)
# Классификатор — единственный файл, чьё имя зависит от точности: int8 (путь M-1) или
# fp32 (русская модель, §Task 3). Имя выбирает флаг `--precision`, поэтому в `ИМЕНА` его
# нет — его подставляет `stage`.
ФАЙЛ_КЛАССИФИКАТОРА = {
    "int8": "classifier_int8.tflite",
    "fp32": "classifier_fp32.tflite",
}
ИМЕНА = {
    "detector": "detector_fp32.tflite",
    "index": "catalog_index.bin",
    "meta": "catalog_meta.json",
    "thresholds": "thresholds.json",
    "calibration": "calibration.json",
}
# Имя манифеста — контракт с Gradle: таск сборки читает файл под этим именем и без
# него роняет сборку (§10 дизайна), поэтому переименование здесь ломает `:demo`.
ИМЯ_МАНИФЕСТА = "models_manifest.json"


class ManifestError(ValueError):
    """Манифест собрать нельзя: контракт неполон."""


# Четвёртая копия `sha256_of_file` в дереве, и снова копия, а не импорт соседней. Модуль
# запускается в обычной venv разработки, а `export/to_tflite.py:57` там не импортируется:
# он тянет `litert_torch` на уровне модуля, плюс `training.dataset` (albumentations) и
# `training.model` (timm) через свои импорты. `training/build_index.py:140` притащил бы
# обучающий код в шаг раскладки ассетов ради семи строк, а хеш файла его контрактом не
# является. Тот же критерий, по которому копия заведена в `scripts/export_yolo.py:46`.
def sha256_of_file(path: Path) -> str:
    хеш = hashlib.sha256()
    with path.open("rb") as файл:
        for кусок in iter(lambda: файл.read(1 << 20), b""):
            хеш.update(кусок)
    return хеш.hexdigest()


# Плейсхолдеры §13.5 и §13.6. Пишутся здесь, а не руками, по той же причине, по которой
# здесь считаются хеши моделей: единственный каталог ассетов сборки должен собираться
# одной командой целиком. Настоящие файлы пишет `training/tune_thresholds.py`, сюда они
# приходят флагами `--thresholds/--calibration` и ложатся поверх; флаг `tuned: false` до
# тех пор обязывает приложение показывать отметку «не откалибровано» (§13.5).
ПЛЕЙСХОЛДЕР_ПОРОГОВ: dict[str, Any] = {
    "version": 1,
    "tuned": False,
    "sim_min": 0.42,
    "margin_min": 0.05,
    "min_observations": 3,
    "min_short_side_px": 120,
    "tuned_on": {"val_n": 0, "neg_n": 0, "achieved_tpr": 0.0, "achieved_fpr": 0.0},
}

# Две точки — минимум, при котором интерполяция вообще определена. Прямая от порога
# отказа до единицы: это заведомо не изотоническая регрессия, и `tuned: false` говорит
# об этом честно.
ПЛЕЙСХОЛДЕР_КАЛИБРОВКИ: dict[str, Any] = {
    "version": 1,
    "method": "isotonic_piecewise_linear",
    "tuned": False,
    "knots": [[0.42, 0.05], [1.0, 0.98]],
    "clip": [0.01, 0.98],
}


def write_placeholders(out: Path) -> dict[str, Path]:
    """Написать `thresholds.json` и `calibration.json` в каталог ассетов сборки."""
    пути: dict[str, Path] = {}
    for ключ, документ in (
        ("thresholds", ПЛЕЙСХОЛДЕР_ПОРОГОВ),
        ("calibration", ПЛЕЙСХОЛДЕР_КАЛИБРОВКИ),
    ):
        путь = out / ИМЕНА[ключ]
        путь.write_text(json.dumps(документ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        пути[ключ] = путь
    return пути


МАКС_УЗЛОВ = 32


def check_tuned_thresholds(документ: Mapping[str, Any]) -> None:
    """Подобранные пороги §13.5: форма плейсхолдера, `tuned: true` и живой `tuned_on`."""
    if set(документ) != set(ПЛЕЙСХОЛДЕР_ПОРОГОВ):
        raise ManifestError(f"thresholds: поля {sorted(документ)} не по форме §13.5")
    if документ["tuned"] is not True:
        raise ManifestError("thresholds: tuned не true — это плейсхолдер, а не подбор")
    for поле in ("val_n", "neg_n"):
        if int(документ["tuned_on"].get(поле, 0)) <= 0:
            raise ManifestError(f"thresholds: tuned_on.{поле} пуст — подбор ни на чём")
    for поле in ("sim_min", "margin_min"):
        if not 0.0 <= float(документ[поле]) <= 1.0:
            raise ManifestError(f"thresholds: {поле}={документ[поле]} вне [0, 1]")


def check_tuned_calibration(документ: Mapping[str, Any]) -> None:
    """Подобранная шкала §13.6: те же правила, что проверяет `parseCalibration` на телефоне."""
    if документ.get("tuned") is not True:
        raise ManifestError("calibration: tuned не true — это плейсхолдер, а не подбор")
    узлы = документ.get("knots") or []
    if not 2 <= len(узлы) <= МАКС_УЗЛОВ:
        raise ManifestError(f"calibration: узлов {len(узлы)}, нужно от 2 до {МАКС_УЗЛОВ}")
    for x, y in узлы:
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ManifestError(f"calibration: узел {[x, y]} вне [0, 1]")
    for (x0, y0), (x1, y1) in itertools.pairwise(узлы):
        if not (x1 > x0 and y1 > y0):
            raise ManifestError(f"calibration: узлы не строго возрастают у {[x1, y1]}")
    lo, hi = документ.get("clip") or [None, None]
    if lo is None or hi is None or not 0.0 <= float(lo) <= float(hi) <= 1.0:
        raise ManifestError(f"calibration: clip {документ.get('clip')} не пара внутри [0, 1]")


def _подобранные(thresholds: Path | None, calibration: Path | None) -> dict[str, Path]:
    """Оба файла или ни одного: подобранные пороги с плейсхолдерной шкалой (и наоборот)
    оставили бы отметку «не откалибровано», а выпуск — бессмысленным."""
    if thresholds is None and calibration is None:
        return {}
    if thresholds is None or calibration is None:
        raise ManifestError("--thresholds и --calibration передаются оба или ни одного")
    check_tuned_thresholds(_обязательный_json(thresholds, "нет файла порогов"))
    check_tuned_calibration(_обязательный_json(calibration, "нет файла шкалы"))
    return {"thresholds": thresholds, "calibration": calibration}


def _контракт(detector_report: Mapping[str, Any]) -> dict[str, Any]:
    контракт = dict(detector_report.get("contract") or {})
    for поле in ОБЯЗАТЕЛЬНЫЕ_ПОЛЯ:
        if поле not in контракт:
            raise ManifestError(f"в контракте детектора нет поля {поле}: экспорт неполон")
        значение = контракт[поле]
        if значение is None or значение == "" or значение == []:
            raise ManifestError(f"поле {поле} контракта детектора пустое: экспорт неполон")
    return контракт


def build_manifest(
    *,
    files: Mapping[str, Path],
    имена: Mapping[str, str],
    export_report: Mapping[str, Any],
    catalog_meta: Mapping[str, Any],
    detector_report: Mapping[str, Any],
    index_manifest: Mapping[str, Any] | None,
    git_sha: str,
) -> dict[str, Any]:
    манифест_экспорта = export_report["manifest"]
    описания = {
        ключ: {
            "name": имя,
            "size": files[ключ].stat().st_size,
            "sha256": sha256_of_file(files[ключ]),
        }
        for ключ, имя in имена.items()
    }
    return {
        "files": описания,
        "provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": манифест_экспорта["checkpoint_sha256"],
            "git_sha": git_sha,
            "num_classes": len(catalog_meta["classes"]),
            "index_source": "tflite" if index_manifest else "checkpoint",
            "detector_weights_sha256": detector_report["weights_sha256"],
            "detector_graph_sha256": detector_report["graph_sha256"],
        },
        "classifier": {
            "img_size": int(манифест_экспорта["img_size"]),
            "mean": [float(значение) for значение in манифест_экспорта["mean"]],
            "std": [float(значение) for значение in манифест_экспорта["std"]],
            "embedding_dim": int(манифест_экспорта["embedding_dim"]),
        },
        "detector": _контракт(detector_report),
    }


# Копия `_load_json` из `export/verify_parity.py:51`, и по той же причине, что копия
# хеша выше: тот модуль импортирует torch и albumentations сверху, в venv разработки он
# не поднимается вовсе.
def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    документ: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return документ


def _обязательный_json(path: Path, подсказка: str) -> dict[str, Any]:
    """Документ прогона или `SystemExit` с путём и командой, которая его пишет."""
    документ = _load_json(path)
    if документ is None:
        raise SystemExit(f"{path}: файла нет — {подсказка}")
    return документ


def _обязательный_файл(path: Path, подсказка: str) -> Path:
    """То же для моделей: иначе о пропаже первым сказал бы `shutil.copy2` без диагноза."""
    if not path.is_file():
        raise SystemExit(f"{path}: файла нет — {подсказка}")
    return path


def _git_sha() -> str:
    """Короткий хеш коммита с суффиксом `-dirty`, если дерево грязное.

    Без суффикса запись `git_sha` означала бы «собрано из этого коммита», а это неправда
    ровно в том случае, когда правки ещё не закоммичены — то есть в большинстве прогонов
    на машине разработчика.
    """
    try:
        коммит = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        правки = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{коммит}-dirty" if правки else коммит


def _каталог_индекса(run: Path, index_dir: str | None) -> Path:
    """Какой индекс раскладывать: названный флагом, принятый паритетом или `index/`.

    Индексов у прогона бывает два — собранный PyTorch (`index/`) и пересобранный на
    выходах квантованной модели (`index_int8/`, §4.7 дизайна M-1.4). Выбирать за
    пользователя молча нельзя, поэтому по умолчанию берётся тот, на котором мерили
    паритет: `export/verify_parity.py:326` пишет путь индекса в `parity.json`.
    """
    if index_dir is not None:
        return run / index_dir
    паритет = _load_json(run / "export" / "parity.json")
    if паритет is None:
        return run / "index"
    названный = паритет.get("manifest", {}).get("index_dir")
    if названный is None:
        raise SystemExit(
            f"{run / 'export' / 'parity.json'}: нет manifest.index_dir — отчёт собран "
            "другой версией пайплайна, укажите каталог индекса флагом --index"
        )
    # Записан путь ровно в том виде, как его набрали в командной строке паритета, а гоняют
    # её из корня репозитория — то есть обычно это `runs/<прогон>/index_int8`, путь от
    # рабочего каталога, а не от прогона. Проверяются оба прочтения, потому что по записи
    # их не различить; абсолютный путь `run / названный` отдаёт неизменным.
    от_прогона = run / str(названный)
    от_каталога = Path(str(названный))
    for кандидат in (от_прогона, от_каталога):
        if кандидат.is_dir():
            return кандидат
    raise SystemExit(
        f"{run / 'export' / 'parity.json'}: записанный index_dir {названный} — не каталог "
        f"ни от прогона ({от_прогона}), ни от текущего рабочего каталога ({от_каталога}); "
        "укажите каталог индекса флагом --index"
    )


def stage(
    run: Path,
    detector: Path,
    out: Path,
    *,
    index_dir: str | None = None,
    precision: str = "int8",
    thresholds: Path | None = None,
    calibration: Path | None = None,
) -> Path:
    """Разложить файлы прогона в `out` и написать рядом манифест. Вернуть его путь.

    Четыре файла копируются из прогона, ещё два — `thresholds.json` и `calibration.json`
    (§13.5, §13.6) — пишутся плейсхолдерами на месте: копировать их неоткуда, пока не
    переданы подобранные файлы.

    `thresholds` и `calibration` — подобранные `thresholds.json`/`calibration.json` от
    `training/tune_thresholds.py`. Их проверяют на годность и кладут поверх плейсхолдеров;
    передаются только парой, потому что подобранные пороги с плейсхолдерной шкалой (и
    наоборот) оставили бы сборку помеченной «не откалибровано». Годность сверяется до
    сноса `out`, как и контракт детектора.

    Ждёт раскладку прогона обучения: `run/export/` с файлом классификатора и
    `export_report.json`, каталог индекса с `catalog_index.bin` и `catalog_meta.json`,
    `detector_report.json` рядом с графом детектора (так его пишет
    `scripts/export_yolo.py`).

    `precision` выбирает, какой классификатор разложить и под каким именем записать в
    манифест: `int8` (`classifier_int8.tflite`, путь M-1) или `fp32`
    (`classifier_fp32.tflite`). Остальные пять имён точности не зависят.

    `index_dir` — путь к каталогу индекса: относительный считается от `run`, абсолютный
    берётся как есть. `None` означает «спросить у `parity.json`»: путь берётся оттуда и
    пробуется от `run` и от рабочего каталога, а если отчёта паритета в прогоне нет — из
    `run/index`.

    Сверка происхождения и полноты контракта детектора идёт до копирования: разложить
    сначала, а отказать потом значило бы оставить в дереве сборки полурассинхронизированный
    набор, которым инструментальные тесты воспользуются как настоящим.
    """
    # Имя классификатора зависит от точности; остальные пять фиксированы. Локальная карта
    # используется всюду вместо глобальной `ИМЕНА`: источник, цели копирования, манифест.
    имена = {**ИМЕНА, "classifier": ФАЙЛ_КЛАССИФИКАТОРА[precision]}
    отчёт_экспорта = _обязательный_json(
        run / "export" / "export_report.json",
        "сначала `python -m export.to_tflite` — без манифеста экспорта происхождение не сверить",
    )
    отчёт_детектора = _обязательный_json(
        detector.with_name("detector_report.json"),
        "рядом с графом детектора нет отчёта — сначала `python -m scripts.export_yolo`",
    )
    индекс = _каталог_индекса(run, index_dir)
    подсказка_индекса = "каталог индекса не собран — сначала `training.build_index`"
    мета = _обязательный_json(индекс / "catalog_meta.json", подсказка_индекса)
    # Манифест индекса читается первым: его наличие переключает `check_provenance` на
    # ветку INT8-индекса, где `report.json` не участвует вовсе, и передавать туда
    # прочитанный отчёт значило бы намекать, что он на что-то влияет. Тот же порядок, что
    # в `export/verify_parity.py:164-170`.
    манифест_индекса = _load_json(индекс / "index_manifest.json")
    отчёт_индекса = None if манифест_индекса is not None else _load_json(индекс / "report.json")
    классификатор = _обязательный_файл(
        run / "export" / имена["classifier"],
        f"нет {имена['classifier']} в run/export: экспорт не дал модель точности {precision}",
    )
    источники = {
        "classifier": классификатор,
        "detector": _обязательный_файл(detector, "граф детектора не экспортирован"),
        "index": _обязательный_файл(индекс / "catalog_index.bin", подсказка_индекса),
        "meta": индекс / "catalog_meta.json",
    }
    check_provenance(
        export_manifest=отчёт_экспорта["manifest"],
        catalog_meta=мета,
        index_report=отчёт_индекса,
        index_manifest=манифест_индекса,
        int8_sha256=sha256_of_file(классификатор),
    )
    # Контракт проверяется здесь, а не только в `build_manifest`: тот вызывается уже после
    # сноса каталога и копирования, и неполный контракт оставил бы в дереве сборки четыре
    # модели без манифеста — ровно то состояние, против которого написан модуль.
    _контракт(отчёт_детектора)
    # Годность подобранных файлов сверяется здесь, рядом с контрактом и до сноса `out`: иначе
    # негодные пороги снесли бы прошлый набор и оставили дерево сборки ни с чем.
    подобранные = _подобранные(thresholds, calibration)

    # Каталог сносится целиком, а не перезаписывается по именам: файл от прошлого прогона
    # с другим именем остался бы лежать в ассетах и уехал бы в APK.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    цели = {ключ: out / имя for ключ, имя in имена.items()}
    for ключ, источник in источники.items():
        shutil.copy2(источник, цели[ключ])
    write_placeholders(out)
    # Подобранные файлы кладутся поверх плейсхолдеров: без них сборка остаётся той же,
    # что до калибровки, и отметка «не откалибровано» в приложении — честной.
    for ключ, источник in подобранные.items():
        shutil.copy2(источник, out / ИМЕНА[ключ])

    # Размеры и хеши считаются по копиям, а не по источникам: манифест описывает то, что
    # уедет на устройство, и оборванное копирование обязано разойтись с ним, а не совпасть.
    манифест = build_manifest(
        files=цели,
        имена=имена,
        export_report=отчёт_экспорта,
        catalog_meta=мета,
        detector_report=отчёт_детектора,
        index_manifest=манифест_индекса,
        git_sha=_git_sha(),
    )
    путь = out / ИМЯ_МАНИФЕСТА
    путь.write_text(json.dumps(манифест, ensure_ascii=False, indent=2), encoding="utf-8")
    return путь


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Раскладка моделей прогона в дерево сборки")
    парсер.add_argument("--run", type=Path, required=True, help="каталог прогона обучения")
    парсер.add_argument("--detector", type=Path, required=True, help="detector_fp32.tflite")
    # Без значения по умолчанию: каталог назначения сносится целиком, и угадывать его за
    # пользователя нельзя. Штатный вызов — `--out app/android/models` (§10 дизайна).
    парсер.add_argument("--out", type=Path, required=True, help="каталог ассетов сборки")
    парсер.add_argument(
        "--index", default=None, help="каталог индекса; по умолчанию принятый в parity.json"
    )
    парсер.add_argument(
        "--precision",
        choices=["int8", "fp32"],
        default="int8",
        help="точность классификатора: какой .tflite разложить (умолчание int8, путь M-1)",
    )
    парсер.add_argument("--thresholds", type=Path, default=None, help="подобранные пороги")
    парсер.add_argument("--calibration", type=Path, default=None, help="подобранная шкала")
    return парсер.parse_args()


def main() -> None:
    аргументы = _parse_args()
    путь = stage(
        аргументы.run,
        аргументы.detector,
        аргументы.out,
        index_dir=аргументы.index,
        precision=аргументы.precision,
        thresholds=аргументы.thresholds,
        calibration=аргументы.calibration,
    )
    print(f"манифест: {путь}")


if __name__ == "__main__":
    main()
