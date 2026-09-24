"""Раскладка моделей в дерево сборки и манифест с контрактами (§10 дизайна M-1.5)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.stage_assets import (
    ИМЕНА,
    ПЛЕЙСХОЛДЕР_ПОРОГОВ,
    ФАЙЛ_КЛАССИФИКАТОРА,
    ManifestError,
    build_manifest,
    stage,
)

ЧЕКПОИНТ_SHA = "aa" * 32

ОТЧЁТ_ЭКСПОРТА: dict[str, Any] = {
    "manifest": {
        "checkpoint_sha256": ЧЕКПОИНТ_SHA,
        "embedding_dim": 256,
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "weights": "raw",
        "preprocess_version": 1,
    }
}
МЕТА: dict[str, Any] = {
    "model_weights_sha256": ЧЕКПОИНТ_SHA,
    "embedding_dim": 256,
    "classes": [
        {
            "class_id": "audi__a5",
            "members": ["audi__a5"],
            "brand": "Audi",
            "display": "Audi A5",
        }
    ],
}
КОНТРАКТ: dict[str, Any] = {
    "imgsz": 416,
    "input_dtype": "float32",
    "input_layout": "NHWC",
    "channel_order": "RGB",
    "input_range": "unit",
    "letterbox_pad_value": 114,
    "output_layout": "CN",
    "box_format": "xywh",
    "coord_units": "normalized",
    "num_classes": 80,
    "vehicle_classes": [2, 5, 7],
    "score_layout": "classes_only",
    "max_det": 100,
}
ОТЧЁТ_ДЕТЕКТОРА: dict[str, Any] = {
    "weights_sha256": "cc" * 32,
    "graph_sha256": "dd" * 32,
    "ultralytics_version": "8.4.120",
    "contract": КОНТРАКТ,
}


def _имена(precision: str = "int8") -> dict[str, str]:
    # Имя классификатора зависит от точности, остальные пять берутся из модуля. Локальная
    # копия словаря разошлась бы с `stage`/`build_manifest` на первом же добавлении файла.
    return {**ИМЕНА, "classifier": ФАЙЛ_КЛАССИФИКАТОРА[precision]}


def _файлы(tmp_path: Path, precision: str = "int8") -> dict[str, Path]:
    пути: dict[str, Path] = {}
    for ключ, имя in _имена(precision).items():
        путь = tmp_path / имя
        путь.write_bytes(имя.encode("utf-8"))
        пути[ключ] = путь
    return пути


def _собрать(tmp_path: Path, *, precision: str = "int8", **правки: Any) -> dict[str, Any]:
    аргументы: dict[str, Any] = {
        "files": _файлы(tmp_path, precision),
        "имена": _имена(precision),
        "export_report": ОТЧЁТ_ЭКСПОРТА,
        "catalog_meta": МЕТА,
        "detector_report": ОТЧЁТ_ДЕТЕКТОРА,
        "index_manifest": None,
        "git_sha": "0123456",
    }
    аргументы.update(правки)
    return build_manifest(**аргументы)


def test_манифест_несёт_четыре_группы(tmp_path: Path) -> None:
    манифест = _собрать(tmp_path)
    assert set(манифест) == {"files", "provenance", "classifier", "detector"}
    assert манифест["classifier"]["img_size"] == 320
    assert манифест["classifier"]["mean"] == [0.485, 0.456, 0.406]
    assert манифест["detector"]["coord_units"] == "normalized"
    assert манифест["provenance"]["checkpoint_sha256"] == ЧЕКПОИНТ_SHA
    assert манифест["provenance"]["num_classes"] == 1


def test_каждый_файл_описан_размером_и_хешем(tmp_path: Path) -> None:
    манифест = _собрать(tmp_path)
    assert set(манифест["files"]) == set(ИМЕНА) | {"classifier"}
    for запись in манифест["files"].values():
        assert set(запись) == {"name", "size", "sha256"}
        assert len(запись["sha256"]) == 64
        assert запись["size"] > 0


def test_индекс_на_int8_помечен_в_происхождении(tmp_path: Path) -> None:
    """Пересобранный на выходах `.tflite` индекс обязан отличаться в манифесте.

    Единственная ветка `build_manifest`, до которой остальные тесты не доходят: с
    `index_manifest` происхождение индекса — не чекпоинт, а квантованная модель
    (§4.7 дизайна M-1.4), и на устройстве это единственный способ узнать, какой из
    двух наборов центроидов приехал.
    """
    манифест = _собрать(tmp_path, index_manifest={"source": "tflite"})
    assert манифест["provenance"]["index_source"] == "tflite"


def test_пустое_поле_контракта_детектора_роняет(tmp_path: Path) -> None:
    отчёт = {**ОТЧЁТ_ДЕТЕКТОРА, "contract": {**КОНТРАКТ, "coord_units": ""}}
    with pytest.raises(ManifestError, match="coord_units"):
        _собрать(tmp_path, detector_report=отчёт)


def test_отсутствующее_поле_контракта_роняет(tmp_path: Path) -> None:
    урезанный = {ключ: значение for ключ, значение in КОНТРАКТ.items() if ключ != "max_det"}
    with pytest.raises(ManifestError, match="max_det"):
        _собрать(tmp_path, detector_report={**ОТЧЁТ_ДЕТЕКТОРА, "contract": урезанный})


def _разложить_прогон(tmp_path: Path, *, мета: dict[str, Any]) -> tuple[Path, Path]:
    прогон = tmp_path / "run"
    (прогон / "export").mkdir(parents=True)
    (прогон / "index").mkdir(parents=True)
    (прогон / "export" / "classifier_int8.tflite").write_bytes(b"int8")
    (прогон / "export" / "export_report.json").write_text(
        json.dumps(ОТЧЁТ_ЭКСПОРТА), encoding="utf-8"
    )
    (прогон / "index" / "catalog_index.bin").write_bytes(b"CLIX")
    (прогон / "index" / "catalog_meta.json").write_text(json.dumps(мета), encoding="utf-8")
    # Отчёт сборки индекса — не украшение фикстуры: `check_provenance` без него роняет
    # прогон на fp32-индексе, потому что один файл чекпоинта содержит и `model_state`, и
    # `ema_state`, а хеш файла их не различает (`export/provenance.py:61-65`). Режим весов
    # тот же, что в манифесте экспорта.
    (прогон / "index" / "report.json").write_text(json.dumps({"weights": "raw"}), encoding="utf-8")
    детектор = tmp_path / "detector_fp32.tflite"
    детектор.write_bytes(b"yolo")
    (tmp_path / "detector_report.json").write_text(json.dumps(ОТЧЁТ_ДЕТЕКТОРА), encoding="utf-8")
    return прогон, детектор


def test_stage_кладёт_модели_плейсхолдеры_и_манифест(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    куда = tmp_path / "models"

    манифест = stage(прогон, детектор, куда)

    assert sorted(путь.name for путь in куда.iterdir()) == [
        "calibration.json",
        "catalog_index.bin",
        "catalog_meta.json",
        "classifier_int8.tflite",
        "detector_fp32.tflite",
        "models_manifest.json",
        "thresholds.json",
    ]
    записанный = json.loads(манифест.read_text(encoding="utf-8"))
    assert записанный["detector"]["imgsz"] == 416


def test_stage_чистит_каталог_от_прошлого_набора(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    куда = tmp_path / "models"
    куда.mkdir()
    # Через `.encode`, а не литералом `b"..."`: байтовый литерал держит только ASCII.
    (куда / "classifier_int8.tflite").write_bytes("старая модель".encode())
    (куда / "лишний.tflite").write_bytes("от прошлого прогона".encode())

    stage(прогон, детектор, куда)

    assert not (куда / "лишний.tflite").exists()
    assert (куда / "classifier_int8.tflite").read_bytes() == b"int8"


def test_stage_роняет_на_расхождении_происхождения(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета={**МЕТА, "model_weights_sha256": "ff" * 32})

    with pytest.raises(Exception, match="sha256"):
        stage(прогон, детектор, tmp_path / "models")


def test_stage_читает_из_паритета_путь_от_рабочего_каталога(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`verify_parity` записывает путь индекса ровно как его набрали в командной строке.

    Гоняют её из корня репозитория, то есть в `parity.json` лежит `runs/<прогон>/index_int8`
    — путь от рабочего каталога, а не от прогона. Прочтение только от прогона давало бы
    `run/runs/<прогон>/index_int8` с диагнозом «индекс не собран», то есть путь по умолчанию,
    ради которого `parity.json` и читается, не работал бы в штатном вызове.
    """
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    # Индекс на выходах квантованной модели: он и отличает выбранный каталог от `index/`
    # — другими байтами и пометкой происхождения в манифесте.
    хеш_int8 = hashlib.sha256(b"int8").hexdigest()
    индекс = прогон / "index_int8"
    индекс.mkdir()
    (индекс / "catalog_index.bin").write_bytes(b"INT8")
    (индекс / "catalog_meta.json").write_text(
        json.dumps({**МЕТА, "model_weights_sha256": хеш_int8}), encoding="utf-8"
    )
    (индекс / "index_manifest.json").write_text(
        json.dumps(
            {
                "source_model_sha256": хеш_int8,
                "checkpoint_sha256": ЧЕКПОИНТ_SHA,
                "preprocess_version": 1,
            }
        ),
        encoding="utf-8",
    )
    (прогон / "export" / "parity.json").write_text(
        json.dumps({"manifest": {"index_dir": "run/index_int8"}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    куда = tmp_path / "models"

    манифест = stage(прогон, детектор, куда)

    assert (куда / "catalog_index.bin").read_bytes() == b"INT8"
    записанный = json.loads(манифест.read_text(encoding="utf-8"))
    assert записанный["provenance"]["index_source"] == "tflite"


def test_stage_называет_оба_прочтения_пути_из_паритета(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Не нашлось ни от прогона, ни от рабочего каталога — отказ обязан назвать оба места.

    В `parity.json` лежит строка без корня, и по ней не видно, от чего её считали. Диагноз
    «индекс не собран» указал бы на один из двух путей, а искать пользователю по обоим.
    """
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    (прогон / "export" / "parity.json").write_text(
        json.dumps({"manifest": {"index_dir": "чужой/index_int8"}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as отказ:
        stage(прогон, детектор, tmp_path / "models")

    текст = str(отказ.value)
    # Ассерты на формулировку, а не только на пути: в до-фиксном сообщении о несобранном
    # индексе оба пути тоже лежали подстроками, и тест прошёл бы на сломанном коде.
    assert "записанный index_dir чужой/index_int8 — не каталог" in текст
    assert f"ни от прогона ({прогон / 'чужой' / 'index_int8'})" in текст
    assert f"ни от текущего рабочего каталога ({Path('чужой') / 'index_int8'})" in текст
    assert "--index" in текст


def test_stage_не_создаёт_каталог_при_неполном_контракте(tmp_path: Path) -> None:
    """Неполный контракт обязан остановить раскладку до сноса каталога назначения.

    Проверка внутри `build_manifest` срабатывает уже после `rmtree` и четырёх `copy2`: в
    дереве сборки остались бы четыре модели без манифеста, а инструментальные тесты приняли
    бы этот набор за настоящий.
    """
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    урезанный = {ключ: значение for ключ, значение in КОНТРАКТ.items() if ключ != "max_det"}
    (tmp_path / "detector_report.json").write_text(
        json.dumps({**ОТЧЁТ_ДЕТЕКТОРА, "contract": урезанный}), encoding="utf-8"
    )
    куда = tmp_path / "models"

    with pytest.raises(ManifestError, match="max_det"):
        stage(прогон, детектор, куда)

    assert not куда.exists()


def test_stage_пишет_плейсхолдеры_порогов(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    выход = tmp_path / "models"

    stage(прогон, детектор, выход)

    пороги = json.loads((выход / "thresholds.json").read_text(encoding="utf-8"))
    assert пороги["tuned"] is False
    assert пороги["sim_min"] == 0.42
    assert пороги["margin_min"] == 0.05
    assert пороги["min_observations"] == 3
    assert пороги["min_short_side_px"] == 120

    калибровка = json.loads((выход / "calibration.json").read_text(encoding="utf-8"))
    assert калибровка["tuned"] is False
    assert len(калибровка["knots"]) >= 2
    assert калибровка["clip"] == [0.01, 0.98]


def test_манифест_описывает_плейсхолдеры(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    выход = tmp_path / "models"

    путь = stage(прогон, детектор, выход)

    файлы = json.loads(путь.read_text(encoding="utf-8"))["files"]
    assert файлы["thresholds"]["name"] == "thresholds.json"
    assert файлы["calibration"]["name"] == "calibration.json"
    assert файлы["thresholds"]["size"] == (выход / "thresholds.json").stat().st_size


def test_stage_fp32_разкладывает_модель_под_именем_точности(tmp_path: Path) -> None:
    """`--precision fp32` кладёт `classifier_fp32.tflite` и пишет это имя в манифест.

    Android читает имя классификатора из манифеста (`files.classifier.name`), а не по
    константе, поэтому именно оно определяет, какой файл поедет на устройство.
    """
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    (прогон / "export" / "classifier_fp32.tflite").write_bytes(b"fp32")
    куда = tmp_path / "models"

    путь = stage(прогон, детектор, куда, precision="fp32")

    имена = sorted(п.name for п in куда.iterdir())
    assert "classifier_fp32.tflite" in имена
    assert "classifier_int8.tflite" not in имена
    записанный = json.loads(путь.read_text(encoding="utf-8"))
    assert записанный["files"]["classifier"]["name"] == "classifier_fp32.tflite"


def test_stage_умолчание_точности_кладёт_int8(tmp_path: Path) -> None:
    """Без флага точности путь M-1 не меняется: разложен `classifier_int8.tflite`."""
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    куда = tmp_path / "models"

    путь = stage(прогон, детектор, куда)

    записанный = json.loads(путь.read_text(encoding="utf-8"))
    assert записанный["files"]["classifier"]["name"] == "classifier_int8.tflite"


ПОРОГИ_ПОДОБРАНЫ: dict[str, Any] = {
    **ПЛЕЙСХОЛДЕР_ПОРОГОВ,
    "tuned": True,
    "sim_min": 0.55,
    "margin_min": 0.03,
    "tuned_on": {"val_n": 1500, "neg_n": 2400, "achieved_tpr": 0.6, "achieved_fpr": 0.08},
}
ШКАЛА_ПОДОБРАНА: dict[str, Any] = {
    "version": 1,
    "method": "isotonic_piecewise_linear",
    "tuned": True,
    "knots": [[0.55, 0.3], [0.7, 0.6], [0.9, 0.95]],
    "clip": [0.01, 0.98],
}


def _json(путь: Path, документ: dict[str, Any]) -> Path:
    путь.write_text(json.dumps(документ), encoding="utf-8")
    return путь


def test_stage_кладёт_подобранные_пороги_и_шкалу(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    выход = tmp_path / "models"

    stage(
        прогон,
        детектор,
        выход,
        thresholds=_json(tmp_path / "t.json", ПОРОГИ_ПОДОБРАНЫ),
        calibration=_json(tmp_path / "c.json", ШКАЛА_ПОДОБРАНА),
    )

    assert json.loads((выход / "thresholds.json").read_text("utf-8")) == ПОРОГИ_ПОДОБРАНЫ
    assert json.loads((выход / "calibration.json").read_text("utf-8")) == ШКАЛА_ПОДОБРАНА


def test_stage_требует_оба_файла_вместе(tmp_path: Path) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    with pytest.raises(ManifestError, match="оба"):
        stage(
            прогон,
            детектор,
            tmp_path / "models",
            thresholds=_json(tmp_path / "t.json", ПОРОГИ_ПОДОБРАНЫ),
        )


@pytest.mark.parametrize(
    ("правка", "ошибка"),
    [
        ({"tuned": False}, "tuned"),
        (
            {"tuned_on": {"val_n": 0, "neg_n": 10, "achieved_tpr": 0.5, "achieved_fpr": 0.1}},
            "val_n",
        ),
        ({"sim_min": 1.2}, "sim_min"),
    ],
)
def test_stage_отвергает_негодные_пороги(
    tmp_path: Path, правка: dict[str, Any], ошибка: str
) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    выход = tmp_path / "models"
    with pytest.raises(ManifestError, match=ошибка):
        stage(
            прогон,
            детектор,
            выход,
            thresholds=_json(tmp_path / "t.json", {**ПОРОГИ_ПОДОБРАНЫ, **правка}),
            calibration=_json(tmp_path / "c.json", ШКАЛА_ПОДОБРАНА),
        )
    assert not выход.exists()


@pytest.mark.parametrize(
    ("правка", "ошибка"),
    [
        ({"tuned": False}, "tuned"),
        ({"knots": [[0.5, 0.3], [0.5, 0.6]]}, "возраст"),
        ({"knots": [[0.5 + i / 100, i / 40] for i in range(33)]}, "32"),
        ({"knots": [[0.5, 0.3], [0.9, 1.2]]}, r"\[0, 1\]"),
    ],
)
def test_stage_отвергает_негодную_шкалу(
    tmp_path: Path, правка: dict[str, Any], ошибка: str
) -> None:
    прогон, детектор = _разложить_прогон(tmp_path, мета=МЕТА)
    with pytest.raises(ManifestError, match=ошибка):
        stage(
            прогон,
            детектор,
            tmp_path / "models",
            thresholds=_json(tmp_path / "t.json", ПОРОГИ_ПОДОБРАНЫ),
            calibration=_json(tmp_path / "c.json", {**ШКАЛА_ПОДОБРАНА, **правка}),
        )
