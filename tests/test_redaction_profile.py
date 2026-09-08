"""Профиль редакции номеров, его провенанс и единая фабрика plate-детекторов.

Профиль проверяется на трёх свойствах, ради которых он и заводится: он называет все
параметры, реально влияющие на детекцию (иначе «тот же профиль» ничего не значит), его
идентификатор меняется от любого свидетеля прогона, и его сериализация не зависит от
порядка полей в памяти.

Фабрика проверяется на подставных классах детекторов: настоящие требуют весов и
`ultralytics`, которых в базовом окружении нет, а доказывать здесь нужно ровно одно —
что параметры профиля доезжают до конструкторов, а не подменяются умолчаниями классов.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.harvest.detectors import DetectorPin
from pipeline.harvest.redaction_profile import (
    CURRENT_REDACTION_PROFILE,
    DETECTOR_STAGE_SOURCES,
    DetectorProfile,
    RedactionProfile,
    build_plate_detectors,
    canonical_profile_json,
    detector_stage_sha256,
    profile_id,
)


def _пин(имя: str, формат: str, sha: str) -> DetectorPin:
    return DetectorPin(
        name=имя,
        format=формат,
        url=f"https://example.invalid/{имя}.{формат}",
        revision="v1",
        license="mit",
        checked="2026-09-07",
        sha256=sha,
        derived_from="",
    )


def _pins(*, plate_sha: str = "1" * 64) -> dict[str, DetectorPin]:
    """Три обязательных пина с фиксированными метаданными; меняется только SHA256."""
    return {
        "plate": _пин("plate", "onnx", plate_sha),
        "plate_hires": _пин("plate_hires", "pt", "2" * 64),
        "face": _пин("face", "onnx", "3" * 64),
    }


def test_current_profile_names_every_effective_parameter() -> None:
    assert CURRENT_REDACTION_PROFILE.plate == DetectorProfile(0.5, 1280, 0.25)
    assert CURRENT_REDACTION_PROFILE.plate_hires == DetectorProfile(0.25, 1280, 0.25)


def test_profile_id_changes_with_parameters_code_weights_and_git() -> None:
    pins = _pins(plate_sha="a" * 64)
    base = profile_id(
        CURRENT_REDACTION_PROFILE,
        pins=pins,
        git_sha="a" * 40,
        stage_sha256="b" * 64,
    )
    changed = replace(
        CURRENT_REDACTION_PROFILE,
        plate=DetectorProfile(0.4, 1280, 0.25),
    )
    assert (
        profile_id(
            changed,
            pins=pins,
            git_sha="a" * 40,
            stage_sha256="b" * 64,
        )
        != base
    )
    assert (
        profile_id(
            CURRENT_REDACTION_PROFILE,
            pins=pins,
            git_sha="c" * 40,
            stage_sha256="b" * 64,
        )
        != base
    )
    assert (
        profile_id(
            CURRENT_REDACTION_PROFILE,
            pins=pins,
            git_sha="a" * 40,
            stage_sha256="d" * 64,
        )
        != base
    )
    assert (
        profile_id(
            CURRENT_REDACTION_PROFILE,
            pins=_pins(plate_sha="e" * 64),
            git_sha="a" * 40,
            stage_sha256="b" * 64,
        )
        != base
    )


def test_canonical_json_is_order_stable() -> None:
    first = canonical_profile_json(CURRENT_REDACTION_PROFILE)
    second = canonical_profile_json(CURRENT_REDACTION_PROFILE)
    assert first == second
    assert first.endswith(b"\n")


def test_идентификатор_профиля_воспроизводится_на_равных_свидетелях() -> None:
    """Без этого предыдущий тест удовлетворяла бы и случайная строка."""
    первый = profile_id(
        CURRENT_REDACTION_PROFILE, pins=_pins(), git_sha="a" * 40, stage_sha256="b" * 64
    )
    второй = profile_id(
        CURRENT_REDACTION_PROFILE, pins=_pins(), git_sha="a" * 40, stage_sha256="b" * 64
    )
    assert первый == второй
    assert len(первый) == 64


@pytest.mark.parametrize(
    "поля",
    [
        (1.5, 1280, 0.25, 0.45),
        (-0.1, 1280, 0.25, 0.45),
        (0.5, 0, 0.25, 0.45),
        (0.5, 1280, 1.0, 0.45),
        (0.5, 1280, 0.25, 0.0),
        (0.5, 1280, 0.25, 1.5),
    ],
)
def test_недопустимые_параметры_детектора_отвергаются(
    поля: tuple[float, int, float, float],
) -> None:
    """Профиль вне области определения — это молча другой прогон, а не оттенок настройки."""
    with pytest.raises(ValueError):
        DetectorProfile(*поля)


def test_стадия_накрывает_все_файлы_детекции() -> None:
    """Хеш стадии обязан покрывать каждый файл, способный изменить найденные боксы."""
    assert {путь.name for путь in DETECTOR_STAGE_SOURCES} == {
        "redact.py",
        "tiling.py",
        "backends.py",
        "yolo.py",
        "redaction_profile.py",
    }
    assert len(detector_stage_sha256()) == 64


def test_хеш_стадии_меняется_с_байтами_исходника(tmp_path: Path) -> None:
    исходник = tmp_path / "stage.py"
    исходник.write_bytes(b"conf = 0.5\n")
    with patch("pipeline.harvest.redaction_profile.DETECTOR_STAGE_SOURCES", (исходник,)):
        до = detector_stage_sha256()
        исходник.write_bytes(b"conf = 0.4\n")
        после = detector_stage_sha256()
    assert до != после


def test_фабрика_передаёт_параметры_профиля_обоим_детекторам() -> None:
    """Параметры берутся из профиля, а не из умолчаний классов детекторов."""
    профиль = RedactionProfile(
        plate=DetectorProfile(0.4, 640, 0.2, 0.4),
        plate_hires=DetectorProfile(0.3, 512, 0.1, 0.6),
    )
    with (
        patch("pipeline.harvest.backends.OnnxBoxDetector") as ssd,
        patch("pipeline.detectors.yolo.YoloDetector") as yolo,
        patch("pipeline.harvest.tiling.TiledDetector") as нарезка,
    ):
        build_plate_detectors(профиль, models_dir=Path("models"), pins=_pins())

    assert ssd.call_args.args[0] == Path("models") / "plate.onnx"
    assert ssd.call_args.kwargs == {"conf": 0.4}
    assert yolo.call_args.kwargs == {
        "weights": str(Path("models") / "plate_hires.pt"),
        "conf": 0.3,
    }
    assert [вызов.kwargs for вызов in нарезка.call_args_list] == [
        {"tile": 640, "overlap": 0.2, "iou_merge": 0.4},
        {"tile": 512, "overlap": 0.1, "iou_merge": 0.6},
    ]


def test_фабрика_текущего_профиля_повторяет_прежние_умолчания() -> None:
    """Шаг 0 не меняет поведение: явные параметры совпадают с прежними умолчаниями."""
    with (
        patch("pipeline.harvest.backends.OnnxBoxDetector") as ssd,
        patch("pipeline.detectors.yolo.YoloDetector") as yolo,
        patch("pipeline.harvest.tiling.TiledDetector") as нарезка,
    ):
        build_plate_detectors(CURRENT_REDACTION_PROFILE, models_dir=Path("models"), pins=_pins())

    assert ssd.call_args.kwargs == {"conf": 0.5}
    assert yolo.call_args.kwargs["conf"] == 0.25
    assert [вызов.kwargs for вызов in нарезка.call_args_list] == [
        {"tile": 1280, "overlap": 0.25, "iou_merge": 0.45},
        {"tile": 1280, "overlap": 0.25, "iou_merge": 0.45},
    ]


def test_фабрика_возвращает_пару_нарезанных_детекторов() -> None:
    with (
        patch("pipeline.harvest.backends.OnnxBoxDetector"),
        patch("pipeline.detectors.yolo.YoloDetector"),
    ):
        детекторы = build_plate_detectors(
            CURRENT_REDACTION_PROFILE, models_dir=Path("models"), pins=_pins()
        )

    assert len(детекторы) == 2
    assert all(hasattr(детектор, "detect") for детектор in детекторы)
