"""Профиль редакции номеров, его провенанс и единая фабрика plate-детекторов.

Профиль проверяется на трёх свойствах, ради которых он и заводится: он называет все
параметры, реально влияющие на детекцию (иначе «тот же профиль» ничего не значит), его
идентификатор меняется от любого свидетеля прогона, и его сериализация не зависит от
порядка записей в памяти.

Фабрика проверяется на подставных классах детекторов: настоящие требуют весов и
`ultralytics`, которых в базовом окружении нет, а доказывать здесь нужно ровно одно —
что параметры профиля доезжают до конструкторов, а не подменяются умолчаниями классов.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.harvest.detectors import DetectorPin
from pipeline.harvest.redaction_profile import (
    CURRENT_REDACTION_PROFILE,
    DETECTOR_STAGE_SOURCES,
    DetectorEntry,
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
    assert len(CURRENT_REDACTION_PROFILE.detectors) == 2
    plate = CURRENT_REDACTION_PROFILE.detectors[0]
    plate_hires = CURRENT_REDACTION_PROFILE.detectors[1]
    assert plate.instance == "plate"
    assert plate.weights == "plate"
    assert plate.params == DetectorProfile(0.5, 1280, 0.25)

    assert plate_hires.instance == "plate_hires"
    assert plate_hires.weights == "plate_hires"
    assert plate_hires.params == DetectorProfile(0.25, 1280, 0.25)


def test_профиль_из_одного_двух_и_трёх_детекторов() -> None:
    p1 = DetectorProfile(0.5, 1280, 0.25)
    p2 = DetectorProfile(0.25, 1280, 0.25)
    p3 = DetectorProfile(0.3, 640, 0.25)

    prof1 = RedactionProfile((DetectorEntry("plate", "plate", p1),))
    assert len(prof1.detectors) == 1
    assert prof1.detectors[0].instance == "plate"

    prof2 = RedactionProfile(
        (
            DetectorEntry("plate", "plate", p1),
            DetectorEntry("plate_hires", "plate_hires", p2),
        )
    )
    assert len(prof2.detectors) == 2

    prof3 = RedactionProfile(
        (
            DetectorEntry("plate", "plate", p1),
            DetectorEntry("plate_hires_1280", "plate_hires", p2),
            DetectorEntry("plate_hires_640", "plate_hires", p3),
        )
    )
    assert len(prof3.detectors) == 3
    assert [d.instance for d in prof3.detectors] == [
        "plate",
        "plate_hires_1280",
        "plate_hires_640",
    ]


def test_запрет_пустого_набора_детекторов() -> None:
    with pytest.raises(ValueError, match="не может быть пустым"):
        RedactionProfile(())


def test_запрет_повторяющихся_имён_детекторов() -> None:
    p = DetectorProfile(0.5, 1280, 0.25)
    with pytest.raises(ValueError, match="повторяющиеся"):
        RedactionProfile(
            (
                DetectorEntry("plate", "plate", p),
                DetectorEntry("plate", "plate", p),
            )
        )


def test_недопустимое_имя_экземпляра() -> None:
    p = DetectorProfile(0.5, 1280, 0.25)
    with pytest.raises(ValueError, match="недопустимое имя экземпляра"):
        DetectorEntry("Plate-1", "plate", p)
    with pytest.raises(ValueError, match="недопустимое имя экземпляра"):
        DetectorEntry("", "plate", p)


def test_канонический_порядок_и_стабильность_json() -> None:
    p1 = DetectorProfile(0.5, 1280, 0.25)
    p2 = DetectorProfile(0.25, 1280, 0.25)
    e1 = DetectorEntry("plate", "plate", p1)
    e2 = DetectorEntry("plate_hires", "plate_hires", p2)

    prof_forward = RedactionProfile((e1, e2))
    prof_backward = RedactionProfile((e2, e1))

    assert prof_forward.detectors == prof_backward.detectors
    json_forward = canonical_profile_json(prof_forward)
    json_backward = canonical_profile_json(prof_backward)
    assert json_forward == json_backward
    assert json_forward.endswith(b"\n")


def test_profile_id_меняется_при_перестановке_имён_и_смене_параметров() -> None:
    pins = _pins(plate_sha="a" * 64)
    base = profile_id(
        CURRENT_REDACTION_PROFILE,
        pins=pins,
        git_sha="a" * 40,
        stage_sha256="b" * 64,
    )

    p_swap = RedactionProfile(
        (
            DetectorEntry(
                "plate",
                "plate",
                DetectorProfile(0.25, 1280, 0.25),
            ),
            DetectorEntry(
                "plate_hires",
                "plate_hires",
                DetectorProfile(0.5, 1280, 0.25),
            ),
        )
    )
    assert (
        profile_id(
            p_swap,
            pins=pins,
            git_sha="a" * 40,
            stage_sha256="b" * 64,
        )
        != base
    )

    p_param = RedactionProfile(
        (
            DetectorEntry(
                "plate",
                "plate",
                DetectorProfile(0.4, 1280, 0.25),
            ),
            CURRENT_REDACTION_PROFILE.detectors[1],
        )
    )
    assert (
        profile_id(
            p_param,
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


def test_profile_id_не_удваивает_свидетель_при_разделении_весов() -> None:
    pins = _pins()
    p1 = DetectorProfile(0.5, 1280, 0.25)
    p2 = DetectorProfile(0.25, 640, 0.25)
    prof = RedactionProfile(
        (
            DetectorEntry("plate_hires_1280", "plate_hires", p1),
            DetectorEntry("plate_hires_640", "plate_hires", p2),
        )
    )
    pid = profile_id(prof, pins=pins, git_sha="a" * 40, stage_sha256="b" * 64)
    assert len(pid) == 64


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


def test_фабрика_передаёт_параметры_профиля_детекторам() -> None:
    """Параметры берутся из профиля, а не из умолчаний классов детекторов."""
    профиль = RedactionProfile(
        (
            DetectorEntry("plate", "plate", DetectorProfile(0.4, 640, 0.2, 0.4)),
            DetectorEntry("plate_hires", "plate_hires", DetectorProfile(0.3, 512, 0.1, 0.6)),
        )
    )
    with (
        patch("pipeline.harvest.backends.OnnxBoxDetector") as ssd,
        patch("pipeline.detectors.yolo.YoloDetector") as yolo,
        patch("pipeline.harvest.tiling.TiledDetector") as нарезка,
    ):
        детекторы = build_plate_detectors(профиль, models_dir=Path("models"), pins=_pins())

    assert len(детекторы) == 2
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


def test_текущий_профиль_повторяет_прежний_набор_детекторов_и_умолчания() -> None:
    """Шаг 0 не меняет поведение: текущий профиль повторяет прежнюю пару детекторов."""
    with (
        patch("pipeline.harvest.backends.OnnxBoxDetector") as ssd,
        patch("pipeline.detectors.yolo.YoloDetector") as yolo,
        patch("pipeline.harvest.tiling.TiledDetector") as нарезка,
    ):
        детекторы = build_plate_detectors(
            CURRENT_REDACTION_PROFILE, models_dir=Path("models"), pins=_pins()
        )

    assert len(детекторы) == 2
    assert [имя for имя, _ in детекторы] == ["plate", "plate_hires"]
    assert all(hasattr(детектор, "detect") for _, детектор in детекторы)

    assert ssd.call_args.kwargs == {"conf": 0.5}
    assert yolo.call_args.kwargs["conf"] == 0.25
    assert [вызов.kwargs for вызов in нарезка.call_args_list] == [
        {"tile": 1280, "overlap": 0.25, "iou_merge": 0.45},
        {"tile": 1280, "overlap": 0.25, "iou_merge": 0.45},
    ]


def test_фабрика_строит_произвольное_число_детекторов() -> None:
    p1 = DetectorProfile(0.5, 1280, 0.25)
    p2 = DetectorProfile(0.25, 1280, 0.25)
    p3 = DetectorProfile(0.25, 640, 0.25)
    профиль = RedactionProfile(
        (
            DetectorEntry("plate", "plate", p1),
            DetectorEntry("plate_hires_1280", "plate_hires", p2),
            DetectorEntry("plate_hires_640", "plate_hires", p3),
        )
    )
    with (
        patch("pipeline.harvest.backends.OnnxBoxDetector"),
        patch("pipeline.detectors.yolo.YoloDetector"),
        patch("pipeline.harvest.tiling.TiledDetector"),
    ):
        детекторы = build_plate_detectors(профиль, models_dir=Path("models"), pins=_pins())

    assert len(детекторы) == 3
    assert [имя for имя, _ in детекторы] == ["plate", "plate_hires_1280", "plate_hires_640"]


def test_фабрика_требует_пин_на_каждое_имя_весов() -> None:
    p = DetectorProfile(0.5, 1280, 0.25)
    профиль = RedactionProfile((DetectorEntry("custom", "missing_weights", p),))
    with pytest.raises(ValueError, match="нет пина для весов"):
        build_plate_detectors(профиль, models_dir=Path("models"), pins=_pins())
