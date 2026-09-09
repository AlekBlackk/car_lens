# tests/test_harvest_rebuild.py
import csv
import hashlib
import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from pipeline.crop import Detection
from pipeline.harvest.cache import cache_path
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.detectors import DetectorPin
from pipeline.harvest.jpeg import encode_jpeg, write_jpeg_atomic
from pipeline.harvest.manifest import HarvestRow, append_row, load_manifest
from pipeline.harvest.redact import PlateDetectionTrace, RawPlateObservation, RedactionResult
from pipeline.harvest.redaction_profile import CURRENT_REDACTION_PROFILE
from pipeline.harvest.tiling import TileObservation
from pipeline.redaction_calibration.model import FrameProfileResult
from scripts.rebuild_redaction import (
    AuditCounts,
    ObservationRecord,
    RebuildDeps,
    RebuildResult,
    _observation_records,
    build_parser,
    main,
    preflight,
    prepare_workspace,
    rebuild_row,
    resolve_manifest_target,
)


class Фикс:
    def __init__(self, боксы: list[list[Detection]]) -> None:
        self.боксы = list(боксы)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.боксы.pop(0) if self.боксы else []


def _оригинал(root: Path, sha: str) -> None:
    кадр = np.full((400, 800, 3), 127, dtype=np.uint8)
    путь = cache_path(root, "commons", sha, "jpg")
    путь.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(путь), кадр)


def _строка(
    sha: str, class_id: str = "lada__granta", basis: str = "category:Lada Granta"
) -> HarvestRow:
    return HarvestRow(
        file=f"{sha}.jpg",
        class_id=class_id,
        source="commons",
        source_item_id="1",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="A",
        license="CC0",
        license_tier=1,
        checked="2026-08-28",
        basis=basis,
        sha256=sha,
        phash=0,
        clip_rank=1,
        clip_margin=0.0,
        plates=0,
        faces=0,
        bbox_x=12.0,
        bbox_y=34.0,
        bbox_w=500.0,
        bbox_h=300.0,
        bbox_area_frac=0.42,
        det_conf=0.91,
        audit="ok",
    )


def _deps(
    input_root: Path,
    plate: Фикс,
    output_root: Path | None = None,
    *,
    profile: str = "test-profile-id",
) -> RebuildDeps:
    return RebuildDeps(
        vehicle_detector=Фикс([[Detection(40.0, 40.0, 360.0, 300.0, 0.95, 2)]]),
        plate_detectors=(plate,),
        face_detector=Фикс([[]]),
        input_root=input_root,
        output_root=output_root or input_root,
        profile_id=profile,
    )


def test_префлайт_ловит_отсутствующий_оригинал(tmp_path: Path) -> None:
    _оригинал(tmp_path, "a" * 64)
    пропало = preflight([_строка("a" * 64), _строка("b" * 64)], tmp_path)
    assert пропало == ["b" * 64]


def test_префлайт_пустой_при_полном_кэше(tmp_path: Path) -> None:
    _оригинал(tmp_path, "a" * 64)
    _оригинал(tmp_path, "b" * 64)
    пропало = preflight([_строка("a" * 64), _строка("b" * 64)], tmp_path)
    assert пропало == []


def _неизменённый_кроп() -> np.ndarray:
    return np.full((348, 418, 3), 127, dtype=np.uint8)


def _старый_кроп(root: Path, row: HarvestRow, image: np.ndarray) -> Path:
    путь = root / "crops" / row.class_id / row.file
    путь.parent.mkdir(parents=True, exist_ok=True)
    write_jpeg_atomic(путь, encode_jpeg(image))
    return путь


@pytest.mark.parametrize(
    "unsafe_file",
    [
        "../../sentinel.jpg",
        "/tmp/sentinel.jpg",
        "..\\..\\sentinel.jpg",
        "\\rooted.jpg",
        "\\\\server\\share\\crop.jpg",
        "C:\\tmp\\sentinel.jpg",
        "C:crop.jpg",
        "crop.jpg:stream",
        "",
        ".",
        "..",
    ],
)
def test_rebuild_отвергает_небезопасный_file_до_записи(tmp_path: Path, unsafe_file: str) -> None:
    sha = "a" * 64
    строка = replace(_строка(sha), file=unsafe_file)
    _оригинал(tmp_path, sha)
    sentinel = tmp_path / "sentinel.jpg"
    sentinel.write_bytes(b"do not overwrite")

    with pytest.raises(ValueError, match="имя кропа"):
        rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex())

    assert sentinel.read_bytes() == b"do not overwrite"
    assert not (tmp_path / "crops").exists()


@pytest.mark.parametrize(
    "unsafe_class",
    ["../../escape", "..", "/abs", "C:\\tmp", "a/b", "a\\b", ""],
)
def test_rebuild_отвергает_небезопасный_class_id_до_записи(
    tmp_path: Path, unsafe_class: str
) -> None:
    # `load_manifest` не ограничивает колонку класса ничем, а `validate` — только
    # непустотой: без этой проверки `..` в классе писал бы кроп мимо workspace.
    sha = "a" * 64
    строка = replace(_строка(sha), class_id=unsafe_class)
    _оригинал(tmp_path, sha)
    sentinel = tmp_path / "sentinel.jpg"
    sentinel.write_bytes(b"do not overwrite")

    with pytest.raises(ValueError, match="имя класса"):
        rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex())

    assert sentinel.read_bytes() == b"do not overwrite"
    assert not (tmp_path / "crops").exists()


def test_rebuild_использует_file_строки_для_старого_и_нового_кропа(tmp_path: Path) -> None:
    sha = "a" * 64
    строка = _строка(sha)
    legacy = replace(строка, file="legacy-import.jpg")
    _оригинал(tmp_path, sha)
    legacy_path = _старый_кроп(tmp_path, legacy, _неизменённый_кроп())
    sha_path = tmp_path / "crops" / legacy.class_id / f"{sha}.jpg"

    причина, новая = rebuild_row(legacy, _deps(tmp_path, Фикс([])), DedupIndex())

    assert причина == "accepted"
    assert новая is not None
    assert новая.file == "legacy-import.jpg"
    assert новая.audit == "ok"
    assert legacy_path.exists()
    assert not sha_path.exists()


def test_rebuild_переносит_audit_только_при_равных_пикселях(tmp_path: Path) -> None:
    sha = "a" * 64
    строка = _строка(sha)
    _оригинал(tmp_path, sha)
    старый = _неизменённый_кроп()
    _старый_кроп(tmp_path, строка, старый)

    причина, новая = rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex())

    assert причина == "accepted"
    assert новая is not None
    assert новая.audit == "ok"


def test_rebuild_сбрасывает_audit_для_изменённых_пикселей(tmp_path: Path) -> None:
    sha = "a" * 64
    строка = _строка(sha)
    _оригинал(tmp_path, sha)
    _старый_кроп(tmp_path, строка, np.zeros((260, 320, 3), dtype=np.uint8))

    причина, новая = rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex())

    assert причина == "accepted"
    assert новая is not None
    assert новая.audit == ""


def test_rebuild_без_старого_кропа_сбрасывает_audit_но_принимает_строку(
    tmp_path: Path,
) -> None:
    sha = "a" * 64
    _оригинал(tmp_path, sha)

    причина, строка = rebuild_row(_строка(sha), _deps(tmp_path, Фикс([])), DedupIndex())

    assert причина == "accepted"
    assert строка is not None
    assert строка.audit == ""


def test_rebuild_с_пустым_старым_кропом_сбрасывает_audit_но_принимает_строку(
    tmp_path: Path,
) -> None:
    sha = "a" * 64
    строка = _строка(sha)
    _оригинал(tmp_path, sha)
    путь = tmp_path / "crops" / строка.class_id / строка.file
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(b"")

    причина, новая = rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex())

    assert причина == "accepted"
    assert новая is not None
    assert новая.audit == ""


def test_rebuild_с_битым_старым_кропом_сбрасывает_audit_но_принимает_строку(
    tmp_path: Path,
) -> None:
    sha = "a" * 64
    строка = _строка(sha)
    _оригинал(tmp_path, sha)
    путь = tmp_path / "crops" / строка.class_id / строка.file
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(b"not a jpeg")

    причина, новая = rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex())

    assert причина == "accepted"
    assert новая is not None
    assert новая.audit == ""


def test_rebuild_reset_audit_сбрасывает_равный_старый_кроп(tmp_path: Path) -> None:
    sha = "a" * 64
    строка = _строка(sha)
    _оригинал(tmp_path, sha)
    _старый_кроп(tmp_path, строка, _неизменённый_кроп())

    причина, новая = rebuild_row(строка, _deps(tmp_path, Фикс([])), DedupIndex(), reset_audit=True)

    assert причина == "accepted"
    assert новая is not None
    assert новая.audit == ""


def test_rebuild_пересчитывает_номера_и_сбрасывает_изменённый_аудит(tmp_path: Path) -> None:
    sha = "a" * 64
    _оригинал(tmp_path, sha)
    plate = Фикс([[Detection(120.0, 210.0, 80.0, 24.0, 0.9, 0)], [], [], []])
    причина, строка = rebuild_row(_строка(sha), _deps(tmp_path, plate), DedupIndex())
    assert причина == "accepted"
    assert строка is not None
    assert строка.plates == 1
    assert строка.audit == "", "изменившиеся пиксели сбрасывают старую отметку аудита"
    сохранённый_кроп = tmp_path / "crops" / "lada__granta" / f"{sha}.jpg"
    assert сохранённый_кроп.exists()


def test_rebuild_отклоняет_при_провале_редакции(tmp_path: Path) -> None:
    sha = "a" * 64
    _оригинал(tmp_path, sha)
    # Переживший бокс лежит в стороне от заливки (108, 206, 212, 237): прежний сценарий
    # из трёх одинаковых боксов моделировал собственную заливку редакции, а не номер,
    # который её пробил.
    plate = Фикс(
        [
            [Detection(120.0, 210.0, 80.0, 24.0, 0.9, 0)],
            [],
            [Detection(500.0, 60.0, 80.0, 24.0, 0.9, 0)],
            [],
        ]
    )
    причина, строка = rebuild_row(_строка(sha), _deps(tmp_path, plate), DedupIndex())
    assert причина == "redaction_failed"
    assert строка is None


def test_rebuild_отклоняет_при_дедупе(tmp_path: Path) -> None:
    sha1 = "a" * 64
    sha2 = "b" * 64
    _оригинал(tmp_path, sha1)
    _оригинал(tmp_path, sha2)
    dedup = DedupIndex()
    plate = Фикс([])
    причина1, строка1 = rebuild_row(_строка(sha1), _deps(tmp_path, plate), dedup)
    assert причина1 == "accepted"
    assert строка1 is not None

    причина2, строка2 = rebuild_row(_строка(sha2), _deps(tmp_path, plate), dedup)
    assert причина2 == "duplicate"
    assert строка2 is None


def test_rebuild_вытеснение_оставляет_категорийную_строку(tmp_path: Path) -> None:
    """§8: при кросс-классовом столкновении выигрывает категорийное заявление.

    Перестройка идёт по манифесту в его порядке, поэтому старая `search:`-строка попадает в
    индекс раньше, а новая `category:` приходит следом и получает `supersede`. Трактовка
    вытеснения как отказа выбрасывала бы из перестройки ровно те категорийные строки, ради
    которых затевался категорийный сбор.
    """
    sha_поисковая = "a" * 64
    sha_категорийная = "b" * 64
    _оригинал(tmp_path, sha_поисковая)
    _оригинал(tmp_path, sha_категорийная)
    dedup = DedupIndex()

    поисковая = _строка(sha_поисковая, "mitsubishi__pajero", "search:Mitsubishi Pajero")
    причина1, строка1 = rebuild_row(поисковая, _deps(tmp_path, Фикс([])), dedup)
    assert причина1 == "accepted"
    assert строка1 is not None

    категорийная = _строка(sha_категорийная, "mazda__cx5", "category:Mazda CX-5 (KE)")
    причина2, строка2 = rebuild_row(категорийная, _deps(tmp_path, Фикс([])), dedup)
    assert причина2 == "accepted", "вытеснение — выигрыш этой строки, а не отказ ей"
    assert строка2 is not None
    assert строка2.class_id == "mazda__cx5"
    assert (tmp_path / "crops" / "mazda__cx5" / f"{sha_категорийная}.jpg").exists()


def test_rebuild_отказывает_при_равных_заявлениях_разных_классов(tmp_path: Path) -> None:
    """Оба заявления категорийные — спор не решён, `conflict` остаётся отказом."""
    sha_первая = "a" * 64
    sha_вторая = "b" * 64
    _оригинал(tmp_path, sha_первая)
    _оригинал(tmp_path, sha_вторая)
    dedup = DedupIndex()

    первая = _строка(sha_первая, "mitsubishi__pajero", "category:Mitsubishi Pajero")
    причина1, _ = rebuild_row(первая, _deps(tmp_path, Фикс([])), dedup)
    assert причина1 == "accepted"

    вторая = _строка(sha_вторая, "mazda__cx5", "category:Mazda CX-5 (KE)")
    причина2, строка2 = rebuild_row(вторая, _deps(tmp_path, Фикс([])), dedup)
    assert причина2 == "conflict"
    assert строка2 is None


def test_rebuild_отклоняет_no_vehicle_и_multi_car(tmp_path: Path) -> None:
    sha = "a" * 64
    _оригинал(tmp_path, sha)

    deps_no_veh = RebuildDeps(
        vehicle_detector=Фикс([[]]),
        plate_detectors=(Фикс([]),),
        face_detector=Фикс([]),
        input_root=tmp_path,
        output_root=tmp_path,
        profile_id="test-profile-id",
    )
    причина, строка = rebuild_row(_строка(sha), deps_no_veh, DedupIndex())
    assert причина == "no_vehicle"
    assert строка is None

    deps_multi = RebuildDeps(
        vehicle_detector=Фикс(
            [
                [
                    Detection(20.0, 20.0, 300.0, 300.0, 0.95, 2),
                    Detection(350.0, 20.0, 300.0, 300.0, 0.95, 2),
                ]
            ]
        ),
        plate_detectors=(Фикс([]),),
        face_detector=Фикс([]),
        input_root=tmp_path,
        output_root=tmp_path,
        profile_id="test-profile-id",
    )
    причина_m, строка_m = rebuild_row(_строка(sha), deps_multi, DedupIndex())
    assert причина_m == "multi_car"
    assert строка_m is None


def test_rebuild_отклоняет_over_redacted(tmp_path: Path) -> None:
    sha = "a" * 64
    _оригинал(tmp_path, sha)
    plate = Фикс([[Detection(40.0, 40.0, 360.0, 300.0, 0.95, 0)], [], [], []])
    причина, строка = rebuild_row(_строка(sha), _deps(tmp_path, plate), DedupIndex())
    assert причина == "over_redacted"
    assert строка is None


def _pin(name: str, format: str, sha: str) -> DetectorPin:
    return DetectorPin(
        name=name,
        format=format,
        url=f"https://example.test/{name}.{format}",
        revision=f"{name}-revision",
        license="mit",
        checked="2026-09-07",
        sha256=sha,
        derived_from="",
    )


def _pins() -> dict[str, DetectorPin]:
    return {
        "face": _pin("face", "onnx", "1" * 64),
        "plate": _pin("plate", "onnx", "2" * 64),
        "plate_hires": _pin("plate_hires", "pt", "3" * 64),
    }


def test_build_parser_requires_out_root_and_removes_old_out() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as ошибка:
        parser.parse_args(["--models", "models_dir", "--data-root", "data_dir"])
    assert ошибка.value.code == 2

    args = parser.parse_args(
        [
            "--models",
            "models_dir",
            "--data-root",
            "data_dir",
            "--out-root",
            "workspace",
            "--reset-audit",
        ]
    )
    assert args.models == Path("models_dir")
    assert args.data_root == Path("data_dir")
    assert args.out_root == Path("workspace")
    assert args.reset_audit
    assert args.out is None
    assert not hasattr(args, "keep_audit")


def test_resolve_manifest_target_по_умолчанию_и_внутри_workspace(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()

    assert resolve_manifest_target(workspace, None) == workspace / "manifest.csv"
    assert resolve_manifest_target(workspace, Path("custom.csv")) == workspace / "custom.csv"
    # Относительный путь считается от workspace, а не от текущего каталога.
    assert resolve_manifest_target(workspace, Path("sub/custom.csv")) == (
        workspace / "sub" / "custom.csv"
    )


@pytest.mark.parametrize(
    "escape",
    ["../escape.csv", "sub/../../escape.csv", "..", "."],
)
def test_resolve_manifest_target_отвергает_выход_за_workspace(tmp_path: Path, escape: str) -> None:
    workspace = (tmp_path / "workspace").resolve()

    with pytest.raises(ValueError, match="за пределы workspace"):
        resolve_manifest_target(workspace, Path(escape))


def test_resolve_manifest_target_отвергает_абсолютный_путь_наружу(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()

    with pytest.raises(ValueError, match="за пределы workspace"):
        resolve_manifest_target(workspace, (tmp_path / "elsewhere.csv").resolve())


def test_resolve_manifest_target_отвергает_симлинк_наружу(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("создание симлинка недоступно в этом окружении")

    with pytest.raises(ValueError, match="за пределы workspace"):
        resolve_manifest_target(workspace, Path("link/escape.csv"))


@pytest.mark.parametrize(
    "reserved",
    [
        "crops",
        "REJECTS.CSV",
        "Report.Json",
        "PROFILE.JSON",
        "Observations.CSV",
        "PIXEL-DIFF.CSV",
        "Quality-Metrics.Json",
    ],
)
def test_resolve_manifest_target_отвергает_служебные_имена_без_учёта_регистра(
    tmp_path: Path, reserved: str
) -> None:
    workspace = (tmp_path / "workspace").resolve()

    with pytest.raises(ValueError, match="служебное имя"):
        resolve_manifest_target(workspace, Path(reserved))


@pytest.mark.parametrize("relation", ["equal", "output_inside_input", "input_inside_output"])
def test_prepare_workspace_rejects_equal_and_nested_roots(tmp_path: Path, relation: str) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    if relation == "equal":
        output_root = input_root
    elif relation == "output_inside_input":
        output_root = input_root / "workspace"
    else:
        output_root = tmp_path / "parent"
        output_root.mkdir()
        input_root = output_root / "input"
        input_root.mkdir()

    with pytest.raises(ValueError, match="корн"):
        prepare_workspace(output_root, input_root)

    if relation == "output_inside_input":
        assert not output_root.exists()


def test_prepare_workspace_rejects_existing_output_even_when_empty(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "workspace"
    input_root.mkdir()
    output_root.mkdir()

    with pytest.raises(ValueError, match="уже существует"):
        prepare_workspace(output_root, input_root)

    assert list(output_root.iterdir()) == []


def test_prepare_workspace_reserves_all_paths_and_creates_only_crops(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "workspace"
    input_root.mkdir()

    paths = prepare_workspace(output_root, input_root)

    assert paths.root == output_root.resolve()
    assert paths.manifest == paths.root / "manifest.csv"
    assert paths.crops == paths.root / "crops"
    assert paths.rejects == paths.root / "rejects.csv"
    assert paths.observations == paths.root / "observations.csv"
    assert paths.report == paths.root / "report.json"
    assert paths.profile == paths.root / "profile.json"
    assert sorted(path.name for path in paths.root.iterdir()) == ["crops"]


def test_rebuild_reads_input_crop_and_writes_output_crop(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    (output_root / "crops").mkdir()
    sha = "a" * 64
    row = _строка(sha)
    _оригинал(input_root, sha)
    old_path = _старый_кроп(input_root, row, _неизменённый_кроп())

    reason, rebuilt = rebuild_row(
        row,
        _deps(input_root, Фикс([]), output_root),
        DedupIndex(),
    )

    assert reason == "accepted"
    assert rebuilt is not None and rebuilt.audit == "ok"
    assert old_path.read_bytes() == (output_root / "crops" / row.class_id / row.file).read_bytes()


def _run_main(
    manifest: Path,
    input_root: Path,
    output_root: Path,
    models: Path,
    rows: list[tuple[str, HarvestRow | None]],
    *,
    input_count: int | None = None,
    out: str | None = None,
) -> int:
    with (
        patch("pipeline.harvest.detectors.verify_weights"),
        patch("pipeline.harvest.detectors.load_detectors_manifest", return_value=_pins()),
        patch("pipeline.harvest.redaction_profile.detector_stage_sha256", return_value="4" * 64),
        patch("pipeline.harvest.redaction_profile.build_plate_detectors") as plate_factory,
        patch("pipeline.detectors.yolo.YoloDetector") as vehicle_factory,
        patch("pipeline.harvest.backends.YuNetFaceDetector") as face_factory,
        patch("scripts.rebuild_redaction.current_git_sha", return_value="5" * 40),
        patch("scripts.rebuild_redaction.current_timestamp", return_value="2026-09-07T12:00:00Z"),
        patch(
            "scripts.rebuild_redaction.rebuild_row",
            side_effect=[
                RebuildResult(
                    reason,
                    rebuilt,
                    FrameProfileResult(
                        photo_id=rebuilt.sha256 if rebuilt is not None else "rejected",
                        class_id=rebuilt.class_id if rebuilt is not None else "rejected",
                        outcome="accepted" if rebuilt is not None else "rejected",
                        reject_reason=None if rebuilt is not None else reason,
                        old_black_frac=0.0,
                        new_black_frac=0.0,
                        delta_black=0.0,
                        pixels_equal=True,
                        audit_preserved=bool(rebuilt is not None and rebuilt.audit),
                    ),
                )
                for reason, rebuilt in rows
            ],
        )
        if rows
        else nullcontext(),
    ):
        count = input_count if input_count is not None else len(rows)
        vehicle_factory.return_value = Фикс(
            [[Detection(40.0, 40.0, 360.0, 300.0, 0.95, 2)]] * count
        )
        plate_factory.return_value = (Фикс([]),)
        face_factory.return_value = Фикс([[]] * count)
        аргументы = [
            "--manifest",
            str(manifest),
            "--models",
            str(models),
            "--data-root",
            str(input_root),
            "--out-root",
            str(output_root),
        ]
        if out is not None:
            аргументы += ["--out", out]
        return main(аргументы)


def test_main_preflights_originals_before_creating_output(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    append_row(manifest, _строка("a" * 64))
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "workspace"

    code = main(
        [
            "--manifest",
            str(manifest),
            "--models",
            str(tmp_path / "models"),
            "--data-root",
            str(input_root),
            "--out-root",
            str(output_root),
        ]
    )

    assert code == 2
    assert not output_root.exists()


def test_main_writes_closed_workspace_artifacts_without_success_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    accepted = _строка("a" * 64)
    rejected = _строка("b" * 64)
    append_row(manifest, accepted)
    append_row(manifest, rejected)
    input_root = tmp_path / "input"
    _оригинал(input_root, accepted.sha256)
    _оригинал(input_root, rejected.sha256)
    output_root = tmp_path / "workspace"
    models = tmp_path / "models"
    models.mkdir()
    real_rebuild_row = rebuild_row

    def fake_rebuild_row(
        row: HarvestRow,
        deps: RebuildDeps,
        dedup: DedupIndex,
        reset_audit: bool = False,
        trace: bool = False,
    ) -> RebuildResult:
        if row.sha256 == rejected.sha256:
            return RebuildResult(
                "redaction_failed",
                None,
                FrameProfileResult(
                    photo_id=row.sha256,
                    class_id=row.class_id,
                    outcome="rejected",
                    reject_reason="redaction_failed",
                    old_black_frac=None,
                    new_black_frac=None,
                    delta_black=None,
                    pixels_equal=None,
                    audit_preserved=False,
                ),
            )
        return real_rebuild_row(row, deps, dedup, reset_audit, trace=trace)

    monkeypatch.setattr("scripts.rebuild_redaction.rebuild_row", fake_rebuild_row)
    code = _run_main(
        manifest,
        input_root,
        output_root,
        models,
        [],
        input_count=2,
    )

    assert code == 0
    rows = load_manifest(output_root / "manifest.csv")
    assert len(rows) == 1
    assert rows[0].sha256 == accepted.sha256
    assert rows[0].audit == ""
    with (output_root / "rejects.csv").open(encoding="utf-8", newline="") as file:
        rejects = list(csv.DictReader(file))
    assert rejects == [
        {
            "file": rejected.file,
            "class_id": rejected.class_id,
            "source": rejected.source,
            "source_item_id": rejected.source_item_id,
            "sha256": rejected.sha256,
            "reason": "redaction_failed",
        }
    ]

    profile_bytes = (output_root / "profile.json").read_bytes()
    report_bytes = (output_root / "report.json").read_bytes()
    assert profile_bytes.endswith(b"\n") and report_bytes.endswith(b"\n")
    profile = json.loads(profile_bytes)
    report = json.loads(report_bytes)
    assert profile == {
        "detector_pins": {name: asdict(pin) for name, pin in sorted(_pins().items())},
        "git_sha": "5" * 40,
        "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "input_root": str(input_root.resolve()),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "profile": asdict(CURRENT_REDACTION_PROFILE),
        "profile_id": profile["profile_id"],
        "stage_sha256": "4" * 64,
        "timestamp": "2026-09-07T12:00:00Z",
    }
    assert len(profile["profile_id"]) == 64
    assert report["counts"] == {"accepted": 1, "input": 2, "rejected": 1}
    assert report["reject_reasons"] == {"redaction_failed": 1}
    # Обе строки входа несут отметку `ok`. Принятая теряет её потому, что старого кропа
    # нет, отклонённая — вместе с самой строкой. Корзины обязаны закрывать `in_input`.
    assert report["audit"] == {
        "corrupt": 0,
        "dropped_with_row": 1,
        "in_input": 2,
        "missing": 1,
        "preserved": 0,
        "reset": 0,
    }
    assert report["output_hashes"] == {
        "manifest.csv": hashlib.sha256((output_root / "manifest.csv").read_bytes()).hexdigest(),
        "observations.csv": hashlib.sha256(
            (output_root / "observations.csv").read_bytes()
        ).hexdigest(),
        "pixel-diff.csv": hashlib.sha256((output_root / "pixel-diff.csv").read_bytes()).hexdigest(),
        "profile.json": hashlib.sha256(profile_bytes).hexdigest(),
        "quality-metrics.json": hashlib.sha256(
            (output_root / "quality-metrics.json").read_bytes()
        ).hexdigest(),
        "rejects.csv": hashlib.sha256((output_root / "rejects.csv").read_bytes()).hexdigest(),
    }
    assert "success" not in report and "passed" not in report
    assert report["counts"]["accepted"] + report["counts"]["rejected"] == 2


def test_audit_counts_требует_закрытой_арифметики() -> None:
    AuditCounts(in_input=3, preserved=1, reset=1, missing=1, corrupt=0, dropped_with_row=0)

    with pytest.raises(ValueError, match="не сходятся"):
        AuditCounts(in_input=3, preserved=1, reset=0, missing=0, corrupt=0, dropped_with_row=0)


def test_отчёт_читает_перенос_отметки_с_построенной_строки(tmp_path: Path) -> None:
    # Корзина `preserved` обязана совпадать с тем, что записано в манифест workspace: она
    # читается с построенной строки, а не пересчитывается предикатом равенства пикселей.
    manifest = tmp_path / "manifest.csv"
    строка = _строка("a" * 64)
    append_row(manifest, строка)
    input_root = tmp_path / "input"
    _оригинал(input_root, строка.sha256)
    _старый_кроп(input_root, строка, _неизменённый_кроп())
    output_root = tmp_path / "workspace"
    models = tmp_path / "models"
    models.mkdir()

    код = _run_main(manifest, input_root, output_root, models, [], input_count=1)

    assert код == 0
    отчёт = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    assert отчёт["audit"] == {
        "corrupt": 0,
        "dropped_with_row": 0,
        "in_input": 1,
        "missing": 0,
        "preserved": 1,
        "reset": 0,
    }
    assert load_manifest(output_root / "manifest.csv")[0].audit == "ok"


def test_main_кладёт_манифест_по_out_внутри_workspace(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    строка = _строка("a" * 64)
    append_row(manifest, строка)
    input_root = tmp_path / "input"
    _оригинал(input_root, строка.sha256)
    output_root = tmp_path / "workspace"
    models = tmp_path / "models"
    models.mkdir()

    код = _run_main(
        manifest, input_root, output_root, models, [], input_count=1, out="sub/custom.csv"
    )

    assert код == 0
    assert not (output_root / "manifest.csv").exists()
    assert len(load_manifest(output_root / "sub" / "custom.csv")) == 1
    отчёт = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    assert "sub/custom.csv" in отчёт["output_hashes"]


def test_main_отвергает_out_наружу_до_создания_workspace(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    append_row(manifest, _строка("a" * 64))
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "workspace"

    with pytest.raises(ValueError, match="за пределы workspace"):
        main(
            [
                "--manifest",
                str(manifest),
                "--models",
                str(tmp_path / "models"),
                "--data-root",
                str(input_root),
                "--out-root",
                str(output_root),
                "--out",
                "../escape.csv",
            ]
        )

    assert not output_root.exists()


def test_main_отвергает_дубли_crop_destination_до_первой_мутации(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    первая = replace(_строка("a" * 64), file="shared.jpg")
    вторая = replace(_строка("b" * 64), file="shared.jpg")
    append_row(манифест, первая)
    append_row(манифест, вторая)
    data_root = tmp_path / "data"
    _оригинал(data_root, первая.sha256)
    _оригинал(data_root, вторая.sha256)
    crop = data_root / "crops" / первая.class_id / первая.file
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"old crop")

    with pytest.raises(ValueError, match="повтор пути кропа"):
        main(
            [
                "--manifest",
                str(манифест),
                "--models",
                str(tmp_path / "models"),
                "--data-root",
                str(data_root),
                "--out-root",
                str(tmp_path / "workspace"),
            ]
        )

    assert crop.read_bytes() == b"old crop"
    assert load_manifest(манифест) == [первая, вторая]


def test_main_отвергает_casefold_alias_до_первой_мутации(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    первая = replace(_строка("a" * 64), file="shared.jpg")
    вторая = replace(_строка("b" * 64), file="SHARED.jpg")
    append_row(манифест, первая)
    append_row(манифест, вторая)
    data_root = tmp_path / "data"
    _оригинал(data_root, первая.sha256)
    _оригинал(data_root, вторая.sha256)
    crop = data_root / "crops" / первая.class_id / первая.file
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"old crop")

    with pytest.raises(ValueError, match="повтор пути кропа"):
        main(
            [
                "--manifest",
                str(манифест),
                "--models",
                str(tmp_path / "models"),
                "--data-root",
                str(data_root),
                "--out-root",
                str(tmp_path / "workspace"),
            ]
        )

    assert crop.read_bytes() == b"old crop"
    assert load_manifest(манифест) == [первая, вторая]


def test_main_префлайт_сбой(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    append_row(манифест, _строка("a" * 64))
    (tmp_path / "data").mkdir()
    код = main(
        [
            "--manifest",
            str(манифест),
            "--models",
            str(tmp_path / "models"),
            "--data-root",
            str(tmp_path / "data"),
            "--out-root",
            str(tmp_path / "workspace"),
        ]
    )
    assert код == 2
    assert not (tmp_path / "workspace").exists()


def test_main_успешный_прогон(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    sha = "a" * 64
    append_row(манифест, _строка(sha))
    _оригинал(tmp_path / "data", sha)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("pipeline.harvest.detectors.verify_weights"),
        patch("pipeline.harvest.detectors.load_detectors_manifest", return_value=_pins()),
        patch("pipeline.harvest.redaction_profile.detector_stage_sha256", return_value="4" * 64),
        patch("pipeline.harvest.backends.OnnxBoxDetector"),
        patch("pipeline.detectors.yolo.YoloDetector"),
        patch("pipeline.harvest.backends.YuNetFaceDetector"),
        patch("scripts.rebuild_redaction.current_git_sha", return_value="5" * 40),
        patch("scripts.rebuild_redaction.current_timestamp", return_value="2026-09-07T12:00:00Z"),
        patch(
            "scripts.rebuild_redaction.rebuild_row",
            return_value=RebuildResult(
                "accepted",
                _строка(sha),
                FrameProfileResult(
                    photo_id=sha,
                    class_id="lada__granta",
                    outcome="accepted",
                    reject_reason=None,
                    old_black_frac=0.0,
                    new_black_frac=0.0,
                    delta_black=0.0,
                    pixels_equal=True,
                    audit_preserved=True,
                ),
            ),
        ),
    ):
        код = main(
            [
                "--manifest",
                str(манифест),
                "--models",
                str(models_dir),
                "--data-root",
                str(tmp_path / "data"),
                "--out-root",
                str(tmp_path / "workspace"),
            ]
        )
        assert код == 0
        новые_строки = load_manifest(tmp_path / "workspace" / "manifest.csv")
        assert len(новые_строки) == 1
        assert load_manifest(манифест) == [_строка(sha)]


def test_main_writes_profile_observations_pixel_diff_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    accepted = _строка("a" * 64)
    rejected = _строка("b" * 64)
    append_row(manifest, accepted)
    append_row(manifest, rejected)
    input_root = tmp_path / "input"
    _оригинал(input_root, accepted.sha256)
    _оригинал(input_root, rejected.sha256)
    _старый_кроп(input_root, accepted, np.zeros((348, 418, 3), dtype=np.uint8))
    _старый_кроп(input_root, rejected, np.full((348, 418, 3), 127, dtype=np.uint8))
    output_root = tmp_path / "workspace"
    models = tmp_path / "models"
    models.mkdir()

    merged = Detection(120.0, 210.0, 80.0, 24.0, 0.9, 0)
    raw = RawPlateObservation("plate", (0, 0, 800, 400), (0, 0, 640, 400), merged)
    trace = PlateDetectionTrace(raw=(raw,), merged=(merged,))
    original_redact_frame = cast(
        Callable[..., RedactionResult],
        __import__("scripts.rebuild_redaction", fromlist=["redact_frame"]).redact_frame,
    )
    calls = 0

    def traced_redact_frame(*args: object, **kwargs: object) -> RedactionResult:
        nonlocal calls
        result = original_redact_frame(*args, **kwargs)
        calls += 1
        return replace(result, first_pass=trace) if calls == 1 else result

    monkeypatch.setattr("scripts.rebuild_redaction.redact_frame", traced_redact_frame)

    code = _run_main(manifest, input_root, output_root, models, [], input_count=2)

    assert code == 0
    profile_document = json.loads((output_root / "profile.json").read_text(encoding="utf-8"))
    with (output_root / "observations.csv").open(encoding="utf-8", newline="") as file:
        observations = list(csv.DictReader(file))
    assert observations == [
        {
            "photo_id": accepted.sha256,
            "detector": "plate",
            "profile": profile_document["profile_id"],
            "zone_x0": "0",
            "zone_y0": "0",
            "zone_x1": "800",
            "zone_y1": "400",
            "tile_x0": "0",
            "tile_y0": "0",
            "tile_x1": "640",
            "tile_y1": "400",
            "x": "120.0",
            "y": "210.0",
            "w": "80.0",
            "h": "24.0",
            "confidence": "0.9",
            "class_id": "0",
            "raw": "True",
            "merged": "False",
            "pass": "first",
        },
        {
            "photo_id": accepted.sha256,
            "detector": "merged",
            "profile": profile_document["profile_id"],
            "zone_x0": "0",
            "zone_y0": "0",
            "zone_x1": "0",
            "zone_y1": "0",
            "tile_x0": "0",
            "tile_y0": "0",
            "tile_x1": "0",
            "tile_y1": "0",
            "x": "120.0",
            "y": "210.0",
            "w": "80.0",
            "h": "24.0",
            "confidence": "0.9",
            "class_id": "0",
            "raw": "False",
            "merged": "True",
            "pass": "first",
        },
    ]
    with (output_root / "pixel-diff.csv").open(encoding="utf-8", newline="") as file:
        pixel_rows = list(csv.DictReader(file))
    assert len(pixel_rows) == 2
    assert pixel_rows[0]["photo_id"] == accepted.sha256
    assert pixel_rows[0]["outcome"] == "accepted"
    assert pixel_rows[0]["old_black_frac"] == "1.0"
    assert pixel_rows[0]["new_black_frac"] == "0.0"
    assert pixel_rows[0]["delta_black"] == "0.0"
    assert pixel_rows[0]["pixels_equal"] == "False"
    assert pixel_rows[1]["photo_id"] == rejected.sha256
    metrics = json.loads((output_root / "quality-metrics.json").read_text(encoding="utf-8"))
    assert metrics == {
        "above_quarter": 0,
        "changed_crops": 1,
        "max_delta_black": 0.0,
        "median_delta_black": 0.0,
        "new_over_redacted": 0,
        "p95_delta_black": 0.0,
        "rejects": [["duplicate", 1]],
    }
    report = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    assert report["counts"] == {"accepted": 1, "input": 2, "rejected": 1}
    assert set(report["output_hashes"]) == {
        "manifest.csv",
        "observations.csv",
        "pixel-diff.csv",
        "profile.json",
        "quality-metrics.json",
        "rejects.csv",
    }


def test_rebuild_records_over_redacted_candidate_without_writing_crop(tmp_path: Path) -> None:
    sha = "a" * 64
    row = _строка(sha)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    (output_root / "crops").mkdir()
    _оригинал(input_root, sha)
    _старый_кроп(input_root, row, np.full((348, 418, 3), 127, dtype=np.uint8))
    plate = Фикс([[Detection(40.0, 40.0, 360.0, 300.0, 0.95, 0)], [], [], []])

    outcome = rebuild_row(
        row,
        _deps(input_root, plate, output_root),
        DedupIndex(),
        trace=True,
    )

    assert outcome.reason == "over_redacted"
    assert outcome.rebuilt is None
    assert outcome.profile_result.outcome == "rejected"
    assert outcome.profile_result.reject_reason == "over_redacted"
    assert outcome.profile_result.old_black_frac == 0.0
    assert outcome.profile_result.new_black_frac is not None
    assert outcome.profile_result.new_black_frac > 0.5
    assert outcome.profile_result.delta_black == pytest.approx(
        outcome.profile_result.new_black_frac
    )
    assert outcome.profile_result.pixels_equal is None
    assert not (output_root / "crops" / row.class_id / row.file).exists()


def test_rebuild_over_redacted_checks_before_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "a" * 64
    row = _строка(sha)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    (output_root / "crops").mkdir()
    _оригинал(input_root, sha)
    plate = Фикс([[Detection(40.0, 40.0, 360.0, 300.0, 0.95, 0)], [], [], []])
    encoder = Mock(side_effect=AssertionError("encoder must not run for rejected crop"))
    monkeypatch.setattr("scripts.rebuild_redaction.encode_jpeg", encoder)

    outcome = rebuild_row(row, _deps(input_root, plate, output_root), DedupIndex())

    assert outcome.reason == "over_redacted"
    assert outcome.profile_result.new_black_frac is not None
    assert outcome.profile_result.new_black_frac > 0.5
    assert outcome.profile_result.pixels_equal is None
    encoder.assert_not_called()


def test_rebuild_empty_crop_is_over_redacted_before_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "a" * 64
    row = _строка(sha)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    (output_root / "crops").mkdir()
    _оригинал(input_root, sha)
    monkeypatch.setattr(
        "scripts.rebuild_redaction.crop_image",
        lambda *_: np.empty((0, 0, 3), dtype=np.uint8),
    )
    encoder = Mock(side_effect=AssertionError("encoder must not run for rejected crop"))
    monkeypatch.setattr("scripts.rebuild_redaction.encode_jpeg", encoder)

    outcome = rebuild_row(row, _deps(input_root, Фикс([]), output_root), DedupIndex())

    assert outcome.reason == "over_redacted"
    assert outcome.profile_result.new_black_frac is None
    encoder.assert_not_called()


def test_observations_keep_duplicate_raw_multiplicity_and_separate_merged_record() -> None:
    duplicate = Detection(10.0, 20.0, 30.0, 10.0, 0.9, 0)
    trace = PlateDetectionTrace(
        raw=(
            RawPlateObservation("plate", (0, 0, 100, 100), (0, 0, 60, 60), duplicate),
            RawPlateObservation("plate", (0, 0, 100, 100), (40, 0, 100, 60), duplicate),
        ),
        merged=(duplicate,),
    )

    records = _observation_records("a" * 64, trace, "first", "profile-123")

    assert [(record.raw, record.merged) for record in records] == [
        (True, False),
        (True, False),
        (False, True),
    ]
    assert [record.tile_x0 for record in records] == [0, 40, 0]
    assert {record.profile for record in records} == {"profile-123"}


class ТрассируемыйТайловый:
    def __init__(self, scenarios: list[list[TileObservation]]) -> None:
        self.scenarios = list(scenarios)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [observation.box for observation in self.scenarios.pop(0)]

    def detect_with_tiles(
        self, image: np.ndarray
    ) -> tuple[list[Detection], tuple[TileObservation, ...]]:
        observations = tuple(self.scenarios.pop(0))
        return [observation.box for observation in observations], observations


def test_trace_on_off_preserves_decisions_manifest_crops_and_records_both_passes(
    tmp_path: Path,
) -> None:
    plate = Detection(120.0, 210.0, 80.0, 24.0, 0.95, 0)
    suppressed = Detection(122.0, 211.0, 78.0, 23.0, 0.70, 0)
    residual = Detection(420.0, 60.0, 70.0, 20.0, 0.90, 0)

    def run_workspace(
        name: str, trace: bool
    ) -> tuple[Path, tuple[str, str], tuple[ObservationRecord, ...]]:
        input_root = tmp_path / f"input-{name}"
        output_root = tmp_path / f"output-{name}"
        input_root.mkdir()
        output_root.mkdir()
        (output_root / "crops").mkdir()
        accepted = _строка("a" * 64)
        rejected = _строка("a" * 64)
        _оригинал(input_root, accepted.sha256)
        _оригинал(input_root, rejected.sha256)
        detector = ТрассируемыйТайловый(
            [
                [
                    TileObservation((0, 0, 640, 400), plate),
                    TileObservation((160, 0, 800, 400), suppressed),
                ],
                [],
                [TileObservation((0, 0, 640, 400), residual)],
                [],
                [],
                [],
                [],
                [],
            ]
        )
        deps = RebuildDeps(
            vehicle_detector=Фикс(
                [
                    [Detection(40.0, 40.0, 360.0, 300.0, 0.95, 2)],
                    [Detection(40.0, 40.0, 360.0, 300.0, 0.95, 2)],
                ]
            ),
            plate_detectors=(detector,),
            face_detector=Фикс([[]]),
            input_root=input_root,
            output_root=output_root,
            profile_id="profile-123",
        )
        dedup = DedupIndex(max_distance=64)
        first = rebuild_row(accepted, deps, dedup, trace=trace)
        second = rebuild_row(rejected, deps, dedup, trace=trace)
        assert second.rebuilt is not None
        append_row(output_root / "manifest.csv", second.rebuilt)
        return output_root, (first.reason, second.reason), first.observations + second.observations

    plain, plain_reasons, plain_observations = run_workspace("plain", False)
    traced, traced_reasons, traced_observations = run_workspace("traced", True)

    assert plain_reasons == traced_reasons == ("redaction_failed", "accepted")
    assert plain_observations == ()
    assert (plain / "manifest.csv").read_bytes() == (traced / "manifest.csv").read_bytes()
    plain_crop = next((plain / "crops").rglob("*.jpg"))
    traced_crop = next((traced / "crops").rglob("*.jpg"))
    assert (
        hashlib.sha256(plain_crop.read_bytes()).digest()
        == hashlib.sha256(traced_crop.read_bytes()).digest()
    )
    assert {(record.raw, record.merged, record.pass_name) for record in traced_observations} == {
        (True, False, "first"),
        (False, True, "first"),
        (True, False, "second"),
        (False, True, "second"),
    }
    assert {record.profile for record in traced_observations} == {"profile-123"}
    assert {record.tile_x0 for record in traced_observations if record.raw} == {0, 160}


def test_main_берёт_детекторы_номера_из_единой_фабрики(tmp_path: Path) -> None:
    """Переderivация обязана идти тем же профилем, что боевой харвест (§3.1 дизайна).

    Своя копия конструирования детекторов здесь была второй из трёх: расхождение с боевой
    означало бы кропы, полученные не тем контуром, при том же имени манифеста.
    """
    манифест = tmp_path / "manifest.csv"
    sha = "a" * 64
    append_row(манифест, _строка(sha))
    _оригинал(tmp_path / "data", sha)
    модели = tmp_path / "models"
    модели.mkdir(parents=True, exist_ok=True)
    пины = _pins()

    with (
        patch("pipeline.harvest.detectors.verify_weights"),
        patch("pipeline.harvest.detectors.load_detectors_manifest", return_value=пины),
        patch("pipeline.harvest.redaction_profile.detector_stage_sha256", return_value="4" * 64),
        patch("pipeline.harvest.redaction_profile.build_plate_detectors") as фабрика,
        patch("pipeline.detectors.yolo.YoloDetector"),
        patch("pipeline.harvest.backends.YuNetFaceDetector"),
        patch("scripts.rebuild_redaction.current_git_sha", return_value="5" * 40),
        patch("scripts.rebuild_redaction.current_timestamp", return_value="2026-09-07T12:00:00Z"),
        patch(
            "scripts.rebuild_redaction.rebuild_row",
            return_value=RebuildResult(
                "accepted",
                _строка(sha),
                FrameProfileResult(
                    photo_id=sha,
                    class_id="lada__granta",
                    outcome="accepted",
                    reject_reason=None,
                    old_black_frac=0.0,
                    new_black_frac=0.0,
                    delta_black=0.0,
                    pixels_equal=True,
                    audit_preserved=True,
                ),
            ),
        ),
    ):
        код = main(
            [
                "--manifest",
                str(манифест),
                "--models",
                str(модели),
                "--data-root",
                str(tmp_path / "data"),
                "--out-root",
                str(tmp_path / "workspace"),
            ]
        )

    assert код == 0
    assert фабрика.call_args.args == (CURRENT_REDACTION_PROFILE,)
    assert фабрика.call_args.kwargs == {"models_dir": модели, "pins": пины}
