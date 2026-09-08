# tests/test_harvest_rebuild.py
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pipeline.crop import Detection
from pipeline.harvest.cache import cache_path
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.jpeg import encode_jpeg, write_jpeg_atomic
from pipeline.harvest.manifest import HarvestRow, append_row, load_manifest
from pipeline.harvest.redaction_profile import CURRENT_REDACTION_PROFILE
from scripts.rebuild_redaction import RebuildDeps, build_parser, main, preflight, rebuild_row


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


def _deps(root: Path, plate: Фикс) -> RebuildDeps:
    return RebuildDeps(
        vehicle_detector=Фикс([[Detection(40.0, 40.0, 360.0, 300.0, 0.95, 2)]]),
        plate_detectors=(plate,),
        face_detector=Фикс([[]]),
        root=root,
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
        root=tmp_path,
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
        root=tmp_path,
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


def test_build_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(["--models", "models_dir", "--data-root", "data_dir"])
    assert args.models == Path("models_dir")
    assert args.data_root == Path("data_dir")
    assert args.out is None
    assert not args.reset_audit

    args_out = parser.parse_args(
        ["--models", "m", "--data-root", "d", "--out", "custom.csv", "--reset-audit"]
    )
    assert args_out.out == Path("custom.csv")
    assert args_out.reset_audit


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
            ]
        )

    assert crop.read_bytes() == b"old crop"
    assert load_manifest(манифест) == [первая, вторая]


def test_main_префлайт_сбой(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    append_row(манифест, _строка("a" * 64))
    код = main(
        [
            "--manifest",
            str(манифест),
            "--models",
            str(tmp_path / "models"),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert код == 2


def test_main_успешный_прогон(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    sha = "a" * 64
    append_row(манифест, _строка(sha))
    _оригинал(tmp_path / "data", sha)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("pipeline.harvest.detectors.verify_weights"),
        patch(
            "pipeline.harvest.detectors.load_detectors_manifest",
            return_value={
                "plate": type("Pin", (), {"format": "onnx"})(),
                "plate_hires": type("Pin", (), {"format": "pt"})(),
            },
        ),
        patch("pipeline.harvest.backends.OnnxBoxDetector"),
        patch("pipeline.detectors.yolo.YoloDetector"),
        patch("pipeline.harvest.backends.YuNetFaceDetector"),
        patch("scripts.rebuild_redaction.rebuild_row", return_value=("accepted", _строка(sha))),
    ):
        код = main(
            [
                "--manifest",
                str(манифест),
                "--models",
                str(models_dir),
                "--data-root",
                str(tmp_path / "data"),
            ]
        )
        assert код == 0
        новые_строки = load_manifest(манифест)
        assert len(новые_строки) == 1


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
    пины = {"plate": type("Pin", (), {"format": "onnx"})()}

    with (
        patch("pipeline.harvest.detectors.verify_weights"),
        patch("pipeline.harvest.detectors.load_detectors_manifest", return_value=пины),
        patch("pipeline.harvest.redaction_profile.build_plate_detectors") as фабрика,
        patch("pipeline.detectors.yolo.YoloDetector"),
        patch("pipeline.harvest.backends.YuNetFaceDetector"),
        patch("scripts.rebuild_redaction.rebuild_row", return_value=("accepted", _строка(sha))),
    ):
        код = main(
            [
                "--manifest",
                str(манифест),
                "--models",
                str(модели),
                "--data-root",
                str(tmp_path / "data"),
            ]
        )

    assert код == 0
    assert фабрика.call_args.args == (CURRENT_REDACTION_PROFILE,)
    assert фабрика.call_args.kwargs == {"models_dir": модели, "pins": пины}
