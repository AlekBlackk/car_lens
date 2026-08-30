# tests/test_harvest_detectors.py
import hashlib
from pathlib import Path

import pytest

from pipeline.harvest.detectors import (
    DEFAULT_DETECTORS_MANIFEST,
    DetectorManifestError,
    load_detectors_manifest,
    verify_weights,
)

ШАПКА = "name,format,url,revision,license,checked,sha256,derived_from\n"
ЛИЦО = (
    "face,onnx,https://example.invalid/yunet.onnx,2023mar,apache-2.0,2026-08-28," + "a" * 64 + ",\n"
)
НОМЕР = (
    "plate,onnx,https://example.invalid/plate.onnx,r1,apache-2.0,2026-08-28,"
    + "b" * 64
    + ",ir:"
    + "c" * 64
    + " | ovc plate.xml --output_model plate.onnx\n"
)
НОМЕР_HIRES = (
    "plate_hires,pt,https://example.invalid/lp_yolov8n.pt,v1,agpl-3.0,2026-08-29,"
    + "d" * 64
    + ",\n"
)


def _записать(tmp_path: Path, тело: str) -> Path:
    путь = tmp_path / "detectors_manifest.csv"
    путь.write_text(ШАПКА + тело, encoding="utf-8")
    return путь


def test_полный_манифест_читается(tmp_path: Path) -> None:
    пины = load_detectors_manifest(_записать(tmp_path, ЛИЦО + НОМЕР + НОМЕР_HIRES))
    assert set(пины) == {"face", "plate", "plate_hires"}
    assert пины["plate"].derived_from.startswith("ir:")


def test_отсутствие_детектора_номеров_отвергается(tmp_path: Path) -> None:
    """Правило 1 Р-11: без обоих детекторов номеров харвест не стартует."""
    with pytest.raises(DetectorManifestError, match="plate"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + НОМЕР_HIRES))
    with pytest.raises(DetectorManifestError, match="plate_hires"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + НОМЕР))


def test_несвободная_лицензия_отвергается(tmp_path: Path) -> None:
    строка = НОМЕР.replace("apache-2.0", "cc-by-nc-4.0")
    with pytest.raises(DetectorManifestError, match="лицензия"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка + НОМЕР_HIRES))


def test_недопустимый_формат_отвергается(tmp_path: Path) -> None:
    строка = НОМЕР.replace("plate,onnx", "plate,ir")
    with pytest.raises(DetectorManifestError, match="формат"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка + НОМЕР_HIRES))


def test_пустой_sha256_отвергается(tmp_path: Path) -> None:
    строка = НОМЕР.replace("b" * 64, "")
    with pytest.raises(DetectorManifestError, match="sha256"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка + НОМЕР_HIRES))


def test_производный_файл_без_derived_from_отвергается(tmp_path: Path) -> None:
    """Пин без происхождения не доказывает, что замывали именно этим файлом."""
    строка = НОМЕР.replace(",ir:" + "c" * 64 + " | ovc plate.xml --output_model plate.onnx", ",")
    строка = строка.replace("plate.onnx,r1", "plate-converted.onnx,r1")
    with pytest.raises(DetectorManifestError, match="derived_from"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка + НОМЕР_HIRES))


def test_agpl_допустим_для_конвейерного_детектора(tmp_path: Path) -> None:
    """AGPL-3.0: обязательства при распространении, конвейер не распространяется (Р-4)."""
    пины = load_detectors_manifest(_записать(tmp_path, ЛИЦО + НОМЕР + НОМЕР_HIRES))
    assert пины["plate_hires"].license == "agpl-3.0"
    assert пины["plate_hires"].format == "pt"


def test_pt_формат_ищется_с_правильным_расширением(tmp_path: Path) -> None:
    (tmp_path / "face.onnx").write_bytes(b"face")
    (tmp_path / "plate.onnx").write_bytes(b"plate")
    контент = b"pt-weights"
    (tmp_path / "plate_hires.pt").write_bytes(контент)
    sha_face = hashlib.sha256(b"face").hexdigest()
    sha_plate = hashlib.sha256(b"plate").hexdigest()
    sha_pt = hashlib.sha256(контент).hexdigest()
    манифест = _записать(
        tmp_path,
        ЛИЦО.replace("a" * 64, sha_face)
        + НОМЕР.replace("b" * 64, sha_plate)
        + НОМЕР_HIRES.replace("d" * 64, sha_pt),
    )
    пины = load_detectors_manifest(манифест)
    verify_weights(пины, tmp_path)  # ищет именно plate_hires.pt, а не .onnx

    (tmp_path / "plate_hires.pt").unlink()
    with pytest.raises(DetectorManifestError, match=r"plate_hires\.pt"):
        verify_weights(пины, tmp_path)


def test_verify_weights_ловит_расхождение_sha(tmp_path: Path) -> None:
    файл = tmp_path / "face.onnx"
    контент = "веса".encode()
    файл.write_bytes(контент)
    настоящий = hashlib.sha256(контент).hexdigest()
    манифест = _записать(tmp_path, ЛИЦО.replace("a" * 64, настоящий) + НОМЕР + НОМЕР_HIRES)
    пины = load_detectors_manifest(манифест)
    with pytest.raises(DetectorManifestError, match="plate"):
        verify_weights(пины, tmp_path)


def test_манифест_проекта_полон_и_свободен() -> None:
    """Реальный манифест — входное условие прогона, а не отчётный побочный продукт."""
    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    assert set(пины) == {"face", "plate", "plate_hires"}
    for пин in пины.values():
        assert пин.format in {"onnx", "pt"}
        assert пин.license in {"mit", "apache-2.0", "agpl-3.0"}
        assert len(пин.sha256) == 64
