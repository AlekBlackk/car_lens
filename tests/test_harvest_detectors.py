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
    "face,onnx,https://example.invalid/yunet.onnx,2023mar,apache-2.0,2026-08-28,"
    + "a" * 64
    + ",\n"
)
НОМЕР = (
    "plate,onnx,https://example.invalid/plate.onnx,r1,apache-2.0,2026-08-28,"
    + "b" * 64
    + ",ir:"
    + "c" * 64
    + " | ovc plate.xml --output_model plate.onnx\n"
)


def _записать(tmp_path: Path, тело: str) -> Path:
    путь = tmp_path / "detectors_manifest.csv"
    путь.write_text(ШАПКА + тело, encoding="utf-8")
    return путь


def test_полный_манифест_читается(tmp_path: Path) -> None:
    пины = load_detectors_manifest(_записать(tmp_path, ЛИЦО + НОМЕР))
    assert set(пины) == {"face", "plate"}
    assert пины["plate"].derived_from.startswith("ir:")


def test_отсутствие_детектора_номеров_отвергается(tmp_path: Path) -> None:
    """Правило 1 Р-11 не опционально: без детектора номеров харвест не стартует."""
    with pytest.raises(DetectorManifestError, match="plate"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО))


def test_несвободная_лицензия_отвергается(tmp_path: Path) -> None:
    строка = НОМЕР.replace("apache-2.0", "cc-by-nc-4.0")
    with pytest.raises(DetectorManifestError, match="лицензия"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка))


def test_формат_кроме_onnx_отвергается(tmp_path: Path) -> None:
    строка = НОМЕР.replace("plate,onnx", "plate,ir")
    with pytest.raises(DetectorManifestError, match="формат"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка))


def test_пустой_sha256_отвергается(tmp_path: Path) -> None:
    строка = НОМЕР.replace("b" * 64, "")
    with pytest.raises(DetectorManifestError, match="sha256"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка))


def test_производный_файл_без_derived_from_отвергается(tmp_path: Path) -> None:
    """Пин без происхождения не доказывает, что замывали именно этим файлом."""
    строка = НОМЕР.replace(",ir:" + "c" * 64 + " | ovc plate.xml --output_model plate.onnx", ",")
    строка = строка.replace("plate.onnx,r1", "plate-converted.onnx,r1")
    with pytest.raises(DetectorManifestError, match="derived_from"):
        load_detectors_manifest(_записать(tmp_path, ЛИЦО + строка))


def test_verify_weights_ловит_расхождение_sha(tmp_path: Path) -> None:
    файл = tmp_path / "face.onnx"
    контент = "веса".encode()
    файл.write_bytes(контент)
    настоящий = hashlib.sha256(контент).hexdigest()
    манифест = _записать(tmp_path, ЛИЦО.replace("a" * 64, настоящий) + НОМЕР)
    пины = load_detectors_manifest(манифест)
    with pytest.raises(DetectorManifestError, match="plate"):
        verify_weights(пины, tmp_path)


def test_манифест_проекта_полон_и_свободен() -> None:
    """Реальный манифест — входное условие прогона, а не отчётный побочный продукт."""
    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    assert set(пины) == {"face", "plate"}
    for пин in пины.values():
        assert пин.format == "onnx"
        assert пин.license in {"mit", "apache-2.0"}
        assert len(пин.sha256) == 64
