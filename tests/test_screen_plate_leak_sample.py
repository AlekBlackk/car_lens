"""Тесты модуля слепого VLM-скрининга выборки утечек номеров и лиц."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from scripts.screen_plate_leak_sample import (
    HISTORICAL_PLATE_LEAKS,
    KNOWN_FACE_MISSES,
    KNOWN_PLATE_LEAKS,
    build_report,
    prepare_image_b64,
    query_vlm,
    wilson_ci,
)


def _rec(flag: bool) -> dict[str, bool]:
    """Минимальная запись кадра: build_report читает только поле flag."""
    return {"flag": flag}


def test_prepare_image_b64_resizes_large_image(tmp_path: Path) -> None:
    img_path = tmp_path / "large.jpg"
    img = Image.new("RGB", (3200, 2000), color="blue")
    img.save(img_path)

    b64, orig_size = prepare_image_b64(img_path, long_side=1600)
    assert orig_size == (3200, 2000)

    decoded_bytes = base64.b64decode(b64)
    decoded_img = Image.open(BytesIO(decoded_bytes))
    assert decoded_img.size == (1600, 1000)


def test_prepare_image_b64_preserves_small_image(tmp_path: Path) -> None:
    img_path = tmp_path / "small.jpg"
    img = Image.new("RGB", (800, 600), color="green")
    img.save(img_path)

    b64, orig_size = prepare_image_b64(img_path, long_side=1600)
    assert orig_size == (800, 600)

    decoded_bytes = base64.b64decode(b64)
    decoded_img = Image.open(BytesIO(decoded_bytes))
    assert decoded_img.size == (800, 600)


@patch("urllib.request.urlopen")
def test_query_vlm_parses_json_response(mock_urlopen: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "message": {
                "content": json.dumps(
                    {
                        "has_unredacted_plate": True,
                        "plate_details": "BO 6243 EB",
                        "has_unredacted_face": False,
                        "face_details": None,
                        "flag": True,
                    }
                )
            }
        }
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result = query_vlm("fake_b64")
    assert result["flag"] is True
    assert result["has_unredacted_plate"] is True
    assert result["plate_details"] == "BO 6243 EB"


@patch("urllib.request.urlopen")
def test_query_vlm_strips_markdown_fences(mock_urlopen: MagicMock) -> None:
    raw_content = (
        "```json\n"
        '{"has_unredacted_plate": false, "plate_details": null, '
        '"has_unredacted_face": true, "face_details": "driver face", "flag": true}\n'
        "```"
    )
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"message": {"content": raw_content}}).encode(
        "utf-8"
    )
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result = query_vlm("fake_b64")
    assert result["flag"] is True
    assert result["has_unredacted_face"] is True
    assert result["face_details"] == "driver face"


def test_wilson_ci_empty_sample_is_zero() -> None:
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_brackets_point_estimate() -> None:
    lo, hi = wilson_ci(13, 70)
    # Точечная оценка 18.6% лежит внутри, а порог 15% интервал накрывает —
    # значит по флагам провал ещё не доказан.
    assert lo < 13 / 70 < hi
    assert lo < 0.15 < hi
    assert round(lo, 4) == 0.1119
    assert round(hi, 4) == 0.2923


def test_build_report_incomplete_coverage_leaves_verdict_null() -> None:
    # Зеркалит реальный прогон: из 8 номеров оценён только #6, из 10 лиц — три.
    processed = {
        6: _rec(True),  # утечка номера, поймана
        43: _rec(True),  # исторический номер, пойман
        32: _rec(True),  # пропуск лица, пойман
        36: _rec(False),  # пропуск лица, НЕ пойман
        68: _rec(False),  # пропуск лица, НЕ пойман
        500: _rec(True),  # чистый кадр, ложный флаг
        501: _rec(False),  # чистый кадр
    }
    report = build_report(processed, "test-model", sample_total=300)

    assert report["coverage_complete"] is False
    assert report["sample_total"] == 300
    assert report["total_evaluated"] == 7

    plate = report["plate_leaks"]
    assert plate["known_count"] == len(KNOWN_PLATE_LEAKS)
    assert plate["evaluated"] == 1
    assert plate["hits"] == [6]
    # recall считается по оценённым кадрам, а не по всем 8 известным утечкам.
    assert plate["recall_over_evaluated"] == 1.0

    face = report["face_misses"]
    assert face["evaluated"] == 3
    assert face["hits"] == [32]
    assert face["misses"] == [36, 68]
    assert face["recall_over_evaluated"] == round(1 / 3, 4)

    verdict = report["verdict"]
    # Пока номера покрыты не полностью — критерий 8/8 и вердикт неопределимы.
    assert verdict["criterion_8_of_8_plates"] is None
    assert verdict["triage_works"] is None


def test_build_report_missing_category_recall_is_null() -> None:
    # Ни одного известного номера/лица в прогоне — recall не определён, а не 0.
    report = build_report({900: _rec(False)}, "test-model")
    assert report["plate_leaks"]["evaluated"] == 0
    assert report["plate_leaks"]["recall_over_evaluated"] is None
    assert report["face_misses"]["recall_over_evaluated"] is None


def test_build_report_full_coverage_low_flags_triage_works() -> None:
    processed: dict[int, dict[str, bool]] = {}
    for n in KNOWN_PLATE_LEAKS | HISTORICAL_PLATE_LEAKS | KNOWN_FACE_MISSES:
        processed[n] = _rec(True)  # все утечки пойманы
    # Достаточно чистых непомеченных кадров, чтобы доля флагов упала ниже 15%.
    for n in range(1000, 1000 + 108):
        processed[n] = _rec(False)

    report = build_report(processed, "test-model", sample_total=300)
    verdict = report["verdict"]
    assert report["coverage_complete"] is True
    assert verdict["criterion_8_of_8_plates"] is True
    assert verdict["criterion_flags_leq_15_percent"] is True
    assert verdict["triage_works"] is True


def test_build_report_full_coverage_missed_plate_triage_fails() -> None:
    processed: dict[int, dict[str, bool]] = {}
    for n in KNOWN_PLATE_LEAKS | HISTORICAL_PLATE_LEAKS | KNOWN_FACE_MISSES:
        processed[n] = _rec(True)
    missed = min(KNOWN_PLATE_LEAKS)
    processed[missed] = _rec(False)  # один номер пропущен при полном покрытии
    for n in range(1000, 1000 + 108):
        processed[n] = _rec(False)

    verdict = build_report(processed, "test-model")["verdict"]
    assert verdict["criterion_8_of_8_plates"] is False
    assert verdict["triage_works"] is False


def test_build_report_verdict_keyed_on_plate_coverage_not_faces() -> None:
    # Критерий — «8/8 номеров», лица в него не входят. Поэтому при полном
    # покрытии номеров вердикт определим, даже если лица покрыты не все.
    processed: dict[int, dict[str, bool]] = {}
    for n in KNOWN_PLATE_LEAKS | HISTORICAL_PLATE_LEAKS:
        processed[n] = _rec(True)
    processed[min(KNOWN_FACE_MISSES)] = _rec(True)  # лишь одно лицо из десяти
    for n in range(1000, 1000 + 108):
        processed[n] = _rec(False)

    report = build_report(processed, "test-model")
    assert report["coverage_complete"] is False  # лица покрыты не полностью
    assert report["verdict"]["criterion_8_of_8_plates"] is True
    assert report["verdict"]["triage_works"] is True
