from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from pipeline.redaction_calibration.measure import (
    black_fraction,
    load_frame_profile_results,
    summarize_profile_results,
    write_frame_profile_results,
)
from pipeline.redaction_calibration.model import FrameProfileResult, ProfileMetrics


def _result(
    *,
    photo_id: str,
    old: float | None,
    new: float | None,
    equal: bool | None,
    reason: str | None = None,
) -> FrameProfileResult:
    delta = None if old is None or new is None else max(0.0, new - old)
    return FrameProfileResult(
        photo_id=photo_id,
        class_id="lada__granta",
        outcome="rejected" if reason else "accepted",
        reject_reason=reason,
        old_black_frac=old,
        new_black_frac=new,
        delta_black=delta,
        pixels_equal=equal,
        audit_preserved=False,
    )


def test_black_fraction_использует_grayscale_и_строгий_порог_8() -> None:
    image = np.array([[[0, 0, 20], [0, 0, 30], [8, 8, 8]]], dtype=np.uint8)

    assert black_fraction(image) == pytest.approx(1 / 3)


def test_profile_metrics_считает_положительный_delta_и_строгие_границы() -> None:
    results = [
        _result(photo_id="a", old=0.40, new=0.30, equal=True),
        _result(photo_id="b", old=0.00, new=0.25, equal=False),
        _result(photo_id="c", old=0.10, new=0.36, equal=False, reason="duplicate"),
        _result(photo_id="d", old=0.00, new=0.60, equal=False, reason="over_redacted"),
        _result(photo_id="e", old=None, new=None, equal=None, reason="decode_failed"),
    ]

    assert summarize_profile_results(results) == ProfileMetrics(
        median_delta_black=0.255,
        p95_delta_black=0.60,
        max_delta_black=0.60,
        above_quarter=2,
        new_over_redacted=1,
        changed_crops=3,
        rejects=(("decode_failed", 1), ("duplicate", 1), ("over_redacted", 1)),
    )


def test_profile_metrics_p95_использует_nearest_rank() -> None:
    results = [
        _result(photo_id=str(index), old=0.0, new=index / 100, equal=False)
        for index in range(1, 21)
    ]

    metrics = summarize_profile_results(results)

    assert metrics.p95_delta_black == pytest.approx(0.19)
    assert metrics.median_delta_black == pytest.approx(0.105)


def test_profile_metrics_отвергает_пустой_ввод() -> None:
    with pytest.raises(ValueError, match="пуст"):
        summarize_profile_results([])


def test_profile_result_csv_roundtrip_имеет_строгую_схему(tmp_path: Path) -> None:
    path = tmp_path / "pixel-diff.csv"
    expected = [
        _result(photo_id="a" * 64, old=0.1, new=0.2, equal=False),
        _result(
            photo_id="b" * 64,
            old=None,
            new=None,
            equal=None,
            reason="decode_failed",
        ),
    ]

    write_frame_profile_results(path, expected)

    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == [
        field.name for field in fields(FrameProfileResult)
    ]
    assert load_frame_profile_results(path) == expected

    path.write_text(
        path.read_text(encoding="utf-8").replace("audit_preserved", "unexpected"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="колон"):
        load_frame_profile_results(path)


def test_profile_result_csv_rejects_blank_required_audit_preserved(tmp_path: Path) -> None:
    path = tmp_path / "pixel-diff.csv"
    write_frame_profile_results(
        path,
        [_result(photo_id="a" * 64, old=0.1, new=0.2, equal=False)],
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(",False\n", ",\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="логическ"):
        load_frame_profile_results(path)
