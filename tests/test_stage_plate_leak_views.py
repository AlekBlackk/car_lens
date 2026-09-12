"""Тесты подготовки просмотровых видов выборки (§4 отчёта, впаянный номер)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageFont

from scripts.stage_plate_leak_views import (
    build_parser,
    caption_text,
    main,
    party_caption_text,
    render_view,
)


def test_подпись_несёт_номер_класс_числа_и_размер() -> None:
    текст = caption_text(7, "bmw__5_series", plates=4, faces=0, width=3000, height=2000)
    for подстрока in ("#007", "bmw__5_series", "plates=4", "faces=0", "3000x2000"):
        assert подстрока in текст


def test_подпись_дополняет_номер_нулями_до_трёх_знаков() -> None:
    assert caption_text(7, "c", 0, 0, 1, 1).startswith("#007")
    assert caption_text(123, "c", 0, 0, 1, 1).startswith("#123")


def test_рендер_уменьшает_длинную_сторону_и_добавляет_полосу() -> None:
    исходник = Image.new("RGB", (3200, 2000), "red")
    шрифт = ImageFont.load_default()

    вид = render_view(исходник, "#001 test", long_side=1600, font=шрифт)

    # Длинная сторона 3200 -> 1600, короткая пропорционально 2000 -> 1000,
    # плюс полоса подписи сверху.
    assert вид.size == (1600, 1000 + 46)


def test_рендер_не_увеличивает_кадр_меньше_long_side() -> None:
    исходник = Image.new("RGB", (800, 600), "blue")
    шрифт = ImageFont.load_default()

    вид = render_view(исходник, "#002 test", long_side=1600, font=шрифт)

    assert вид.size == (800, 600 + 46)


def test_рендер_полного_разрешения_не_уменьшает_и_впаивает_подпись() -> None:
    исходник = Image.new("RGB", (3200, 2000), "red")
    шрифт = ImageFont.load_default()

    вид = render_view(
        исходник,
        "plate-001 sha=aaaa #007 (10,20,30,40)",
        long_side=None,
        font=шрифт,
    )

    assert вид.size == (3200, 2000 + 46)
    assert вид.getpixel((10, 8)) != исходник.getpixel((10, 8))


def test_main_роняет_на_черновике_без_crop_path(tmp_path: Path) -> None:
    выборка = tmp_path / "sample.csv"
    pd.DataFrame(
        {"номер": [1], "sha256": ["a"], "класс": ["c"], "plates": [0], "faces": [0]}
    ).to_csv(выборка, index=False)

    with pytest.raises(ValueError, match="crop_path"):
        main(["--sample", str(выборка)])


def test_разбор_флагов_даёт_значения_по_умолчанию() -> None:
    аргументы = build_parser().parse_args(["--sample", "s.csv"])
    assert аргументы.long_side == 1600
    assert аргументы.crops_root is None
    assert аргументы.out_dir is None


def test_party_caption_burns_required_identity_and_detector_fields() -> None:
    caption = party_caption_text(
        party_number=2,
        frame_number=7,
        photo_id="abcdef0123456789",
        class_id="lada__vesta",
        pipeline_outcome="accepted",
        plates=3,
        faces=1,
        width=2400,
        height=1600,
    )

    for expected in (
        "party=002",
        "frame=007",
        "photo=abcdef012345",
        "lada__vesta",
        "outcome=accepted",
        "plates=3",
        "faces=1",
        "2400x1600",
    ):
        assert expected in caption


def _write_party_ledger(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "photo_id": "accepted-photo-id",
                "pipeline_outcome": "accepted",
                "reject_reason": "",
                "review_status": "pending",
                "plate_verdict": "",
                "note": "",
                "crop_path": "accepted.jpg",
                "class_id": "class-a",
                "plates": 1,
                "faces": 0,
            },
            {
                "photo_id": "rejected-photo-id",
                "pipeline_outcome": "rejected",
                "reject_reason": "over_redacted",
                "review_status": "resolved",
                "plate_verdict": "",
                "note": "",
                "crop_path": "",
                "class_id": "class-b",
                "plates": 0,
                "faces": 2,
            },
            {
                "photo_id": "missing-photo-id",
                "pipeline_outcome": "accepted",
                "reject_reason": "",
                "review_status": "pending",
                "plate_verdict": "",
                "note": "",
                "crop_path": "missing.jpg",
                "class_id": "class-c",
                "plates": 2,
                "faces": 0,
            },
        ]
    ).to_csv(path, index=False)


def test_party_mode_stages_full_resolution_views_and_status_tiles(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    crops = tmp_path / "crops"
    output = tmp_path / "views"
    crops.mkdir()
    _write_party_ledger(ledger)
    Image.new("RGB", (3200, 2000), "red").save(crops / "accepted.jpg")

    assert (
        main(
            [
                "--party-ledger",
                str(ledger),
                "--party-number",
                "2",
                "--crops-root",
                str(crops),
                "--out-dir",
                str(output),
                "--font",
                "default",
            ]
        )
        == 0
    )

    records = pd.read_csv(output / "views.csv", keep_default_na=False)
    assert records[["frame_number", "photo_id", "view_status"]].to_dict("records") == [
        {"frame_number": 1, "photo_id": "accepted-photo-id", "view_status": "image"},
        {"frame_number": 2, "photo_id": "rejected-photo-id", "view_status": "rejected_tile"},
        {
            "frame_number": 3,
            "photo_id": "missing-photo-id",
            "view_status": "operational_error",
        },
    ]
    accepted = Image.open(output / str(records.loc[0, "file"]))
    assert accepted.size == (3200, 2000 + 46)
    rejected = Image.open(output / str(records.loc[1, "file"]))
    missing = Image.open(output / str(records.loc[2, "file"]))
    assert rejected.size == missing.size
    assert rejected.size == (1600, 900 + 46)
    assert records.loc[2, "pipeline_outcome"] == "operational_error"
    assert records.loc[2, "reject_reason"] == "missing_crop"


def test_party_mode_rejects_ledger_with_duplicate_photo_ids(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame(
        [
            {"photo_id": "same", "pipeline_outcome": "rejected", "reject_reason": "x"},
            {"photo_id": "same", "pipeline_outcome": "rejected", "reject_reason": "x"},
        ]
    ).to_csv(ledger, index=False)

    with pytest.raises(ValueError, match="дубликат"):
        main(
            [
                "--party-ledger",
                str(ledger),
                "--party-number",
                "1",
                "--crops-root",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "views"),
                "--font",
                "default",
            ]
        )
