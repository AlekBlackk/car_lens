"""Тесты подготовки просмотровых видов выборки (§4 отчёта, впаянный номер)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageChops, ImageFont

from scripts.stage_plate_leak_views import (
    CAPTION_HEIGHT,
    CAPTION_MARGIN,
    STATUS_TILE_MARGIN,
    STATUS_TILE_SIZE,
    _status_tile,
    build_parser,
    caption_text,
    fit_text_lines,
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


def _ledger_row(
    photo_id: str,
    outcome: str,
    *,
    reason: str = "",
    crop_path: str = "",
    class_id: str = "class-a",
    plates: object = 0,
    faces: object = 0,
) -> dict[str, object]:
    """Строка ledger в том же составе колонок, который пишет measure CLI."""
    return {
        "photo_id": photo_id,
        "pipeline_outcome": outcome,
        "reject_reason": reason,
        "review_status": "resolved" if outcome == "rejected" else "pending",
        "plate_verdict": "",
        "note": "",
        "crop_path": crop_path,
        "class_id": class_id,
        "plates": plates,
        "faces": faces,
    }


def _write_party_ledger(path: Path) -> None:
    pd.DataFrame(
        [
            _ledger_row(
                "accepted-photo-id",
                "accepted",
                crop_path="accepted.jpg",
                class_id="class-a",
                plates=1,
            ),
            _ledger_row(
                "rejected-photo-id",
                "rejected",
                reason="over_redacted",
                class_id="class-b",
                faces=2,
            ),
            _ledger_row(
                "missing-photo-id",
                "accepted",
                crop_path="missing.jpg",
                class_id="class-c",
                plates=2,
            ),
        ]
    ).to_csv(path, index=False)


def _stage(ledger: Path, crops: Path, output: Path, party_number: int = 2) -> int:
    return main(
        [
            "--party-ledger",
            str(ledger),
            "--party-number",
            str(party_number),
            "--crops-root",
            str(crops),
            "--out-dir",
            str(output),
            "--font",
            "default",
        ]
    )


def _ink_box(tile: Image.Image) -> tuple[int, int, int, int] | None:
    """Рамка тёмного текста на сером фоне плитки, без шума JPEG."""
    background = Image.new("RGB", tile.size, "#eeeeee")
    difference = ImageChops.difference(tile, background).convert("L")
    return difference.point(lambda value: 255 if value > 40 else 0).getbbox()


def _tile_ink_box(view: Image.Image) -> tuple[int, int, int, int] | None:
    """То же для сохранённого вида: полоса подписи сверху не учитывается."""
    top = view.size[1] - STATUS_TILE_SIZE[1]
    return _ink_box(view.crop((0, top, view.size[0], view.size[1])))


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
    pd.DataFrame([_ledger_row("same", "rejected", reason="x")] * 2).to_csv(ledger, index=False)

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


def test_party_mode_rejects_ledger_without_audit_columns(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame(
        [{"photo_id": "id-a", "pipeline_outcome": "accepted", "reject_reason": ""}]
    ).to_csv(ledger, index=False)

    with pytest.raises(ValueError, match="crop_path"):
        _stage(ledger, tmp_path, tmp_path / "views")


@pytest.mark.parametrize("crop_path", ["../outside.jpg", "nested/../../outside.jpg"])
def test_party_mode_rejects_crop_path_leaving_crops_root(tmp_path: Path, crop_path: str) -> None:
    ledger = tmp_path / "ledger.csv"
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.new("RGB", (40, 30), "red").save(tmp_path / "outside.jpg")
    pd.DataFrame([_ledger_row("id-a", "accepted", crop_path=crop_path)]).to_csv(ledger, index=False)

    with pytest.raises(ValueError, match="crops-root"):
        _stage(ledger, crops, tmp_path / "views")


def test_party_mode_rejects_absolute_crop_path(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    crops = tmp_path / "crops"
    crops.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (40, 30), "red").save(outside)
    pd.DataFrame([_ledger_row("id-a", "accepted", crop_path=str(outside))]).to_csv(
        ledger, index=False
    )

    with pytest.raises(ValueError, match="crops-root"):
        _stage(ledger, crops, tmp_path / "views")


def test_party_mode_keeps_ledger_operational_error_reason(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    crops = tmp_path / "crops"
    output = tmp_path / "views"
    crops.mkdir()
    pd.DataFrame([_ledger_row("id-a", "operational_error", reason="redaction_failed")]).to_csv(
        ledger, index=False
    )

    assert _stage(ledger, crops, output) == 0

    records = pd.read_csv(output / "views.csv", keep_default_na=False)
    assert records.loc[0, "pipeline_outcome"] == "operational_error"
    assert records.loc[0, "reject_reason"] == "redaction_failed"
    assert records.loc[0, "detail"] == "OPERATIONAL ERROR: redaction_failed"
    assert "missing accepted crop" not in str(records.loc[0, "detail"])


def test_party_mode_treats_blank_detector_cells_as_zero(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    crops = tmp_path / "crops"
    output = tmp_path / "views"
    crops.mkdir()
    pd.DataFrame(
        [_ledger_row("id-a", "rejected", reason="over_redacted", plates="", faces="")]
    ).to_csv(ledger, index=False)

    assert _stage(ledger, crops, output) == 0

    records = pd.read_csv(output / "views.csv", keep_default_na=False)
    assert records.loc[0, "plates"] == 0
    assert records.loc[0, "faces"] == 0
    assert "plates=0 faces=0" in str(records.loc[0, "caption"])


def test_status_tile_keeps_long_detail_whole_and_inside_its_width() -> None:
    font = ImageFont.load_default()
    detail = "OPERATIONAL ERROR: missing accepted crop " + "/very/long/workspace/segment" * 20
    max_width = STATUS_TILE_SIZE[0] - 2 * STATUS_TILE_MARGIN

    lines = fit_text_lines(detail, font, max_width)
    tile = _status_tile(detail, font)

    assert "".join(lines) == detail
    assert len(lines) > 1
    box = _ink_box(tile)
    assert box is not None
    assert box[0] >= STATUS_TILE_MARGIN - 1
    assert box[2] <= STATUS_TILE_SIZE[0] - STATUS_TILE_MARGIN
    assert box[3] <= STATUS_TILE_SIZE[1]


def test_party_tile_details_are_rendered_inside_the_tile(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    crops = tmp_path / "crops"
    output = tmp_path / "views"
    crops.mkdir()
    long_reason = "over_redacted_" + "reason_segment_" * 25
    pd.DataFrame(
        [
            _ledger_row("rejected-id", "rejected", reason=long_reason),
            _ledger_row(
                "missing-id", "accepted", crop_path="deeply/" * 45 + "missing_crop_file.jpg"
            ),
        ]
    ).to_csv(ledger, index=False)

    assert _stage(ledger, crops, output) == 0

    records = pd.read_csv(output / "views.csv", keep_default_na=False)
    for index in (0, 1):
        view = Image.open(output / str(records.loc[index, "file"]))
        box = _tile_ink_box(view)
        assert box is not None
        assert box[2] <= STATUS_TILE_SIZE[0] - STATUS_TILE_MARGIN
        assert box[3] <= STATUS_TILE_SIZE[1]
        assert box[3] - box[1] > 20, "деталь должна переноситься на несколько строк"
    assert long_reason in str(records.loc[0, "detail"])
    assert "missing_crop_file.jpg" in str(records.loc[1, "detail"])


def test_long_caption_stays_one_line_without_wrapping() -> None:
    """Прежние вызовы render_view обязаны сохранить однострочную полосу."""
    font = ImageFont.load_default()
    tile = Image.new("RGB", (200, 120), "red")

    view = render_view(tile, "very " * 200, long_side=None, font=font)

    assert view.size == (200, 120 + CAPTION_HEIGHT)


def test_long_caption_is_wrapped_into_a_taller_strip() -> None:
    font = ImageFont.load_default()
    tile = Image.new("RGB", STATUS_TILE_SIZE, "#eeeeee")
    caption = party_caption_text(
        party_number=7,
        frame_number=4,
        photo_id="dd99ee00ff11aa22",
        class_id="renault__logan_" + "extra_long_class_identifier_" * 12,
        pipeline_outcome="operational_error",
        plates=2,
        faces=0,
        width=1600,
        height=900,
    )

    view = render_view(tile, caption, long_side=None, font=font, wrap_caption=True)

    lines = fit_text_lines(caption, font, STATUS_TILE_SIZE[0] - 2 * CAPTION_MARGIN)
    assert "".join(lines) == caption
    assert len(lines) > 1
    assert view.size == (STATUS_TILE_SIZE[0], STATUS_TILE_SIZE[1] + CAPTION_HEIGHT * len(lines))
    strip = view.crop((0, 0, view.size[0], CAPTION_HEIGHT * len(lines)))
    background = Image.new("RGB", strip.size, "white")
    box = ImageChops.difference(strip, background).convert("L")
    ink = box.point(lambda value: 255 if value > 40 else 0).getbbox()
    assert ink is not None
    assert ink[2] <= STATUS_TILE_SIZE[0] - 1
