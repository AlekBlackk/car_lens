# tests/test_harvest_audit_gallery.py
from pathlib import Path

import cv2
import numpy as np

from pipeline.harvest.manifest import HarvestRow, append_row
from scripts.audit_gallery import (
    build_parser,
    main,
    render_gallery,
    отобрать_для_показа,
)


def _строка(
    file: str,
    *,
    plates: int = 0,
    faces: int = 0,
    class_id: str = "kia__rio",
    clip_rank: int = 1,
    sha256: str | None = None,
) -> HarvestRow:
    sha = sha256 or file.split(".")[0].rjust(64, "0")
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id="1",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="A",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis="category:X",
        sha256=sha,
        phash=1,
        clip_rank=clip_rank,
        clip_margin=0.0,
        plates=plates,
        faces=faces,
        bbox_x=12.0,
        bbox_y=34.0,
        bbox_w=500.0,
        bbox_h=300.0,
        bbox_area_frac=0.42,
        det_conf=0.91,
        audit="",
    )


def test_галерея_содержит_все_обязательные_кадры() -> None:
    строки = [
        _строка("a.jpg", plates=1),
        _строка("b.jpg", class_id="lada__granta"),
        _строка("c.jpg"),
    ]
    html = render_gallery(строки, {"a.jpg", "b.jpg"}, lambda с: f"thumbs/{с.file}")
    assert "a.jpg" in html and "b.jpg" in html and "c.jpg" in html
    assert "thumbs/a.jpg" in html


def test_обязательные_кадры_идут_первыми_и_сортируются_по_убыванию_clip_rank() -> None:
    строки = [
        _строка("clean_low.jpg", clip_rank=2),
        _строка("clean_high.jpg", clip_rank=10),
        _строка("req_low.jpg", clip_rank=1, plates=1),
        _строка("req_high.jpg", clip_rank=5, plates=1),
    ]
    required = {"req_low.jpg", "req_high.jpg"}
    html = render_gallery(строки, required, lambda с: с.file)

    pos_req_high = html.index('data-f="req_high.jpg"')
    pos_req_low = html.index('data-f="req_low.jpg"')
    pos_clean_high = html.index('data-f="clean_high.jpg"')
    pos_clean_low = html.index('data-f="clean_low.jpg"')

    assert pos_req_high < pos_req_low < pos_clean_high < pos_clean_low


def test_есть_кнопка_выгрузки_и_формат_csv_и_горячие_клавиши() -> None:
    html = render_gallery([_строка("a.jpg")], set(), lambda с: с.file)
    assert "verdicts.csv" in html
    assert "file,verdict" in html
    assert "drop:plate" in html
    assert "drop:face" in html
    assert "drop:wrong_model" in html
    assert "conflict" in html
    assert "ok" in html


def test_экранирование_html() -> None:
    строка = _строка('<test"&>.jpg', class_id='<script>alert("x")</script>')
    html = render_gallery([строка], set(), lambda с: с.file)
    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;" in html
    assert '&lt;test"&amp;&gt;.jpg' in html or "&quot;" in html


def test_build_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(["--manifest", "m.csv", "--data-root", "data_dir"])
    assert args.manifest == Path("m.csv")
    assert args.data_root == Path("data_dir")
    assert args.thumb_px == 480
    assert args.seed == 0
    assert args.verdicts == []
    assert args.only_missing is False

    args_custom = parser.parse_args(
        [
            "--manifest",
            "m.csv",
            "--data-root",
            "d",
            "--thumb-px",
            "320",
            "--seed",
            "42",
            "--verdicts",
            "v1.csv",
            "v2.csv",
            "--only-missing",
        ]
    )
    assert args_custom.thumb_px == 320
    assert args_custom.seed == 42
    assert args_custom.verdicts == [Path("v1.csv"), Path("v2.csv")]
    assert args_custom.only_missing is True



def test_main_ужимает_миниатюры_и_сохраняет_галерею(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "manifest.csv"

    sha1 = "1" * 64
    sha2 = "2" * 64
    row1 = _строка("car1.jpg", class_id="kia__rio", sha256=sha1, plates=1)
    row2 = _строка("car2.jpg", class_id="lada__granta", sha256=sha2)

    append_row(manifest_path, row1)
    append_row(manifest_path, row2)

    crop_dir1 = data_root / "crops" / "kia__rio"
    crop_dir2 = data_root / "crops" / "lada__granta"
    crop_dir1.mkdir(parents=True, exist_ok=True)
    crop_dir2.mkdir(parents=True, exist_ok=True)

    img_large = np.zeros((600, 800, 3), dtype=np.uint8)
    img_small = np.zeros((100, 100, 3), dtype=np.uint8)

    cv2.imwrite(str(crop_dir1 / f"{sha1}.jpg"), img_large)
    cv2.imwrite(str(crop_dir2 / f"{sha2}.jpg"), img_small)

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--thumb-px",
            "300",
            "--seed",
            "0",
        ]
    )
    assert exit_code == 0

    index_html = data_root / "audit" / "index.html"
    assert index_html.exists()
    html_content = index_html.read_text(encoding="utf-8")
    assert "car1.jpg" in html_content
    assert "car2.jpg" in html_content

    thumb1 = data_root / "audit" / "thumbs" / "car1.jpg"
    thumb2 = data_root / "audit" / "thumbs" / "car2.jpg"
    assert thumb1.exists()
    assert thumb2.exists()

    saved_thumb1 = cv2.imread(str(thumb1))
    assert saved_thumb1 is not None
    assert max(saved_thumb1.shape[:2]) <= 300


def test_показываются_только_кадры_без_вердикта() -> None:
    строки = [_строка("a.jpg"), _строка("b.jpg")]
    видно = отобрать_для_показа(строки, {"a.jpg": "ok"}, only_missing=True)
    assert [с.file for с in видно] == ["b.jpg"]


def test_без_флага_показываются_все() -> None:
    строки = [_строка("a.jpg"), _строка("b.jpg")]
    видно = отобрать_для_показа(строки, {"a.jpg": "ok"}, only_missing=False)
    assert [с.file for с in видно] == ["a.jpg", "b.jpg"]

