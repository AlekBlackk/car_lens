"""Тесты подготовки просмотровых видов выборки (§4 отчёта, впаянный номер)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageFont

from scripts.stage_plate_leak_views import build_parser, caption_text, main, render_view


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
