"""Тесты зума спорного места кадра выборки (§4 отчёта)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageFont

from scripts.zoom_plate_leak_frame import build_parser, expand_box, fit_scale, main, render_zoom


def test_расширение_бокса_добавляет_поле() -> None:
    # Бокс — центральная половина кадра 100x100, поле 25% от размера бокса (50x50).
    box = expand_box((0.25, 0.25, 0.75, 0.75), (100, 100), margin=0.25)
    assert box == (12, 12, 88, 88)


def test_расширение_бокса_не_вылезает_за_границы_кадра() -> None:
    box = expand_box((0.0, 0.0, 0.1, 0.1), (100, 100), margin=1.0)
    assert box[0] == 0
    assert box[1] == 0


def test_масштаб_сжимает_большую_сторону() -> None:
    assert fit_scale((3000, 2000), max_side=1500, min_side=700) == pytest.approx(0.5)


def test_масштаб_увеличивает_маленькую_сторону_не_больше_трёх() -> None:
    assert fit_scale((100, 80), max_side=1500, min_side=700) == pytest.approx(3.0)


def test_масштаб_не_трогает_кадр_в_целевой_полосе() -> None:
    assert fit_scale((900, 800), max_side=1500, min_side=700) == 1.0


def test_рендер_зума_вырезает_и_подписывает() -> None:
    исходник = Image.new("RGB", (2000, 2000), "green")
    шрифт = ImageFont.load_default()

    зум = render_zoom(
        исходник, (0, 0, 200, 200), "#001 zoom вырезка 200x200 из 2000x2000", font=шрифт
    )

    # Вырезка 200x200 меньше MIN_SIDE=700 -> увеличена в 3x (потолок), плюс полоса подписи.
    assert зум.size == (600, 600 + 40)


def test_main_роняет_на_неизвестном_номере(tmp_path: Path) -> None:
    выборка = tmp_path / "sample.csv"
    pd.DataFrame({"номер": [1], "crop_path": ["c/a.jpg"]}).to_csv(выборка, index=False)

    with pytest.raises(ValueError, match="номера 2 нет"):
        main(["--sample", str(выборка), "--number", "2", "--box", "0", "0", "1", "1"])


def test_разбор_флагов_даёт_значения_по_умолчанию() -> None:
    аргументы = build_parser().parse_args(
        ["--sample", "s.csv", "--number", "5", "--box", "0.1", "0.1", "0.9", "0.9"]
    )
    assert аргументы.label == "zoom"
    assert аргументы.crops_root is None
