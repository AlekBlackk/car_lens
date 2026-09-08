import csv
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pipeline.crop import Detection
from pipeline.harvest.redaction_profile import CURRENT_REDACTION_PROFILE
from scripts.measure_redaction import (
    ЯЧЕЙКА_H,
    ЯЧЕЙКА_W,
    Остаток,
    main,
    записать_замер,
    записать_лист,
    измерить_кадр,
    корзина,
    листы,
    ячейка,
)

# Контракт CSV для задачи 5: имена и порядок захардкожены нарочно. Сравнение шапки с
# `MEASURE_COLUMNS` было бы самореферентным — переименование поля `Остаток` уехало бы
# в обе стороны сравнения, тест остался бы зелёным, а читатель CSV сломался.
КОЛОНКИ_ЗАМЕРА = [
    "source_item_id",
    "class_id",
    "sha256",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "aspect",
    "overlap",
    "filled_count",
]


class Список:
    def __init__(self, боксы: list[list[Detection]]) -> None:
        self.боксы = list(боксы)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.боксы.pop(0) if self.боксы else []


def _остаток(aspect: float, overlap: float) -> Остаток:
    return Остаток(
        source_item_id="1",
        class_id="lifan__x60",
        sha256="ab",
        x=0,
        y=0,
        w=int(100 * aspect),
        h=100,
        conf=0.5,
        aspect=aspect,
        overlap=overlap,
        filled_count=1,
    )


def test_корзина_A_когда_остаток_под_заливкой() -> None:  # noqa: N802
    assert корзина(_остаток(0.5, 0.9), aspect_min=1.0, overlap_min=0.5) == "A"


def test_корзина_B_когда_форма_невозможна_а_заливки_нет() -> None:  # noqa: N802
    assert корзина(_остаток(0.5, 0.0), aspect_min=1.0, overlap_min=0.5) == "B"


def test_корзина_V_когда_не_гасится_ничем() -> None:  # noqa: N802
    assert корзина(_остаток(4.0, 0.0), aspect_min=1.0, overlap_min=0.5) == "V"


def test_корзина_A_на_ровной_границе_перекрытия() -> None:  # noqa: N802
    """Ровные границы — это и есть выбираемые пороги: потолки Global Constraints."""
    assert корзина(_остаток(4.0, 0.5), aspect_min=1.0, overlap_min=0.5) == "A"


def test_корзина_V_на_ровной_границе_формы() -> None:  # noqa: N802
    assert корзина(_остаток(1.0, 0.0), aspect_min=1.0, overlap_min=0.5) == "V"


def test_измерение_кадра_считает_aspect_и_overlap_по_остатку() -> None:
    кадр = np.full((300, 600, 3), 200, dtype=np.uint8)
    номер = Detection(100.0, 100.0, 80.0, 20.0, 0.9, 0)
    остаток_в_заливке = Detection(100.0, 100.0, 40.0, 20.0, 0.4, 0)

    _, остатки = измерить_кадр(
        кадр,
        [],
        [Список([[номер], [остаток_в_заливке]])],
        source_item_id="1",
        class_id="lifan__x60",
        sha256="ab",
    )

    assert len(остатки) == 1
    assert остатки[0].aspect == 2.0
    assert остатки[0].overlap == 1.0
    assert остатки[0].filled_count == 1


def test_листы_разбиваются_по_сто_ячеек() -> None:
    ячейки = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(101)]

    страницы = листы(ячейки)

    assert len(страницы) == 2
    assert страницы[0].shape == (1200, 1600, 3)


def test_ячейка_берёт_поле_контекста_вокруг_бокса() -> None:
    кадр = np.full((300, 600, 3), 50, dtype=np.uint8)
    кадр[100:120, 100:200] = 200
    # `ячейка` читает только геометрию остатка, поэтому подменяется именно она.
    остаток = replace(_остаток(5.0, 0.0), x=100, y=100, w=100, h=20)

    плитка = ячейка(кадр, остаток)

    assert плитка.shape == (ЯЧЕЙКА_H, ЯЧЕЙКА_W, 3)
    assert плитка[60, 80].tolist() == [200, 200, 200], "середина плитки — сам бокс"
    assert плитка[60, 5].tolist() == [50, 50, 50], "слева от бокса — фон кадра"
    assert плитка[60, 150].tolist() == [50, 50, 50], "справа от бокса — фон кадра"


def test_ячейка_вырожденного_бокса_остаётся_чёрной_плиткой() -> None:
    """Ветка пустого среза защитная: без теста ревью справедливо назовёт её мёртвой."""
    кадр = np.full((300, 600, 3), 200, dtype=np.uint8)
    нулевой = _остаток(0.0, 0.0)  # ширина 0 на левой границе кадра — срез пустой

    плитка = ячейка(кадр, нулевой)

    assert плитка.shape == (ЯЧЕЙКА_H, ЯЧЕЙКА_W, 3)
    assert not плитка[20:100].any(), "пустой срез даёт чёрную плитку, а не падение"


def test_csv_замера_читается_обратно_целиком(tmp_path: Path) -> None:
    """Формат держится тестом, а не докстрингом: этот файл читает задача 5."""
    путь = tmp_path / "measure" / "reject_measure.csv"
    остаток = _остаток(2.0, 0.75)

    записать_замер(путь, [остаток])

    строки = путь.read_text(encoding="utf-8").splitlines()
    assert строки[0].split(",") == КОЛОНКИ_ЗАМЕРА
    assert len(строки) == 2, "пустых строк между записями нет"
    assert list(csv.DictReader(строки)) == [
        {имя: str(getattr(остаток, имя)) for имя in КОЛОНКИ_ЗАМЕРА}
    ]


def test_подпись_ячейки_обведена_чёрным() -> None:
    """На светлой ячейке одна зелёная подпись теряется, а по ней ячейка ищется в CSV."""
    кадр = np.full((300, 600, 3), 255, dtype=np.uint8)

    полоса = ячейка(кадр, _остаток(2.0, 0.5))[:20]

    assert (полоса == (0, 0, 0)).all(axis=2).any(), "обводка подписи чёрная"
    assert (полоса == (0, 255, 0)).all(axis=2).any(), "поверх обводки — зелёный текст"


def test_лист_пишется_через_imencode_когда_imwrite_отказал(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`imwrite` отдаёт False на не-ASCII пути молча — страница обязана лечь всё равно."""
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    путь = tmp_path / "sheet_A_01.png"

    записать_лист(путь, np.zeros((ЯЧЕЙКА_H, ЯЧЕЙКА_W, 3), dtype=np.uint8))

    assert путь.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_берёт_детекторы_номера_из_единой_фабрики(tmp_path: Path) -> None:
    """Замер обязан мерить тем же контуром, каким кадры отбивались (§3.1 дизайна).

    Карта берётся пустой: доказывается маршрут детекторов, а не цикл по кадрам, — а сам
    цикл в этом файле уже разобран на `измерить_кадр` без весов.
    """
    корень = tmp_path / "data"
    корень.mkdir(parents=True)
    (корень / "reject_index.csv").write_text("source_item_id,class_id,sha256\n", encoding="utf-8")
    модели = tmp_path / "models"
    пины = {"plate": type("Pin", (), {"format": "onnx"})()}

    with (
        patch("pipeline.harvest.detectors.verify_weights"),
        patch("pipeline.harvest.detectors.load_detectors_manifest", return_value=пины),
        patch("pipeline.harvest.redaction_profile.build_plate_detectors") as фабрика,
        patch("pipeline.detectors.yolo.YoloDetector"),
    ):
        код = main(
            [
                "--data-root",
                str(корень),
                "--models",
                str(модели),
                "--out",
                str(tmp_path / "measure"),
            ]
        )

    assert код == 0
    assert фабрика.call_args.args == (CURRENT_REDACTION_PROFILE,)
    assert фабрика.call_args.kwargs == {"models_dir": модели, "pins": пины}
