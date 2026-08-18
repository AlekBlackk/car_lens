"""Тесты отбора representative dataset (§4.3 дизайна M-1.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from export.calibration import Calibration, CalibrationError, select_calibration
from training.samples import Sample

КЛАССОВ = 171


def _индекс(классов: int = КЛАССОВ) -> dict[str, int]:
    return {f"brand__model_{номер:03d}": номер for номер in range(классов)}


def _образцы(классов: int = КЛАССОВ, на_класс: int = 5) -> list[Sample]:
    образцы = []
    for класс in range(классов):
        for номер in range(на_класс):
            образцы.append(
                Sample(
                    path=Path(f"crops/stanford/{класс:03d}/{номер:02d}.jpg"),
                    class_idx=класс,
                    is_multi_car=False,
                )
            )
    return образцы


def test_итог_ровно_512() -> None:
    итог = select_calibration(_образцы(), class_index=_индекс(), total=512)
    assert len(итог.samples) == 512
    assert sum(итог.per_class_counts.values()) == 512


def test_остаток_раздаётся_по_возрастанию_класса() -> None:
    """171 класс и 512 кропов: 170 классов по 3, один — по 2. Равной квоты не бывает."""
    итог = select_calibration(_образцы(), class_index=_индекс(), total=512)
    счётчики = [итог.per_class_counts[f"brand__model_{номер:03d}"] for номер in range(КЛАССОВ)]
    assert счётчики[:170] == [3] * 170
    assert счётчики[170] == 2


def test_порядок_строк_на_выбор_не_влияет() -> None:
    """Сид что-то значит только при стабильной сортировке до розыгрыша."""
    образцы = _образцы()
    перемешанные = list(reversed(образцы))
    первый = select_calibration(образцы, class_index=_индекс(), total=512)
    второй = select_calibration(перемешанные, class_index=_индекс(), total=512)
    assert [образец.path for образец in первый.samples] == [
        образец.path for образец in второй.samples
    ]


def test_другой_сид_даёт_другой_набор() -> None:
    первый = select_calibration(_образцы(), class_index=_индекс(), total=512, seed=1337)
    второй = select_calibration(_образцы(), class_index=_индекс(), total=512, seed=7)
    assert [о.path for о in первый.samples] != [о.path for о in второй.samples]


def test_недобор_отдаёт_что_есть_и_попадает_в_счётчик() -> None:
    """Соседи за бедный класс не добирают: квота на класс и есть стратификация."""
    образцы = [о for о in _образцы() if not (о.class_idx == 5 and о.path.name != "00.jpg")]
    итог = select_calibration(образцы, class_index=_индекс(), total=512)
    assert итог.per_class_counts["brand__model_005"] == 1
    assert len(итог.samples) == 510
    assert итог.requested_total == 512


def test_класс_без_кропов_падает() -> None:
    """Ради этой проверки class_index и передаётся: иначе класс просто не виден."""
    образцы = [о for о in _образцы() if о.class_idx != 9]
    with pytest.raises(CalibrationError, match="brand__model_009"):
        select_calibration(образцы, class_index=_индекс(), total=512)


def test_метка_вне_индекса_падает() -> None:
    образцы = [*_образцы(классов=3, на_класс=4), Sample(Path("x.jpg"), 99, False)]
    with pytest.raises(CalibrationError, match="99"):
        select_calibration(образцы, class_index=_индекс(классов=3), total=6)


def test_total_меньше_числа_классов_падает() -> None:
    """При квоте ноль часть классов не попала бы в калибровку вовсе."""
    with pytest.raises(CalibrationError, match="меньше числа классов"):
        select_calibration(_образцы(классов=4, на_класс=3), class_index=_индекс(4), total=3)


def test_счётчики_ключуются_по_model_id() -> None:
    итог = select_calibration(_образцы(классов=3, на_класс=4), class_index=_индекс(3), total=6)
    assert isinstance(итог, Calibration)
    assert set(итог.per_class_counts) == set(_индекс(3))
    assert итог.seed == 1337
