"""Тесты модуля pipeline/class_purity."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from pipeline.class_purity import (
    CONFIDENCES,
    DROP_COLUMNS,
    TIERS,
    ClassPurityError,
    DropRow,
    class_impact,
    cleaned_pool,
    read_drop_list,
    select_tiers,
    unknown_ids,
)


def _make_row(
    photo_id: str = "p1",
    model_id: str = "bmw__3_series",
    sheet: int = 1,
    cell: int = 1,
    tier: int = 1,
    category: str = "fragment",
    confidence: str = "high",
    reason: str = "деталь",
    crop_path: str = "bmw__3_series/p1.jpg",
) -> DropRow:
    return DropRow(
        photo_id=photo_id,
        model_id=model_id,
        sheet=sheet,
        cell=cell,
        tier=tier,
        category=category,
        confidence=confidence,
        reason=reason,
        crop_path=crop_path,
    )


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def test_константы_модуля() -> None:
    assert DROP_COLUMNS == (
        "photo_id",
        "model_id",
        "sheet",
        "cell",
        "tier",
        "category",
        "confidence",
        "reason",
        "crop_path",
    )
    assert TIERS == (1, 2, 3)
    assert CONFIDENCES == ("high", "medium")


def test_drop_row_валидация_полей() -> None:
    row = _make_row()
    assert row.photo_id == "p1"
    assert row.sheet == 1
    assert row.tier == 1

    # Пустой photo_id
    with pytest.raises(ClassPurityError, match="photo_id"):
        _make_row(photo_id="")
    with pytest.raises(ClassPurityError, match="photo_id"):
        _make_row(photo_id="   ")

    # tier=9 вне TIERS
    with pytest.raises(ClassPurityError, match="tier"):
        _make_row(tier=9)

    # confidence='maybe' вне CONFIDENCES
    with pytest.raises(ClassPurityError, match="confidence"):
        _make_row(confidence="maybe")


def test_read_drop_list_круговое_чтение(tmp_path: Path) -> None:
    csv_path = tmp_path / "drop.csv"
    rows_data: list[list[object]] = [
        ["id1", "audi__a4", 1, 2, 1, "fragment", "high", "фара", "audi/id1.jpg"],
        ["id2", "bmw__3", 1, 3, 2, "nameplate_split", "medium", "модель", "bmw/id2.jpg"],
    ]
    _write_csv(csv_path, list(DROP_COLUMNS), rows_data)

    rows = read_drop_list(csv_path)
    assert len(rows) == 2
    assert rows[0] == DropRow(
        photo_id="id1",
        model_id="audi__a4",
        sheet=1,
        cell=2,
        tier=1,
        category="fragment",
        confidence="high",
        reason="фара",
        crop_path="audi/id1.jpg",
    )
    assert rows[1].tier == 2
    assert rows[1].confidence == "medium"


def test_read_drop_list_чужая_шапка_ошибка(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_header.csv"
    bad_header = list(DROP_COLUMNS)
    bad_header[0] = "wrong_id"
    _write_csv(csv_path, bad_header, [["id1", "audi", 1, 1, 1, "f", "high", "r", "c"]])

    with pytest.raises(ClassPurityError, match=r"(?i)шапка"):
        read_drop_list(csv_path)


def test_read_drop_list_дубль_photo_id_ошибка(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicate.csv"
    rows_data: list[list[object]] = [
        ["dup_id", "audi__a4", 1, 2, 1, "fragment", "high", "фара", "audi/id1.jpg"],
        ["dup_id", "bmw__3", 1, 3, 2, "nameplate", "medium", "модель", "bmw/id2.jpg"],
    ]
    _write_csv(csv_path, list(DROP_COLUMNS), rows_data)

    with pytest.raises(ClassPurityError, match=r"(?i)дважды"):
        read_drop_list(csv_path)


def test_select_tiers() -> None:
    rows = [
        _make_row(photo_id="p1", tier=1),
        _make_row(photo_id="p2", tier=2),
        _make_row(photo_id="p3", tier=3),
        _make_row(photo_id="p4", tier=1),
    ]

    assert select_tiers(rows, [1]) == frozenset({"p1", "p4"})
    assert select_tiers(rows, [1, 2]) == frozenset({"p1", "p2", "p4"})
    assert select_tiers(rows, [3]) == frozenset({"p3"})
    assert select_tiers(rows, []) == frozenset()


def test_unknown_ids() -> None:
    rows_ids = ["p1", "p2", "p3", "p4"]
    pool_ids = ["p1", "p3", "p5"]

    # p2 и p4 отсутствуют в пуле; должны вернуться отсортированными
    missing = unknown_ids(rows_ids, pool_ids)
    assert missing == ("p2", "p4")

    # Все есть
    assert unknown_ids(["p1", "p3"], pool_ids) == ()

    # Пустой список кандидатов
    assert unknown_ids([], pool_ids) == ()


def test_cleaned_pool_выкидывает_строки_и_сохраняет_колонки() -> None:
    pool = pd.DataFrame(
        {
            "photo_id": ["p1", "p2", "p3", "p4"],
            "model_id": ["m1", "m1", "m2", "m2"],
            "other_col": [10, 20, 30, 40],
        }
    )

    cleaned = cleaned_pool(pool, ["p2", "p3"])
    assert list(cleaned.columns) == list(pool.columns)
    assert cleaned["photo_id"].tolist() == ["p1", "p4"]
    assert cleaned["model_id"].tolist() == ["m1", "m2"]
    assert list(cleaned.index) == [0, 1]

    # Если drop_ids не пересекаются с пулом
    unchanged = cleaned_pool(pool, ["unknown_id"])
    assert len(unchanged) == 4
    assert unchanged["photo_id"].tolist() == ["p1", "p2", "p3", "p4"]


def test_class_impact() -> None:
    pool = pd.DataFrame(
        {
            "photo_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "model_id": ["audi", "audi", "bmw", "bmw", "lada", "lada"],
        }
    )

    # Дропаем одну audi (p1) и обе bmw (p3, p4). Lada не затронута.
    impact = class_impact(pool, ["p1", "p3", "p4"])

    assert impact == {
        "audi": (2, 1),
        "bmw": (2, 0),
    }
    assert "lada" not in impact

    # Если ничего не удалено
    empty_impact = class_impact(pool, [])
    assert empty_impact == {}


def test_чтение_реального_файла_кандидатов() -> None:
    real_csv = Path(__file__).resolve().parent.parent / "bench" / "class-purity-drop-candidates.csv"
    if not real_csv.exists():
        pytest.skip("bench/class-purity-drop-candidates.csv не найден")

    rows = read_drop_list(real_csv)
    assert len(rows) == 196
    t1_ids = select_tiers(rows, [1])
    assert len(t1_ids) == 67
    t2_ids = select_tiers(rows, [2])
    assert len(t2_ids) == 104
    t3_ids = select_tiers(rows, [3])
    assert len(t3_ids) == 25
