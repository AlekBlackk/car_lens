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
    with pytest.raises(ClassPurityError, match="photo_id"):
        _make_row(photo_id=123)  # type: ignore[arg-type]

    # Некорректный sheet / cell (должны быть строго int, не bool и не str)
    with pytest.raises(ClassPurityError, match="sheet"):
        _make_row(sheet="1")  # type: ignore[arg-type]
    with pytest.raises(ClassPurityError, match="sheet"):
        _make_row(sheet=True)
    with pytest.raises(ClassPurityError, match="cell"):
        _make_row(cell="2")  # type: ignore[arg-type]
    with pytest.raises(ClassPurityError, match="cell"):
        _make_row(cell=False)

    # tier вне TIERS или не int (bool/float не допускаются)
    with pytest.raises(ClassPurityError, match="tier"):
        _make_row(tier=9)
    with pytest.raises(ClassPurityError, match="tier"):
        _make_row(tier=True)
    with pytest.raises(ClassPurityError, match="tier"):
        _make_row(tier=1.0)  # type: ignore[arg-type]

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


def test_read_drop_list_поддержка_str_и_utf8_bom(tmp_path: Path) -> None:
    csv_path = tmp_path / "bom.csv"
    row = "id1,audi__a4,1,2,1,fragment,high,фара,audi/id1.jpg\n"
    content = "\ufeff" + ",".join(DROP_COLUMNS) + "\n" + row
    csv_path.write_bytes(content.encode("utf-8"))

    # Вызов как через Path, так и через str
    rows_path = read_drop_list(csv_path)
    assert len(rows_path) == 1
    assert rows_path[0].photo_id == "id1"

    rows_str = read_drop_list(str(csv_path))
    assert len(rows_str) == 1
    assert rows_str[0].photo_id == "id1"


def test_read_drop_list_пустой_файл_ошибка(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    with pytest.raises(ClassPurityError, match=r"(?i)шапка"):
        read_drop_list(empty_csv)


def test_read_drop_list_пропуск_tier_0(tmp_path: Path) -> None:
    csv_path = tmp_path / "struck_out.csv"
    rows_data: list[list[object]] = [
        ["id1", "audi__a4", 1, 1, 1, "fragment", "high", "фара", "audi/id1.jpg"],
        ["id0", "bmw__3", 1, 2, 0, "fragment", "high", "вычеркнут", "bmw/id0.jpg"],
        ["id2", "bmw__3", 1, 3, 2, "nameplate", "medium", "модель", "bmw/id2.jpg"],
    ]
    _write_csv(csv_path, list(DROP_COLUMNS), rows_data)

    rows = read_drop_list(csv_path)
    # tier 0 должен быть пропущен как вычеркнутый
    assert [r.photo_id for r in rows] == ["id1", "id2"]


def test_read_drop_list_битые_строки_ошибка(tmp_path: Path) -> None:
    # Меньше колонок
    bad_cols = tmp_path / "bad_cols.csv"
    _write_csv(bad_cols, list(DROP_COLUMNS), [["id1", "audi", 1]])
    with pytest.raises(ClassPurityError, match=r"колонок"):
        read_drop_list(bad_cols)

    # Нечисловое значение в int колонке
    bad_int = tmp_path / "bad_int.csv"
    bad_row: list[object] = ["id1", "audi", "не_число", 1, 1, "f", "high", "r", "c"]
    _write_csv(bad_int, list(DROP_COLUMNS), [bad_row])
    with pytest.raises(ClassPurityError, match=r"ошибка преобразования чисел"):
        read_drop_list(bad_int)


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

    # Владелец правит список руками (вычёркивает строки или ставит tier=0), поэтому
    # проверяются свойства файла, а не его размер: шапку, дубли, tier и confidence уже
    # сторожит read_drop_list, здесь — связность строк и то, что чистить есть что.
    # Каталог кропа не обязан совпадать с model_id (бывают групповые `grp__*`),
    # а имя файла — всегда photo_id.
    rows = read_drop_list(real_csv)
    assert select_tiers(rows, [1])
    for row in rows:
        assert Path(row.crop_path).name == f"{row.photo_id}.jpg", row.photo_id
