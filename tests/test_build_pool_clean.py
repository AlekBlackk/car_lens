"""Тесты CLI сборки чистого пула scripts/build_pool_clean.py."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from pipeline.class_purity import DROP_COLUMNS, ClassPurityError
from pipeline.schema import COLUMNS, read_split, write_split
from scripts.build_pool_clean import main


def _make_synthetic_pool(path: Path, items: list[tuple[str, str]]) -> None:
    """Создать валидный сплит с заданными (photo_id, model_id)."""
    records = [
        {
            "photo_id": pid,
            "listing_id": f"list_{i}",
            "source": "commons",
            "crop_path": f"{mid}/{pid}.jpg",
            "brand": "Lada",
            "model_id": mid,
            "class_idx": 0,
            "body_style": None,
            "year": None,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "bbox_area_frac": 0.5,
            "det_conf": 0.9,
            "phash": i,
            "raw_model_string": None,
            "is_multi_car": False,
        }
        for i, (pid, mid) in enumerate(items)
    ]
    write_split(pd.DataFrame(records, columns=list(COLUMNS)), path)


def _make_drop_csv(path: Path, items: list[tuple[str, str, int]]) -> None:
    """Создать CSV кандидатов с (photo_id, model_id, tier)."""
    rows = [
        [pid, mid, 1, 1, tier, "fragment", "high", "деталь", f"{mid}/{pid}.jpg"]
        for pid, mid, tier in items
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(DROP_COLUMNS))
        for r in rows:
            writer.writerow(r)


def test_сборка_на_синтетическом_пуле(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pool_path = tmp_path / "pool.parquet"
    drop_path = tmp_path / "drop.csv"
    out_path = tmp_path / "pool-clean.parquet"

    _make_synthetic_pool(
        pool_path,
        [("p1", "bmw"), ("p2", "bmw"), ("p3", "audi"), ("p4", "audi")],
    )
    _make_drop_csv(drop_path, [("p2", "bmw", 1)])

    ret = main(
        [
            "--split",
            str(pool_path),
            "--drop-list",
            str(drop_path),
            "--out",
            str(out_path),
        ]
    )
    assert ret == 0
    assert out_path.exists()

    cleaned = read_split(out_path)
    assert sorted(cleaned["photo_id"].tolist()) == ["p1", "p3", "p4"]

    out = capsys.readouterr().out
    assert "удалено 1, осталось 3" in out
    assert "затронутых классов: 1" in out


def test_отказ_писать_в_исходный_сплит(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.parquet"
    drop_path = tmp_path / "drop.csv"

    _make_synthetic_pool(pool_path, [("p1", "bmw")])
    _make_drop_csv(drop_path, [("p1", "bmw", 1)])

    with pytest.raises(ValueError, match=r"(?i)исходный пул"):
        main(
            [
                "--split",
                str(pool_path),
                "--drop-list",
                str(drop_path),
                "--out",
                str(pool_path),
            ]
        )


def test_неизвестный_id_ошибка_и_файл_не_создан(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.parquet"
    drop_path = tmp_path / "drop.csv"
    out_path = tmp_path / "pool-clean.parquet"

    _make_synthetic_pool(pool_path, [("p1", "bmw")])
    _make_drop_csv(drop_path, [("unknown_id", "bmw", 1)])

    with pytest.raises(ClassPurityError):
        main(
            [
                "--split",
                str(pool_path),
                "--drop-list",
                str(drop_path),
                "--out",
                str(out_path),
            ]
        )

    assert not out_path.exists()


def test_предупреждение_по_min_class(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pool_path = tmp_path / "pool.parquet"
    drop_path = tmp_path / "drop.csv"
    out_path = tmp_path / "pool-clean.parquet"

    # m1: 15 кадров, m2: 2 кадра
    items = [(f"m1_{i}", "model_1") for i in range(15)]
    items.extend([("m2_0", "model_2"), ("m2_1", "model_2")])
    _make_synthetic_pool(pool_path, items)

    # Дропаем 1 кадр из model_2: было 2, стало 1
    _make_drop_csv(drop_path, [("m2_0", "model_2", 1)])

    ret = main(
        [
            "--split",
            str(pool_path),
            "--drop-list",
            str(drop_path),
            "--out",
            str(out_path),
            "--min-class",
            "10",
        ]
    )
    assert ret == 0

    out = capsys.readouterr().out
    assert "model_2: 2 -> 1" in out
    assert "model_1" not in out


def test_tiers_берёт_несколько_уровней(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.parquet"
    drop_path = tmp_path / "drop.csv"
    out_path = tmp_path / "pool-clean.parquet"

    _make_synthetic_pool(
        pool_path,
        [("p1", "m1"), ("p2", "m1"), ("p3", "m2"), ("p4", "m2")],
    )
    _make_drop_csv(
        drop_path,
        [("p1", "m1", 1), ("p2", "m1", 2), ("p3", "m2", 3)],
    )

    # Исключаем tier 1 и tier 2
    ret = main(
        [
            "--split",
            str(pool_path),
            "--drop-list",
            str(drop_path),
            "--out",
            str(out_path),
            "--tiers",
            "1",
            "2",
        ]
    )
    assert ret == 0

    cleaned = read_split(out_path)
    # p1 (tier 1) и p2 (tier 2) удалены, p3 (tier 3) и p4 (нет в drop) остались
    assert sorted(cleaned["photo_id"].tolist()) == ["p3", "p4"]
