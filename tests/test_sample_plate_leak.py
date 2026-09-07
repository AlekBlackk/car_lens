"""Тесты забора черновой выборки для замера утечки номера (§4 отчёта)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.schema import COLUMNS, write_split
from scripts.sample_plate_leak import build_parser, draw_sample, join_manifest, main


def сделать_объединённый_кадр(n: int) -> pd.DataFrame:
    """`n` строк, уже сведённых со столбцами манифеста — как после `join_manifest`."""
    return pd.DataFrame(
        {
            "photo_id": [f"p{i}" for i in range(n)],
            "crop_path": [f"c{i % 3}/p{i}.jpg" for i in range(n)],
            "sha256": [f"p{i}" for i in range(n)],
            "class_id": [f"c{i % 3}" for i in range(n)],
            "source_item_id": [f"item{i}" for i in range(n)],
            "plates": [i % 2 for i in range(n)],
            "faces": [0] * n,
            "audit": ["ok"] * n,
        }
    )


def test_отбор_воспроизводим_по_сиду() -> None:
    кадр = сделать_объединённый_кадр(50)

    первый = draw_sample(кадр, size=10, seed=1337)
    второй = draw_sample(кадр, size=10, seed=1337)
    третий = draw_sample(кадр, size=10, seed=7)

    assert list(первый["sha256"]) == list(второй["sha256"])
    assert list(первый["sha256"]) != list(третий["sha256"])


def test_отбор_нумерует_с_единицы_и_переименовывает_колонки() -> None:
    выборка = draw_sample(сделать_объединённый_кадр(20), size=5, seed=1337)

    assert list(выборка["номер"]) == [1, 2, 3, 4, 5]
    assert list(выборка.columns) == [
        "номер",
        "sha256",
        "pageid",
        "класс",
        "plates",
        "faces",
        "штамп",
        "crop_path",
    ]


def test_отбор_роняет_если_размер_больше_населения() -> None:
    with pytest.raises(ValueError, match="больше населения"):
        draw_sample(сделать_объединённый_кадр(5), size=10, seed=1337)


def test_join_manifest_роняет_на_строке_без_пары() -> None:
    сплит = pd.DataFrame({"photo_id": ["a", "b"]})
    манифест = pd.DataFrame(
        {
            "sha256": ["a"],
            "class_id": ["c"],
            "source_item_id": ["i"],
            "plates": [0],
            "faces": [0],
            "audit": ["ok"],
        }
    )

    with pytest.raises(ValueError, match="без пары в манифесте"):
        join_manifest(сплит, манифест)


def _записать_pool_parquet(путь: Path, n: int) -> None:
    строки = [
        {
            "photo_id": f"p{i}",
            "listing_id": f"l{i}",
            "source": "commons",
            "crop_path": f"c{i % 3}/p{i}.jpg",
            "brand": "Lada",
            "model_id": "lada__2107",
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
        for i in range(n)
    ]
    write_split(pd.DataFrame(строки, columns=list(COLUMNS)), путь)


def _записать_манифест(путь: Path, n: int) -> None:
    шапка = "sha256,class_id,source_item_id,plates,faces,audit"
    строки = [шапка] + [f"p{i},lada__2107,item{i},{i % 2},0,ok" for i in range(n)]
    путь.write_text("\n".join(строки) + "\n", encoding="utf-8")


def test_main_пишет_выборку_и_паспорт(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))
    _записать_pool_parquet(tmp_path / "splits" / "pool.parquet", n=20)
    манифест = tmp_path / "manifest.csv"
    _записать_манифест(манифест, n=20)
    выход = tmp_path / "draft.csv"

    код = main(
        [
            "--manifest",
            str(манифест),
            "--size",
            "5",
            "--seed",
            "1337",
            "--out",
            str(выход),
        ]
    )

    assert код == 0
    выборка = pd.read_csv(выход)
    assert len(выборка) == 5
    assert list(выборка.columns) == [
        "номер",
        "sha256",
        "pageid",
        "класс",
        "plates",
        "faces",
        "штамп",
        "crop_path",
    ]

    паспорт = json.loads((tmp_path / "draft-meta.json").read_text(encoding="utf-8"))
    assert паспорт["сид"] == 1337
    assert паспорт["выборка"]["размер"] == 5
    assert паспорт["сплит"]["строк"] == 20


def test_разбор_флагов_даёт_значения_по_умолчанию() -> None:
    аргументы = build_parser().parse_args([])
    assert аргументы.split == "pool"
    assert аргументы.size == 300
    assert аргументы.seed == 1337
    assert аргументы.out == Path("bench") / "plate-leak-sample-draft.csv"
