"""Тесты стейджа галереи сплошного аудита пула."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from pipeline.harvest.cache import cache_path
from pipeline.pool_audit import FACE_KEYS, PLATE_KEYS, REGISTRY_COLUMNS, file_sha256
from pipeline.schema import COLUMNS, write_split
from scripts.stage_pool_audit import (
    StagedFrame,
    context_box,
    main,
    render_page,
    stage_frame,
)

КОНФИГ = re.compile(r'<script type="application/json" id="config">(.*?)</script>', re.DOTALL)


def _jpeg(путь: Path, размер: tuple[int, int]) -> Path:
    путь.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", размер, "red").save(путь)
    return путь


def _подготовить(tmp_path: Path, photo_id: str, *, кроп: bool, оригинал: bool) -> StagedFrame:
    out = tmp_path / "out"
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    (out / "context").mkdir(exist_ok=True)
    if кроп:
        _jpeg(tmp_path / "crops" / "cls" / f"{photo_id}.jpg", (1200, 600))
    if оригинал:
        _jpeg(cache_path(tmp_path / "harvest", "commons", photo_id, "jpg"), (3000, 2000))
    return stage_frame(
        photo_id=photo_id,
        source="commons",
        crop_path=f"cls/{photo_id}.jpg",
        class_id="cls",
        bbox=[100.0, 200.0, 1200.0, 600.0],
        session=1,
        out=out,
        crops_root=tmp_path / "crops",
        raw_root=tmp_path / "harvest",
    )


def test_кадр_получает_миниатюру_контекст_и_зум_на_сам_кроп(tmp_path: Path) -> None:
    кадр = _подготовить(tmp_path, "p0", кроп=True, оригинал=True)
    out = tmp_path / "out"
    кроп = tmp_path / "crops" / "cls" / "p0.jpg"
    assert кадр.problem == ""
    with Image.open(out / кадр.thumb) as миниатюра:
        assert max(миниатюра.size) == 480
    with Image.open(out / кадр.context) as контекст:
        assert контекст.size == (1420, 720)
    assert (out / кадр.zoom).resolve() == кроп.resolve()
    assert кадр.crop_sha256 == file_sha256(кроп)


def test_бокс_контекста_не_выходит_за_кадр() -> None:
    assert context_box([0.0, 0.0, 100.0, 100.0], (100, 100)) == (0, 0, 100, 100)
    assert context_box([100.0, 200.0, 1200.0, 600.0], (3000, 2000)) == (0, 140, 1420, 860)


def test_пропавший_кроп_это_проблема_а_не_исключение(tmp_path: Path) -> None:
    кадр = _подготовить(tmp_path, "p1", кроп=False, оригинал=True)
    assert кадр.problem == "crop_missing"
    assert (кадр.crop_sha256, кадр.thumb, кадр.zoom) == ("", "", "")


def test_пропавший_оригинал_оставляет_кадр_в_работе(tmp_path: Path) -> None:
    кадр = _подготовить(tmp_path, "p2", кроп=True, оригинал=False)
    assert кадр.problem == "original_missing"
    assert кадр.context == ""
    assert кадр.thumb and кадр.zoom


def _конфиг(страница: str) -> dict[str, Any]:
    найдено = КОНФИГ.search(страница)
    assert найдено is not None
    результат: dict[str, Any] = json.loads(найдено.group(1))
    return результат


def _кадр(class_id: str = "cls") -> StagedFrame:
    return StagedFrame(
        photo_id="p0",
        session=1,
        class_id=class_id,
        crop_sha256="a" * 64,
        thumb="thumbs/p0.jpg",
        context="context/p0.jpg",
        zoom="../crops/cls/p0.jpg",
        problem="",
    )


def test_страница_несёт_клавиши_колонки_и_ключ_хранилища() -> None:
    страница = render_page(
        [_кадр()],
        title="t",
        storage_prefix="poolaudit:pilot:abc:",
        layout="grid",
        export_name="export-pilot-session-01.csv",
    )
    конфиг = _конфиг(страница)
    assert конфиг["plate_keys"] == dict(PLATE_KEYS)
    assert конфиг["face_keys"] == dict(FACE_KEYS)
    assert конфиг["columns"] == list(REGISTRY_COLUMNS)
    assert конфиг["prefix"] == "poolaudit:pilot:abc:"
    assert конфиг["export_name"] == "export-pilot-session-01.csv"
    кадр = конфиг["frames"][0]
    assert кадр["image"] == "thumbs/p0.jpg"
    assert кадр["zoom"] == "../crops/cls/p0.jpg"


def test_одиночная_раскладка_показывает_сам_кроп() -> None:
    страница = render_page(
        [_кадр()], title="t", storage_prefix="x:", layout="single", export_name="e.csv"
    )
    assert _конфиг(страница)["frames"][0]["image"] == "../crops/cls/p0.jpg"
    assert '<body class="single">' in страница


def test_данные_не_закрывают_тег_script() -> None:
    страница = render_page(
        [_кадр("</script><b>")],
        title="t",
        storage_prefix="x:",
        layout="grid",
        export_name="e.csv",
    )
    assert "</script><b>" not in страница
    assert _конфиг(страница)["frames"][0]["class_id"] == "</script><b>"


def test_неизвестная_раскладка_отвергается() -> None:
    with pytest.raises(ValueError, match="раскладка"):
        render_page([_кадр()], title="t", storage_prefix="x:", layout="list", export_name="e")


def _пул(путь: Path, n: int) -> None:
    строки = [
        {
            "photo_id": f"p{i}",
            "listing_id": f"l{i}",
            "source": "commons",
            "crop_path": f"cls/p{i}.jpg",
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


def _аргументы(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--split",
        str(tmp_path / "pool.parquet"),
        "--out",
        str(tmp_path / "out"),
        "--crops-root",
        str(tmp_path / "crops"),
        "--raw-root",
        str(tmp_path / "harvest"),
        *extra,
    ]


def _окружение(tmp_path: Path) -> Path:
    _пул(tmp_path / "pool.parquet", 4)
    for i in range(4):
        _jpeg(tmp_path / "crops" / "cls" / f"p{i}.jpg", (64, 32))
    _jpeg(cache_path(tmp_path / "harvest", "commons", "p0", "jpg"), (64, 64))
    ids = tmp_path / "ids.csv"
    ids.write_text("номер,sha256\n1,p1\n2,p2\n", encoding="utf-8")
    return ids


def test_main_стейджит_пилот_по_списку(tmp_path: Path) -> None:
    ids = _окружение(tmp_path)
    assert main(_аргументы(tmp_path, "--mode", "pilot", "--only-ids", str(ids))) == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.glob("*.html")) == ["session-01.html"]
    with (out / "stage.csv").open(encoding="utf-8", newline="") as поток:
        assert [r["photo_id"] for r in csv.DictReader(поток)] == ["p1", "p2"]
    with (out / "problems.csv").open(encoding="utf-8", newline="") as поток:
        assert {r["problem"] for r in csv.DictReader(поток)} == {"original_missing"}
    префикс = _конфиг((out / "session-01.html").read_text(encoding="utf-8"))["prefix"]
    assert префикс == f"poolaudit:pilot:{file_sha256(tmp_path / 'pool.parquet')[:16]}:"


def test_main_не_пишет_в_существующий_каталог(tmp_path: Path) -> None:
    _окружение(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        main(_аргументы(tmp_path, "--mode", "main"))


def test_main_отвергает_id_не_из_пула(tmp_path: Path) -> None:
    _окружение(tmp_path)
    чужие = tmp_path / "bad.csv"
    чужие.write_text("sha256\npX\n", encoding="utf-8")
    with pytest.raises(ValueError, match="не из пула"):
        main(_аргументы(tmp_path, "--mode", "pilot", "--only-ids", str(чужие)))
