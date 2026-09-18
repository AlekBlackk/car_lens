"""Тесты финализации сплошного аудита пула."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from pipeline.pool_audit import (
    PENDING,
    PoolAuditError,
    RegistryRow,
    file_sha256,
    write_registry,
)
from pipeline.schema import COLUMNS, read_split, write_split
from scripts.finalize_pool_audit import main


def _окружение(tmp_path: Path, n: int = 4) -> dict[str, str]:
    строки = [
        {
            "photo_id": f"p{i}",
            "listing_id": f"l{i}",
            "source": "commons",
            "crop_path": f"cls{i % 2}/p{i}.jpg",
            "brand": "Lada",
            "model_id": f"cls{i % 2}",
            "class_idx": i % 2,
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
    write_split(pd.DataFrame(строки, columns=list(COLUMNS)), tmp_path / "pool.parquet")
    sha: dict[str, str] = {}
    for i in range(n):
        кроп = tmp_path / "crops" / f"cls{i % 2}" / f"p{i}.jpg"
        кроп.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (i, 0, 0)).save(кроп)
        sha[f"p{i}"] = file_sha256(кроп)
    return sha


def _выгрузка(путь: Path, строки: list[RegistryRow]) -> Path:
    write_registry(путь, строки)
    return путь


def _аргументы(tmp_path: Path, *выгрузки: Path) -> list[str]:
    return [
        "--exports",
        *map(str, выгрузки),
        "--split",
        str(tmp_path / "pool.parquet"),
        "--crops-root",
        str(tmp_path / "crops"),
        "--registry",
        str(tmp_path / "registry.csv"),
        "--audited-out",
        str(tmp_path / "pool-audited.parquet"),
        "--witness",
        str(tmp_path / "witness.json"),
        "--registry-copy",
        str(tmp_path / "registry-copy.csv"),
    ]


def test_полный_аудит_собирает_pool_audited_и_свидетеля(tmp_path: Path) -> None:
    sha = _окружение(tmp_path)
    первая = _выгрузка(
        tmp_path / "e1.csv",
        [
            RegistryRow("p0", 1, "clean_filled", "clean", sha["p0"]),
            RegistryRow("p1", 1, "leak", "clean", sha["p1"]),
        ],
    )
    вторая = _выгрузка(
        tmp_path / "e2.csv",
        [
            RegistryRow("p2", 2, "clean_no_plate", "leak", sha["p2"]),
            RegistryRow("p3", 2, "clean_partial", "clean", sha["p3"]),
        ],
    )
    assert main(_аргументы(tmp_path, первая, вторая)) == 0

    audited = read_split(tmp_path / "pool-audited.parquet")
    assert sorted(audited["photo_id"]) == ["p0", "p3"]
    свидетель = json.loads((tmp_path / "witness.json").read_text(encoding="utf-8"))
    assert свидетель["full_pool_audited"] is True
    assert свидетель["training_authorized"] is False
    assert свидетель["pool"]["frames"] == 4
    assert свидетель["pool"]["sha256"] == file_sha256(tmp_path / "pool.parquet")
    assert свидетель["pool_audited"]["frames"] == 2
    assert свидетель["pool_audited"]["sha256"] == file_sha256(tmp_path / "pool-audited.parquet")
    assert свидетель["registry_sha256"] == file_sha256(tmp_path / "registry.csv")
    assert свидетель["excluded"]["total"] == 2
    assert свидетель["excluded"]["plate_leak"] == 1
    assert свидетель["excluded"]["face_leak"] == 1
    assert свидетель["excluded"]["by_class"] == {
        "cls0": {"excluded": 1, "pool": 2},
        "cls1": {"excluded": 1, "pool": 2},
    }
    копия = (tmp_path / "registry-copy.csv").read_bytes()
    assert копия == (tmp_path / "registry.csv").read_bytes()


def test_pending_не_даёт_собрать_pool_audited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sha = _окружение(tmp_path, n=2)
    выгрузка = _выгрузка(
        tmp_path / "e.csv",
        [
            RegistryRow("p0", 1, "clean_filled", "clean", sha["p0"]),
            RegistryRow("p1", 1, PENDING, "clean", sha["p1"]),
        ],
    )
    assert main(_аргументы(tmp_path, выгрузка)) == 1
    assert not (tmp_path / "pool-audited.parquet").exists()
    assert not (tmp_path / "witness.json").exists()
    assert (tmp_path / "registry.csv").exists()
    assert "pending: 1" in capsys.readouterr().out


def test_неполное_покрытие_пула_не_проходит(tmp_path: Path) -> None:
    sha = _окружение(tmp_path, n=2)
    выгрузка = _выгрузка(
        tmp_path / "e.csv",
        [
            RegistryRow("p0", 1, "clean_filled", "clean", sha["p0"]),
        ],
    )
    assert main(_аргументы(tmp_path, выгрузка)) == 1


def test_кроп_изменился_после_вердикта(tmp_path: Path) -> None:
    sha = _окружение(tmp_path, n=2)
    выгрузка = _выгрузка(
        tmp_path / "e.csv",
        [
            RegistryRow("p0", 1, "clean_filled", "clean", sha["p0"]),
            RegistryRow("p1", 1, "clean_filled", "clean", sha["p1"]),
        ],
    )
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "crops" / "cls1" / "p1.jpg")
    assert main(_аргументы(tmp_path, выгрузка)) == 1


def test_кадр_в_двух_выгрузках_ошибка(tmp_path: Path) -> None:
    sha = _окружение(tmp_path, n=1)
    строка = RegistryRow("p0", 1, "clean_filled", "clean", sha["p0"])
    первая = _выгрузка(tmp_path / "e1.csv", [строка])
    вторая = _выгрузка(tmp_path / "e2.csv", [строка])
    with pytest.raises(PoolAuditError, match="дважды"):
        main(_аргументы(tmp_path, первая, вторая))


def test_нельзя_перезаписать_исходный_пул(tmp_path: Path) -> None:
    sha = _окружение(tmp_path, n=1)
    выгрузка = _выгрузка(
        tmp_path / "e.csv",
        [
            RegistryRow("p0", 1, "clean_filled", "clean", sha["p0"]),
        ],
    )
    аргументы = _аргументы(tmp_path, выгрузка)
    аргументы[аргументы.index("--audited-out") + 1] = str(tmp_path / "pool.parquet")
    with pytest.raises(ValueError, match="исходный пул"):
        main(аргументы)
