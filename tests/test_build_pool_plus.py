"""Склейка внешних источников (фильтры 7–8) и отчёт `bench/external-ingest.md`."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, Catalog, load_catalog
from pipeline.ingest.external import (
    GAP_CLASSES_PATH,
    ExternalIngestError,
    merge_external,
    read_gap_classes,
)
from pipeline.schema import COLUMNS
from scripts.build_pool_plus import render_report


def строка(photo_id: str, class_idx: int, phash: int) -> dict[str, Any]:
    источник = photo_id.split(":")[0]
    return {
        "photo_id": photo_id,
        "listing_id": photo_id,
        "source": источник,
        "crop_path": f"external/{источник}/{photo_id}.jpg",
        "brand": "Audi",
        "model_id": "audi__a5",
        "class_idx": class_idx,
        "body_style": None,
        "year": None,
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "bbox_area_frac": 0.5,
        "det_conf": 0.9,
        "phash": phash,
        "raw_model_string": None,
        "is_multi_car": False,
    }


def кадр(*строки: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(строки), columns=list(COLUMNS))


# Хеши, далёкие друг от друга: у каждого свой байт, остальные нули — расстояние 8+ бит.
def далёкий(n: int) -> int:
    return 0xFF << (8 * n)


def test_из_близкой_пары_остаётся_приоритетный_источник() -> None:
    итог, отказы = merge_external(
        {
            "autoevolution": кадр(строка("autoevolution:a", 0, далёкий(0))),
            "manual": кадр(строка("manual:m", 0, далёкий(0) ^ 1)),  # расстояние 1 бит
        },
        max_per_class=150,
        seed=1337,
    )
    assert list(итог["photo_id"]) == ["manual:m"]
    assert отказы == {"dup_cross": 1}


def test_потолок_заполняется_по_приоритету() -> None:
    авто = [строка(f"autoevolution:{i}", 0, далёкий(i)) for i in range(4)]
    ручные = [строка(f"manual:{i}", 0, далёкий(4 + i)) for i in range(2)]
    итог, отказы = merge_external(
        {"autoevolution": кадр(*авто), "manual": кадр(*ручные)}, max_per_class=3, seed=1337
    )
    assert sorted(итог["source"]) == ["autoevolution", "manual", "manual"]
    assert отказы == {"cap": 3}
    assert list(итог.columns) == list(COLUMNS)


def test_потолок_не_зависит_от_порядка_строк() -> None:
    строки = [строка(f"autoevolution:{i}", 0, далёкий(i)) for i in range(6)]
    перемешанные = строки.copy()
    random.Random(7).shuffle(перемешанные)
    первый, _ = merge_external({"autoevolution": кадр(*строки)}, max_per_class=3, seed=1)
    второй, _ = merge_external({"autoevolution": кадр(*перемешанные)}, max_per_class=3, seed=1)
    assert set(первый["photo_id"]) == set(второй["photo_id"])


def test_неизвестный_источник_отвергается() -> None:
    with pytest.raises(ExternalIngestError, match="flickr"):
        merge_external({"flickr": кадр()}, max_per_class=150, seed=1337)


def test_пустой_вход_даёт_пустой_кадр() -> None:
    итог, отказы = merge_external({"manual": кадр()}, max_per_class=150, seed=1337)
    assert итог.empty
    assert отказы == {}


@pytest.fixture(scope="module")
def каталог() -> Catalog:
    return load_catalog(DEFAULT_CATALOG_PATH)


def test_список_дыры_из_каталога(каталог: Catalog) -> None:
    дыра = read_gap_classes(GAP_CLASSES_PATH, каталог)
    assert len(дыра) == 66
    assert "lada__vesta" in дыра


def test_список_дыры_с_чужим_классом_отвергается(tmp_path: Path, каталог: Catalog) -> None:
    путь = tmp_path / "gap.txt"
    путь.write_text("# шапка\nlada__vesta\nlada__vesta_ng\n", encoding="utf-8")
    with pytest.raises(ExternalIngestError, match="lada__vesta_ng"):
        read_gap_classes(путь, каталог)


def test_отчёт_считает_по_классам_и_источникам(каталог: Catalog) -> None:
    пул = кадр(строка("commons:1", 0, 1), строка("commons:2", 0, 2))
    пул["source"] = "commons"
    внешние = кадр(строка("manual:1", 0, 3), строка("autoevolution:1", 0, 4))
    текст = render_report(
        каталог,
        пул,
        внешние,
        {"manual": {"seen": 5, "kept": 1, "rejected": {"no_vehicle": 4}}},
        {"cap": 2},
        frozenset({каталог.classes[0].class_id}),
    )
    первый = каталог.classes[0].class_id
    assert f"| `{первый}` | да | 2 | 1 | 0 | 1 | 4 |" in текст
    assert "| manual | 5 | 1 | no_vehicle 4 |" in текст
    assert "cap 2" in текст
