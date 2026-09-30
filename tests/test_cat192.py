"""Сплиты каталога 192: перенумерация, пересборка пула и проверки без утечки."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.cat192 import (
    Cat192Error,
    build_external_pool,
    leaking_cars,
    pick_one_frame_per_car,
    remap_class_idx,
)
from pipeline.catalog import Catalog, load_catalog
from pipeline.schema import COLUMNS
from scripts.build_cat192_splits import own_car_reason

YAML = """
version: 1
brands:
  Audi:
    models:
      a3: {display: Audi A3}
      a4: {display: Audi A4}
      a6: {display: Audi A6}
"""

# Индексы в каталоге из трёх классов: a3 = 0, a4 = 1, a6 = 2. В «старом» каталоге из двух классов
# a4 был 0, a6 — 1: новый класс a3 сдвигает остальные.
СТАРЫЙ = {"audi__a4": 0, "audi__a6": 1}
НОВЫЙ = {"audi__a3": 0, "audi__a4": 1, "audi__a6": 2}


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    путь = tmp_path / "canonical.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return load_catalog(путь)


def строка(photo_id: str, model_id: str | None, phash: int, *, listing: str = "") -> dict[str, Any]:
    источник = photo_id.split(":")[0]
    return {
        "photo_id": photo_id,
        "listing_id": listing or photo_id,
        "source": источник,
        "crop_path": f"external/{источник}/{photo_id}.jpg",
        "brand": None if model_id is None else "Audi",
        "model_id": model_id,
        "class_idx": None if model_id is None else СТАРЫЙ.get(model_id, -5),
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


def далёкий(n: int) -> int:
    return 0xFF << (8 * n)


def test_индекс_пересчитывается_по_model_id(каталог: Catalog) -> None:
    итог = remap_class_idx(
        кадр(строка("m:1", "audi__a4", далёкий(0)), строка("m:2", "audi__a6", далёкий(1))),
        каталог,
    )
    assert list(итог["class_idx"]) == [1, 2]


def test_негатив_без_model_id_остаётся_без_класса(каталог: Catalog) -> None:
    итог = remap_class_idx(кадр(строка("n:1", None, далёкий(0))), каталог)
    assert итог["class_idx"].isna().all()


def test_model_id_вне_каталога_роняет_пересчёт(каталог: Catalog) -> None:
    with pytest.raises(Cat192Error, match="audi__a8"):
        remap_class_idx(кадр(строка("m:1", "audi__a8", далёкий(0))), каталог)


def test_пул_пересобирает_только_классы_с_новыми_кадрами(каталог: Catalog) -> None:
    старые = кадр(
        строка("mad_cars:a4-old", "audi__a4", далёкий(0)),
        строка("mad_cars:a6", "audi__a6", далёкий(1)),
        строка("manual:a4", "audi__a4", далёкий(2)),
    )
    новые = кадр(
        строка("mad_cars:a3", "audi__a3", далёкий(3)),
        строка("mad_cars:a4-new", "audi__a4", далёкий(4)),
    )
    итог = build_external_pool(
        старые, новые, каталог, max_per_class=150, own_caps={"mad_cars": 150}, seed=1
    )
    assert set(итог.external["photo_id"]) == {
        "mad_cars:a3",
        "mad_cars:a4-new",
        "mad_cars:a6",
        "manual:a4",
    }
    assert итог.changed_classes == {0, 1}
    assert list(итог.external.loc[итог.external["model_id"] == "audi__a3", "class_idx"]) == [0]


def test_склейка_не_должна_менять_прочие_классы(каталог: Catalog) -> None:
    # Два кадра a6 уже близки: прошлая склейка такого не оставила бы, и сборка падает.
    старые = кадр(
        строка("mad_cars:a6-1", "audi__a6", далёкий(1)),
        строка("mad_cars:a6-2", "audi__a6", далёкий(1) ^ 1),
    )
    новые = кадр(строка("mad_cars:a3", "audi__a3", далёкий(3)))
    with pytest.raises(Cat192Error, match="вне пересобираемых"):
        build_external_pool(
            старые, новые, каталог, max_per_class=150, own_caps={"mad_cars": 150}, seed=1
        )


def test_один_кадр_на_машину_детерминированно() -> None:
    кадры = кадр(
        строка("mad_cars:x1", "audi__a4", далёкий(0), listing="mad_cars:audi__a4:7"),
        строка("mad_cars:x2", "audi__a4", далёкий(1), listing="mad_cars:audi__a4:7"),
        строка("mad_cars:y1", "audi__a4", далёкий(2), listing="mad_cars:audi__a4:8"),
    )
    первый = pick_one_frame_per_car(кадры, seed=5)
    второй = pick_one_frame_per_car(кадры.iloc[::-1], seed=5)
    assert len(первый) == 2
    assert set(первый["photo_id"]) == set(второй["photo_id"])


def test_утечка_выбрасывает_машину_целиком() -> None:
    проверка = кадр(
        строка("e:1", "audi__a4", далёкий(0), listing="e:car1"),
        строка("e:2", "audi__a4", далёкий(1), listing="e:car1"),
        строка("e:3", "audi__a4", далёкий(2), listing="e:car2"),
    )
    обучение = [далёкий(0) ^ 1, далёкий(5)]
    assert leaking_cars(проверка, обучение) == {"e:car1"}


def test_свой_негатив_по_игле_и_по_папке_mad() -> None:
    иглы = {"audi__a3": ("a3",)}
    точные = {("audi", "a3")}
    без_спорных: frozenset[tuple[str, str]] = frozenset()
    assert own_car_reason(
        "AUDI_A3_Sedan_2020Present",
        mad_folder=False,
        exact=точные,
        disputed=без_спорных,
        needles=иглы,
    )
    assert (
        own_car_reason(
            "audi__a3", mad_folder=True, exact=точные, disputed=без_спорных, needles=иглы
        )
        == "exact"
    )
    assert (
        own_car_reason(
            "audi__a8", mad_folder=True, exact=точные, disputed=без_спорных, needles=иглы
        )
        is None
    )
    assert (
        own_car_reason(
            "000000478314.jpg", mad_folder=False, exact=точные, disputed=без_спорных, needles=иглы
        )
        is None
    )
