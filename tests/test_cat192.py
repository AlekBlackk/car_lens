"""Сплиты каталога 192: перенумерация, пересборка пула и проверки без утечки."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.cat192 import (
    Cat192Error,
    build_external_pool,
    cap_per_class,
    leaking_cars,
    pick_one_frame_per_car,
    remap_class_idx,
    split_groups,
    without_close_frames,
    without_close_twins,
)
from pipeline.catalog import Catalog, load_catalog
from pipeline.schema import COLUMNS, read_split, write_split
from scripts.build_cat192_splits import main as build_main
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


def каталог_путь(tmp_path: Path) -> Path:
    путь = tmp_path / "canonical.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return путь


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    return load_catalog(каталог_путь(tmp_path))


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


def с_классом(frame: pd.DataFrame) -> pd.DataFrame:
    """Кадры в формате сплита: `class_idx` по каталогу из трёх классов, у негативов — пусто."""
    результат = frame.copy()
    результат["class_idx"] = pd.array(
        [None if модель is None else НОВЫЙ[модель] for модель in frame["model_id"]], dtype="Int32"
    )
    return результат


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


def test_отложенные_группы_четверть_и_только_у_классов_от_четырёх_групп() -> None:
    группы = {f"A4_{n}": "audi__a4" for n in range(8)} | {f"A6_{n}": "audi__a6" for n in range(3)}
    отложены = split_groups(группы, salt="s")
    assert len(отложены) == 2
    assert all(папка.startswith("A4_") for папка in отложены)
    assert split_groups(dict(reversed(группы.items())), salt="s") == отложены


def test_соль_меняет_отложенные_группы() -> None:
    группы = {f"A4_{n}": "audi__a4" for n in range(40)}
    assert split_groups(группы, salt="s1") != split_groups(группы, salt="s2")


def test_кадры_близкие_к_эталону_выбрасываются() -> None:
    кадры = кадр(
        строка("e:1", "audi__a4", далёкий(0)),
        строка("e:2", "audi__a4", далёкий(1)),
    )
    итог = without_close_frames(кадры, [далёкий(0) ^ 1, далёкий(5)])
    assert list(итог["photo_id"]) == ["e:2"]


def test_потолок_на_класс_детерминирован() -> None:
    кадры = кадр(*(строка(f"e:{n}", "audi__a4", далёкий(n)) for n in range(5)))
    первый = cap_per_class(кадры, 2, seed=3)
    второй = cap_per_class(кадры.iloc[::-1], 2, seed=3)
    assert len(первый) == 2
    assert set(первый["photo_id"]) == set(второй["photo_id"])


def test_новый_источник_пересобирает_только_классы_с_новыми_кадрами(каталог: Catalog) -> None:
    старые = кадр(
        строка("mad_cars:a3", "audi__a3", далёкий(0)),
        строка("mad_cars:a4", "audi__a4", далёкий(1)),
    )
    новые = кадр(строка("autoevolution:a3", "audi__a3", далёкий(2)))
    итог = build_external_pool(
        старые,
        новые,
        каталог,
        max_per_class=150,
        own_caps={"mad_cars": 150},
        seed=1,
        source="autoevolution",
    )
    assert set(итог.external["photo_id"]) == {"mad_cars:a3", "mad_cars:a4", "autoevolution:a3"}
    assert итог.changed_classes == {0}


def test_studio_без_eval_ext_оставляет_студийную_проверку_и_сторонится_её_кадров(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.build_cat192_splits.DEFAULT_CATALOG_PATH", каталог_путь(tmp_path))
    основа = tmp_path / "base"
    основа.mkdir()

    def хеш(n: int) -> int:
        # Блоки по 4 бита: расстояние между разными — 8 (> 6), а значения умещаются в int64.
        return 0xF << (4 * n)

    for имя, данные in {
        "pool-ext": кадр(
            строка("mad_cars:a3", "audi__a3", хеш(0)),
            строка("mad_cars:a4", "audi__a4", хеш(1)),
        ),
        "pool-clean": кадр(строка("street:a6", "audi__a6", хеш(2))),
        "eval-studio": кадр(строка("autoevolution:studio", "audi__a3", хеш(3))),
        "eval-new": кадр(строка("mad_cars:new", "audi__a4", хеш(4))),
        "eval-old": кадр(строка("mad_cars:old", "audi__a6", хеш(5))),
        "holdout_ru": кадр(строка("street:holdout", "audi__a6", хеш(6))),
        "negatives": кадр(строка("stanford:n1", None, хеш(7))),
        "negatives-street": кадр(строка("street:n2", None, хеш(8))),
    }.items():
        write_split(с_классом(данные), основа / f"{имя}.parquet")
    принятые = tmp_path / "manual.parquet"
    write_split(
        с_классом(
            кадр(
                строка("manual:fresh", "audi__a3", хеш(9)),
                строка("manual:like-studio", "audi__a3", хеш(3)),
            )
        ),
        принятые,
    )

    итог = tmp_path / "out"

    build_main(
        [
            "studio",
            "--base",
            str(основа),
            "--train-ext",
            str(принятые),
            "--source",
            "manual",
            "--out",
            str(итог),
        ]
    )

    пул = read_split(итог / "pool-ext.parquet")
    assert set(пул["photo_id"]) == {"mad_cars:a3", "mad_cars:a4", "manual:fresh"}
    assert list(read_split(итог / "eval-studio.parquet")["photo_id"]) == ["autoevolution:studio"]

    # Кадр, близкий к дополнительной проверке (`T_held`), в обучение не попадает.
    чужая_проверка = tmp_path / "held.parquet"
    write_split(с_классом(кадр(строка("manual:held", "audi__a3", хеш(9)))), чужая_проверка)
    без_близких = tmp_path / "out2"
    build_main(
        [
            "studio",
            "--base",
            str(основа),
            "--train-ext",
            str(принятые),
            "--source",
            "manual",
            "--also-avoid",
            str(чужая_проверка),
            "--out",
            str(без_близких),
        ]
    )
    assert "manual:fresh" not in set(read_split(без_близких / "pool-ext.parquet")["photo_id"])


def test_близнецы_одной_машины_остаются_одним_кадром_а_другой_класс_не_трогается() -> None:
    кадры = с_классом(
        кадр(
            строка("manual:a", "audi__a3", 0xF),
            строка("manual:b", "audi__a3", 0xFFF),  # 8 бит от первого
            строка("manual:c", "audi__a3", 0xFFFF << 40),  # далеко от обоих
            строка("manual:d", "audi__a4", 0xF),  # тот же хеш, но другой класс
        )
    )
    итог = without_close_twins(кадры, max_distance=10, seed=1)
    оставлено = set(итог["photo_id"])
    assert {"manual:c", "manual:d"} <= оставлено
    assert len(оставлено & {"manual:a", "manual:b"}) == 1
    повтор = without_close_twins(кадры.iloc[::-1].reset_index(drop=True), max_distance=10, seed=1)
    assert set(повтор["photo_id"]) == оставлено


def test_heldout_убирает_кадры_смоука_и_близнецов(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.build_cat192_splits.DEFAULT_CATALOG_PATH", каталог_путь(tmp_path))
    проверка, смоук, итог = (
        tmp_path / "eval.parquet",
        tmp_path / "f9.parquet",
        tmp_path / "held.parquet",
    )
    write_split(
        с_классом(
            кадр(
                строка("manual:smoke-twin", "audi__a3", 0xFFFF << 40),
                строка("manual:one", "audi__a3", 0xF),
                строка("manual:twin", "audi__a3", 0xFFF),
                строка("manual:other", "audi__a4", 0xF << 20),
            )
        ),
        проверка,
    )
    write_split(с_классом(кадр(строка("manual:smoke", "audi__a3", 0xFFFF << 40))), смоук)

    build_main(["heldout", "--eval-ext", str(проверка), "--fresh9", str(смоук), "--out", str(итог)])

    оставлено = set(read_split(итог)["photo_id"])
    assert "manual:smoke-twin" not in оставлено
    assert "manual:other" in оставлено
    assert len(оставлено & {"manual:one", "manual:twin"}) == 1


def test_groups_делит_только_группы_новых_классов(tmp_path: Path) -> None:
    таблица = tmp_path / "ae.csv"
    строки = ["dir_path,model_id,brand,model,years,images"]
    строки += [f"A3_{n},audi__a3,Audi,a3,,10" for n in range(8)]
    строки += [f"A4_{n},audi__a4,Audi,a4,,10" for n in range(8)]
    таблица.write_text("\n".join(строки) + "\n", encoding="utf-8")
    новые = tmp_path / "new.txt"
    новые.write_text("# новые\naudi__a3\n", encoding="utf-8")
    обучение, проверка = tmp_path / "train.csv", tmp_path / "eval.csv"

    build_main(
        [
            "groups",
            "--mapping",
            str(таблица),
            "--new-classes",
            str(новые),
            "--train-map",
            str(обучение),
            "--eval-map",
            str(проверка),
        ]
    )

    в_обучении = pd.read_csv(обучение)["dir_path"]
    в_проверке = pd.read_csv(проверка)["dir_path"]
    assert len(в_проверке) == 2
    assert set(в_обучении).isdisjoint(в_проверке)
    assert len(в_обучении) + len(в_проверке) == 8
    assert all(папка.startswith("A3_") for папка in [*в_обучении, *в_проверке])
