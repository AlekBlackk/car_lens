"""MAD-Cars: таблица соответствия каталогу и выборка кадров обхода."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.catalog import Catalog, load_catalog
from pipeline.ingest.external import ExternalIngestError
from pipeline.ingest.mad_cars import (
    SAMPLE_COLUMNS,
    FetchReport,
    fetch_frames,
    frame_path,
    read_excluded,
    read_mad_mapping,
    sample_frames,
)

YAML = """
version: 1
brands:
  Audi:
    models:
      a4: {display: Audi A4}
      a6: {display: Audi A6}
"""


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    путь = tmp_path / "canonical.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return load_catalog(путь)


def таблица(tmp_path: Path, строки: str) -> Path:
    путь = tmp_path / "mad_cars.csv"
    путь.write_text("mad_brand,mad_model,model_id\n" + строки, encoding="utf-8")
    return путь


def манифест(*машины: tuple[str, str, str, int]) -> pd.DataFrame:
    """(brand, model, car_id, число кадров) → строки `mad.csv`."""
    строки = [
        {"brand": b, "model": m, "car_id": c, "view_id": str(v), "url": f"u/{c}/{v}.jpg"}
        for b, m, c, n in машины
        for v in range(n)
    ]
    return pd.DataFrame(строки)


def test_таблица_читается_в_пары(tmp_path: Path, каталог: Catalog) -> None:
    путь = таблица(tmp_path, "audi,a4,audi__a4\naudi,a4_allroad,audi__a4\n")
    assert read_mad_mapping(путь, каталог) == {
        ("audi", "a4"): "audi__a4",
        ("audi", "a4_allroad"): "audi__a4",
    }


def test_чужой_model_id_роняет_загрузку(tmp_path: Path, каталог: Catalog) -> None:
    with pytest.raises(ExternalIngestError, match="audi__a8"):
        read_mad_mapping(таблица(tmp_path, "audi,a8,audi__a8\n"), каталог)


def test_повтор_пары_роняет_загрузку(tmp_path: Path, каталог: Catalog) -> None:
    путь = таблица(tmp_path, "audi,a4,audi__a4\naudi,a4,audi__a6\n")
    with pytest.raises(ExternalIngestError, match="audi,a4"):
        read_mad_mapping(путь, каталог)


def test_три_кадра_с_шагом_треть_обхода() -> None:
    выборка = sample_frames(
        манифест(("audi", "a4", "7", 90)),
        {("audi", "a4"): "audi__a4"},
        cars_per_class=10,
        seed=1,
    )
    assert list(выборка.columns) == list(SAMPLE_COLUMNS)
    assert list(выборка["view_id"]) == ["0", "30", "60"]


def test_короткий_обход_без_дублей() -> None:
    выборка = sample_frames(
        манифест(("audi", "a4", "7", 2), ("audi", "a4", "8", 1)),
        {("audi", "a4"): "audi__a4"},
        cars_per_class=10,
        seed=1,
    )
    assert sorted(zip(выборка["car_id"], выборка["view_id"], strict=True)) == [
        ("7", "0"),
        ("7", "1"),
        ("8", "0"),
    ]


def test_порядок_view_id_числовой_а_не_строковый() -> None:
    # "10" < "9" строкой: при строковой сортировке шаг N/3 прыгал бы по кругу обхода.
    выборка = sample_frames(
        манифест(("audi", "a4", "7", 12)),
        {("audi", "a4"): "audi__a4"},
        cars_per_class=10,
        seed=1,
    )
    assert list(выборка["view_id"]) == ["0", "4", "8"]


def test_потолок_машин_и_детерминизм() -> None:
    машины = [("audi", "a4", str(i), 3) for i in range(20)]
    таблица_ = {("audi", "a4"): "audi__a4"}
    первая = sample_frames(манифест(*машины), таблица_, cars_per_class=5, seed=3)
    вторая = sample_frames(манифест(*reversed(машины)), таблица_, cars_per_class=5, seed=3)
    assert первая["car_id"].nunique() == 5
    assert set(первая["car_id"]) == set(вторая["car_id"])


def test_пары_без_соответствия_не_берутся() -> None:
    выборка = sample_frames(
        манифест(("audi", "a4", "7", 3), ("audi", "q7", "8", 3)),
        {("audi", "a4"): "audi__a4"},
        cars_per_class=10,
        seed=1,
    )
    assert set(выборка["model_id"]) == {"audi__a4"}


def выборка_из(*строки: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(m, c, v, f"https://x/{c}/{v}.jpg") for m, c, v in строки], columns=list(SAMPLE_COLUMNS)
    )


def test_выброшенные_машины_читаются_множеством(tmp_path: Path) -> None:
    путь = tmp_path / "excluded.csv"
    путь.write_text("model_id,car_id,reason\naudi__a4,7,повёрнут\n", encoding="utf-8")
    assert read_excluded(путь) == frozenset({"7"})


def test_скачивание_пропускает_лежащие_и_удаляет_выброшенные(tmp_path: Path) -> None:
    лежит = frame_path(tmp_path, "audi__a4", "1", "0")
    лежит.parent.mkdir(parents=True)
    лежит.write_bytes(b"jpg")
    выброшенный = frame_path(tmp_path, "audi__a4", "9", "0")
    выброшенный.write_bytes(b"jpg")
    скачанные: list[str] = []

    def скачать(url: str, куда: Path) -> None:
        скачанные.append(url)
        куда.write_bytes(b"jpg")

    отчёт = fetch_frames(
        выборка_из(("audi__a4", "1", "0"), ("audi__a4", "2", "5"), ("audi__a4", "9", "0")),
        tmp_path,
        frozenset({"9"}),
        скачать,
    )
    assert отчёт == FetchReport(downloaded=1, present=1, removed=1, failed=[])
    assert скачанные == ["https://x/2/5.jpg"]
    assert not выброшенный.exists()


def test_сбой_скачивания_не_оставляет_пустой_файл(tmp_path: Path) -> None:
    def сломанное(url: str, куда: Path) -> None:
        куда.write_bytes(b"")
        raise OSError("503")

    отчёт = fetch_frames(выборка_из(("audi__a4", "1", "0")), tmp_path, frozenset(), сломанное)
    assert отчёт.failed == ["https://x/1/0.jpg"]
    assert not frame_path(tmp_path, "audi__a4", "1", "0").exists()


def test_лист_машин_три_кадра_в_строку(tmp_path: Path) -> None:
    from PIL import Image

    from scripts.mad_cars import CELL, render_car_sheet

    машины = []
    for n in range(3):
        кадры = []
        for v in range(3):
            путь = tmp_path / f"{n}_{v}.jpg"
            Image.new("RGB", (64, 36), "white").save(путь)
            кадры.append(путь)
        машины.append((str(100 + n), кадры))
    лист = render_car_sheet(машины, first_number=0)
    # Две колонки машин по 3 кадра; 3 машины → 2 ряда.
    assert лист.size == (CELL[0] * 6, CELL[1] * 2)
