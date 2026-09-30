"""MAD-Cars: таблица соответствия каталогу и выборка кадров обхода."""

from __future__ import annotations

from collections.abc import Iterator
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
    read_car_ids,
    read_excluded,
    read_mad_disputed,
    read_mad_mapping,
    sample_eval_frames,
    sample_frames,
    sample_negative_frames,
    sample_reject_pool_frames,
    select_off_catalog_pairs,
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


def test_занятые_машины_в_выборку_не_попадают() -> None:
    машины = [("audi", "a4", str(i), 3) for i in range(10)]
    выборка = sample_frames(
        манифест(*машины),
        {("audi", "a4"): "audi__a4"},
        cars_per_class=10,
        seed=1,
        exclude_cars={"0", "1", "2"},
    )
    assert set(выборка["car_id"]) == {str(i) for i in range(3, 10)}


ТАБЛИЦА_ПРОВЕРКИ = {("audi", "a4"): "audi__a4", ("audi", "a4_allroad"): "audi__a4"}


def test_проверка_по_названию_берёт_машины_каждого_названия() -> None:
    машины = [("audi", "a4", str(i), 8) for i in range(10)] + [
        ("audi", "a4_allroad", str(100 + i), 8) for i in range(10)
    ]
    выборка = sample_eval_frames(
        манифест(*машины), ТАБЛИЦА_ПРОВЕРКИ, cars_per_group=3, by_name=True, seed=1
    )
    по_названиям = выборка.groupby("mad_name")["car_id"].nunique()
    assert dict(по_названиям) == {"audi/a4": 3, "audi/a4_allroad": 3}
    assert set(выборка["model_id"]) == {"audi__a4"}


def test_проверка_по_классу_делит_потолок_между_названиями() -> None:
    машины = [("audi", "a4", str(i), 8) for i in range(10)] + [
        ("audi", "a4_allroad", str(100 + i), 8) for i in range(10)
    ]
    выборка = sample_eval_frames(
        манифест(*машины), ТАБЛИЦА_ПРОВЕРКИ, cars_per_group=3, by_name=False, seed=1
    )
    assert выборка["car_id"].nunique() == 3


def test_проверка_три_кадра_обхода_и_колонки() -> None:
    выборка = sample_eval_frames(
        манифест(("audi", "a4", "7", 8)),
        {("audi", "a4"): "audi__a4"},
        cars_per_group=5,
        by_name=True,
        seed=1,
    )
    assert list(выборка.columns) == [*SAMPLE_COLUMNS, "mad_name"]
    assert list(выборка["view_id"]) == ["2", "4", "6"]


def test_проверка_обходит_занятые_машины_и_не_зависит_от_порядка() -> None:
    машины = [("audi", "a4", str(i), 8) for i in range(20)]
    таблица_ = {("audi", "a4"): "audi__a4"}
    первая = sample_eval_frames(
        манифест(*машины), таблица_, cars_per_group=5, by_name=True, seed=3, exclude_cars={"0"}
    )
    вторая = sample_eval_frames(
        манифест(*reversed(машины)),
        таблица_,
        cars_per_group=5,
        by_name=True,
        seed=3,
        exclude_cars={"0"},
    )
    assert "0" not in set(первая["car_id"])
    assert set(первая["car_id"]) == set(вторая["car_id"])


def test_car_id_выборок_объединяются(tmp_path: Path) -> None:
    первая, вторая = tmp_path / "a.csv", tmp_path / "b.csv"
    строки = ["model_id,car_id,view_id,url"]
    первая.write_text("\n".join([*строки, "x,1,0,u", "x,2,0,u"]), encoding="utf-8")
    вторая.write_text("\n".join([*строки, "y,2,0,u", "y,3,0,u"]), encoding="utf-8")
    assert read_car_ids(первая, вторая) == {"1", "2", "3"}


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


def test_скачивание_повторяет_попытку_при_обрыве_тела(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import http.client

    from scripts import mad_cars

    class ОтветОбрыв:
        def __enter__(self) -> ОтветОбрыв:
            return self

        def __exit__(self, *исключение: object) -> None:
            return None

        def read(self) -> bytes:
            raise http.client.IncompleteRead(b"")

    class ОтветЦелый:
        def __enter__(self) -> ОтветЦелый:
            return self

        def __exit__(self, *исключение: object) -> None:
            return None

        def read(self) -> bytes:
            return b"jpg"

    ответы: Iterator[ОтветОбрыв | ОтветЦелый] = iter([ОтветОбрыв(), ОтветЦелый()])

    def подставной_urlopen(url: str, timeout: int) -> ОтветОбрыв | ОтветЦелый:
        return next(ответы)

    monkeypatch.setattr("urllib.request.urlopen", подставной_urlopen)
    путь = tmp_path / "out.jpg"
    mad_cars.download_url("https://x/1/0.jpg", путь)
    assert путь.read_bytes() == b"jpg"


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


def спорные(tmp_path: Path, строки: str) -> Path:
    путь = tmp_path / "disputed.csv"
    путь.write_text("mad_brand,mad_model,model_id,reason\n" + строки, encoding="utf-8")
    return путь


ИГЛЫ = {"audi__a4": ("a4",), "audi__a6": ("a6",)}


def test_спорные_пары_читаются(tmp_path: Path, каталог: Catalog) -> None:
    путь = спорные(tmp_path, "audi,s4,audi__a4,спортивная версия\n")
    assert read_mad_disputed(путь, каталог) == frozenset({("audi", "s4")})


def test_спорная_пара_без_причины_роняет_загрузку(tmp_path: Path, каталог: Catalog) -> None:
    with pytest.raises(ExternalIngestError, match="без причины"):
        read_mad_disputed(спорные(tmp_path, "audi,s4,audi__a4, \n"), каталог)


def test_чужой_класс_в_спорных_роняет_загрузку(tmp_path: Path, каталог: Catalog) -> None:
    with pytest.raises(ExternalIngestError, match="audi__a8"):
        read_mad_disputed(спорные(tmp_path, "audi,s8,audi__a8,версия\n"), каталог)


def test_чужими_остаются_пары_без_соответствия_и_иглы() -> None:
    пары = [("audi", "a4"), ("audi", "s4"), ("audi", "a6_allroad"), ("audi", "q7")]
    чужие, отброшены = select_off_catalog_pairs(
        пары, {("audi", "a4")}, frozenset({("audi", "s4")}), ИГЛЫ
    )
    assert чужие == [("audi", "q7")]
    assert отброшены == {
        ("audi", "a4"): "exact",
        ("audi", "s4"): "disputed",
        ("audi", "a6_allroad"): "needle:audi__a6",
    }


def test_спорной_пары_нет_в_mad_cars_отбор_падает() -> None:
    with pytest.raises(ExternalIngestError, match="audi/s4"):
        select_off_catalog_pairs([("audi", "q7")], set(), frozenset({("audi", "s4")}), ИГЛЫ)


def test_негативы_средний_кадр_обхода_и_потолок_на_модель() -> None:
    кадр = манифест(("audi", "q7", "1", 5), ("audi", "q7", "2", 4), ("audi", "q7", "3", 2))
    выборка = sample_negative_frames(кадр, [("audi", "q7")], cars_per_model=2, total=100, seed=1)
    assert list(выборка.columns) == list(SAMPLE_COLUMNS)
    assert len(выборка) == 2
    assert set(выборка["model_id"]) == {"audi__q7"}
    середины = {"1": "2", "2": "2", "3": "1"}
    for машина, ракурс in zip(выборка["car_id"], выборка["view_id"], strict=True):
        assert ракурс == середины[машина]


def test_негативы_общий_потолок_и_независимость_от_порядка() -> None:
    кадр = манифест(*[("audi", f"m{i}", str(i), 3) for i in range(30)])
    пары = [("audi", f"m{i}") for i in range(30)]
    прямой = sample_negative_frames(кадр, пары, cars_per_model=10, total=7, seed=5)
    обратный = sample_negative_frames(
        кадр.iloc[::-1].reset_index(drop=True),
        list(reversed(пары)),
        cars_per_model=10,
        total=7,
        seed=5,
    )
    assert len(прямой) == 7
    pd.testing.assert_frame_equal(прямой, обратный)


def test_негативы_берут_только_чужие_пары() -> None:
    кадр = манифест(("audi", "a4", "1", 3), ("audi", "q7", "2", 3))
    выборка = sample_negative_frames(кадр, [("audi", "q7")], cars_per_model=10, total=10, seed=1)
    assert list(выборка["car_id"]) == ["2"]


def test_подкоманда_negatives_sample_пишет_выборку(
    tmp_path: Path, каталог: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.mad_cars as cli

    monkeypatch.setattr(cli, "load_catalog", lambda _путь: каталог)
    monkeypatch.setattr(cli, "load_alias_table", lambda _путь: ИГЛЫ)
    манифест(("audi", "a4", "1", 3), ("audi", "q7", "2", 3)).to_csv(
        tmp_path / "mad.csv", index=False
    )
    выход = tmp_path / "neg.csv"

    код = cli.main(
        [
            "negatives-sample",
            "--manifest",
            str(tmp_path / "mad.csv"),
            "--mapping",
            str(таблица(tmp_path, "audi,a4,audi__a4\n")),
            "--disputed",
            str(спорные(tmp_path, "")),
            "--seed",
            "1",
            "--out",
            str(выход),
        ]
    )

    assert код == 0
    выборка = pd.read_csv(выход, dtype=str)
    assert list(выборка["model_id"]) == ["audi__q7"]


def test_подкоманда_negatives_sample_пропускает_машины_без_марки(
    tmp_path: Path, каталог: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # В настоящем mad.csv 8 машин с пустыми brand и model: чужой такую не назовёшь.
    import scripts.mad_cars as cli

    monkeypatch.setattr(cli, "load_catalog", lambda _путь: каталог)
    monkeypatch.setattr(cli, "load_alias_table", lambda _путь: ИГЛЫ)
    манифест(("", "", "1", 3), ("audi", "q7", "2", 3)).to_csv(tmp_path / "mad.csv", index=False)
    выход = tmp_path / "neg.csv"

    код = cli.main(
        [
            "negatives-sample",
            "--manifest",
            str(tmp_path / "mad.csv"),
            "--mapping",
            str(таблица(tmp_path, "audi,a4,audi__a4\n")),
            "--disputed",
            str(спорные(tmp_path, "")),
            "--seed",
            "1",
            "--out",
            str(выход),
        ]
    )

    assert код == 0
    assert list(pd.read_csv(выход, dtype=str)["car_id"]) == ["2"]


def test_пул_не_берёт_модели_замера() -> None:
    кадр = манифест(("audi", "q7", "1", 4), ("audi", "q5", "2", 4))
    пул = sample_reject_pool_frames(
        кадр, [("audi", "q7"), ("audi", "q5")], {"audi__q7"}, cars_per_model=10, seed=1
    )
    assert set(пул["model_id"]) == {"audi__q5"}


def test_пул_три_кадра_на_четвертях_обхода() -> None:
    кадр = манифест(("audi", "q7", "1", 8), ("audi", "q7", "2", 2), ("audi", "q7", "3", 1))
    пул = sample_reject_pool_frames(кадр, [("audi", "q7")], set(), cars_per_model=10, seed=1)
    ракурсы = {м: sorted(г["view_id"].astype(int)) for м, г in пул.groupby("car_id")}
    assert ракурсы == {"1": [2, 4, 6], "2": [0, 1], "3": [0]}
    assert list(пул.columns) == list(SAMPLE_COLUMNS)


def test_пул_потолок_машин_на_модель() -> None:
    кадр = манифест(*[("audi", "q7", str(i), 4) for i in range(15)])
    пул = sample_reject_pool_frames(кадр, [("audi", "q7")], set(), cars_per_model=10, seed=3)
    assert пул["car_id"].nunique() == 10
    assert len(пул) == 30


def test_пул_не_зависит_от_порядка_строк() -> None:
    кадр = манифест(*[("audi", f"m{i}", str(i), 5) for i in range(12)])
    пары = [("audi", f"m{i}") for i in range(12)]
    прямой = sample_reject_pool_frames(кадр, пары, {"audi__m0"}, cars_per_model=10, seed=5)
    обратный = sample_reject_pool_frames(
        кадр.iloc[::-1].reset_index(drop=True),
        list(reversed(пары)),
        {"audi__m0"},
        cars_per_model=10,
        seed=5,
    )
    pd.testing.assert_frame_equal(прямой, обратный)


def test_подкоманда_reject_pool_sample_исключает_замер(
    tmp_path: Path, каталог: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.mad_cars as cli

    monkeypatch.setattr(cli, "load_catalog", lambda _путь: каталог)
    monkeypatch.setattr(cli, "load_alias_table", lambda _путь: ИГЛЫ)
    манифест(("audi", "a4", "1", 3), ("audi", "q7", "2", 3), ("audi", "q5", "3", 3)).to_csv(
        tmp_path / "mad.csv", index=False
    )
    замер = tmp_path / "measured.csv"
    pd.DataFrame([("audi__q7", "2", "1", "u")], columns=list(SAMPLE_COLUMNS)).to_csv(
        замер, index=False
    )
    выход = tmp_path / "pool.csv"

    код = cli.main(
        [
            "reject-pool-sample",
            "--manifest",
            str(tmp_path / "mad.csv"),
            "--mapping",
            str(таблица(tmp_path, "audi,a4,audi__a4\n")),
            "--disputed",
            str(спорные(tmp_path, "")),
            "--measured",
            str(замер),
            "--seed",
            "1",
            "--out",
            str(выход),
        ]
    )

    assert код == 0
    пул = pd.read_csv(выход, dtype=str)
    assert set(пул["model_id"]) == {"audi__q5"}
    assert list(пул["view_id"]) == ["0", "1", "2"]
