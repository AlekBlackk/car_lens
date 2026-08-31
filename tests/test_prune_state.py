from pathlib import Path

import pytest

from pipeline.harvest.manifest import HarvestRow, write_manifest
from pipeline.harvest.state import HarvestState, is_class_done, load_state, save_state
from scripts.prune_state import main, prune


def _строка(**правки: object) -> HarvestRow:
    поля: dict[str, object] = {
        "file": "ab.jpg",
        "class_id": "lada__granta",
        "source": "commons",
        "source_item_id": "12345",
        "page_url": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "file_url": "https://upload.wikimedia.org/X.jpg",
        "author": "Иван Петров",
        "license": "CC BY 4.0",
        "license_tier": 2,
        "checked": "2026-08-28",
        "basis": "category:Lada Granta",
        "sha256": "ab" * 32,
        "phash": -12345,
        "clip_rank": 4,
        "clip_margin": 0.12,
        "plates": 1,
        "faces": 0,
        "bbox_x": 12.0,
        "bbox_y": 34.0,
        "bbox_w": 500.0,
        "bbox_h": 300.0,
        "bbox_area_frac": 0.42,
        "det_conf": 0.91,
        "audit": "",
    }
    поля.update(правки)
    return HarvestRow(**поля)  # type: ignore[arg-type]


def _записать_состояние(путь: Path) -> None:
    """Состояние с мусором во всех прополаемых полях и с непустым `accepted`."""
    состояние = HarvestState(
        seen={"kia__rio": 120},
        accepted={"kia__rio": 26},
        tokens={"kia__rio|category:Kia Rio": "page|1"},
        done_items={"commons:1", "commons:2", "commons:3"},
    )
    состояние.categories["kia__rio|Kia Rio"] = ("Kia Rio",)
    save_state(путь, состояние)


def _записать_манифест(путь: Path) -> None:
    write_manifest(путь, [_строка(source="commons", source_item_id="2")])


def test_остаются_только_принятые() -> None:
    состояние = HarvestState(
        seen={"kia__rio": 120},
        accepted={"kia__rio": 26},
        tokens={"kia__rio|category:Kia Rio": "page|1"},
        done_items={"commons:1", "commons:2", "commons:3"},
    )
    состояние.categories["kia__rio|Kia Rio"] = ("Kia Rio",)
    строки = [_строка(source="commons", source_item_id="2")]

    итог = prune(состояние, строки)

    assert итог.done_items == {"commons:2"}
    assert итог.seen == {}
    assert итог.tokens == {}
    assert итог.categories == {}
    # Единственная строка манифеста принадлежит `lada__granta`, поэтому счётчик пересобирается
    # на неё, а `kia__rio`, чьих строк в манифесте нет, из него исчезает.
    assert итог.accepted == {"lada__granta": 1}


def test_завышенный_счётчик_снова_открывает_класс() -> None:
    состояние = HarvestState({}, {"lada__granta": 30}, {}, set())
    строки = [
        _строка(source="commons", source_item_id="2"),
        _строка(source="commons", source_item_id="3"),
    ]

    итог = prune(состояние, строки)

    assert итог.accepted == {"lada__granta": 2}
    assert is_class_done(итог, "lada__granta") is False


def test_класс_без_строк_в_манифесте_исчезает_из_счётчика() -> None:
    состояние = HarvestState({}, {"chevrolet__lanos": 30, "lada__granta": 30}, {}, set())
    строки = [_строка(source="commons", source_item_id="2")]

    итог = prune(состояние, строки)

    assert "chevrolet__lanos" not in итог.accepted
    assert итог.accepted == {"lada__granta": 1}


def test_прополка_идемпотентна() -> None:
    состояние = HarvestState({}, {"kia__rio": 1}, {}, {"commons:2"})
    строки = [_строка(source="commons", source_item_id="2")]

    assert prune(prune(состояние, строки), строки).done_items == {"commons:2"}


def test_сухой_прогон_ничего_не_пишет(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    состояние_путь = tmp_path / "state.json"
    манифест_путь = tmp_path / "harvest_manifest.csv"
    _записать_состояние(состояние_путь)
    _записать_манифест(манифест_путь)
    было = состояние_путь.read_bytes()

    код = main(["--state", str(состояние_путь), "--manifest", str(манифест_путь)])

    assert код == 0
    assert состояние_путь.read_bytes() == было
    assert not (tmp_path / "state.json.bak").exists()
    вывод = capsys.readouterr().out
    assert "было 3, станет 1" in вывод
    assert "accepted: сумма было 26, станет 1" in вывод
    assert "сухой прогон" in вывод


def test_write_пишет_прополотое_и_кладёт_прежнее_в_bak(tmp_path: Path) -> None:
    состояние_путь = tmp_path / "state.json"
    манифест_путь = tmp_path / "harvest_manifest.csv"
    _записать_состояние(состояние_путь)
    _записать_манифест(манифест_путь)
    было = состояние_путь.read_text(encoding="utf-8")

    код = main(["--state", str(состояние_путь), "--manifest", str(манифест_путь), "--write"])

    assert код == 0
    assert (tmp_path / "state.json.bak").read_text(encoding="utf-8") == было
    записанное = load_state(состояние_путь)
    assert записанное.done_items == {"commons:2"}
    assert записанное.seen == {}
    assert записанное.tokens == {}
    assert записанное.categories == {}
    assert записанное.accepted == {"lada__granta": 1}


def test_write_без_файла_состояния_не_падает(tmp_path: Path) -> None:
    состояние_путь = tmp_path / "state.json"
    манифест_путь = tmp_path / "harvest_manifest.csv"
    _записать_манифест(манифест_путь)

    код = main(["--state", str(состояние_путь), "--manifest", str(манифест_путь), "--write"])

    assert код == 0
    assert not (tmp_path / "state.json.bak").exists()
    assert load_state(состояние_путь).done_items == {"commons:2"}
