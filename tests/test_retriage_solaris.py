from pathlib import Path

import pytest

from pipeline.harvest.cache import cache_path
from pipeline.harvest.manifest import HarvestRow, write_manifest
from scripts.retriage_solaris import QUEUE_NAME, main, to_review


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


def _положить_в_кэш(корень: Path, строка: HarvestRow) -> None:
    """Подделать оригинал в кэше: скрипт смотрит на наличие файла, а не на пиксели."""
    путь = cache_path(корень, строка.source, строка.sha256, "jpg")
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(b"")


def test_отбираются_только_затронутые_классы() -> None:
    строки = [
        _строка(file="a.jpg", class_id="hyundai__accent"),
        _строка(file="b.jpg", class_id="grp__solaris_hs"),
        _строка(file="c.jpg", class_id="kia__rio"),
    ]

    assert [с.file for с in to_review(строки, ("hyundai__accent", "grp__solaris_hs"))] == [
        "a.jpg",
        "b.jpg",
    ]


def test_пустой_список_классов_ничего_не_берёт() -> None:
    assert to_review([_строка(file="a.jpg", class_id="hyundai__accent")], ()) == []


def test_в_очередь_идут_только_затронутые_классы_с_кэшем(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    манифест = tmp_path / "harvest_manifest.csv"
    корень = tmp_path / "harvest"
    # Порядок в манифесте намеренно перемешивает классы: очередь обязана повторять его,
    # а не группировать кадры по классу.
    строки = [
        _строка(file="c.jpg", class_id="grp__solaris_hs", sha256="c3" * 32),
        _строка(file="a.jpg", class_id="hyundai__accent", sha256="a1" * 32),
        _строка(file="d.jpg", class_id="hyundai__accent", sha256="d4" * 32),
        _строка(file="b.jpg", class_id="kia__rio", sha256="b2" * 32),
    ]
    write_manifest(манифест, строки)
    for строка in (строки[0], строки[1], строки[3]):  # `d.jpg` оставлен без оригинала
        _положить_в_кэш(корень, строка)

    код = main(["--manifest", str(манифест), "--root", str(корень)])

    assert код == 0
    assert (корень / QUEUE_NAME).read_text(encoding="utf-8") == "c.jpg\na.jpg\n"
    вывод = capsys.readouterr().out
    assert "в очередь: 2; нет в кэше: 1" in вывод
    # Подсказка обязана быть запускаемой: у галереи `--manifest` и `--data-root`
    # обязательные, и команда без них падает на разборе аргументов.
    assert "-m scripts.audit_gallery" in вывод
    for флаг in ("--manifest", "--data-root", "--only-files"):
        assert флаг in вывод


def test_пустая_очередь_даёт_файл_нулевой_длины(tmp_path: Path) -> None:
    манифест = tmp_path / "harvest_manifest.csv"
    корень = tmp_path / "harvest"
    write_manifest(манифест, [_строка(file="a.jpg", class_id="kia__rio")])

    код = main(["--manifest", str(манифест), "--root", str(корень)])

    assert код == 0
    assert (корень / QUEUE_NAME).stat().st_size == 0
