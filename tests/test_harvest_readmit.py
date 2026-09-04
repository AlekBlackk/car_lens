"""Переприём: источник из кэша и префлайт.

Ни сети, ни моделей: кэш подкладывается в `tmp_path` штатным `cache_path`, потому что он
шардирован по двум первым символам хеша, и тест, склеивший путь руками, зелёный при
сломанном коде.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.harvest.cache import cache_path
from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.readmit import (
    CachedSource,
    Readmission,
    accepted_delta,
    accepted_regressions,
    merge_verdicts_file,
    patch_state,
    plan_readmission,
)
from pipeline.harvest.rejects import Reject
from pipeline.harvest.sources import Candidate, SourceQuery
from pipeline.harvest.state import HarvestState

SHA = "ab" + "c" * 62


def _отказ(pageid: str, class_id: str = "lifan__x60") -> Reject:
    return Reject(
        class_id=class_id,
        basis="category:Lifan X60",
        source="commons",
        source_item_id=pageid,
        reason="redaction_failed",
        checked="2026-09-01",
    )


def _кандидат(pageid: str) -> Candidate:
    return Candidate(
        source="commons",
        source_item_id=pageid,
        class_id="lifan__x60",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        license_raw="CC BY 4.0",
        author="Иван Петров",
        title="File:X.jpg",
        width=1600,
        height=1200,
        basis="category:Lifan X60",
    )


def _положить_в_кэш(root: Path, sha: str, данные: bytes = b"jpeg") -> Path:
    путь = cache_path(root, "commons", sha, "jpg")
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(данные)
    return путь


def test_источник_читает_кэш_по_шардированному_пути(tmp_path: Path) -> None:
    _положить_в_кэш(tmp_path, SHA, "пиксели".encode())
    источник = CachedSource("commons", tmp_path, {"111": SHA})

    assert источник.fetch(_кандидат("111")) == "пиксели".encode()
    assert источник.name == "commons"


def test_источник_отдаёт_пустые_байты_если_файла_нет(tmp_path: Path) -> None:
    источник = CachedSource("commons", tmp_path, {"111": SHA})
    assert источник.fetch(_кандидат("111")) == b""


def test_источник_отдаёт_пустые_байты_если_страницы_нет_в_карте(tmp_path: Path) -> None:
    источник = CachedSource("commons", tmp_path, {})
    assert источник.fetch(_кандидат("999")) == b""


def test_поиск_в_кэш_источнике_не_поддержан(tmp_path: Path) -> None:
    источник = CachedSource("commons", tmp_path, {})
    with pytest.raises(NotImplementedError):
        источник.search(SourceQuery("lifan__x60", None, "lifan x60", 50, None))


def test_префлайт_пропускает_кандидата_с_оригиналом(tmp_path: Path) -> None:
    _положить_в_кэш(tmp_path, SHA)
    готовые, отброшенные = plan_readmission([_отказ("111")], {"111": SHA}, tmp_path, "commons")

    assert готовые == (Readmission("111", "lifan__x60", "category:Lifan X60", SHA),)
    assert отброшенные == ()


def test_префлайт_называет_кандидата_без_sha256(tmp_path: Path) -> None:
    готовые, отброшенные = plan_readmission([_отказ("111")], {}, tmp_path, "commons")

    assert готовые == ()
    assert len(отброшенные) == 1
    assert отброшенные[0].startswith("111:")


def test_префлайт_называет_кандидата_без_файла_в_кэше(tmp_path: Path) -> None:
    готовые, отброшенные = plan_readmission([_отказ("111")], {"111": SHA}, tmp_path, "commons")

    assert готовые == ()
    assert отброшенные[0].startswith("111:")
    assert SHA[:8] in отброшенные[0]


def _строка(file: str, class_id: str, pageid: str) -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id=pageid,
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="Иван Петров",
        license="CC BY 4.0",
        license_tier=2,
        checked="2026-09-04",
        basis="category:Lifan X60",
        sha256=file.removesuffix(".jpg").rjust(64, "0"),
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=1,
        faces=0,
        bbox_x=1.0,
        bbox_y=2.0,
        bbox_w=300.0,
        bbox_h=200.0,
        bbox_area_frac=0.4,
        det_conf=0.9,
        audit="",
    )


def _состояние() -> HarvestState:
    return HarvestState(
        seen={"lifan__x60": 12, "kia__rio": 400},
        accepted={"lifan__x60": 1, "kia__rio": 30},
        tokens={"kia__rio|category:Kia Rio": "page|1"},
        done_items={"commons:111", "commons:777"},
        categories={"kia__rio|Kia Rio": ("Kia Rio",)},
    )


def test_accepted_выводится_из_манифеста_целиком() -> None:
    манифест = [
        _строка("a.jpg", "lifan__x60", "111"),
        _строка("b.jpg", "lifan__x60", "222"),
        _строка("c.jpg", "kia__rio", "333"),
    ]
    новое = patch_state(_состояние(), манифест)

    assert новое.accepted == {"lifan__x60": 2, "kia__rio": 1}


def test_метки_добавляются_а_прежние_остаются() -> None:
    новое = patch_state(_состояние(), [_строка("a.jpg", "lifan__x60", "222")])

    assert новое.done_items == {"commons:111", "commons:777", "commons:222"}


def test_стоп_критерий_не_трогается() -> None:
    прежнее = _состояние()
    новое = patch_state(прежнее, [_строка("a.jpg", "lifan__x60", "222")])

    assert новое.seen == прежнее.seen
    assert новое.tokens == прежнее.tokens
    assert новое.categories == прежнее.categories


def test_патч_идемпотентен() -> None:
    манифест = [_строка("a.jpg", "lifan__x60", "222")]
    один = patch_state(_состояние(), манифест)
    два = patch_state(один, манифест)

    assert два.accepted == один.accepted
    assert два.done_items == один.done_items


def test_дельта_называет_только_изменившиеся_классы() -> None:
    манифест = [
        _строка("a.jpg", "lifan__x60", "111"),
        _строка("b.jpg", "lifan__x60", "222"),
        *[_строка(f"r{номер}.jpg", "kia__rio", str(номер)) for номер in range(30)],
    ]
    дельта = accepted_delta(_состояние(), манифест)

    assert дельта == {"lifan__x60": (1, 2)}


def test_уменьшение_accepted_попадает_в_регресс() -> None:
    дельта = {"lifan__x60": (1, 4), "kia__rio": (30, 28)}
    assert accepted_regressions(дельта) == ["kia__rio"]


def test_вердикты_дедупа_переливаются_в_боевой_файл(tmp_path: Path) -> None:
    временный = tmp_path / "verdicts.dedup.readmit.csv"
    боевой = tmp_path / "verdicts.dedup.csv"
    временный.write_text("file,verdict\nx.jpg,drop:superseded\n", encoding="utf-8")

    assert merge_verdicts_file(временный, боевой) == 1
    assert боевой.read_text(encoding="utf-8").splitlines()[0] == "file,verdict"
    assert "x.jpg,drop:superseded" in боевой.read_text(encoding="utf-8")


def test_перелив_не_плодит_шапку_в_непустом_файле(tmp_path: Path) -> None:
    временный = tmp_path / "verdicts.dedup.readmit.csv"
    боевой = tmp_path / "verdicts.dedup.csv"
    боевой.write_text("file,verdict\ny.jpg,conflict\n", encoding="utf-8")
    временный.write_text("file,verdict\nx.jpg,drop:superseded\n", encoding="utf-8")

    merge_verdicts_file(временный, боевой)

    строки = [
        строка for строка in боевой.read_text(encoding="utf-8").splitlines() if строка.strip()
    ]
    assert строки.count("file,verdict") == 1
    assert len(строки) == 3


def test_перелив_пустого_и_отсутствующего_файла_ничего_не_создаёт(tmp_path: Path) -> None:
    боевой = tmp_path / "verdicts.dedup.csv"
    только_шапка = tmp_path / "шапка.csv"
    только_шапка.write_text("file,verdict\n", encoding="utf-8")

    assert merge_verdicts_file(tmp_path / "нет.csv", боевой) == 0
    assert merge_verdicts_file(только_шапка, боевой) == 0
    assert not боевой.exists()
