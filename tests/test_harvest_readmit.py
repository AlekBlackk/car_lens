"""Переприём: источник из кэша и префлайт.

Ни сети, ни моделей: кэш подкладывается в `tmp_path` штатным `cache_path`, потому что он
шардирован по двум первым символам хеша, и тест, склеивший путь руками, зелёный при
сломанном коде.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.harvest.cache import cache_path
from pipeline.harvest.readmit import CachedSource, Readmission, plan_readmission
from pipeline.harvest.rejects import Reject
from pipeline.harvest.sources import Candidate, SourceQuery

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
