"""Фикстура в дереве репозитория совпадает с тем, что даёт генератор.

Без этого теста правка формата не роняет ничего: фикстура остаётся старой, Node-тест
читает её и зеленеет, а настоящий индекс уже другой.
"""

from __future__ import annotations

from pathlib import Path

from scripts.gen_index_fixture import FIXTURE_DIR, build_fixture


def test_фикстура_совпадает_с_генератором(tmp_path: Path) -> None:
    build_fixture(tmp_path)
    for имя in ("catalog_index.bin", "catalog_meta.json"):
        assert (tmp_path / имя).read_bytes() == (FIXTURE_DIR / имя).read_bytes(), (
            f"{имя} в tests/fixtures/index устарел: пересобрать "
            f"`uv run python -m scripts.gen_index_fixture`"
        )


def test_размер_фикстуры_по_формуле() -> None:
    assert (FIXTURE_DIR / "catalog_index.bin").stat().st_size == 64 + 2 * 4 * 256 * 4
