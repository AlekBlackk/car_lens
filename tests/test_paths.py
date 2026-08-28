"""Тесты разрешения путей к данным."""

from pathlib import Path

import pytest

from pipeline import paths


def test_data_root_берёт_значение_из_переменной_среды(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", "/mnt/data/carlens")
    assert paths.data_root() == Path("/mnt/data/carlens")


def test_data_root_падает_на_умолчание_без_переменной(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CARLENS_DATA_ROOT", raising=False)
    assert paths.data_root() == Path.home() / "carlens-data"


def test_data_root_раскрывает_тильду(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", "~/somewhere")
    assert paths.data_root() == Path.home() / "somewhere"


def test_подкаталоги_лежат_под_корнем(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", "/tmp/cl")
    assert paths.raw_dir() == Path("/tmp/cl/raw")
    assert paths.crops_dir() == Path("/tmp/cl/crops")
    assert paths.splits_dir() == Path("/tmp/cl/splits")
    assert paths.runs_dir() == Path("/tmp/cl/runs")
    assert paths.harvest_dir() == Path("/tmp/cl/harvest")


def test_ensure_dirs_создаёт_всё_дерево(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path / "root"))
    paths.ensure_dirs()
    for sub in ("raw", "crops", "splits", "runs"):
        assert (tmp_path / "root" / sub).is_dir()
