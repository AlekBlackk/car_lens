"""Тесты CLI счёта пилота аудита пула."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.pool_audit import PENDING, RegistryRow, pilot_truth, write_registry
from scripts.score_pool_audit_pilot import main

ВЫБОРКА = Path("bench") / "plate-leak-sample.csv"


def _выгрузка(путь: Path, *, пропустить: str | None = None, pending: bool = False) -> Path:
    выборка = pd.read_csv(ВЫБОРКА, dtype={"sha256": str})
    эталон = pilot_truth(выборка)
    строки: list[RegistryRow] = []
    for sha in выборка["sha256"]:
        plate = "leak" if sha in эталон.plate_ids and sha != пропустить else "clean_filled"
        face = "leak" if sha in эталон.face_ids else "clean"
        строки.append(RegistryRow(sha, 1, plate, face, "s"))
    if pending:
        строки[-1] = RegistryRow(строки[-1].photo_id, 1, PENDING, "clean", "s")
    write_registry(путь, строки)
    return путь


def test_пройденный_пилот_код_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--export", str(_выгрузка(tmp_path / "e.csv"))]) == 0
    вывод = capsys.readouterr().out
    assert "номера: 9/9" in вывод
    assert "лица: 10/10" in вывод


def test_пропуск_номера_код_1_и_номер_выборки_в_выводе(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    выборка = pd.read_csv(ВЫБОРКА, dtype={"sha256": str})
    sha_43 = выборка.loc[выборка["номер"] == 43, "sha256"].item()
    assert main(["--export", str(_выгрузка(tmp_path / "e.csv", пропустить=sha_43))]) == 1
    вывод = capsys.readouterr().out
    assert "номера: 8/9" in вывод
    assert "#043" in вывод


def test_незаконченный_пилот_код_2(tmp_path: Path) -> None:
    assert main(["--export", str(_выгрузка(tmp_path / "e.csv", pending=True))]) == 2
