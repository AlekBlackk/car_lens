"""Разбор флагов паритета. Модуль тянет litert-torch, поэтому тест — только в venv экспорта."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.export


def test_флаг_корня_кропов_разбирается(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--crops-root` доезжает до паритета как `Path`, без флага — `None` (Stanford)."""
    pytest.importorskip("litert_torch")
    from export.verify_parity import _parse_args

    базовые = [
        "verify_parity.py",
        "--export",
        "/e",
        "--index",
        "/i",
        "--checkpoint",
        "/c.pt",
    ]
    monkeypatch.setattr(sys, "argv", [*базовые, "--crops-root", "/x/y"])
    assert _parse_args().crops_root == Path("/x/y")

    monkeypatch.setattr(sys, "argv", базовые)
    assert _parse_args().crops_root is None


def test_флаг_точности_разбирается(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--precision` разбирается, умолчание — `int8` (путь Stanford не меняется)."""
    pytest.importorskip("litert_torch")
    from export.verify_parity import _parse_args

    базовые = [
        "verify_parity.py",
        "--export",
        "/e",
        "--index",
        "/i",
        "--checkpoint",
        "/c.pt",
    ]
    monkeypatch.setattr(sys, "argv", [*базовые, "--precision", "fp32"])
    assert _parse_args().precision == "fp32"

    monkeypatch.setattr(sys, "argv", базовые)
    assert _parse_args().precision == "int8"
