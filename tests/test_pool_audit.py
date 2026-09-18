"""Тесты чистой логики сплошного аудита приватности пула."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

from pipeline.pool_audit import (
    FACE_KEYS,
    PENDING,
    PLATE_KEYS,
    PLATE_VERDICTS,
    REGISTRY_COLUMNS,
    PoolAuditError,
    RegistryRow,
    assign_sessions,
    file_sha256,
    read_registry,
    write_registry,
)


def _строка(
    photo_id: str = "p0",
    plate: str = "clean_filled",
    face: str = "clean",
    sha: str = "s0",
    session: int = 1,
) -> RegistryRow:
    return RegistryRow(photo_id, session, plate, face, sha)


def test_клавиши_номера_идут_по_таксономии() -> None:
    assert dict(PLATE_KEYS) == {
        "1": "leak",
        "2": "leak_hist",
        "3": "clean_partial",
        "4": "clean_unreadable",
        "5": "clean_filled",
        "6": "clean_no_plate",
    }
    assert set(PLATE_KEYS.values()) == set(PLATE_VERDICTS)


def test_клавиши_лица_и_номера_не_пересекаются() -> None:
    assert dict(FACE_KEYS) == {"f": "leak", "g": "clean"}
    assert not set(PLATE_KEYS) & set(FACE_KEYS)


@pytest.mark.parametrize(
    ("plate", "face"),
    [("ok", "clean"), ("clean_filled", "unclear"), ("", "clean"), ("clean_filled", "")],
)
def test_неизвестный_вердикт_отвергается(plate: str, face: str) -> None:
    with pytest.raises(PoolAuditError):
        _строка(plate=plate, face=face)


def test_пустой_photo_id_отвергается() -> None:
    with pytest.raises(PoolAuditError):
        _строка(photo_id="")


@pytest.mark.parametrize(
    ("plate", "face", "pending"),
    [
        (PENDING, "clean", True),
        ("clean_filled", PENDING, True),
        (PENDING, PENDING, True),
        ("clean_no_plate", "clean", False),
    ],
)
def test_непросмотренная_ось_это_pending_а_не_clean(plate: str, face: str, pending: bool) -> None:
    строка = _строка(plate=plate, face=face)
    assert строка.pending is pending
    assert строка.excluded is False


@pytest.mark.parametrize(
    ("plate", "face", "excluded"),
    [
        ("leak", "clean", True),
        ("leak_hist", "clean", True),
        ("clean_no_plate", "leak", True),
        ("clean_partial", "clean", False),
        ("clean_unreadable", "clean", False),
        ("clean_filled", "clean", False),
        ("clean_no_plate", "clean", False),
    ],
)
def test_правило_исключения(plate: str, face: str, excluded: bool) -> None:
    assert _строка(plate=plate, face=face).excluded is excluded


def test_поля_строки_идут_в_порядке_колонок_реестра() -> None:
    assert tuple(поле.name for поле in fields(RegistryRow)) == REGISTRY_COLUMNS


def test_реестр_круговой(tmp_path: Path) -> None:
    строки = [
        _строка("p0", "leak", "clean", "a" * 64, 1),
        _строка("p1", PENDING, PENDING, "", 2),
    ]
    путь = tmp_path / "registry.csv"
    write_registry(путь, строки)
    assert read_registry(путь) == строки
    assert путь.read_text(encoding="utf-8").splitlines()[0] == ",".join(REGISTRY_COLUMNS)


def test_чужая_шапка_реестра_отвергается(tmp_path: Path) -> None:
    путь = tmp_path / "registry.csv"
    путь.write_text("file,verdict\np0,ok\n", encoding="utf-8")
    with pytest.raises(PoolAuditError, match="шапка"):
        read_registry(путь)


def test_сеансы_по_300_в_порядке_photo_id() -> None:
    ids = [f"{i:064x}" for i in range(2863)]
    сеансы = assign_sessions(reversed(ids))
    assert max(сеансы.values()) == 10
    assert sum(1 for s in сеансы.values() if s == 10) == 163
    assert {i for i, s in сеансы.items() if s == 1} == set(sorted(ids)[:300])


def test_дубликат_photo_id_в_сеансах_отвергается() -> None:
    with pytest.raises(PoolAuditError, match="дважды"):
        assign_sessions(["p0", "p1", "p0"])


def test_sha_кропа_считается_по_байтам(tmp_path: Path) -> None:
    путь = tmp_path / ("0" * 64 + ".jpg")
    путь.write_bytes(b"first")
    первый = file_sha256(путь)
    путь.write_bytes(b"second")
    assert file_sha256(путь) != первый
    assert file_sha256(путь) == hashlib.sha256(b"second").hexdigest()
