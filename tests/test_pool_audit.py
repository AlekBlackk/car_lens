"""Тесты чистой логики сплошного аудита приватности пула."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from pipeline.pool_audit import (
    FACE_KEYS,
    KNOWN_FACE_LEAK_NUMBERS,
    PENDING,
    PLATE_KEYS,
    PLATE_VERDICTS,
    REGISTRY_COLUMNS,
    PilotTruth,
    PoolAuditError,
    RegistryCheck,
    RegistryRow,
    assign_sessions,
    audited_mismatch,
    audited_pool,
    check_registry,
    excluded_ids,
    file_sha256,
    index_registry,
    pilot_truth,
    read_registry,
    score_pilot,
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


def _чистый_реестр(n: int) -> list[RegistryRow]:
    return [_строка(f"p{i}", sha=f"s{i}") for i in range(n)]


def _текущие_sha(n: int) -> dict[str, str | None]:
    return {f"p{i}": f"s{i}" for i in range(n)}


def test_полный_чистый_реестр_проходит_приёмку() -> None:
    проверка = check_registry(
        _чистый_реестр(3), pool_ids=["p0", "p1", "p2"], current_crop_sha256=_текущие_sha(3)
    )
    assert проверка == RegistryCheck((), (), (), ())
    assert проверка.passed


def test_кадр_пула_без_строки_и_лишняя_строка_ловятся() -> None:
    проверка = check_registry(
        _чистый_реестр(3), pool_ids=["p0", "p1", "p9"], current_crop_sha256=_текущие_sha(3)
    )
    assert проверка.missing_ids == ("p9",)
    assert проверка.extra_ids == ("p2",)
    assert not проверка.passed


@pytest.mark.parametrize(("plate", "face"), [(PENDING, "clean"), ("clean_filled", PENDING)])
def test_pending_блокирует_приёмку(plate: str, face: str) -> None:
    строки = [*_чистый_реестр(2), _строка("p2", plate, face, "s2")]
    проверка = check_registry(
        строки, pool_ids=["p0", "p1", "p2"], current_crop_sha256=_текущие_sha(3)
    )
    assert проверка.pending_ids == ("p2",)
    assert not проверка.passed


def test_протухший_и_пропавший_кроп_ловятся() -> None:
    текущие = _текущие_sha(3)
    текущие["p1"] = "изменился"
    текущие["p2"] = None
    проверка = check_registry(
        _чистый_реестр(3), pool_ids=["p0", "p1", "p2"], current_crop_sha256=текущие
    )
    assert проверка.stale_ids == ("p1", "p2")
    assert not проверка.passed


def test_дубликат_в_реестре_ошибка_а_не_последний_выигрывает() -> None:
    with pytest.raises(PoolAuditError, match="дважды"):
        index_registry([_строка("p0"), _строка("p0", plate="leak")])


def test_исключение_вычисляется_из_вердиктов() -> None:
    строки = [
        _строка("p0", plate="leak"),
        _строка("p1", plate="leak_hist"),
        _строка("p2", face="leak"),
        _строка("p3", plate="clean_partial"),
    ]
    assert excluded_ids(строки) == frozenset({"p0", "p1", "p2"})


def test_audited_pool_выкидывает_только_утечки() -> None:
    пул = pd.DataFrame({"photo_id": ["p0", "p1", "p2"], "model_id": ["a", "b", "c"]})
    строки = [_строка("p0"), _строка("p1", plate="leak"), _строка("p2")]
    итог = audited_pool(пул, строки)
    assert list(итог["photo_id"]) == ["p0", "p2"]
    assert list(итог.index) == [0, 1]


def test_audited_mismatch_ловит_оставленную_утечку_и_потерянный_чистый() -> None:
    строки = [_строка("p0"), _строка("p1", plate="leak"), _строка("p2")]
    пул = ["p0", "p1", "p2"]
    assert audited_mismatch(строки, pool_ids=пул, audited_ids=["p0", "p2"]) == ()
    assert audited_mismatch(строки, pool_ids=пул, audited_ids=["p0", "p1", "p2"]) == ("p1",)
    assert audited_mismatch(строки, pool_ids=пул, audited_ids=["p0"]) == ("p2",)


ВЫБОРКА = Path("bench") / "plate-leak-sample.csv"


def _выборка() -> pd.DataFrame:
    return pd.read_csv(ВЫБОРКА, dtype={"sha256": str})


def test_эталон_пилота_из_настоящей_выборки() -> None:
    выборка = _выборка()
    эталон = pilot_truth(выборка)
    assert len(эталон.plate_ids) == 9
    assert len(эталон.face_ids) == 10
    assert not эталон.plate_ids & эталон.face_ids
    номер_43 = выборка.loc[выборка["номер"] == 43, "sha256"].item()
    assert номер_43 in эталон.plate_ids


def test_список_лиц_не_совпавший_с_выборкой_отвергается() -> None:
    выборка = _выборка()
    обрезанная = выборка[выборка["номер"] != KNOWN_FACE_LEAK_NUMBERS[0]]
    with pytest.raises(PoolAuditError, match="лиц"):
        pilot_truth(обрезанная)


def _идеальная_выгрузка(эталон: PilotTruth, чистых: int = 3) -> list[RegistryRow]:
    строки = [_строка(i, plate="leak") for i in sorted(эталон.plate_ids)]
    строки += [_строка(i, face="leak") for i in sorted(эталон.face_ids)]
    строки += [_строка(f"c{i}") for i in range(чистых)]
    return строки


def test_идеальная_выгрузка_проходит_пилот() -> None:
    эталон = PilotTruth(frozenset({"n0", "n1"}), frozenset({"f0"}))
    счёт = score_pilot(_идеальная_выгрузка(эталон), эталон)
    assert счёт.passed
    assert (счёт.plate_expected, счёт.face_expected) == (2, 1)


def test_пропуск_номера_проваливает_пилот() -> None:
    эталон = PilotTruth(frozenset({"n0", "n1"}), frozenset({"f0"}))
    строки = [
        _строка("n0", plate="leak_hist"),
        _строка("n1", plate="clean_partial"),
        _строка("f0", face="leak"),
    ]
    счёт = score_pilot(строки, эталон)
    assert счёт.plate_misses == ("n1",)
    assert not счёт.passed


def test_пропуск_лица_проваливает_пилот_а_лишнее_лицо_нет() -> None:
    эталон = PilotTruth(frozenset({"n0"}), frozenset({"f0", "f1"}))
    строки = [
        _строка("n0", plate="leak"),
        _строка("f0", face="leak"),
        _строка("f1", face="clean"),
        _строка("x0", face="leak"),
    ]
    счёт = score_pilot(строки, эталон)
    assert счёт.face_misses == ("f1",)
    assert not счёт.passed


def test_кадр_эталона_без_строки_это_промах() -> None:
    эталон = PilotTruth(frozenset({"n0"}), frozenset({"f0"}))
    счёт = score_pilot([_строка("f0", face="leak")], эталон)
    assert счёт.plate_misses == ("n0",)


def test_pending_не_даёт_пройти_пилот() -> None:
    эталон = PilotTruth(frozenset({"n0"}), frozenset({"f0"}))
    строки = [*_идеальная_выгрузка(эталон), _строка("c9", plate=PENDING)]
    счёт = score_pilot(строки, эталон)
    assert счёт.pending_ids == ("c9",)
    assert not счёт.passed
