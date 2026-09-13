"""Снимок продвижения: полнота, журналируемая установка и полный откат."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pipeline.redaction_calibration.promotion import (
    INDEX_FILES,
    SPLIT_FILES,
    PreviousSnapshotRequest,
    PromotionDestinations,
    PromotionError,
    SnapshotTier,
    build_snapshot,
    confirmation_token,
    load_snapshot,
    promote,
    rollback,
    validate_snapshot,
)
from pipeline.redaction_calibration.state import (
    CalibrationStateError,
    PromotionRecord,
    PromotionState,
)
from scripts.promote_redaction_profile import (
    _witnesses_from_arguments,
    is_git_clean,
    main,
    write_record,
)

CREATED_AT = "2026-09-13T00:00:00Z"
PROFILE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64
CANONICAL_SHA = "c" * 64
GIT_SHA = "9" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _make_index(root: Path, marker: bytes) -> Path:
    """Каталог few-shot индекса, чей `report.json` честно ссылается на свои же файлы."""
    root.mkdir(parents=True, exist_ok=True)
    index_bin = _write(root / "catalog_index.bin", b"index-" + marker)
    meta = _write(root / "catalog_meta.json", b'{"classes": ["' + marker + b'"]}')
    _write(root / "fewshot_metrics.json", b'{"top1": 0.5}')
    _write(
        root / "report.json",
        json.dumps(
            {"index_sha256": _sha256(index_bin), "meta_sha256": _sha256(meta)},
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    return root


def _make_sources(root: Path, marker: bytes = b"new") -> dict[str, Path]:
    """Готовый набор входов снимка: манифест, кропы, сплиты и индекс."""
    manifest = _write(root / "manifest.csv", b"sha256,class_id\n" + marker + b",lada__vesta\n")
    crops = root / "crops"
    _write(crops / "lada__vesta" / "one.jpg", b"crop-one-" + marker)
    _write(crops / "gaz__gazel" / "two.jpg", b"crop-two-" + marker)
    splits = root / "splits"
    for name in SPLIT_FILES:
        _write(splits / name, name.encode("utf-8") + b"-" + marker)
    index = _make_index(root / "index", marker)
    return {"manifest": manifest, "crops": crops, "splits": splits, "index": index}


def _witnesses(sources: dict[str, Path]) -> dict[str, str | bool]:
    return {
        "profile_id": "profile-x",
        "profile_sha256": PROFILE_SHA,
        "geometry_witness_sha256": GEOMETRY_SHA,
        "canonical_sha256": CANONICAL_SHA,
        "git_sha": GIT_SHA,
        "git_clean": True,
        "index_pool_sha256": _sha256(sources["splits"] / "pool.parquet"),
    }


def _build(
    tmp_path: Path,
    *,
    snapshot_id: str = "snap-new",
    marker: bytes = b"new",
    complete: bool = True,
    witnesses: dict[str, str | bool] | None = None,
) -> tuple[Path, dict[str, Path]]:
    sources = _make_sources(tmp_path / f"sources-{snapshot_id}", marker)
    snapshot_root = tmp_path / "snapshots" / snapshot_id
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    build_snapshot(
        snapshot_root,
        snapshot_id=snapshot_id,
        manifest_path=sources["manifest"],
        crops_root=sources["crops"],
        splits_root=sources["splits"] if complete else None,
        index_root=sources["index"] if complete else None,
        witnesses=witnesses if witnesses is not None else _witnesses(sources),
        created_at=CREATED_AT,
    )
    return snapshot_root, sources


def _destinations(
    tmp_path: Path, *, marker: bytes = b"old", derived: bool = True
) -> tuple[PromotionDestinations, dict[str, Path]]:
    live = _make_sources(tmp_path / "live", marker)
    if not derived:
        _remove_tree(live["splits"])
        _remove_tree(live["index"])
    return (
        PromotionDestinations(
            manifest=live["manifest"],
            crops=live["crops"],
            splits=live["splits"],
            index=live["index"],
            journal=tmp_path / "journal.json",
        ),
        live,
    )


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.rmdir() if child.is_dir() else child.unlink()
    path.rmdir()


def _promotable(snapshot_result_sha256: str) -> PromotionRecord:
    return PromotionRecord(
        state=PromotionState.PROMOTABLE,
        profile_id="profile-x",
        profile_sha256=PROFILE_SHA,
        geometry_witness_sha256=GEOMETRY_SHA,
        snapshot_result_sha256=snapshot_result_sha256,
    )


def _previous_request(tmp_path: Path, name: str = "snap-displaced") -> PreviousSnapshotRequest:
    """Куда продвижение обязано сложить вытесняемые боевые данные."""
    return PreviousSnapshotRequest(
        root=tmp_path / "snapshots" / name,
        snapshot_id=name,
        witnesses={
            "profile_id": "profile-displaced",
            "profile_sha256": "e" * 64,
            "geometry_witness_sha256": "f" * 64,
            "canonical_sha256": CANONICAL_SHA,
            "git_sha": GIT_SHA,
            "git_clean": True,
        },
        created_at=CREATED_AT,
    )


def _promote(
    snapshot: Any,
    destinations: PromotionDestinations,
    tmp_path: Path,
    **kwargs: Any,
) -> Any:
    """Продвижение с обязательным снимком вытесняемых данных и верным токеном."""
    kwargs.setdefault("record", _promotable(snapshot.result_sha256))
    kwargs.setdefault("confirmation_token", confirmation_token(snapshot, operation="promote"))
    kwargs.setdefault("previous", _previous_request(tmp_path))
    return promote(snapshot, destinations, **kwargs)


def _live_state(destinations: PromotionDestinations) -> dict[str, bytes]:
    """Наблюдаемое содержимое назначения: файл → байты, целиком по всем группам."""
    state: dict[str, bytes] = {}
    if destinations.manifest.exists():
        state["manifest.csv"] = destinations.manifest.read_bytes()
    for group, root in (
        ("crops", destinations.crops),
        ("splits", destinations.splits),
        ("index", destinations.index),
    ):
        if root is None or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                state[f"{group}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    return state


def _sidecars(destinations: PromotionDestinations) -> list[str]:
    """Служебные следы протокола рядом с назначением: staging, previous и журнал."""
    found: list[str] = []
    if destinations.journal.exists():
        found.append(destinations.journal.name)
    parents = {destinations.manifest.parent, destinations.crops.parent}
    for root in (destinations.splits, destinations.index):
        if root is not None:
            parents.add(root.parent)
    for parent in parents:
        found.extend(
            child.name
            for child in parent.iterdir()
            if child.name.startswith(".")
            and (".staging" in child.name or ".previous" in child.name)
        )
    return sorted(found)


def _fault(name: str) -> Callable[[str], None]:
    def checkpoint(step: str) -> None:
        if step == name:
            raise RuntimeError(f"инъекция отказа на {step}")

    return checkpoint


# --- полнота снимка -------------------------------------------------------------------


def test_build_snapshot_records_complete_tier(tmp_path: Path) -> None:
    snapshot_root, sources = _build(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    assert snapshot.tier is SnapshotTier.COMPLETE
    assert snapshot.snapshot_id == "snap-new"
    assert "manifest.csv" in snapshot.artifacts
    assert "crops/lada__vesta/one.jpg" in snapshot.artifacts
    for name in SPLIT_FILES:
        assert f"splits/{name}" in snapshot.artifacts
    for name in INDEX_FILES:
        assert f"index/{name}" in snapshot.artifacts
    assert snapshot.hashes["manifest.csv"] == _sha256(sources["manifest"])
    assert len(snapshot.result_sha256) == 64
    assert (snapshot_root / "provenance.json").is_file()


def test_build_snapshot_records_base_tier(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path, complete=False)
    snapshot = load_snapshot(snapshot_root)

    assert snapshot.tier is SnapshotTier.BASE
    assert not any(name.startswith(("splits/", "index/")) for name in snapshot.artifacts)


def test_build_snapshot_rejects_half_derived_input(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    with pytest.raises(PromotionError, match="сплиты и индекс"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=None,
            witnesses=_witnesses(sources),
            created_at=CREATED_AT,
        )


def test_build_snapshot_rejects_missing_split(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    witnesses = _witnesses(sources)
    (sources["splits"] / "pool.parquet").unlink()
    with pytest.raises(PromotionError, match=r"pool\.parquet"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=witnesses,
            created_at=CREATED_AT,
        )


def test_build_snapshot_rejects_index_from_wrong_pool(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    witnesses = _witnesses(sources) | {"index_pool_sha256": "d" * 64}
    with pytest.raises(PromotionError, match="index_pool_sha256"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=witnesses,
            created_at=CREATED_AT,
        )


def test_build_snapshot_rejects_index_report_not_describing_its_index(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    (sources["index"] / "catalog_index.bin").write_bytes(b"index-tampered")
    with pytest.raises(PromotionError, match="index_sha256"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=_witnesses(sources),
            created_at=CREATED_AT,
        )


def test_build_snapshot_refuses_existing_root(tmp_path: Path) -> None:
    snapshot_root, sources = _build(tmp_path)
    with pytest.raises(PromotionError, match="уже существует"):
        build_snapshot(
            snapshot_root,
            snapshot_id="snap-new",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=_witnesses(sources),
            created_at=CREATED_AT,
        )


def test_build_snapshot_leaves_no_staging_behind(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    siblings = [child.name for child in snapshot_root.parent.iterdir()]
    assert siblings == ["snap-new"]


# --- валидация ------------------------------------------------------------------------


def test_validate_snapshot_detects_changed_crop(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    (snapshot_root / "crops" / "lada__vesta" / "one.jpg").write_bytes(b"tampered")

    with pytest.raises(PromotionError, match=r"crops/lada__vesta/one\.jpg"):
        validate_snapshot(snapshot, destinations)


def test_validate_snapshot_does_not_write_to_destination(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    before = _live_state(destinations)

    validate_snapshot(load_snapshot(snapshot_root), destinations)

    assert _live_state(destinations) == before
    assert _sidecars(destinations) == []


def test_validate_snapshot_rejects_leftover_journal(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    destinations.journal.write_text("{}", encoding="utf-8")

    with pytest.raises(PromotionError, match="журнал"):
        validate_snapshot(load_snapshot(snapshot_root), destinations)


def test_validate_snapshot_rejects_destination_collision(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    collision = destinations.crops.parent / ".crops.promote-snap-new.staging"
    collision.mkdir()

    with pytest.raises(PromotionError, match="staging"):
        validate_snapshot(load_snapshot(snapshot_root), destinations)


# --- подтверждение --------------------------------------------------------------------


def test_promote_requires_snapshot_specific_token(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    with pytest.raises(PromotionError, match="подтверждени"):
        _promote(
            snapshot,
            destinations,
            tmp_path,
            confirmation_token=confirmation_token(snapshot, operation="rollback"),
        )

    assert _live_state(destinations) == before
    assert _sidecars(destinations) == []


def test_promote_requires_promotable_record(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    record = PromotionRecord(state=PromotionState.NOT_SELECTED)

    with pytest.raises(CalibrationStateError, match="promotable"):
        _promote(snapshot, destinations, tmp_path, record=record)

    assert _sidecars(destinations) == []


def test_promote_requires_record_bound_to_this_snapshot(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    with pytest.raises(CalibrationStateError, match="snapshot_result_sha256"):
        _promote(snapshot, destinations, tmp_path, record=_promotable("f" * 64))


# --- установка ------------------------------------------------------------------------


def test_promote_installs_every_group_of_a_complete_snapshot(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    result = _promote(snapshot, destinations, tmp_path)

    assert result.switched == ("manifest", "crops", "splits", "index")
    assert len(result.result_sha256) == 64
    live = _live_state(destinations)
    assert live["manifest.csv"] == (snapshot_root / "manifest.csv").read_bytes()
    assert live["crops/lada__vesta/one.jpg"] == b"crop-one-new"
    assert live["splits/pool.parquet"] == b"pool.parquet-new"
    assert live["index/catalog_index.bin"] == b"index-new"
    assert _sidecars(destinations) == []


def test_promote_of_base_snapshot_leaves_derived_destinations_untouched(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path, complete=False)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    result = _promote(snapshot, destinations, tmp_path)

    assert result.switched == ("manifest", "crops")
    live = _live_state(destinations)
    assert live["crops/lada__vesta/one.jpg"] == b"crop-one-new"
    assert live["splits/pool.parquet"] == b"pool.parquet-old"
    assert live["index/catalog_index.bin"] == b"index-old"


# --- откат ----------------------------------------------------------------------------


def test_rollback_restores_manifest_crops_splits_and_index_together(tmp_path: Path) -> None:
    previous_root, _ = _build(tmp_path, snapshot_id="snap-old", marker=b"old")
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path, marker=b"live")
    new = load_snapshot(snapshot_root)
    _promote(new, destinations, tmp_path)

    previous = load_snapshot(previous_root)
    result = rollback(
        previous,
        destinations,
        confirmation_token=confirmation_token(previous, operation="rollback"),
    )

    assert result.switched == ("manifest", "crops", "splits", "index")
    live = _live_state(destinations)
    assert live["crops/lada__vesta/one.jpg"] == b"crop-one-old"
    assert live["splits/holdout_ru.parquet"] == b"holdout_ru.parquet-old"
    assert live["index/fewshot_metrics.json"] == b'{"top1": 0.5}'
    assert live["manifest.csv"] == (previous_root / "manifest.csv").read_bytes()
    assert _sidecars(destinations) == []


def test_rollback_rejects_base_snapshot_once_derived_artifacts_exist(tmp_path: Path) -> None:
    previous_root, _ = _build(tmp_path, snapshot_id="snap-old", marker=b"old", complete=False)
    destinations, _ = _destinations(tmp_path)
    previous = load_snapshot(previous_root)
    before = _live_state(destinations)

    with pytest.raises(PromotionError, match="частичный откат"):
        rollback(
            previous,
            destinations,
            confirmation_token=confirmation_token(previous, operation="rollback"),
        )

    assert _live_state(destinations) == before
    assert _sidecars(destinations) == []


def test_rollback_accepts_base_snapshot_before_derived_artifacts_exist(tmp_path: Path) -> None:
    previous_root, _ = _build(tmp_path, snapshot_id="snap-old", marker=b"old", complete=False)
    destinations, _ = _destinations(tmp_path, derived=False)
    previous = load_snapshot(previous_root)

    result = rollback(
        previous,
        destinations,
        confirmation_token=confirmation_token(previous, operation="rollback"),
    )

    assert result.switched == ("manifest", "crops")
    assert _live_state(destinations)["crops/gaz__gazel/two.jpg"] == b"crop-two-old"


# --- инъекция отказов -----------------------------------------------------------------

STAGE_FAULTS = [
    "staged:manifest",
    "staged:crops",
    "staged:splits",
    "staged:index",
    "journal_written",
    "before_switch:manifest",
]

SWITCH_FAULTS = [
    "switching:manifest",
    "after_switch:manifest",
    "before_switch:crops",
    "switching:crops",
    "after_switch:crops",
    "before_switch:splits",
    "switching:splits",
    "after_switch:splits",
    "before_switch:index",
    "switching:index",
    "after_switch:index",
]


@pytest.mark.parametrize("step", STAGE_FAULTS + SWITCH_FAULTS)
def test_promote_leaves_destination_wholly_old_or_wholly_new(tmp_path: Path, step: str) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    with pytest.raises(RuntimeError, match="инъекция отказа"):
        _promote(snapshot, destinations, tmp_path, checkpoint=_fault(step))

    after = _live_state(destinations)
    assert after == before, f"назначение смешалось после отказа на {step}"


@pytest.mark.parametrize("step", STAGE_FAULTS)
def test_promote_removes_staging_on_pre_switch_failure(tmp_path: Path, step: str) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    with pytest.raises(RuntimeError):
        _promote(snapshot, destinations, tmp_path, checkpoint=_fault(step))

    assert not any(".staging" in name for name in _sidecars(destinations))


def test_promote_keeps_the_journal_when_a_switch_fails(tmp_path: Path) -> None:
    """Оборванное переключение обязано оставить след: журнал блокирует слепой повтор."""
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    with pytest.raises(RuntimeError):
        _promote(snapshot, destinations, tmp_path, checkpoint=_fault("after_switch:crops"))

    assert destinations.journal.is_file()
    journal = json.loads(destinations.journal.read_text(encoding="utf-8"))
    assert journal["snapshot_id"] == "snap-new"
    assert journal["groups"] == ["manifest", "crops", "splits", "index"]

    with pytest.raises(PromotionError, match="журнал"):
        validate_snapshot(snapshot, destinations)


# --- CLI ------------------------------------------------------------------------------


def test_cli_dry_run_prints_token_and_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    code = main(_cli_promote_argv(snapshot_root, destinations))

    assert code == 0
    assert confirmation_token(snapshot, operation="promote") in capsys.readouterr().out
    assert _live_state(destinations) == before


def test_cli_promotes_only_with_the_printed_token(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    record_path = tmp_path / "promotion.json"
    record_path.write_text(
        json.dumps(
            {
                "state": "promotable",
                "profile_id": "profile-x",
                "profile_sha256": PROFILE_SHA,
                "geometry_witness_sha256": GEOMETRY_SHA,
                "snapshot_result_sha256": snapshot.result_sha256,
            }
        ),
        encoding="utf-8",
    )
    argv = [
        *_cli_promote_argv(snapshot_root, destinations),
        "--record",
        str(record_path),
        "--previous-root",
        str(tmp_path / "snapshots" / "snap-displaced"),
        "--previous-witnesses",
        str(_previous_witnesses_file(tmp_path)),
        "--confirm",
        confirmation_token(snapshot, operation="promote"),
    ]

    assert main(argv) == 0

    assert destinations.crops.joinpath("lada__vesta", "one.jpg").read_bytes() == b"crop-one-new"
    updated = json.loads(record_path.read_text(encoding="utf-8"))
    assert updated["state"] == "promoted"
    assert len(updated["promotion_result_sha256"]) == 64


def test_cli_rejects_a_wrong_token(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    before = _live_state(destinations)
    argv = [*_cli_promote_argv(snapshot_root, destinations), "--confirm", "0" * 16]

    assert main(argv) == 2
    assert _live_state(destinations) == before


def _cli_destination_argv(destinations: PromotionDestinations) -> list[str]:
    assert destinations.splits is not None
    assert destinations.index is not None
    return [
        "--manifest-dest",
        str(destinations.manifest),
        "--crops-dest",
        str(destinations.crops),
        "--splits-dest",
        str(destinations.splits),
        "--index-dest",
        str(destinations.index),
        "--journal",
        str(destinations.journal),
    ]


def _cli_promote_argv(snapshot_root: Path, destinations: PromotionDestinations) -> list[str]:
    return [
        "promote",
        "--snapshot",
        str(snapshot_root),
        *_cli_destination_argv(destinations),
    ]


# --- свидетель версии кода (§13) ------------------------------------------------------


def test_build_snapshot_requires_a_code_version_witness(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    witnesses = {name: value for name, value in _witnesses(sources).items() if name != "git_sha"}
    with pytest.raises(PromotionError, match="git_sha"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=witnesses,
            created_at=CREATED_AT,
        )


def test_build_snapshot_rejects_a_short_git_sha(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    witnesses = _witnesses(sources) | {"git_sha": "9" * 7}
    with pytest.raises(PromotionError, match="git_sha"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=witnesses,
            created_at=CREATED_AT,
        )


def test_snapshot_carries_the_code_version_into_provenance(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    document = json.loads((snapshot_root / "provenance.json").read_text(encoding="utf-8"))

    assert document["input_witnesses"]["git_sha"] == GIT_SHA
    # Именно булев `true`, а не строка: в JSON эти два значения различимы, и описи
    # положено нести то же, что несёт паспорт базиса.
    assert document["input_witnesses"]["git_clean"] is True
    restored = load_snapshot(snapshot_root)
    assert restored.input_witnesses["git_sha"] == GIT_SHA
    assert restored.input_witnesses["git_clean"] is True


# --- сохранённый снимок вытесняемых данных --------------------------------------------


def test_promote_retains_the_displaced_data_as_a_complete_snapshot(tmp_path: Path) -> None:
    """После продвижения откатываться есть куда, даже если снимок «до» никто не собирал."""
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    result = _promote(snapshot, destinations, tmp_path)

    assert result.previous_snapshot_id == "snap-displaced"
    retained = load_snapshot(tmp_path / "snapshots" / "snap-displaced")
    assert retained.tier is SnapshotTier.COMPLETE
    assert retained.result_sha256 == result.previous_snapshot_result_sha256
    # Хеши сохранённого снимка описывают именно те байты, что стояли до продвижения.
    for name, data in before.items():
        assert retained.hashes[name] == hashlib.sha256(data).hexdigest()


def test_rollback_on_the_retained_snapshot_restores_the_pre_promotion_bytes(
    tmp_path: Path,
) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    _promote(snapshot, destinations, tmp_path)
    assert _live_state(destinations) != before

    retained = load_snapshot(tmp_path / "snapshots" / "snap-displaced")
    rollback(
        retained,
        destinations,
        confirmation_token=confirmation_token(retained, operation="rollback"),
    )

    assert _live_state(destinations) == before
    assert _sidecars(destinations) == []


def test_promote_retains_a_base_snapshot_when_the_destination_has_no_derived_data(
    tmp_path: Path,
) -> None:
    snapshot_root, _ = _build(tmp_path, complete=False)
    destinations, _ = _destinations(tmp_path, derived=False)
    snapshot = load_snapshot(snapshot_root)

    _promote(snapshot, destinations, tmp_path)

    assert load_snapshot(tmp_path / "snapshots" / "snap-displaced").tier is SnapshotTier.BASE


def test_promote_refuses_when_the_displaced_data_cannot_be_captured(tmp_path: Path) -> None:
    """Не собрать снимок «до» — значит откатываться будет некуда: продвигать нельзя."""
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)
    occupied = tmp_path / "snapshots" / "snap-displaced"
    occupied.mkdir(parents=True)

    with pytest.raises(PromotionError, match="уже существует"):
        _promote(snapshot, destinations, tmp_path)

    assert _live_state(destinations) == before
    assert _sidecars(destinations) == []


# --- профиль записи против свидетеля снимка -------------------------------------------


def test_promote_refuses_a_record_promotable_for_another_profile(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    foreign = PromotionRecord(
        state=PromotionState.PROMOTABLE,
        profile_id="profile-y",
        profile_sha256=PROFILE_SHA,
        geometry_witness_sha256=GEOMETRY_SHA,
        snapshot_result_sha256=snapshot.result_sha256,
    )

    with pytest.raises(CalibrationStateError, match="profile_id"):
        _promote(snapshot, destinations, tmp_path, record=foreign)

    assert _sidecars(destinations) == []


def test_promote_refuses_a_record_with_a_stale_geometry_witness(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    stale = PromotionRecord(
        state=PromotionState.PROMOTABLE,
        profile_id="profile-x",
        profile_sha256=PROFILE_SHA,
        geometry_witness_sha256="1" * 64,
        snapshot_result_sha256=snapshot.result_sha256,
    )

    with pytest.raises(CalibrationStateError, match="geometry_witness_sha256"):
        _promote(snapshot, destinations, tmp_path, record=stale)


# --- окно между двумя переименованиями ------------------------------------------------


def test_a_failing_second_rename_leaves_the_destination_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отказ ОС на втором переименовании не имеет права стереть назначение.

    Между двумя `os.replace` назначения не существует: старое уже в `.previous`, новое
    ещё в `.staging`. Именно здесь настоящая ошибка ОС (EACCES на открытом дереве кропов,
    ENOSPC на манифесте) способна уничтожить группу, и наружный `_restore` её не спасёт —
    группа ещё не числится переключённой.
    """
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)
    real_replace = os.replace

    def failing_replace(source: Any, target: Any) -> None:
        if Path(target) == destinations.crops and ".staging" in str(source):
            raise OSError(13, "инъекция отказа ОС на втором переименовании")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="инъекция отказа ОС"):
        _promote(snapshot, destinations, tmp_path)

    monkeypatch.undo()
    assert destinations.crops.is_dir(), "дерево кропов исчезло вместе с отказом"
    assert _live_state(destinations) == before
    assert not any(".staging" in name for name in _sidecars(destinations))
    assert not any(".previous" in name for name in _sidecars(destinations))


# --- CLI: устойчивость и предупреждение отката ----------------------------------------


def test_cli_reports_a_state_machine_refusal_as_an_exit_code(tmp_path: Path) -> None:
    """Самая вероятная ошибка оператора не должна выглядеть падением интерпретатора."""
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    record_path = tmp_path / "promotion.json"
    record_path.write_text(json.dumps({"state": "not_selected"}), encoding="utf-8")
    before = _live_state(destinations)

    assert main(_cli_promote_full_argv(snapshot_root, destinations, tmp_path, record_path)) == 2
    assert _live_state(destinations) == before


def test_cli_reports_a_corrupt_record_as_an_exit_code(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    record_path = tmp_path / "promotion.json"
    record_path.write_text("{не json", encoding="utf-8")

    assert main(_cli_promote_full_argv(snapshot_root, destinations, tmp_path, record_path)) == 2


def test_cli_promote_refuses_without_a_place_for_the_displaced_data(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)
    argv = [
        *_cli_promote_argv(snapshot_root, destinations),
        "--confirm",
        confirmation_token(snapshot, operation="promote"),
    ]

    assert main(argv) == 2
    assert _live_state(destinations) == before


def test_cli_rollback_names_the_stale_record_and_the_new_campaign_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    record_path = tmp_path / "promotion.json"
    _promote(snapshot, destinations, tmp_path)
    retained = load_snapshot(tmp_path / "snapshots" / "snap-displaced")
    write_record(
        record_path,
        PromotionRecord(
            state=PromotionState.PROMOTED,
            profile_id="profile-x",
            profile_sha256=PROFILE_SHA,
            geometry_witness_sha256=GEOMETRY_SHA,
            snapshot_result_sha256=snapshot.result_sha256,
            promotion_result_sha256="7" * 64,
        ),
    )
    capsys.readouterr()

    code = main(
        [
            "rollback",
            "--snapshot",
            str(tmp_path / "snapshots" / "snap-displaced"),
            *_cli_destination_argv(destinations),
            "--record",
            str(record_path),
            "--confirm",
            confirmation_token(retained, operation="rollback"),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "ВНИМАНИЕ" in out
    assert "НОВОЙ КАМПАНИЕЙ" in out
    assert str(record_path) in out
    assert "promoted" in out
    # Откат данные вернул, но запись автомата не тронул: обратного ребра у §11.3 нет.
    assert json.loads(record_path.read_text(encoding="utf-8"))["state"] == "promoted"


def test_cli_rollback_warns_even_without_a_named_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    previous_root, _ = _build(tmp_path, snapshot_id="snap-old", marker=b"old")
    destinations, _ = _destinations(tmp_path)
    previous = load_snapshot(previous_root)
    capsys.readouterr()

    code = main(
        [
            "rollback",
            "--snapshot",
            str(previous_root),
            *_cli_destination_argv(destinations),
            "--confirm",
            confirmation_token(previous, operation="rollback"),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "ВНИМАНИЕ" in out
    assert "--record" in out


def _previous_witnesses_file(tmp_path: Path) -> Path:
    path = tmp_path / "previous-witnesses.json"
    path.write_text(
        json.dumps(
            {
                "profile_id": "profile-displaced",
                "profile_sha256": "e" * 64,
                "geometry_witness_sha256": "f" * 64,
                "canonical_sha256": CANONICAL_SHA,
                "git_sha": GIT_SHA,
                "git_clean": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _cli_promote_full_argv(
    snapshot_root: Path, destinations: PromotionDestinations, tmp_path: Path, record: Path
) -> list[str]:
    snapshot = load_snapshot(snapshot_root)
    return [
        *_cli_promote_argv(snapshot_root, destinations),
        "--record",
        str(record),
        "--previous-root",
        str(tmp_path / "snapshots" / "snap-displaced"),
        "--previous-witnesses",
        str(_previous_witnesses_file(tmp_path)),
        "--confirm",
        confirmation_token(snapshot, operation="promote"),
    ]


# --- чистота дерева как вторая половина свидетеля версии кода -------------------------


def test_build_snapshot_requires_the_clean_tree_witness(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources")
    witnesses = {name: value for name, value in _witnesses(sources).items() if name != "git_clean"}
    with pytest.raises(PromotionError, match="git_clean"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=witnesses,
            created_at=CREATED_AT,
        )


def test_build_snapshot_rejects_a_stringly_typed_clean_flag(tmp_path: Path) -> None:
    """`"false"` — непустая строка, то есть истина при небрежном чтении. Нужен bool."""
    sources = _make_sources(tmp_path / "sources")
    witnesses = _witnesses(sources) | {"git_clean": "false"}
    with pytest.raises(PromotionError, match="git_clean"):
        build_snapshot(
            tmp_path / "snap",
            snapshot_id="snap",
            manifest_path=sources["manifest"],
            crops_root=sources["crops"],
            splits_root=sources["splits"],
            index_root=sources["index"],
            witnesses=witnesses,
            created_at=CREATED_AT,
        )


def test_a_dirty_tree_snapshot_records_the_flag_and_survives_the_round_trip(
    tmp_path: Path,
) -> None:
    """Собрать снимок грязным деревом можно: перебор профилей идёт до коммита."""
    sources = _make_sources(tmp_path / "sources")
    snapshot_root = tmp_path / "snapshots" / "snap-dirty"
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    build_snapshot(
        snapshot_root,
        snapshot_id="snap-dirty",
        manifest_path=sources["manifest"],
        crops_root=sources["crops"],
        splits_root=sources["splits"],
        index_root=sources["index"],
        witnesses=_witnesses(sources) | {"git_clean": False},
        created_at=CREATED_AT,
    )

    document = json.loads((snapshot_root / "provenance.json").read_text(encoding="utf-8"))
    assert document["input_witnesses"]["git_clean"] is False
    assert load_snapshot(snapshot_root).input_witnesses["git_clean"] is False


def test_promote_refuses_a_snapshot_built_from_a_dirty_tree(tmp_path: Path) -> None:
    """А вот боевыми данными результат грязного дерева не становится: воспроизвести нечем."""
    sources = _make_sources(tmp_path / "sources-dirty")
    snapshot_root = tmp_path / "snapshots" / "snap-dirty"
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    build_snapshot(
        snapshot_root,
        snapshot_id="snap-dirty",
        manifest_path=sources["manifest"],
        crops_root=sources["crops"],
        splits_root=sources["splits"],
        index_root=sources["index"],
        witnesses=_witnesses(sources) | {"git_clean": False},
        created_at=CREATED_AT,
    )
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    with pytest.raises(PromotionError, match="грязным деревом"):
        _promote(snapshot, destinations, tmp_path)

    assert _live_state(destinations) == before
    assert _sidecars(destinations) == []
    assert not (tmp_path / "snapshots" / "snap-displaced").exists()


def test_the_dirty_tree_refusal_reaches_the_cli_as_an_exit_code(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "sources-dirty")
    snapshot_root = tmp_path / "snapshots" / "snap-dirty"
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    build_snapshot(
        snapshot_root,
        snapshot_id="snap-dirty",
        manifest_path=sources["manifest"],
        crops_root=sources["crops"],
        splits_root=sources["splits"],
        index_root=sources["index"],
        witnesses=_witnesses(sources) | {"git_clean": False},
        created_at=CREATED_AT,
    )
    destinations, _ = _destinations(tmp_path)
    record_path = tmp_path / "promotion.json"
    write_record(record_path, _promotable(load_snapshot(snapshot_root).result_sha256))
    before = _live_state(destinations)

    assert main(_cli_promote_full_argv(snapshot_root, destinations, tmp_path, record_path)) == 2
    assert _live_state(destinations) == before


# --- CLI определяет обе половины свидетеля сам ----------------------------------------


def _fake_git(clean: bool) -> Callable[..., Any]:
    """Подставной git: `rev-parse` даёт SHA, `status --porcelain` — чистоту дерева."""
    real_run = subprocess.run

    def run(command: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(command, list) and command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"{GIT_SHA}\n", stderr="")
        if isinstance(command, list) and command[:2] == ["git", "status"]:
            output = "" if clean else " M pipeline/redaction_calibration/promotion.py\n"
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        return real_run(command, *args, **kwargs)

    return run


@pytest.mark.parametrize("clean", [True, False])
def test_cli_derives_both_halves_of_the_code_version_witness(
    monkeypatch: pytest.MonkeyPatch, clean: bool
) -> None:
    """Свидетель считается командой; опции, которой его можно объявить, не существует."""
    monkeypatch.setattr(subprocess, "run", _fake_git(clean))
    arguments = argparse.Namespace(
        profile_id="profile-x",
        profile_sha256=PROFILE_SHA,
        geometry_witness_sha256=GEOMETRY_SHA,
        canonical_sha256=CANONICAL_SHA,
        index_pool_sha256=None,
    )

    witnesses = _witnesses_from_arguments(arguments)

    assert witnesses["git_sha"] == GIT_SHA
    assert witnesses["git_clean"] is clean
    assert is_git_clean() is clean


def test_the_snapshot_command_has_no_option_to_declare_the_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _make_sources(tmp_path / "sources")
    monkeypatch.setattr(subprocess, "run", _fake_git(False))
    argv = [
        "snapshot",
        "--snapshot",
        str(tmp_path / "snapshots" / "snap-cli"),
        "--manifest",
        str(sources["manifest"]),
        "--crops",
        str(sources["crops"]),
        "--profile-id",
        "profile-x",
        "--profile-sha256",
        PROFILE_SHA,
        "--geometry-witness-sha256",
        GEOMETRY_SHA,
        "--canonical-sha256",
        CANONICAL_SHA,
    ]

    with pytest.raises(SystemExit):
        main([*argv, "--git-sha", "0" * 40])

    (tmp_path / "snapshots").mkdir(parents=True, exist_ok=True)
    assert main(argv) == 0
    written = load_snapshot(tmp_path / "snapshots" / "snap-cli")
    assert written.input_witnesses["git_sha"] == GIT_SHA
    assert written.input_witnesses["git_clean"] is False
