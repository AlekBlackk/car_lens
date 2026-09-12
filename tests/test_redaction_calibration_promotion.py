"""Снимок продвижения: полнота, журналируемая установка и полный откат."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline.redaction_calibration.promotion import (
    INDEX_FILES,
    SPLIT_FILES,
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
from scripts.promote_redaction_profile import main

CREATED_AT = "2026-09-13T00:00:00Z"
PROFILE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64
CANONICAL_SHA = "c" * 64


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


def _witnesses(sources: dict[str, Path]) -> dict[str, str]:
    return {
        "profile_id": "profile-x",
        "profile_sha256": PROFILE_SHA,
        "geometry_witness_sha256": GEOMETRY_SHA,
        "canonical_sha256": CANONICAL_SHA,
        "index_pool_sha256": _sha256(sources["splits"] / "pool.parquet"),
    }


def _build(
    tmp_path: Path,
    *,
    snapshot_id: str = "snap-new",
    marker: bytes = b"new",
    complete: bool = True,
    witnesses: dict[str, str] | None = None,
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
        promote(
            snapshot,
            destinations,
            record=_promotable(snapshot.result_sha256),
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
        promote(
            snapshot,
            destinations,
            record=record,
            confirmation_token=confirmation_token(snapshot, operation="promote"),
        )

    assert _sidecars(destinations) == []


def test_promote_requires_record_bound_to_this_snapshot(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    with pytest.raises(CalibrationStateError, match="snapshot_result_sha256"):
        promote(
            snapshot,
            destinations,
            record=_promotable("f" * 64),
            confirmation_token=confirmation_token(snapshot, operation="promote"),
        )


# --- установка ------------------------------------------------------------------------


def test_promote_installs_every_group_of_a_complete_snapshot(tmp_path: Path) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    result = promote(
        snapshot,
        destinations,
        record=_promotable(snapshot.result_sha256),
        confirmation_token=confirmation_token(snapshot, operation="promote"),
    )

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

    result = promote(
        snapshot,
        destinations,
        record=_promotable(snapshot.result_sha256),
        confirmation_token=confirmation_token(snapshot, operation="promote"),
    )

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
    promote(
        new,
        destinations,
        record=_promotable(new.result_sha256),
        confirmation_token=confirmation_token(new, operation="promote"),
    )

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
    "after_switch:manifest",
    "before_switch:crops",
    "after_switch:crops",
    "before_switch:splits",
    "after_switch:splits",
    "before_switch:index",
    "after_switch:index",
]


@pytest.mark.parametrize("step", STAGE_FAULTS + SWITCH_FAULTS)
def test_promote_leaves_destination_wholly_old_or_wholly_new(tmp_path: Path, step: str) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)
    before = _live_state(destinations)

    with pytest.raises(RuntimeError, match="инъекция отказа"):
        promote(
            snapshot,
            destinations,
            record=_promotable(snapshot.result_sha256),
            confirmation_token=confirmation_token(snapshot, operation="promote"),
            checkpoint=_fault(step),
        )

    after = _live_state(destinations)
    assert after == before, f"назначение смешалось после отказа на {step}"


@pytest.mark.parametrize("step", STAGE_FAULTS)
def test_promote_removes_staging_on_pre_switch_failure(tmp_path: Path, step: str) -> None:
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    with pytest.raises(RuntimeError):
        promote(
            snapshot,
            destinations,
            record=_promotable(snapshot.result_sha256),
            confirmation_token=confirmation_token(snapshot, operation="promote"),
            checkpoint=_fault(step),
        )

    assert not any(".staging" in name for name in _sidecars(destinations))


def test_promote_keeps_the_journal_when_a_switch_fails(tmp_path: Path) -> None:
    """Оборванное переключение обязано оставить след: журнал блокирует слепой повтор."""
    snapshot_root, _ = _build(tmp_path)
    destinations, _ = _destinations(tmp_path)
    snapshot = load_snapshot(snapshot_root)

    with pytest.raises(RuntimeError):
        promote(
            snapshot,
            destinations,
            record=_promotable(snapshot.result_sha256),
            confirmation_token=confirmation_token(snapshot, operation="promote"),
            checkpoint=_fault("after_switch:crops"),
        )

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


def _cli_promote_argv(snapshot_root: Path, destinations: PromotionDestinations) -> list[str]:
    assert destinations.splits is not None
    assert destinations.index is not None
    return [
        "promote",
        "--snapshot",
        str(snapshot_root),
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
