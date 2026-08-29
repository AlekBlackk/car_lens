# tests/test_harvest_audit.py
from pathlib import Path

import pytest

from pipeline.harvest.audit import (
    AuditError,
    apply_audit,
    audit_counts,
    load_audit,
    required_audit_files,
)
from pipeline.harvest.manifest import HarvestRow


def _строка(
    file: str,
    *,
    plates: int = 0,
    faces: int = 0,
    class_id: str = "kia__rio",
    audit: str = "",
) -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id="1",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="A",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis="category:X",
        sha256=file.split(".")[0].rjust(64, "0"),
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=plates,
        faces=faces,
        audit=audit,
    )


def test_вердикты_читаются(tmp_path: Path) -> None:
    p = tmp_path / "verdicts.csv"
    p.write_text("file,verdict\na.jpg,ok\nb.jpg,drop:plate\nc.jpg,conflict\n", encoding="utf-8")
    assert load_audit(p) == {"a.jpg": "ok", "b.jpg": "drop:plate", "c.jpg": "conflict"}


def test_неизвестный_вердикт_отвергается(tmp_path: Path) -> None:
    p = tmp_path / "verdicts.csv"
    p.write_text("file,verdict\na.jpg,maybe\n", encoding="utf-8")
    with pytest.raises(AuditError, match="maybe"):
        load_audit(p)


def test_apply_проставляет_отметку_и_выкидывает_dirty() -> None:
    строки = [_строка("a.jpg"), _строка("b.jpg", plates=1), _строка("c.jpg")]
    итог = apply_audit(строки, {"a.jpg": "ok", "b.jpg": "drop:plate"})
    файлы = {с.file: с.audit for с in итог}
    assert "b.jpg" not in файлы, "кадр с читаемым номером удаляется из манифеста"
    assert файлы["a.jpg"] == "ok"
    assert файлы["c.jpg"] == ""


def test_счётчики_h3_считают_dirty_до_удаления() -> None:
    строки = [
        _строка("a.jpg", audit="ok"),
        _строка("b.jpg", audit="drop:plate"),
        _строка("c.jpg", audit="drop:face"),
        _строка("d.jpg", audit="drop:wrong_model"),
        _строка("e.jpg", audit=""),
    ]
    total, dirty = audit_counts(строки)
    assert total == 4
    assert dirty == 2, "drop:wrong_model — не нарушение приватности, в dirty не идёт"


def test_обязательный_охват_аудита() -> None:
    строки = (
        [_строка(f"p{n}.jpg", plates=1) for n in range(5)]
        + [_строка(f"f{n}.jpg", faces=2) for n in range(5)]
        + [_строка(f"core{n}.jpg", class_id="lada__granta") for n in range(3)]
        + [_строка(f"clean{n}.jpg") for n in range(500)]
    )
    надо = required_audit_files(строки, seed=0)
    assert all(f"p{n}.jpg" in надо for n in range(5))
    assert all(f"f{n}.jpg" in надо for n in range(5))
    assert all(f"core{n}.jpg" in надо for n in range(3))
    assert sum(1 for f in надо if f.startswith("clean")) == 200
