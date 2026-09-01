"""CLI харвеста: порядок слияния вердиктов и защита отметки человека.

`scripts/harvest.py::main` до этого не был покрыт ничем, а именно в нём живёт решение о
том, может ли автомат перебить человека (§8 дизайна C1). Прогон берётся в режиме
`--report-only`: он проходит весь путь вердиктов, но не требует ни сети, ни весов CLIP.
Все пути в теле `main` — модульные константы, поэтому тест подменяет их на `tmp_path`:
иначе он переписал бы настоящий манифест провенанса.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.harvest.audit import load_applied
from pipeline.harvest.manifest import HarvestRow, load_manifest, write_manifest
from scripts.harvest import main


def _строка(file: str, class_id: str = "kia__rio", audit: str = "") -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id=file.removesuffix(".jpg"),
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="A",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis="category:Kia Rio",
        sha256=file.removesuffix(".jpg").rjust(64, "0"),
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=0,
        faces=0,
        bbox_x=12.0,
        bbox_y=34.0,
        bbox_w=500.0,
        bbox_h=300.0,
        bbox_area_frac=0.42,
        det_conf=0.91,
        audit=audit,
    )


@pytest.fixture
def стенд(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Подменить все пути `main` на временные и вернуть их."""
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path / "data"))
    пути = {
        "манифест": tmp_path / "harvest_manifest.csv",
        "журнал": tmp_path / "dedup_verdicts.csv",
        "отчёт": tmp_path / "harvest-m0.md",
        "атрибуция": tmp_path / "NOTICES.md",
        "вердикты": tmp_path / "data" / "harvest" / "verdicts.dedup.csv",
    }
    monkeypatch.setattr("scripts.harvest.DEFAULT_MANIFEST_PATH", пути["манифест"])
    monkeypatch.setattr("scripts.harvest.DEFAULT_APPLIED_PATH", пути["журнал"])
    monkeypatch.setattr("scripts.harvest.REPORT_PATH", пути["отчёт"])
    monkeypatch.setattr("scripts.harvest.DEFAULT_NOTICES_PATH", пути["атрибуция"])
    пути["вердикты"].parent.mkdir(parents=True, exist_ok=True)
    return пути


def test_автовердикт_не_сносит_строку_с_отметкой_человека(стенд: dict[str, Path]) -> None:
    """Вытеснение проверенной строки — это удаление провенанса решением автомата."""
    write_manifest(стенд["манифест"], [_строка("a.jpg", audit="ok"), _строка("b.jpg", audit="")])
    стенд["вердикты"].write_text(
        "file,verdict\na.jpg,drop:superseded\nb.jpg,drop:superseded\n", encoding="utf-8"
    )

    main(["--contact", "mail@example.com", "--report-only"])

    отметки = {с.file: с.audit for с in load_manifest(стенд["манифест"])}
    assert отметки == {"a.jpg": "conflict"}, "проверенная строка цела, непроверенная снесена"


def test_вердикт_человека_побеждает_автовердикт(стенд: dict[str, Path]) -> None:
    """Порядок слияния: файл дедупа ранний, `--audit` поздний, поздний побеждает."""
    write_manifest(стенд["манифест"], [_строка("a.jpg", audit="")])
    стенд["вердикты"].write_text("file,verdict\na.jpg,drop:superseded\n", encoding="utf-8")
    ручной = стенд["манифест"].parent / "verdicts.csv"
    ручной.write_text("file,verdict\na.jpg,ok\n", encoding="utf-8")

    main(["--contact", "mail@example.com", "--report-only", "--audit", str(ручной)])

    отметки = {с.file: с.audit for с in load_manifest(стенд["манифест"])}
    assert отметки == {"a.jpg": "ok"}, "человек возвращает кадр, снесённый автоматом"


def test_применённый_автовердикт_попадает_в_журнал_и_усекается(стенд: dict[str, Path]) -> None:
    """След удаления обязан остаться в git, а сам вердикт — не примениться второй раз."""
    write_manifest(
        стенд["манифест"],
        [_строка("a.jpg", class_id="mitsubishi__pajero"), _строка("b.jpg", audit="ok")],
    )
    стенд["вердикты"].write_text(
        "file,verdict\na.jpg,drop:superseded\nb.jpg,drop:superseded\n", encoding="utf-8"
    )

    main(["--contact", "mail@example.com", "--report-only"])

    журнал = {з.file: (з.class_id, з.verdict) for з in load_applied(стенд["журнал"])}
    assert журнал["a.jpg"] == ("mitsubishi__pajero", "drop:superseded")
    assert журнал["b.jpg"][1] == "conflict", "в журнал идёт применённый вердикт, а не исходный"
    assert стенд["вердикты"].read_text(encoding="utf-8") == ""


def test_второй_прогон_не_сносит_принятый_заново_кадр(стенд: dict[str, Path]) -> None:
    """Цикл потери: прополка возвращает кадр в кандидаты, а старый вердикт ждёт его снова."""
    write_manifest(стенд["манифест"], [_строка("a.jpg"), _строка("b.jpg")])
    стенд["вердикты"].write_text("file,verdict\na.jpg,drop:superseded\n", encoding="utf-8")
    main(["--contact", "mail@example.com", "--report-only"])
    assert [с.file for с in load_manifest(стенд["манифест"])] == ["b.jpg"]

    # Кадр принят заново под тем же детерминированным именем `{sha}.jpg`.
    write_manifest(стенд["манифест"], [_строка("a.jpg"), _строка("b.jpg")])
    main(["--contact", "mail@example.com", "--report-only"])

    assert [с.file for с in load_manifest(стенд["манифест"])] == ["a.jpg", "b.jpg"]
