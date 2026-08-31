from pathlib import Path

from pipeline.harvest.manifest import HarvestRow, write_manifest
from pipeline.harvest.rejects import Reject, append_reject_row
from scripts.report_c1 import G_C1_MAX_MISSING, C1Numbers, check_c1, collect, main, render


def _строка(**правки: object) -> HarvestRow:
    поля: dict[str, object] = {
        "file": "ab.jpg",
        "class_id": "lada__granta",
        "source": "commons",
        "source_item_id": "12345",
        "page_url": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "file_url": "https://upload.wikimedia.org/X.jpg",
        "author": "Иван Петров",
        "license": "CC BY 4.0",
        "license_tier": 2,
        "checked": "2026-08-28",
        "basis": "category:Lada Granta",
        "sha256": "ab" * 32,
        "phash": -12345,
        "clip_rank": 4,
        "clip_margin": 0.12,
        "plates": 1,
        "faces": 0,
        "bbox_x": 12.0,
        "bbox_y": 34.0,
        "bbox_w": 500.0,
        "bbox_h": 300.0,
        "bbox_area_frac": 0.42,
        "det_conf": 0.91,
        "audit": "",
    }
    поля.update(правки)
    return HarvestRow(**поля)  # type: ignore[arg-type]


def _отказ(**правки: object) -> Reject:
    поля: dict[str, object] = {
        "class_id": "kia__rio",
        "basis": "category:Kia Rio",
        "source": "commons",
        "source_item_id": "1",
        "reason": "over_redacted",
        "checked": "2026-08-31",
    }
    поля.update(правки)
    return Reject(**поля)  # type: ignore[arg-type]


def _числа(**правки: object) -> C1Numbers:
    поля: dict[str, object] = {
        "rows": 10,
        "category_rows": 4,
        "category_share": 0.4,
        "classes_with_category": 3,
        "classes_without_holdout": 6,
        "conflicts": 0,
        "supersedes": 0,
        "reasons": {},
    }
    поля.update(правки)
    return C1Numbers(**поля)  # type: ignore[arg-type]


def _подготовить(tmp_path: Path) -> tuple[list[str], Path]:
    """Манифест, журнал отказов и вердикты дедупа в `tmp_path`; флаги путей и путь отчёта."""
    манифест = tmp_path / "harvest_manifest.csv"
    write_manifest(
        манифест,
        [
            _строка(basis="category:Kia Rio"),
            _строка(file="cd.jpg", sha256="cd" * 32, basis="search:Kia Rio"),
        ],
    )
    журнал = tmp_path / "reject_log.csv"
    append_reject_row(журнал, _отказ(reason="over_redacted"))
    append_reject_row(журнал, _отказ(reason="not_a_car_photo", source_item_id="2"))
    append_reject_row(журнал, _отказ(reason="not_a_car_photo", source_item_id="3"))
    вердикты = tmp_path / "verdicts.dedup.csv"
    вердикты.write_text(
        "file,verdict\na.jpg,conflict\nb.jpg,drop:superseded\nc.jpg,drop:superseded\n",
        encoding="utf-8",
    )
    отчёт = tmp_path / "harvest-c1.md"
    флаги = [
        "--report",
        str(отчёт),
        "--manifest",
        str(манифест),
        "--reject-log",
        str(журнал),
        "--dedup-verdicts",
        str(вердикты),
    ]
    return флаги, отчёт


def test_гейт_блокирует_по_классам_без_эталона() -> None:
    числа = collect(rows=[], rejects=[], verdicts={}, classes_without_holdout=16)

    прошёл, провалы = check_c1(числа)

    assert прошёл is False
    assert провалы == [f"G_C1: классов без эталона — 16 > {G_C1_MAX_MISSING}"]


def test_гейт_проходит_на_шести() -> None:
    числа = collect(rows=[], rejects=[], verdicts={}, classes_without_holdout=6)

    assert check_c1(числа) == (True, [])


def test_доля_категорийного_базиса_считается_но_не_блокирует() -> None:
    строки = [_строка(basis="category:Kia Rio"), _строка(basis="search:Kia Rio")]

    числа = collect(rows=строки, rejects=[], verdicts={}, classes_without_holdout=6)

    assert числа.category_rows == 1
    assert числа.category_share == 0.5
    assert check_c1(числа)[0] is True


def test_отчёт_на_красном_гейте_печатает_провал_и_вердикт() -> None:
    числа = _числа(classes_without_holdout=16)
    прошёл, провалы = check_c1(числа)

    текст = render(числа, прошёл, провалы, "abc1234", "2026-08-31")

    assert "- вердикт гейта: НЕ ПРОЙДЕН" in текст
    assert f"- G_C1: классов без эталона — 16 > {G_C1_MAX_MISSING}" in текст


def test_отчёт_на_зелёном_гейте_печатает_нет_вместо_провалов() -> None:
    числа = _числа(classes_without_holdout=6)
    прошёл, провалы = check_c1(числа)

    текст = render(числа, прошёл, провалы, "abc1234", "2026-08-31")

    assert "## Блокирующие провалы\n\nНет.\n" in текст
    assert "- вердикт гейта: пройден" in текст
    assert "НЕ ПРОЙДЕН" not in текст


def test_причины_отказа_идут_по_убыванию_числа_кандидатов() -> None:
    отказы = [
        _отказ(reason="alias_mismatch"),
        *(_отказ(reason="not_a_car_photo") for _ in range(3)),
        *(_отказ(reason="over_redacted") for _ in range(2)),
    ]
    числа = collect(rows=[], rejects=отказы, verdicts={}, classes_without_holdout=6)

    текст = render(числа, True, [], "abc1234", "2026-08-31")

    assert [с for с in текст.splitlines() if с.startswith("| `")] == [
        "| `not_a_car_photo` | 3 |",
        "| `over_redacted` | 2 |",
        "| `alias_mismatch` | 1 |",
    ]


def test_main_на_красном_гейте_пишет_отчёт_и_возвращает_1(tmp_path: Path) -> None:
    флаги, отчёт = _подготовить(tmp_path)

    код = main([*флаги, "--classes-without-holdout", "16"])

    assert код == 1
    текст = отчёт.read_text(encoding="utf-8")
    assert "- вердикт гейта: НЕ ПРОЙДЕН" in текст
    assert f"- G_C1: классов без эталона — 16 > {G_C1_MAX_MISSING}" in текст
    assert "| споров `conflict` | 0 | 1 |" in текст
    assert "| вытеснений `supersede` | 0 | 2 |" in текст
    assert "| `not_a_car_photo` | 2 |" in текст


def test_main_на_зелёном_гейте_возвращает_0(tmp_path: Path) -> None:
    флаги, отчёт = _подготовить(tmp_path)

    код = main([*флаги, "--classes-without-holdout", "6"])

    assert код == 0
    текст = отчёт.read_text(encoding="utf-8")
    assert "- вердикт гейта: пройден" in текст
    assert "- строк в манифесте: 2" in текст
    assert "50.0%" in текст


def test_main_без_файла_вердиктов_считает_их_пустыми(tmp_path: Path) -> None:
    флаги, отчёт = _подготовить(tmp_path)
    флаги[флаги.index("--dedup-verdicts") + 1] = str(tmp_path / "нет.csv")

    код = main([*флаги, "--classes-without-holdout", "6"])

    assert код == 0
    текст = отчёт.read_text(encoding="utf-8")
    assert "| споров `conflict` | 0 | 0 |" in текст
    assert "| вытеснений `supersede` | 0 | 0 |" in текст
