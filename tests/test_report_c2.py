from pathlib import Path

import pytest

from pipeline.harvest.manifest import HarvestRow, write_manifest
from scripts.report_c2 import (
    G_C2_MAX_MISSING,
    DiscriminationRow,
    build_parser,
    discrimination_counts,
    gate_failures,
    load_discrimination,
    main,
    render,
)


def _строка(**правки: object) -> HarvestRow:
    поля: dict[str, object] = {
        "file": "ab.jpg",
        "class_id": "chevrolet__lanos",
        "source": "flickr",
        "source_item_id": "12345",
        "page_url": "https://www.flickr.com/photos/user/12345",
        "file_url": "https://live.staticflickr.com/12345.jpg",
        "author": "Иван Петров",
        "license": "CC BY 2.0",
        "license_tier": 2,
        "checked": "2026-09-02",
        "basis": "search:Chevrolet Lanos",
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
        "audit": "ok",
    }
    поля.update(правки)
    return HarvestRow(**поля)  # type: ignore[arg-type]


def test_порог_ноль_назначен_до_прогона() -> None:
    assert G_C2_MAX_MISSING == 0


def test_гейт_молчит_на_нуле() -> None:
    assert gate_failures(0) == []


def test_гейт_блокирует_на_любом_остатке() -> None:
    провалы = gate_failures(2)
    assert len(провалы) == 1
    assert "G_C2" in провалы[0]
    assert "2" in провалы[0]


def test_протокол_разведения_читается(tmp_path: Path) -> None:
    путь = tmp_path / "lanos_discrimination.csv"
    путь.write_text(
        "file,basis,note\n"
        "аа.jpg,badge,шильдик Chevrolet на решётке\n"
        "бб.jpg,context,украинский номер 2007 года\n"
        "вв.jpg,uncertain,корма, шильдика не видно\n",
        encoding="utf-8",
    )
    строки = load_discrimination(путь)
    assert len(строки) == 3
    assert строки[2].basis == "uncertain"
    assert строки[2].note == "корма, шильдика не видно"


def test_свод_считает_три_основания(tmp_path: Path) -> None:
    путь = tmp_path / "lanos_discrimination.csv"
    путь.write_text(
        "file,basis,note\nа,badge,\nб,badge,\nв,context,\nг,uncertain,\n", encoding="utf-8"
    )
    assert discrimination_counts(load_discrimination(путь)) == {
        "badge": 2,
        "context": 1,
        "uncertain": 1,
    }


def test_неизвестное_основание_отвергается(tmp_path: Path) -> None:
    """Свободная строка вместо словаря сделала бы число непосчитанным молча."""
    путь = tmp_path / "lanos_discrimination.csv"
    путь.write_text("file,basis,note\nа,точно она,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="точно она"):
        load_discrimination(путь)


def test_отсутствующий_файл_даёт_пустой_протокол(tmp_path: Path) -> None:
    """До ручного прохода файла нет, и отчёт обязан собираться без него."""
    assert load_discrimination(tmp_path / "нет.csv") == ()


def test_рендер_отчёта_с_дефицитом() -> None:
    протокол = [
        DiscriminationRow("1.jpg", "badge", "шильдик"),
        DiscriminationRow("2.jpg", "context", "номер"),
    ]
    текст = render(
        classes_without_holdout=["chevrolet__lanos", "lifan__x60"],
        holdout_frames=100,
        manifest_rows=200,
        discrimination=протокол,
        git_sha="abc1234",
        ran_at="2026-09-02",
    )
    assert "вердикт гейта: НЕ ПРОЙДЕН" in текст
    assert "- G_C2: классов без эталона — 2 > 0" in текст
    assert "| `badge` — виден шильдик или решётка | 1 |" in текст
    assert "| `context` — номер, год, география | 1 |" in текст
    assert "| `uncertain` — принят без разведения | 0 |" in текст
    assert "## Классы, оставшиеся без эталона" in текст
    assert "- `chevrolet__lanos`" in текст
    assert "- `lifan__x60`" in текст


def test_рендер_отчёта_без_дефицита() -> None:
    текст = render(
        classes_without_holdout=(),
        holdout_frames=120,
        manifest_rows=220,
        discrimination=(),
        git_sha="abc1234",
        ran_at="2026-09-02",
    )
    assert "вердикт гейта: пройден" in текст
    assert "## Блокирующие провалы\n\nНет." in текст
    assert "## Классы, оставшиеся без эталона" not in текст


def test_main_пишет_отчёт_и_возвращает_1_при_дефиците(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    write_manifest(манифест, [_строка()])
    дискриминация = tmp_path / "discrimination.csv"
    дискриминация.write_text("file,basis,note\nab.jpg,badge,шильдик\n", encoding="utf-8")
    выход = tmp_path / "bench" / "harvest-c2.md"

    код = main(
        [
            "--manifest",
            str(манифест),
            "--discrimination",
            str(дискриминация),
            "--out",
            str(выход),
            "--git-sha",
            "test_sha",
            "--ran-at",
            "2026-09-02",
        ]
    )
    assert код == 1
    assert выход.exists()
    содержимое = выход.read_text(encoding="utf-8")
    assert "НЕ ПРОЙДЕН" in содержимое
    assert "test_sha" in содержимое
    assert "- `chevrolet__lanos`" in содержимое


def test_main_возвращает_0_когда_дефицита_нет(tmp_path: Path) -> None:
    манифест = tmp_path / "manifest.csv"
    # Класс набирает >= 9 кадров (5 в эталон, 4 в пул) от разных авторов
    строки = [
        _строка(
            file=f"{i:02d}.jpg",
            source_item_id=str(i),
            sha256=f"{i:064x}",
            author=f"автор_{i}",
            audit="ok",
        )
        for i in range(10)
    ]
    write_manifest(манифест, строки)
    выход = tmp_path / "bench" / "harvest-c2.md"

    код = main(
        [
            "--manifest",
            str(манифест),
            "--out",
            str(выход),
            "--git-sha",
            "test_sha",
            "--ran-at",
            "2026-09-02",
        ]
    )
    assert код == 0
    assert выход.exists()
    содержимое = выход.read_text(encoding="utf-8")
    assert "вердикт гейта: пройден" in содержимое


def test_парсер_аргументов() -> None:
    парсер = build_parser()
    аргументы = парсер.parse_args(["--git-sha", "1234567", "--ran-at", "2026-09-02"])
    assert аргументы.git_sha == "1234567"
    assert аргументы.ran_at == "2026-09-02"
    assert аргументы.out == Path("bench/harvest-c2.md")
