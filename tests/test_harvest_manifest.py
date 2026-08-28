from pathlib import Path

import pytest

from pipeline.harvest.manifest import (
    COLUMNS,
    HarvestRow,
    ManifestError,
    append_row,
    holdout_candidates,
    load_manifest,
    strip_html,
    validate,
)


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
        "audit": "",
    }
    поля.update(правки)
    return HarvestRow(**поля)  # type: ignore[arg-type]


def test_запись_и_чтение_переживают_кириллицу_и_числа(tmp_path: Path) -> None:
    путь = tmp_path / "harvest_manifest.csv"
    append_row(путь, _строка())
    append_row(путь, _строка(file="cd.jpg", sha256="cd" * 32, phash=7))
    строки = load_manifest(путь)
    assert [c.strip() for c in путь.read_text(encoding="utf-8").splitlines()[0].split(",")] == list(
        COLUMNS
    )
    assert len(строки) == 2
    assert строки[0].author == "Иван Петров"
    assert строки[0].phash == -12345
    assert строки[1].clip_margin == pytest.approx(0.12)


def test_шапка_пишется_один_раз(tmp_path: Path) -> None:
    путь = tmp_path / "harvest_manifest.csv"
    append_row(путь, _строка())
    append_row(путь, _строка(file="cd.jpg", sha256="cd" * 32))
    assert путь.read_text(encoding="utf-8").count("class_id") == 1


def test_повтор_sha256_отвергается() -> None:
    with pytest.raises(ManifestError, match="повтор"):
        validate([_строка(), _строка(file="cd.jpg")], allow_sa=False)


def test_share_alike_без_флага_отвергается() -> None:
    строка = _строка(license="CC BY-SA 4.0", license_tier=3)
    with pytest.raises(ManifestError, match="CC BY-SA"):
        validate([строка], allow_sa=False)
    validate([строка], allow_sa=True)


def test_несвободная_лицензия_отвергается_всегда() -> None:
    with pytest.raises(ManifestError, match="не разрешена"):
        validate([_строка(license="CC BY-NC 4.0", license_tier=2)], allow_sa=True)


def test_тир_обязан_совпадать_с_лицензией() -> None:
    with pytest.raises(ManifestError, match="тир"):
        validate([_строка(license="CC BY 4.0", license_tier=1)], allow_sa=False)


def test_пустой_провенанс_отвергается() -> None:
    with pytest.raises(ManifestError, match="page_url"):
        validate([_строка(page_url="")], allow_sa=False)


def test_в_holdout_идут_только_аудированные() -> None:
    """Правило 2 Р-11: эталонный сплит держится на проверенных руками кадрах."""
    строки = [
        _строка(audit="ok"),
        _строка(file="cd.jpg", sha256="cd" * 32, audit=""),
        _строка(file="ef.jpg", sha256="ef" * 32, audit="drop:не та модель"),
        _строка(file="12.jpg", sha256="12" * 32, audit="conflict"),
    ]
    отобранные = holdout_candidates(строки)
    assert [с.file for с in отобранные] == ["ab.jpg"]


def test_неизвестная_отметка_аудита_отвергается() -> None:
    with pytest.raises(ManifestError, match="audit"):
        validate([_строка(audit="да вроде норм")], allow_sa=False)


def test_разметка_автора_снимается() -> None:
    """`extmetadata.Artist` приходит с HTML, а манифест — это текст для человека."""
    assert strip_html('<a href="/wiki/User:X" title="X">Иван Петров</a>') == "Иван Петров"
