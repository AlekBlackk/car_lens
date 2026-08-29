from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.notices import render_notices
from scripts.harvest import build_parser


def _строка(автор: str, лицензия: str, тир: int, файл: str) -> HarvestRow:
    return HarvestRow(
        file=файл,
        class_id="lada__granta",
        source="commons",
        source_item_id="1",
        page_url=f"https://commons.wikimedia.org/wiki/File:{файл}",
        file_url="https://upload.wikimedia.org/X.jpg",
        author=автор,
        license=лицензия,
        license_tier=тир,
        checked="2026-08-28",
        basis="category:Lada Granta",
        sha256="ab" * 32,
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=0,
        faces=0,
        audit="ok",
    )


def test_атрибуция_перечисляет_авторов_и_ссылки() -> None:
    текст = render_notices([_строка("Иван Петров", "CC BY 4.0", 2, "a.jpg")])
    assert "Иван Петров" in текст
    assert "CC BY 4.0" in текст
    assert "https://commons.wikimedia.org/wiki/File:a.jpg" in текст


def test_share_alike_вынесен_отдельным_разделом() -> None:
    """Tier 3 изолируется, чтобы риск снимался фильтром, а не пересбором выборки."""
    текст = render_notices(
        [
            _строка("Автор Один", "CC BY 4.0", 2, "a.jpg"),
            _строка("Автор Два", "CC BY-SA 4.0", 3, "b.jpg"),
        ]
    )
    сa_раздел = текст.split("ShareAlike")[1]
    assert "Автор Два" in сa_раздел
    assert "Автор Один" not in сa_раздел


def test_общественное_достояние_не_требует_атрибуции() -> None:
    текст = render_notices([_строка("Неизвестен", "CC0", 1, "a.jpg")])
    assert "CC0" in текст


def test_cli_разбирает_флаги() -> None:
    аргументы = build_parser().parse_args(
        ["--contact", "mail@example.com", "--classes", "lada__granta", "--allow-sa"]
    )
    assert аргументы.contact == "mail@example.com"
    assert аргументы.classes == ["lada__granta"]
    assert аргументы.allow_sa is True


def test_cli_требует_контакт() -> None:
    """Без контакта в User-Agent запросы к хранилищу отправлять нельзя (§3.2)."""
    парсер = build_parser()
    try:
        парсер.parse_args(["--classes", "lada__granta"])
    except SystemExit as выход:
        assert выход.code == 2
    else:  # pragma: no cover
        raise AssertionError("контакт обязан быть обязательным аргументом")


def test_cli_принимает_путь_аудита_и_режим_отчёта() -> None:
    аргументы = build_parser().parse_args(
        ["--contact", "m@e.com", "--audit", "verdicts.csv", "--report-only"]
    )
    assert аргументы.audit == "verdicts.csv"
    assert аргументы.report_only is True

