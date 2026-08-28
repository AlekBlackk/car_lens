import pytest

from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery, basis_of


def test_basis_у_категории() -> None:
    запрос = SourceQuery("lada__granta", category="Lada Granta", text=None, limit=50,
                         continue_token=None)
    assert basis_of(запрос) == "category:Lada Granta"


def test_basis_у_поиска() -> None:
    """Строки поиска аудируются в первую очередь, поэтому происхождение метки видно."""
    запрос = SourceQuery("lada__granta", category=None, text="Lada Granta", limit=50,
                         continue_token=None)
    assert basis_of(запрос) == "search:Lada Granta"


def test_запрос_без_категории_и_текста_отвергается() -> None:
    with pytest.raises(ValueError, match="ни категории, ни текста"):
        basis_of(SourceQuery("lada__granta", None, None, 50, None))


def test_кандидат_неизменяем() -> None:
    кандидат = Candidate(
        source="commons",
        source_item_id="12345",
        class_id="lada__granta",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        license_raw="CC BY-SA 4.0",
        author="Автор",
        title="File:X.jpg",
        width=1200,
        height=900,
        basis="category:Lada Granta",
    )
    with pytest.raises(AttributeError):
        кандидат.class_id = "uaz__patriot"  # type: ignore[misc]


def test_страница_несёт_токен_продолжения() -> None:
    страница = SearchPage(candidates=(), continue_token="gcmcontinue|123")
    assert страница.continue_token == "gcmcontinue|123"
