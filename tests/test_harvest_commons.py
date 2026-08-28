import json
from collections.abc import Mapping

import pytest

from pipeline.harvest.commons import (
    BACKOFF_SECONDS,
    CommonsSource,
    HarvestHttpError,
    HttpResponse,
    request_with_backoff,
)
from pipeline.harvest.sources import SourceQuery

ОТВЕТ = {
    "continue": {"gcmcontinue": "page|0810|999"},
    "query": {
        "pages": {
            "12345": {
                "pageid": 12345,
                "title": "File:Lada Granta 2019.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/Lada_Granta.jpg",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Lada.jpg",
                        "width": 1600,
                        "height": 1200,
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "Иван Петров"},
                        },
                    }
                ],
            },
            "67890": {
                "pageid": 67890,
                "title": "File:Granta NC.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/Granta_NC.jpg",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:NC.jpg",
                        "width": 1600,
                        "height": 1200,
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-NC 4.0"},
                            "Artist": {"value": "Кто-то"},
                        },
                    }
                ],
            },
        }
    },
}


class ЗаписанныйТранспорт:
    """Подставной транспорт: отдаёт заготовленные ответы, считает вызовы."""

    def __init__(self, ответы: list[HttpResponse]) -> None:
        self.ответы = list(ответы)
        self.вызовы: list[Mapping[str, str]] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.вызовы.append(dict(params))
        return self.ответы.pop(0)


def _ок(данные: object) -> HttpResponse:
    return HttpResponse(200, json.dumps(данные).encode("utf-8"))


def test_backoff_повторяет_на_429_и_возвращает_ответ() -> None:
    транспорт = ЗаписанныйТранспорт(
        [HttpResponse(429, b""), HttpResponse(503, b""), _ок({"ok": True})]
    )
    паузы: list[float] = []
    ответ = request_with_backoff(транспорт, "https://x.invalid", {}, sleep=паузы.append)
    assert ответ.status == 200
    assert паузы == list(BACKOFF_SECONDS[:2])


def test_backoff_сдаётся_после_трёх_попыток() -> None:
    транспорт = ЗаписанныйТранспорт([HttpResponse(503, b"")] * 4)
    with pytest.raises(HarvestHttpError, match="503"):
        request_with_backoff(транспорт, "https://x.invalid", {}, sleep=lambda _: None)


def test_ошибка_не_из_списка_не_повторяется() -> None:
    """404 повторять бессмысленно: он не пройдёт и на восьмой секунде."""
    транспорт = ЗаписанныйТранспорт([HttpResponse(404, b"")])
    with pytest.raises(HarvestHttpError, match="404"):
        request_with_backoff(транспорт, "https://x.invalid", {}, sleep=lambda _: None)
    assert len(транспорт.вызовы) == 1


def test_поиск_по_категории_отдаёт_кандидатов_и_токен() -> None:
    источник = CommonsSource(ЗаписанныйТранспорт([_ок(ОТВЕТ)]), sleep=lambda _: None)
    страница = источник.search(
        SourceQuery("lada__granta", "Lada Granta", None, 50, None)
    )
    assert страница.continue_token == "page|0810|999"
    assert len(страница.candidates) == 1, "кандидат под NC обязан отсеяться до скачивания"
    кандидат = страница.candidates[0]
    assert кандидат.source_item_id == "12345"
    assert кандидат.class_id == "lada__granta"
    assert кандидат.license_raw == "CC BY-SA 4.0"
    assert кандидат.author == "Иван Петров"
    assert кандидат.basis == "category:Lada Granta"


def test_токен_продолжения_уходит_в_параметры() -> None:
    транспорт = ЗаписанныйТранспорт([_ок(ОТВЕТ)])
    источник = CommonsSource(транспорт, sleep=lambda _: None)
    источник.search(SourceQuery("lada__granta", "Lada Granta", None, 50, "page|0810|999"))
    assert транспорт.вызовы[0]["gcmcontinue"] == "page|0810|999"


def test_разметка_из_поиска_помечается_как_search() -> None:
    источник = CommonsSource(ЗаписанныйТранспорт([_ок(ОТВЕТ)]), sleep=lambda _: None)
    страница = источник.search(SourceQuery("lada__granta", None, "Lada Granta", 50, None))
    assert страница.candidates[0].basis == "search:Lada Granta"
