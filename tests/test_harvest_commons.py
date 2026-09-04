import json
from collections.abc import Mapping

import pytest

from pipeline.harvest.commons import (
    BACKOFF_SECONDS,
    CommonsSource,
    HarvestHttpError,
    HttpResponse,
    category_is_populated,
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
    страница = источник.search(SourceQuery("lada__granta", "Lada Granta", None, 50, None))
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


def test_непустая_категория_подтверждается() -> None:
    транспорт = ЗаписанныйТранспорт([_ок({"query": {"pages": {"1": {"pageid": 1}}}})])
    assert category_is_populated(транспорт, "Lada Granta", sleep=lambda _: None) is True


def test_категория_только_с_подкатегориями_подтверждается() -> None:
    """Категория жива, даже если файлы лежат в подкатегориях (напр. по годам/поколениям)."""
    транспорт = ЗаписанныйТранспорт(
        [
            _ок(
                {
                    "query": {
                        "pages": {"-1": {"ns": 14, "title": "Category:BMW 5 Series by generation"}}
                    }
                }
            )
        ]
    )
    assert category_is_populated(транспорт, "BMW 5 Series", sleep=lambda _: None) is True
    # Сторож стоит на запросе, а не на ответе: подставной транспорт отдаёт заготовку при
    # любом `gcmtype`, поэтому утверждение про `ns: 14` описывает форму ответа и регрессию
    # не ловит. Возврат к `gcmtype=file` — это 56 живых категорий, объявленных мёртвыми.
    assert транспорт.вызовы[-1]["gcmtype"] == "file|subcat"


def test_пустая_или_несуществующая_категория_отвергается() -> None:
    """Ответ без `pages` — это и «категории нет», и «в ней ничего»: разницы для нас нет."""
    транспорт = ЗаписанныйТранспорт([_ок({"batchcomplete": ""})])
    assert category_is_populated(транспорт, "VAZ-2113", sleep=lambda _: None) is False


def _страница(pageid: int, лицензия: str = "CC BY 4.0") -> dict[str, object]:
    return {
        "pageid": pageid,
        "title": f"File:Car {pageid}.jpg",
        "imageinfo": [
            {
                "url": f"https://upload.wikimedia.org/{pageid}.jpg",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{pageid}.jpg",
                "width": 1600,
                "height": 1200,
                "extmetadata": {
                    "LicenseShortName": {"value": лицензия},
                    "Artist": {"value": "Иван Петров"},
                },
            }
        ],
    }


def test_lookup_батчит_pageid_по_пятьдесят() -> None:
    страницы = {str(номер): ("lada__granta", "category:Lada Granta") for номер in range(1, 121)}
    партии = [list(страницы)[:50], list(страницы)[50:100], list(страницы)[100:]]
    ответы = [
        _ок({"query": {"pages": {ключ: _страница(int(ключ)) for ключ in партия}}})
        for партия in партии
    ]
    транспорт = ЗаписанныйТранспорт(ответы)
    кандидаты = CommonsSource(транспорт, sleep=lambda _: None).lookup(страницы)

    assert len(транспорт.вызовы) == 3
    assert транспорт.вызовы[0]["pageids"].count("|") == 49
    assert транспорт.вызовы[2]["pageids"] == "|".join(партии[2])
    assert len(кандидаты) == 120


def test_lookup_держит_порядок_ключей_а_не_ответа() -> None:
    страницы = {
        "300": ("lada__granta", "category:Lada Granta"),
        "100": ("uaz__patriot", "search:uaz patriot"),
        "200": ("lada__granta", "category:Lada Granta"),
    }
    # Порядок ключей в ответе API произволен и здесь намеренно не тот, что на входе.
    ответ = _ок(
        {"query": {"pages": {"100": _страница(100), "200": _страница(200), "300": _страница(300)}}}
    )
    кандидаты = CommonsSource(ЗаписанныйТранспорт([ответ]), sleep=lambda _: None).lookup(страницы)

    assert [кандидат.source_item_id for кандидат in кандидаты] == ["300", "100", "200"]
    assert кандидаты[1].class_id == "uaz__patriot"
    assert кандидаты[1].basis == "search:uaz patriot"


def test_lookup_роняет_страницу_с_несвободной_лицензией() -> None:
    страницы = {
        "1": ("lada__granta", "category:Lada Granta"),
        "2": ("lada__granta", "category:Lada Granta"),
    }
    ответ = _ок(
        {"query": {"pages": {"1": _страница(1), "2": _страница(2, лицензия="All rights reserved")}}}
    )
    кандидаты = CommonsSource(ЗаписанныйТранспорт([ответ]), sleep=lambda _: None).lookup(страницы)

    assert [кандидат.source_item_id for кандидат in кандидаты] == ["1"]
