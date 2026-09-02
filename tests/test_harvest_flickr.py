import json
from collections.abc import Mapping

import pytest

from pipeline.harvest.commons import HttpResponse
from pipeline.harvest.flickr import FlickrSource, license_string, requested_licenses
from pipeline.harvest.licenses import LicenseTier, is_allowed, tier_of
from pipeline.harvest.sources import SourceQuery


@pytest.mark.parametrize(
    ("идентификатор", "ожидаемый_тир"),
    [
        ("9", LicenseTier.PUBLIC_DOMAIN),
        ("10", LicenseTier.PUBLIC_DOMAIN),
        ("4", LicenseTier.ATTRIBUTION),
        ("5", LicenseTier.SHARE_ALIKE),
        (" 9 ", LicenseTier.PUBLIC_DOMAIN),
    ],
)
def test_разрешённые_id_ложатся_в_известный_тир(
    идентификатор: str, ожидаемый_тир: LicenseTier
) -> None:
    """Строка адаптера обязана быть понятна `tier_of` без правок `licenses.py`."""
    строка = license_string(идентификатор)
    assert строка is not None
    assert tier_of(строка) is ожидаемый_тир


@pytest.mark.parametrize(
    "идентификатор", ["0", "1", "2", "3", "6", "7", "8", "", "  ", "42", "нет"]
)
def test_прочие_id_отвергаются(идентификатор: str) -> None:
    """`7` — заявление о неизвестности статуса, `8` — юрисдикция вне закрытого списка."""
    assert license_string(идентификатор) is None


def test_пятёрка_без_флага_отвергается_общим_гейтом() -> None:
    """Условную разрешённость SA решает не адаптер, а гейт приёма в `run.py`."""
    строка = license_string("5")
    assert строка is not None
    assert is_allowed(строка, allow_sa=False) is False
    assert is_allowed(строка, allow_sa=True) is True


def test_запрошенные_лицензии_без_sa_не_содержат_пятёрку() -> None:
    assert requested_licenses(allow_sa=False) == "9,10,4"


def test_запрошенные_лицензии_с_sa_содержат_пятёрку() -> None:
    assert requested_licenses(allow_sa=True) == "9,10,4,5"


ОТВЕТ = {
    "stat": "ok",
    "photos": {
        "page": 1,
        "pages": 3,
        "perpage": 2,
        "total": 5,
        "photo": [
            {
                "id": "111",
                "owner": "77777@N00",
                "ownername": "Иван Петров",
                "title": "DSC_0412",
                "tags": "chevrolet lanos zaz t150 запорожье",
                "license": "4",
                "url_o": "https://live.staticflickr.com/111_o.jpg",
                "width_o": "2048",
                "height_o": "1536",
                "url_l": "https://live.staticflickr.com/111_l.jpg",
                "width_l": "1024",
                "height_l": "768",
            },
            {
                "id": "222",
                "owner": "88888@N00",
                "ownername": "Кто-то",
                "title": "Daewoo Lanos NC",
                "tags": "daewoo lanos",
                "license": "2",
                "url_l": "https://live.staticflickr.com/222_l.jpg",
                "width_l": "1024",
                "height_l": "768",
            },
        ],
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


def _источник(
    ответы: list[HttpResponse], allow_sa: bool = False
) -> tuple[FlickrSource, ЗаписанныйТранспорт]:
    """Источник и его транспорт: проверять состав запроса надо, лезть в приватное — нет."""
    транспорт = ЗаписанныйТранспорт(ответы)
    return FlickrSource(транспорт, "КЛЮЧ", allow_sa=allow_sa, sleep=lambda _: None), транспорт


def test_несвободная_лицензия_не_становится_кандидатом() -> None:
    """Фильтр при разборе: даже если выдача вернула лишнее, оно не пройдёт."""
    источник, _ = _источник([_ок(ОТВЕТ)])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    assert [к.source_item_id for к in страница.candidates] == ["111"]


def test_кандидат_несёт_провенанс_и_наибольший_размер() -> None:
    источник, _ = _источник([_ок(ОТВЕТ)])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    кандидат = страница.candidates[0]
    assert кандидат.source == "flickr"
    assert кандидат.author == "Иван Петров"
    assert кандидат.page_url == "https://www.flickr.com/photos/77777@N00/111"
    assert кандидат.license_raw == "CC BY 2.0"
    assert кандидат.basis == "search:Chevrolet Lanos"
    # Наибольший доступный вариант и его собственные размеры: гейт MIN_SIDE стоит до
    # скачивания и сравнивает именно те числа, которые описывают скачиваемый файл.
    assert кандидат.file_url == "https://live.staticflickr.com/111_o.jpg"
    assert (кандидат.width, кандидат.height) == (2048, 1536)


def test_шлюз_игл_получает_заголовок_и_теги() -> None:
    """На Flickr марка живёт в тегах, а не в заголовке; ложный отказ невосстановим."""
    источник, _ = _источник([_ок(ОТВЕТ)])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    assert страница.candidates[0].title == "DSC_0412 chevrolet lanos zaz t150 запорожье"


def test_запрос_несёт_ключ_и_список_лицензий() -> None:
    источник, транспорт = _источник([_ок(ОТВЕТ)])
    источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    вызов = транспорт.вызовы[0]
    assert вызов["api_key"] == "КЛЮЧ"
    assert вызов["license"] == "9,10,4"
    assert вызов["text"] == "Chevrolet Lanos"
    assert вызов["method"] == "flickr.photos.search"


def test_токен_продолжения_есть_пока_есть_страницы() -> None:
    источник, _ = _источник([_ок(ОТВЕТ)])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    assert страница.continue_token == "2"


def test_на_последней_странице_токена_нет() -> None:
    последняя = {"stat": "ok", "photos": {"page": 3, "pages": 3, "photo": []}}
    источник, _ = _источник([_ок(последняя)])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, "3"))
    assert страница.continue_token is None


def test_токен_продолжения_доезжает_до_запроса() -> None:
    """Рестарт обязан продолжать с сохранённой страницы, а не начинать заново."""
    источник, транспорт = _источник([_ок(ОТВЕТ)])
    источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, "2"))
    assert транспорт.вызовы[0]["page"] == "2"


def test_фото_без_годного_размера_пропускается() -> None:
    без_размеров = {
        "stat": "ok",
        "photos": {
            "page": 1,
            "pages": 1,
            "photo": [
                {"id": "333", "owner": "9@N00", "ownername": "Х", "title": "т", "license": "4"}
            ],
        },
    }
    источник, _ = _источник([_ок(без_размеров)])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    assert страница.candidates == ()


def test_скачивание_возвращает_байты() -> None:
    источник, _ = _источник([_ок(ОТВЕТ), HttpResponse(200, b"JPEG")])
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    assert источник.fetch(страница.candidates[0]) == b"JPEG"


def test_неудачное_скачивание_даёт_пустые_байты() -> None:
    """Пустые байты — отказ решает вызывающий, как и у Commons."""
    источник, _ = _источник([_ок(ОТВЕТ)] + [HttpResponse(404, b"")] * 4)
    страница = источник.search(SourceQuery("chevrolet__lanos", None, "Chevrolet Lanos", 2, None))
    assert источник.fetch(страница.candidates[0]) == b""
