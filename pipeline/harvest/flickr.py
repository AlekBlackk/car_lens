"""Адаптер Flickr: второй источник харвеста под протоколом `HarvestSource`.

Всё знание про Flickr живёт здесь. `licenses.py` не трогается: его список закрыт
осознанно, и расширять его источником-специфичными кодами значило бы размазать
юридическое решение по двум модулям. Адаптер переводит числовой код выдачи в ту самую
строку, которую `tier_of` уже понимает, а всё непонятое не превращает в кандидата вовсе.

Интервал между запросами адаптер не считает: его держит транспорт одним пейсером на
домен, общим для поиска и скачивания (§3.2 SPEC, `UrllibTransport`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from pipeline.harvest.commons import HarvestHttpError, Transport, request_with_backoff
from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery, basis_of

# Отображение кодов Flickr в строки закрытого списка `licenses.py`. Отсутствующий здесь
# код — отказ, а не догадка: `7` («No known copyright restrictions») это заявление о
# неизвестности статуса, а не выдача прав, `8` («United States Government Work») — вопрос
# юрисдикции, которого закрытый список не решает.
_ЛИЦЕНЗИИ: dict[str, str] = {
    "9": "CC0 1.0",
    "10": "PDM 1.0",
    "4": "CC BY 2.0",
    "5": "CC BY-SA 2.0",
}

_БЕЗ_SA = ("9", "10", "4")
_ВКЛЮЧАЯ_SA = (*_БЕЗ_SA, "5")


def license_string(license_id: str) -> str | None:
    """Строка лицензии для `tier_of` или `None`, если код не разрешён решением Р-11.

    Разрешённость ShareAlike условна, но условие снимается не здесь: код `5` даёт
    `CC BY-SA 2.0` и без `--allow-sa`. Общий гейт приёма применяет флаг прогона —
    `process_candidate` (`run.py:137`) зовёт `is_allowed` с `allow_sa=deps.allow_sa` для
    кандидата любого источника, и Commons устроен тем же порядком (`commons.py:210` зовёт
    `is_allowed(..., allow_sa=True)` на своей стороне). Адаптеру звать `is_allowed` самому
    незачем: тир решает источник, режим прогона — общий код. Без флага код `5` у API и не
    запрашивается вовсе, см. `requested_licenses`.
    """
    return _ЛИЦЕНЗИИ.get(license_id.strip())


def requested_licenses(allow_sa: bool) -> str:
    """Значение параметра `license` запроса: несвободное не должно приезжать вовсе.

    Фильтр стоит дважды — здесь и при разборе ответа. Запрос экономит трафик и время
    человека, разбор защищает от того, что выдача однажды вернёт что-то за пределами
    запрошенного.
    """
    return ",".join(_ВКЛЮЧАЯ_SA if allow_sa else _БЕЗ_SA)


API_URL = "https://api.flickr.com/services/rest/"

# Порядок значим: берётся первый вариант, у которого есть и URL, и собственные размеры.
# Подставить размеры оригинала к ссылке на уменьшенную копию нельзя — гейт `MIN_SIDE`
# стоит до скачивания и сломался бы молча в обе стороны.
_РАЗМЕРЫ = ("o", "l", "c")


def _наибольший(фото: Mapping[str, Any]) -> tuple[str, int, int] | None:
    """URL и его родные размеры или `None`, если ни один вариант не полон."""
    for суффикс in _РАЗМЕРЫ:
        url = фото.get(f"url_{суффикс}")
        ширина = фото.get(f"width_{суффикс}")
        высота = фото.get(f"height_{суффикс}")
        if url and ширина and высота:
            try:
                return str(url), int(ширина), int(высота)
            except (TypeError, ValueError):
                continue
    return None


def search_text(фото: Mapping[str, Any]) -> str:
    """Что увидит шлюз игл: заголовок вместе с тегами.

    На Commons заголовок — каталожное имя; на Flickr это подпись владельца («DSC_0412»),
    а марка и модель живут в тегах. Вердикт `unknown` бракуется так же жёстко, как
    `foreign`, поэтому одного заголовка хватило бы на массовый ложный отказ. Цена лишнего
    кандидата — секунды человека в галерее аудита, цена потерянного — кадр, о котором
    никто никогда не узнает.
    """
    заголовок = str(фото.get("title") or "").strip()
    теги = str(фото.get("tags") or "").strip()
    return f"{заголовок} {теги}".strip()


class FlickrSource:
    """Поиск и скачивание в Flickr через REST API.

    Интервала между запросами источник не считает: его держит транспорт одним пейсером на
    домен. API и CDN картинок — разные домены, и пейсятся независимо, что и есть
    формулировка §3.2 SPEC «≤ 1 запрос / 2 сек на домен».
    """

    name = "flickr"

    def __init__(
        self,
        transport: Transport,
        api_key: str,
        allow_sa: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._api_key = api_key
        self._allow_sa = allow_sa
        self._sleep = sleep

    def search(self, query: SourceQuery) -> SearchPage:
        """Одна страница выдачи вместе с номером следующей."""
        ответ = request_with_backoff(
            self._transport, API_URL, self._параметры(query), sleep=self._sleep
        )
        данные: Any = json.loads(ответ.body.decode("utf-8"))
        блок = данные.get("photos") or {}
        основание = basis_of(query)
        кандидаты = tuple(
            собранный
            for фото in блок.get("photo", [])
            if (собранный := self._кандидат(фото, query.class_id, основание)) is not None
        )
        страница = int(блок.get("page", 1) or 1)
        всего = int(блок.get("pages", 1) or 1)
        токен = str(страница + 1) if страница < всего else None
        return SearchPage(кандидаты, токен)

    def fetch(self, candidate: Candidate) -> bytes:
        """Скачать файл. Пустые байты — запрос не удался, отказ решает вызывающий."""
        try:
            ответ = request_with_backoff(self._transport, candidate.file_url, {}, sleep=self._sleep)
        except HarvestHttpError:
            return b""
        return ответ.body

    def _параметры(self, query: SourceQuery) -> dict[str, str]:
        if not query.text:
            raise ValueError(f"{query.class_id}: у Flickr нет категорий, нужен текстовый запрос")
        параметры = {
            "method": "flickr.photos.search",
            "api_key": self._api_key,
            "text": query.text,
            "license": requested_licenses(self._allow_sa),
            "extras": "license,owner_name,tags,url_l,url_c,url_o",
            "per_page": str(query.limit),
            "media": "photos",
            "content_type": "1",
            "safe_search": "1",
            "sort": "relevance",
            "format": "json",
            "nojsoncallback": "1",
        }
        if query.continue_token:
            параметры["page"] = query.continue_token
        return параметры

    def _кандидат(self, фото: Mapping[str, Any], class_id: str, основание: str) -> Candidate | None:
        """Кандидат или `None`, если лицензия не разрешена либо размеров нет."""
        лицензия = license_string(str(фото.get("license", "")))
        if лицензия is None:
            return None
        размер = _наибольший(фото)
        if размер is None:
            return None
        url, ширина, высота = размер
        владелец = str(фото.get("owner", ""))
        идентификатор = str(фото.get("id", ""))
        return Candidate(
            source=self.name,
            source_item_id=идентификатор,
            class_id=class_id,
            page_url=f"https://www.flickr.com/photos/{владелец}/{идентификатор}",
            file_url=url,
            license_raw=лицензия,
            author=str(фото.get("ownername", "")),
            title=search_text(фото),
            width=ширина,
            height=высота,
            basis=основание,
        )
