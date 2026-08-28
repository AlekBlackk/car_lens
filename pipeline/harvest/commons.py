"""Адаптер Wikimedia Commons к протоколу `HarvestSource`.

Сеть спрятана за `Transport`, поэтому тесты идут на записанных ответах и без единого
запроса — иначе они перестанут запускаться примерно через неделю.

Лицензия проверяется **до** скачивания: качать то, что нельзя использовать,
бессмысленно и невежливо к хранилищу.

Категория первична, полнотекстовый поиск — добор: категория Commons размечена людьми и
надёжнее строки запроса. Оба пути помечаются в `basis`, чтобы аудит знал, чему верить
меньше.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pipeline.harvest.licenses import is_allowed
from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery, basis_of

API_URL = "https://commons.wikimedia.org/w/api.php"
RETRY_STATUSES = frozenset({429, 503})
BACKOFF_SECONDS = (2.0, 4.0, 8.0)
MIN_INTERVAL_SECONDS = 2.0


class HarvestHttpError(RuntimeError):
    """Запрос не удался и повторять его больше нечем."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Ответ транспорта: код и тело, без разбора."""

    status: int
    body: bytes


class Transport(Protocol):
    """Минимальный HTTP-клиент. User-Agent с контактом (§3.2) задаётся реализацией."""

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:  # pragma: no cover
        ...


def request_with_backoff(
    transport: Transport,
    url: str,
    params: Mapping[str, str],
    sleep: Callable[[float], None] = time.sleep,
) -> HttpResponse:
    """Запрос с тремя попытками и паузами 2 / 4 / 8 с на кодах 429 и 503.

    Повторяются только эти два кода: они означают «сейчас занято». Повторять 404
    бессмысленно, он не пройдёт и на восьмой секунде, а лишние запросы к хранилищу —
    это ровно то, чего просит не делать §3.2.
    """
    последний = 0
    for попытка, пауза in enumerate(BACKOFF_SECONDS):
        ответ = transport.get(url, params)
        if ответ.status == 200:
            return ответ
        последний = ответ.status
        if ответ.status not in RETRY_STATUSES:
            break
        if попытка < len(BACKOFF_SECONDS) - 1:
            sleep(пауза)
    raise HarvestHttpError(f"{url}: код {последний}, попытки исчерпаны")


class CommonsSource:
    """Поиск и скачивание в Commons через Action API."""

    name = "commons"

    def __init__(
        self, transport: Transport, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self._transport = transport
        self._sleep = sleep
        self._последний_запрос = 0.0

    def search(self, query: SourceQuery) -> SearchPage:
        """Одна страница выдачи вместе с токеном продолжения."""
        ответ = request_with_backoff(
            self._transport, API_URL, self._параметры(query), sleep=self._sleep
        )
        данные: Any = json.loads(ответ.body.decode("utf-8"))
        основание = basis_of(query)
        кандидаты = tuple(
            self._кандидат(страница, query.class_id, основание)
            for страница in (данные.get("query") or {}).get("pages", {}).values()
            if self._годится(страница)
        )
        токен = (данные.get("continue") or {}).get("gcmcontinue") or (
            данные.get("continue") or {}
        ).get("gsroffset")
        return SearchPage(кандидаты, str(токен) if токен is not None else None)

    def fetch(self, candidate: Candidate) -> bytes:
        """Скачать файл, соблюдая паузу в две секунды между запросами (§3.2)."""
        прошло = time.monotonic() - self._последний_запрос
        if прошло < MIN_INTERVAL_SECONDS:
            self._sleep(MIN_INTERVAL_SECONDS - прошло)
        ответ = request_with_backoff(
            self._transport, candidate.file_url, {}, sleep=self._sleep
        )
        self._последний_запрос = time.monotonic()
        return ответ.body

    def _параметры(self, query: SourceQuery) -> dict[str, str]:
        общие = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
        }
        if query.category:
            общие |= {
                "generator": "categorymembers",
                "gcmtitle": f"Category:{query.category}",
                "gcmtype": "file",
                "gcmlimit": str(query.limit),
            }
            if query.continue_token:
                общие["gcmcontinue"] = query.continue_token
            return общие
        общие |= {
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query.text}",
            "gsrnamespace": "6",
            "gsrlimit": str(query.limit),
        }
        if query.continue_token:
            общие["gsroffset"] = query.continue_token
        return общие

    @staticmethod
    def _инфо(страница: Any) -> dict[str, Any]:
        сведения = страница.get("imageinfo") or [{}]
        первое: dict[str, Any] = сведения[0]
        return первое

    def _годится(self, страница: Any) -> bool:
        мета = self._инфо(страница).get("extmetadata") or {}
        лицензия = (мета.get("LicenseShortName") or {}).get("value", "")
        # `allow_sa=True`: тир решается здесь, а флаг прогона применяется в `run.py`,
        # иначе источник знал бы про режим запуска, который его не касается.
        return is_allowed(str(лицензия), allow_sa=True)

    def _кандидат(self, страница: Any, class_id: str, basis: str) -> Candidate:
        инфо = self._инфо(страница)
        мета = инфо.get("extmetadata") or {}
        return Candidate(
            source=self.name,
            source_item_id=str(страница.get("pageid")),
            class_id=class_id,
            page_url=str(инфо.get("descriptionurl", "")),
            file_url=str(инфо.get("url", "")),
            license_raw=str((мета.get("LicenseShortName") or {}).get("value", "")),
            author=str((мета.get("Artist") or {}).get("value", "")),
            title=str(страница.get("title", "")),
            width=int(инфо.get("width", 0)),
            height=int(инфо.get("height", 0)),
            basis=basis,
        )
