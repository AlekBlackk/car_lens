"""Адаптер Wikimedia Commons к протоколу `HarvestSource`.

Сеть спрятана за `Transport`, поэтому тесты идут на записанных ответах и без единого
запроса — иначе они перестанут запускаться примерно через неделю.

Лицензия проверяется **до** скачивания: качать то, что нельзя использовать,
бессмысленно и невежливо к хранилищу.

Категория первична, полнотекстовый поиск — добор: категория Commons размечена людьми и
надёжнее строки запроса. Оба пути помечаются в `basis`, чтобы аудит знал, чему верить
меньше.

Интервал между запросами (§3.2: ≤ 1 запрос / 2 с на домен) держит `DomainPacer` — один
на прогон, общий для поиска, скачивания и обхода категорий, и вызывает его транспорт,
единственное место, где запрос действительно уходит в сеть.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from pipeline.harvest.licenses import is_allowed
from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery, basis_of

API_URL = "https://commons.wikimedia.org/w/api.php"
# Предел `pageids` у Action API для анонимного клиента — 50 идентификаторов на запрос.
LOOKUP_BATCH = 50
RETRY_STATUSES = frozenset({429, 503})
BACKOFF_SECONDS = (2.0, 4.0, 8.0)
MIN_INTERVAL_SECONDS = 2.0


class HarvestHttpError(RuntimeError):
    """Запрос не удался и повторять его больше нечем."""


class DomainPacer:
    """Один интервал между запросами к одному домену: ≤ 1 запрос / 2 с (§3.2 SPEC).

    Пейсер один на весь прогон и общий для всех путей — поиска, скачивания и обхода
    категорий. До него интервал считался в двух местах по отдельности, и оба не работали:
    `CommonsSource.search` не держал его вовсе (счётчик обновлялся только в `fetch`), а
    `expand` заводил счётчик заново на каждый вызов, поэтому первый запрос вызова уходил
    без паузы — а в отгруженной конфигурации (`depth` не задан ни одному классу, значит 1)
    каждый вызов делает ровно один запрос, то есть пауза не срабатывала ни разу. Отсюда и
    форма правки: состояние обязано жить дольше вызова, а вежливость — не зависеть от того,
    вспомнил ли о ней очередной вызывающий.

    Ключ — домен, а не адрес: ограничение §3.2 сформулировано на домен, и страницы API
    (`commons.wikimedia.org`) не должны копить долг из-за скачиваний файлов
    (`upload.wikimedia.org`).

    Отметка ставится **после** ответа (`mark`), а не перед запросом: интервал получается
    «старт-до-старта» длиннее нужного на время самого запроса. Это консервативно в нужную
    сторону, и менять это здесь не задача.

    Часы и сон впрыскиваются: тест обязан мерить арифметику интервала, а не ждать секунды.
    """

    def __init__(
        self,
        min_interval: float = MIN_INTERVAL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._последний: dict[str, float] = {}

    def wait(self, url: str) -> None:
        """Додержать интервал перед запросом к домену `url`. Первый запрос уходит сразу."""
        домен = self._домен(url)
        если_был = self._последний.get(домен)
        if если_был is None:
            return
        прошло = self._clock() - если_был
        if прошло < self._min_interval:
            self._sleep(self._min_interval - прошло)

    def mark(self, url: str) -> None:
        """Отметить, что запрос к домену `url` только что завершился."""
        self._последний[self._домен(url)] = self._clock()

    @staticmethod
    def _домен(url: str) -> str:
        return urlsplit(url).netloc.lower()


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
    """Поиск и скачивание в Commons через Action API.

    Интервала между запросами источник не считает: его держит транспорт одним пейсером на
    домен (`DomainPacer`), общим с обходом категорий. `sleep` остаётся — им пользуется
    `request_with_backoff` для пауз 2/4/8 с на кодах 429 и 503, а это другая величина.
    """

    name = "commons"

    def __init__(self, transport: Transport, sleep: Callable[[float], None] = time.sleep) -> None:
        self._transport = transport
        self._sleep = sleep

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
        """Скачать файл. Пустые байты — запрос не удался, отказ решает вызывающий."""
        try:
            ответ = request_with_backoff(self._transport, candidate.file_url, {}, sleep=self._sleep)
        except HarvestHttpError:
            return b""
        return ответ.body

    def lookup(self, pages: Mapping[str, tuple[str, str]]) -> tuple[Candidate, ...]:
        """Кандидаты по известным `pageid`: переприём не ищет, а перечитывает страницы.

        Ключ — `pageid`, значение — `(class_id, basis)`: и класс, и основание известны из
        журнала отказов, а Commons про них не знает.

        Порядок результата — порядок ключей, а не ответа API: страницы в JSON приходят как
        попало, а прогон обязан воспроизводиться. Страница, которой API не отдал (файл
        удалён) или которую отсеяла лицензия, в результат не попадает; недостачу считает
        вызывающий по разнице ключей — молчаливая потеря выглядела бы как отказ цепочки.
        """
        ключи = list(pages)
        собрано: dict[str, Candidate] = {}
        for начало in range(0, len(ключи), LOOKUP_BATCH):
            батч = ключи[начало : начало + LOOKUP_BATCH]
            ответ = request_with_backoff(
                self._transport,
                API_URL,
                {
                    "action": "query",
                    "format": "json",
                    "pageids": "|".join(батч),
                    "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata",
                },
                sleep=self._sleep,
            )
            данные: Any = json.loads(ответ.body.decode("utf-8"))
            for ключ, страница in ((данные.get("query") or {}).get("pages") or {}).items():
                заявка = pages.get(str(ключ))
                if заявка is None or not self._годится(страница):
                    continue
                class_id, основание = заявка
                собрано[str(ключ)] = self._кандидат(страница, class_id, основание)
        return tuple(собрано[ключ] for ключ in ключи if ключ in собрано)

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
            author=str(
                (мета.get("Artist") or {}).get("value", "")
                or (мета.get("Credit") or {}).get("value", "")
                or "Неизвестен"
            ),
            title=str(страница.get("title", "")),
            width=int(инфо.get("width", 0)),
            height=int(инфо.get("height", 0)),
            basis=basis,
        )


def category_is_populated(
    transport: Transport, category: str, sleep: Callable[[float], None] = time.sleep
) -> bool:
    """Есть ли в категории Commons хотя бы один файл или подкатегория.

    Существование и непустота здесь одно и то же: класс, чья категория пуста, и класс,
    чьей категории нет, получают одинаковый ноль принятых строк, и лечатся оба правкой
    имени в `harvest_queries.yaml`. Наличие подкатегорий означает живую категорию
    (файлы разложены по поколениям/годам), требующую рекурсивного сбора.
    """
    ответ = request_with_backoff(
        transport,
        API_URL,
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file|subcat",
            "gcmlimit": "1",
        },
        sleep=sleep,
    )
    данные: Any = json.loads(ответ.body.decode("utf-8"))
    return bool((данные.get("query") or {}).get("pages"))
