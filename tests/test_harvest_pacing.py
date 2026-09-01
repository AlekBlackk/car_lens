"""Интервал §3.2: ≤ 1 запрос / 2 с на домен, один пейсер на поиск, скачивание и обход."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from pipeline.harvest import backends
from pipeline.harvest.backends import UrllibTransport
from pipeline.harvest.commons import (
    API_URL,
    MIN_INTERVAL_SECONDS,
    CommonsSource,
    DomainPacer,
)
from pipeline.harvest.sources import Candidate, SourceQuery

ФАЙЛ = "https://upload.wikimedia.org/Lada_Granta.jpg"


class Часы:
    """Подставные монотонные часы: тест двигает время сам, а не ждёт секунды."""

    def __init__(self) -> None:
        self.сейчас = 0.0

    def __call__(self) -> float:
        return self.сейчас


class Ответ:
    """Минимальный ответ `urlopen`: контекстный менеджер со статусом и телом."""

    def __init__(self, тело: bytes) -> None:
        self.status = 200
        self._тело = тело

    def read(self) -> bytes:
        return self._тело

    def __enter__(self) -> Ответ:
        return self

    def __exit__(self, *_исключение: object) -> None:
        return None


def _подменить_сеть(monkeypatch: pytest.MonkeyPatch, тело: bytes) -> None:
    def открыть(_запрос: Any, timeout: float = 0.0) -> Ответ:
        return Ответ(тело)

    monkeypatch.setattr(backends, "urlopen", открыть)


def _пейсер(паузы: list[float], часы: Часы) -> DomainPacer:
    return DomainPacer(sleep=паузы.append, clock=часы)


def test_первый_запрос_к_домену_уходит_без_паузы() -> None:
    паузы: list[float] = []
    пейсер = _пейсер(паузы, Часы())

    пейсер.wait(API_URL)

    assert паузы == []


def test_второй_запрос_ждёт_остаток_интервала() -> None:
    паузы: list[float] = []
    часы = Часы()
    пейсер = _пейсер(паузы, часы)

    пейсер.wait(API_URL)
    пейсер.mark(API_URL)
    часы.сейчас = 0.5
    пейсер.wait(API_URL)

    assert паузы == [MIN_INTERVAL_SECONDS - 0.5]


def test_запрос_после_интервала_не_ждёт() -> None:
    паузы: list[float] = []
    часы = Часы()
    пейсер = _пейсер(паузы, часы)

    пейсер.mark(API_URL)
    часы.сейчас = MIN_INTERVAL_SECONDS
    пейсер.wait(API_URL)

    assert паузы == []


def test_разные_домены_не_ждут_друг_друга() -> None:
    """Ограничение §3.2 — на домен: страницы API не платят за скачивание файлов."""
    паузы: list[float] = []
    пейсер = _пейсер(паузы, Часы())

    пейсер.mark(API_URL)
    пейсер.wait(ФАЙЛ)

    assert паузы == []


def test_интервал_переживает_вызовы() -> None:
    """Суть находки I6: счётчик обнулялся на каждом вызове `expand`, и пауза не срабатывала.

    Пейсер один на прогон, поэтому три запроса подряд к одному домену дают две паузы,
    сколько бы разных вызывающих их ни отправляло.
    """
    паузы: list[float] = []
    пейсер = _пейсер(паузы, Часы())

    for _ in range(3):
        пейсер.wait(API_URL)
        пейсер.mark(API_URL)

    assert паузы == [MIN_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS]


def test_транспорт_пейсит_каждый_запрос(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пейсинг стоит в единственной точке, где запрос уходит в сеть."""
    _подменить_сеть(monkeypatch, b"{}")
    паузы: list[float] = []
    транспорт = UrllibTransport("почта@пример.рф", pacer=_пейсер(паузы, Часы()))
    параметры: Mapping[str, str] = {}

    транспорт.get(API_URL, параметры)
    транспорт.get(API_URL, параметры)

    assert паузы == [MIN_INTERVAL_SECONDS]


def test_провал_запроса_тоже_отмечается(monkeypatch: pytest.MonkeyPatch) -> None:
    """Неудачный запрос хранилище обслужило: частить после ошибки §3.2 не разрешает."""

    def упасть(_запрос: Any, timeout: float = 0.0) -> Ответ:
        raise TimeoutError

    monkeypatch.setattr(backends, "urlopen", упасть)
    паузы: list[float] = []
    транспорт = UrllibTransport("почта@пример.рф", pacer=_пейсер(паузы, Часы()))

    assert транспорт.get(API_URL, {}).status == 599
    транспорт.get(API_URL, {})

    assert паузы == [MIN_INTERVAL_SECONDS]


def test_поиск_пейсится_наравне_со_скачиванием(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CommonsSource.search` не держал интервала вовсе: счётчик обновлялся только в `fetch`.

    За прогон C1b это ~1271 категорийная страница и ~900 текстовых, ушедших пачками.
    """
    _подменить_сеть(monkeypatch, b'{"query": {"pages": {}}}')
    паузы: list[float] = []
    транспорт = UrllibTransport("почта@пример.рф", pacer=_пейсер(паузы, Часы()))
    источник = CommonsSource(транспорт, sleep=lambda _: None)
    запрос = SourceQuery("lada__granta", "Lada Granta", None, 50, None)

    источник.search(запрос)
    источник.search(запрос)

    assert паузы == [MIN_INTERVAL_SECONDS]


def test_поиск_и_скачивание_считают_разные_домены(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один пейсер, но поиск в API и скачивание файла — разные домены, а не общая очередь."""
    _подменить_сеть(monkeypatch, b'{"query": {"pages": {}}}')
    паузы: list[float] = []
    транспорт = UrllibTransport("почта@пример.рф", pacer=_пейсер(паузы, Часы()))
    источник = CommonsSource(транспорт, sleep=lambda _: None)

    источник.search(SourceQuery("lada__granta", "Lada Granta", None, 50, None))
    источник.fetch(
        Candidate(
            source="commons",
            source_item_id="1",
            class_id="lada__granta",
            page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
            file_url=ФАЙЛ,
            license_raw="CC0",
            author="А",
            title="File:X.jpg",
            width=1600,
            height=1200,
            basis="category:Lada Granta",
        )
    )

    assert паузы == []
