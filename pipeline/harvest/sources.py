"""Граница между харвестом и хранилищем.

Ядро конвейера не знает, откуда пришёл кандидат: Commons сегодня, Flickr и Openverse
завтра (Р-11) — каждый добавляется одним файлом, без правки цепочки решений. Приём тот
же, что у `Detector` в `crop.py`.

Поиск отдаёт страницу, а не поток: токен продолжения обязан быть виден снаружи, иначе
его нечего сохранять в состояние прогона и рестарт начнёт выборку заново.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SourceQuery:
    """Один запрос к хранилищу: либо категория, либо текст, плюс место в пагинации."""

    class_id: str
    category: str | None
    text: str | None
    limit: int
    continue_token: str | None


@dataclass(frozen=True, slots=True)
class Candidate:
    """Кандидат до скачивания: провенанс есть, пикселей нет."""

    source: str
    source_item_id: str
    class_id: str
    page_url: str
    file_url: str
    license_raw: str
    author: str
    title: str
    width: int
    height: int
    basis: str


@dataclass(frozen=True, slots=True)
class SearchPage:
    """Страница выдачи и токен, с которого продолжать."""

    candidates: tuple[Candidate, ...]
    continue_token: str | None


class HarvestSource(Protocol):
    """Что угодно, умеющее искать и отдавать файлы по свободной лицензии."""

    name: str

    def search(self, query: SourceQuery) -> SearchPage:  # pragma: no cover
        ...

    def fetch(self, candidate: Candidate) -> bytes:  # pragma: no cover
        ...


SEARCH_BASIS_PREFIX = "search:"
CATEGORY_BASIS_PREFIX = "category:"


def basis_of(query: SourceQuery) -> str:
    """Основание разметки для манифеста: откуда взялась метка класса.

    Префиксы объявлены здесь, у производителя: какие строки считаются поисковыми, решают
    сразу два потребителя — шлюз на приёме (`run.py`) и триаж манифеста (`alignment.py`),
    — и разойтись им нельзя.
    """
    if query.category:
        return f"{CATEGORY_BASIS_PREFIX}{query.category}"
    if query.text:
        return f"{SEARCH_BASIS_PREFIX}{query.text}"
    raise ValueError(f"{query.class_id}: запрос без ни категории, ни текста")
