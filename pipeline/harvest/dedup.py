"""Дедуп принятых фотографий по перцептивному хешу кропа.

Хеш считается по кропу, а не по кадру, — причина записана в `clean.py`: одинаковый фон
двух разных машин ближе по хешу кадра, чем две фотографии одной машины на разном фоне.

Дедуп двухуровневый, и второй уровень важнее первого. Один и тот же снимок, принятый в
два класса, — это **улика ошибки разметки**, а не дубль: молча оставлять одну из двух
строк нельзя, потому что неизвестно, какая права. Обе уходят человеку — если оба
заявления подкреплены одинаково.

Четвёртый исход — `supersede` — снимает часть этих споров автоматически. Категория
Commons размечена людьми, поисковая строка — нет: когда новый кандидат заявлен категорией,
а старая принятая строка держится только поисковым запросом, спор решён в пользу
разметки, и старая строка вытесняется, а не уходит в конфликт наравне с новой.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.clean import DEFAULT_PHASH_HAMMING
from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.sources import CATEGORY_BASIS_PREFIX, SEARCH_BASIS_PREFIX
from pipeline.hashing import hamming


@dataclass(frozen=True, slots=True)
class DedupVerdict:
    """Что делать с кандидатом: принять, отбросить как дубль или отдать в аудит."""

    status: str
    other: str | None


class DedupIndex:
    """Накопитель принятых хешей с проверкой близости и разбором спора классов."""

    def __init__(self, max_distance: int = DEFAULT_PHASH_HAMMING) -> None:
        self._max_distance = max_distance
        self._принятые: list[tuple[int, str, str, str]] = []

    @classmethod
    def from_manifest(
        cls, rows: Sequence[HarvestRow], max_distance: int = DEFAULT_PHASH_HAMMING
    ) -> DedupIndex:
        """Засеять индекс уже принятыми строками.

        Без засева второй прогон принимает во второй класс кадр, который первый прогон уже
        принял в первый: «кто собрался раньше» становится произвольным даже не в рамках
        линейки, а в рамках одного запуска.
        """
        индекс = cls(max_distance)
        for строка in rows:
            индекс.add(строка.phash, строка.class_id, строка.file, строка.basis)
        return индекс

    def check(self, phash: int, class_id: str, basis: str) -> DedupVerdict:
        """Сверить хеш с уже принятыми.

        Категория Commons размечена людьми, поисковая строка — нет. Когда один кадр заявлен
        двумя классами и одно заявление подкреплено человеческой разметкой, а второе строкой
        запроса, спор решён в пользу разметки. Когда оба подкреплены одинаково, спор не
        решён, и кадр уходит человеку.
        """
        for хеш, класс, файл, чужой_базис in self._принятые:
            if hamming(хеш, phash) > self._max_distance:
                continue
            if класс == class_id:
                return DedupVerdict("duplicate", файл)
            новый_категорийный = basis.startswith(CATEGORY_BASIS_PREFIX)
            старый_поисковый = чужой_базис.startswith(SEARCH_BASIS_PREFIX)
            if новый_категорийный and старый_поисковый:
                return DedupVerdict("supersede", файл)
            return DedupVerdict("conflict", файл)
        return DedupVerdict("new", None)

    def add(self, phash: int, class_id: str, file: str, basis: str) -> None:
        """Запомнить принятую фотографию."""
        self._принятые.append((phash, class_id, file, basis))
