"""Дедуп принятых фотографий по перцептивному хешу кропа.

Хеш считается по кропу, а не по кадру, — причина записана в `clean.py`: одинаковый фон
двух разных машин ближе по хешу кадра, чем две фотографии одной машины на разном фоне.

Дедуп двухуровневый, и второй уровень важнее первого. Один и тот же снимок, принятый в
два класса, — это **улика ошибки разметки**, а не дубль: молча оставлять одну из двух
строк нельзя, потому что неизвестно, какая права. Обе уходят человеку.

Сравнение линейное. Принятых фотографий по стоп-критерию не больше 15 × 150 = 2250, то
есть худший случай — два с половиной миллиона сравнений целых за весь прогон, на фоне
которого одна только детекция считает минуты. Корзинный метод `hashing.close_pairs`
здесь не нужен: он для пакетной обработки готовой выборки, а не для потока.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.clean import DEFAULT_PHASH_HAMMING
from pipeline.hashing import hamming


@dataclass(frozen=True, slots=True)
class DedupVerdict:
    """Что делать с кандидатом: принять, отбросить как дубль или отдать в аудит."""

    status: str
    other: str | None


class DedupIndex:
    """Накопитель принятых хешей с проверкой близости."""

    def __init__(self, max_distance: int = DEFAULT_PHASH_HAMMING) -> None:
        self._max_distance = max_distance
        self._принятые: list[tuple[int, str, str]] = []

    def check(self, phash: int, class_id: str) -> DedupVerdict:
        """Сверить хеш с уже принятыми."""
        for хеш, класс, файл in self._принятые:
            if hamming(хеш, phash) <= self._max_distance:
                return DedupVerdict("duplicate" if класс == class_id else "conflict", файл)
        return DedupVerdict("new", None)

    def add(self, phash: int, class_id: str, file: str) -> None:
        """Запомнить принятую фотографию."""
        self._принятые.append((phash, class_id, file))
