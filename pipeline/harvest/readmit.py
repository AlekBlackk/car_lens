"""Переприём отбитых кандидатов: провенанс из журнала, пиксели из кэша.

Правило значимости остатка (`redact.py`) приняло кадры, которые прежняя редакция отбивала,
но приносить их некому: `rebuild_redaction.py` ходит по манифесту, а отбракованный кандидат
в манифесте не лежал никогда. Возврат идёт не повторным харвестом, и это замер, а не вкус:
метки кандидатов стёрла прополка 2026-09-02, классы закрыты стоп-критерием
(`accepted >= 30` у 49 из 60), а попадание в ту же выдачу не гарантировано уехавшими
токенами. Оригиналы при этом лежат в кэше все до одного, поэтому сеть нужна только на
метаданные страницы.

Цепочку решений модуль не повторяет: кандидатов ведёт `run.process_candidate`. Здесь только
источник, читающий кэш, префлайт и правка состояния.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.harvest.cache import cache_path
from pipeline.harvest.rejects import Reject
from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery


class CachedSource:
    """`HarvestSource`, отдающий байты из кэша оригиналов вместо сети."""

    def __init__(self, source: str, root: Path, sha256_by_page: Mapping[str, str]) -> None:
        # Имя обёрнутого источника: атрибут требует протокол `HarvestSource`, и без него
        # `mypy` не пустит класс в `HarvestDeps.source`. Экземплярный, а не классовый, как у
        # `CommonsSource`: какой источник обёрнут, известно только вызывающему.
        self.name = source
        self._root = root
        self._sha256_by_page = dict(sha256_by_page)

    def search(self, query: SourceQuery) -> SearchPage:
        """Переприём не ищет. Пустая страница была бы ложью про хранилище."""
        raise NotImplementedError("переприём читает кэш по известным pageid, а не ищет")

    def fetch(self, candidate: Candidate) -> bytes:
        """Байты оригинала из кэша. Пустые — файла нет, отказ решает цепочка."""
        sha = self._sha256_by_page.get(candidate.source_item_id, "")
        if not sha:
            return b""
        путь = cache_path(self._root, candidate.source, sha, "jpg")
        return путь.read_bytes() if путь.exists() else b""


@dataclass(frozen=True, slots=True)
class Readmission:
    """Один кандидат переприёма: провенанс из журнала, пиксели из кэша."""

    source_item_id: str
    class_id: str
    basis: str
    sha256: str


def plan_readmission(
    rejects: Sequence[Reject],
    sha256_by_page: Mapping[str, str],
    root: Path,
    source: str,
) -> tuple[tuple[Readmission, ...], tuple[str, ...]]:
    """Кандидаты, у которых есть и основание в журнале, и оригинал в кэше.

    Второй кортеж — отброшенные, строками `pageid: причина`. Поимённо, а не счётчиком: без
    имени непонятно, какую страницу качать и какой файл кэша чистить.

    Путь к оригиналу строит `cache_path` и только он: кэш шардирован по двум первым символам
    хеша, и склейка `raw/<source>/<sha>.jpg` забраковала бы целый кэш целиком.
    """
    готовые: list[Readmission] = []
    отброшенные: list[str] = []
    for отказ in rejects:
        sha = sha256_by_page.get(отказ.source_item_id, "")
        if not sha:
            отброшенные.append(f"{отказ.source_item_id}: нет sha256 в карте отбитых")
            continue
        if not cache_path(root, source, sha, "jpg").exists():
            отброшенные.append(f"{отказ.source_item_id}: нет файла в кэше ({sha[:8]})")
            continue
        готовые.append(Readmission(отказ.source_item_id, отказ.class_id, отказ.basis, sha))
    return tuple(готовые), tuple(отброшенные)
