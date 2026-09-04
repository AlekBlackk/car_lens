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

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.harvest.audit import load_audit
from pipeline.harvest.cache import cache_path
from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.rejects import Reject
from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery
from pipeline.harvest.state import HarvestState


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


def patch_state(state: HarvestState, manifest_rows: Sequence[HarvestRow]) -> HarvestState:
    """Состояние после переприёма: `accepted` выводится из манифеста, метки добавляются.

    `manifest_rows` — **весь** манифест после дописывания, а не дельта прогона. Имя параметра
    называет это, потому что ошибка тихая по форме и громкая по последствиям: на дельте из
    ста строк `Counter` пересчитает `accepted` по затронутым классам, а остальные обнулит.

    Выводится, а не накапливается: сложение уехало бы при обрыве между записью манифеста и
    записью состояния, а вывод из строк идемпотентен — повторный прогон даёт то же самое.

    `seen`, `tokens` и `categories` не трогаются. Это и есть отличие от `prune_state.py`,
    который обнуляет их все и потому для возврата отбитых кадров не годится.
    """
    метки = {f"{строка.source}:{строка.source_item_id}" for строка in manifest_rows}
    return HarvestState(
        seen=dict(state.seen),
        accepted=dict(Counter(строка.class_id for строка in manifest_rows)),
        tokens=dict(state.tokens),
        done_items=state.done_items | метки,
        categories=dict(state.categories),
    )


def accepted_delta(
    state: HarvestState, manifest_rows: Sequence[HarvestRow]
) -> dict[str, tuple[int, int]]:
    """Классы, у которых `accepted` расходится с числом строк: `класс -> (было, станет)`."""
    станет = Counter(строка.class_id for строка in manifest_rows)
    классы = set(state.accepted) | set(станет)
    return {
        класс: (state.accepted.get(класс, 0), станет.get(класс, 0))
        for класс in sorted(классы)
        if state.accepted.get(класс, 0) != станет.get(класс, 0)
    }


def accepted_regressions(delta: Mapping[str, tuple[int, int]]) -> list[str]:
    """Классы, теряющие принятые кадры. Переприём на них останавливается без записи.

    Уменьшение означает расхождение манифеста со счётчиком — например, строки удалил аудит.
    Это предмет прополки, а не переприёма: молча переоткрывать чужой класс нельзя.
    """
    return [класс for класс, (было, станет) in sorted(delta.items()) if станет < было]


def merge_verdicts_file(source: Path, target: Path) -> int:
    """Перелить вердикты дедупа из временного файла в боевой. Возвращает число строк.

    Временный файл нужен потому, что `process_candidate` дописывает `supersede`/`conflict` сам,
    по ходу цикла: в боевом файле такой вердикт приговорил бы живую строку манифеста ещё до
    того, как переприём решил фиксироваться, и обрыв оставил бы приговор без новой строки
    взамен. Перелив идёт третьим шагом фиксации, после манифеста.

    Читается штатным `load_audit` — тот же формат `file,verdict`, что ждёт `--audit`. Пишется
    идиомой `run._записать_вердикт`: шапка появляется на пустом или отсутствующем файле, потому
    что журнал укорачивают `> файл`, и без проверки размера первая строка стала бы шапкой.
    """
    вердикты = load_audit(source) if source.exists() else {}
    if not вердикты:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    новый = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=["file", "verdict"])
        if новый:
            писатель.writeheader()
        for имя, вердикт in вердикты.items():
            писатель.writerow({"file": имя, "verdict": вердикт})
    return len(вердикты)
