"""Разворачивание категории Commons в список подкатегорий, годных для сбора.

Commons раскладывает автомобили по поколениям: у `Kia Rio` ноль файлов на верхнем уровне
и шесть подкатегорий, у `Ford Focus` — ноль и восемь. Сбор `gcmtype: file` видит оглавление,
а не полку, и это причина, по которой у 82 классов ноль строк категорийного базиса (§2.1
дизайна C1).

Модуль не знает про классы: он получает имя категории и правила допуска. Знание о том,
какому классу принадлежит категория, живёт в `harvest_queries.yaml` и в `run.py`.

Бюджет запросов не украшение: `Lada Niva by country` разворачивается в дерево на сотни
узлов, и обход без потолка выест чужое хранилище за один класс.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pipeline.harvest.aliases import normalize_filename
from pipeline.harvest.commons import API_URL, Transport, request_with_backoff

MAX_SUBCAT_REQUESTS = 12
MAX_DEPTH = 2

# Не попытка перечислить весь мусор Commons, а защита от заноса в поддеревья, заведомо не
# дающие кадров каталожного домена: верная машина в чужом домене съёмки для CLIP законна.
SUBCAT_STOPWORDS = (
    "by country",
    "in competition",
    "rally",
    "racing",
    "race",
    "taxi",
    "police",
    "military",
    "ambulance",
    "fire",
    "crashed",
    "crash",
    "wrecked",
    "abandoned",
    "museum",
    "interior",
    "engine",
    "tuning",
    "models",
    "toys",
)


@dataclass(frozen=True, slots=True)
class ExpandResult:
    """Что дал обход и чего он стоил.

    Счётчики живут здесь, а не в вызывающем коде: только обход знает, что он отверг, а
    отчёт C1b обязан показать, чем именно защита от заноса срезала выборку.
    """

    categories: tuple[str, ...]
    requests: int
    rejected_stopword: int
    rejected_exclude: int
    budget_exhausted: bool


def _подкатегории(
    transport: Transport, category: str, sleep: Callable[[float], None]
) -> tuple[str, ...]:
    """Один запрос: имена подкатегорий данной категории."""
    ответ = request_with_backoff(
        transport,
        API_URL,
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "subcat",
            "gcmlimit": "50",
        },
        sleep=sleep,
    )
    данные: Any = json.loads(ответ.body.decode("utf-8"))
    страницы = (данные.get("query") or {}).get("pages") or {}
    return tuple(
        str(страница.get("title", "")).removeprefix("Category:") for страница in страницы.values()
    )


def is_stopword_category(name: str) -> bool:
    """Содержит ли имя категории маркер чужого домена съёмки."""
    нормальное = normalize_filename(name)
    return any(маркер in нормальное for маркер in SUBCAT_STOPWORDS)


def is_child_of(parent: str, child: str) -> bool:
    """Является ли категория подполкой родителя, а не перекрёстной ссылкой."""
    return normalize_filename(parent) in normalize_filename(child)


def expand(
    transport: Transport,
    category: str,
    depth: int = 1,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> ExpandResult:
    """Категория и допущенные потомки, в порядке обхода в ширину.

    Родитель идёт первым всегда, даже если файлов у него нет: пустая выдача стоит один
    запрос и не стоит отдельной проверки. `include` пробивает все правила допуска —
    явно названная человеком категория обходу не подчиняется.
    """
    запрещено = {normalize_filename(имя) for имя in exclude}
    итог: list[str] = [category]
    итог.extend(имя for имя in include if имя != category)
    видели = {normalize_filename(имя) for имя in итог}
    запросов = 0
    стоп_словом = 0
    исключением = 0
    исчерпан = False

    очередь: deque[tuple[str, int]] = deque([(category, 0)])
    while очередь:
        родитель, уровень = очередь.popleft()
        if уровень >= min(depth, MAX_DEPTH):
            continue
        if запросов >= MAX_SUBCAT_REQUESTS:
            исчерпан = True
            break
        запросов += 1
        for потомок in _подкатегории(transport, родитель, sleep):
            имя = normalize_filename(потомок)
            if имя in видели:
                continue
            if имя in запрещено:
                исключением += 1
                continue
            if is_stopword_category(потомок):
                стоп_словом += 1
                continue
            if not is_child_of(родитель, потомок):
                continue
            видели.add(имя)
            итог.append(потомок)
            очередь.append((потомок, уровень + 1))

    return ExpandResult(tuple(итог), запросов, стоп_словом, исключением, исчерпан)
