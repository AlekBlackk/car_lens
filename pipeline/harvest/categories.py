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
from pipeline.harvest.commons import API_URL, MIN_INTERVAL_SECONDS, Transport, request_with_backoff

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
    rejected_not_own: int


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


def _допустить_всё(_имя: str) -> bool:
    return True


#: Модульная константа для тестов `expand`, которым правило допуска безразлично (глубина,
#: бюджет, стоп-слова, интервал) — `admit` обязателен именно затем, чтобы прод не мог его
#: забыть, а тестам на другое незачем собирать `alias_table` и `claimed`.
ADMIT_ALL: Callable[[str], bool] = _допустить_всё


def expand(
    transport: Transport,
    category: str,
    depth: int = 1,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
    *,
    admit: Callable[[str], bool],
) -> ExpandResult:
    """Категория и допущенные потомки, в порядке обхода в ширину.

    Родитель идёт первым всегда, даже если файлов у него нет: пустая выдача стоит один
    запрос и не стоит отдельной проверки. `include` пробивает все правила допуска —
    явно названная человеком категория обходу не подчиняется.

    `admit` решает, своя ли потомку полка (§5 дизайна C1: заявлена ли она другим классом и
    подтверждена ли иглой) — модуль сам этого решить не может, он не знает про классы.
    Параметр keyword-only и без умолчания: прод не должен иметь возможности забыть его и
    молча собрать дерево без проверки принадлежности.

    Между запросами держится тот же интервал, что у харвеста (`MIN_INTERVAL_SECONDS`,
    §3.2 SPEC): без него обход отправлял бы до `MAX_SUBCAT_REQUESTS` запросов встык.
    Ждёт только переданный `sleep` — тесты подставляют сюда заглушку и идут мгновенно.
    """
    запрещено = {normalize_filename(имя) for имя in exclude}
    итог: list[str] = [category]
    итог.extend(имя for имя in include if имя != category)
    видели = {normalize_filename(имя) for имя in итог}
    запросов = 0
    стоп_словом = 0
    исключением = 0
    не_своим = 0
    исчерпан = False
    последний_запрос = 0.0

    очередь: deque[tuple[str, int]] = deque([(category, 0)])
    while очередь:
        родитель, уровень = очередь.popleft()
        if уровень >= min(depth, MAX_DEPTH):
            continue
        if запросов >= MAX_SUBCAT_REQUESTS:
            исчерпан = True
            break
        прошло = time.monotonic() - последний_запрос
        if прошло < MIN_INTERVAL_SECONDS:
            sleep(MIN_INTERVAL_SECONDS - прошло)
        запросов += 1
        потомки = _подкатегории(transport, родитель, sleep)
        последний_запрос = time.monotonic()
        for потомок in потомки:
            имя = normalize_filename(потомок)
            if имя in видели:
                continue
            if имя in запрещено:
                исключением += 1
                continue
            if is_stopword_category(потомок):
                стоп_словом += 1
                continue
            if not admit(потомок):
                # Отказ считается, а не молчит: без счётчика потеря целой полки (как у
                # `Long-wheelbase Lada Nivas`, задача 12) не оставляла следа ни в прогоне,
                # ни в отчёте.
                не_своим += 1
                continue
            видели.add(имя)
            итог.append(потомок)
            очередь.append((потомок, уровень + 1))

    return ExpandResult(tuple(итог), запросов, стоп_словом, исключением, исчерпан, не_своим)
