"""Разворачивание категории Commons в список подкатегорий, годных для сбора.

Commons раскладывает автомобили по поколениям: у `Kia Rio` ноль файлов на верхнем уровне
и шесть подкатегорий, у `Ford Focus` — ноль и восемь. Сбор `gcmtype: file` видит оглавление,
а не полку, и это причина, по которой у 82 классов ноль строк категорийного базиса (§2.1
дизайна C1).

Модуль не знает про классы: он получает имя категории и готовый предикат допуска
(`admit`). Знание о том, какому классу принадлежит категория, живёт в `harvest_queries.yaml`
и собирается в предикат в `scripts/harvest.py` (`admission.build_admit`); `run.py` только
проносит `class_id` до места сборки, сам решений не принимает.

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
# Сколько подкатегорий берёт один запрос. Продолжение не запрашивается — см. докстринг
# `_подкатегории`, — поэтому величина и есть граница, за которой список приходит усечённым.
SUBCAT_PAGE_LIMIT = 50

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
    # Сколько запросов вернулись усечёнными: у категории больше `SUBCAT_PAGE_LIMIT`
    # подкатегорий, а продолжение не запрашивается. Без счётчика остаток терялся бы молча —
    # тот самый класс дефекта («знаменатель может незаметно сжаться»), на устранение которого
    # ушли задачи 12 и 13.
    truncated: int = 0


def _подкатегории(
    transport: Transport, category: str, sleep: Callable[[float], None]
) -> tuple[tuple[str, ...], bool]:
    """Один запрос: имена подкатегорий и признак того, что список пришёл усечённым.

    Продолжение сознательно не запрашивается: страница подкатегорий — это `gcmlimit`
    штук, и вторая страница означала бы ещё один запрос к чужому хранилищу на категорию,
    у которой бюджет обхода (`MAX_SUBCAT_REQUESTS`) и так вычерпывается вглубь. Но
    молчать об усечении нельзя: потерянный остаток — это исчезнувший знаменатель, и
    отличить «подкатегорий больше нет» от «мы посмотрели пятьдесят» стало бы невозможно.
    Поэтому обход возвращает флаг, счётчик доезжает до `expand_log.csv` и до отчёта, и
    решение «пора листать» принимается по числу, а не по догадке. На снимке C1b усечения
    нет ни одного: максимум — 34 потомка у `Toyota Corolla`.
    """
    ответ = request_with_backoff(
        transport,
        API_URL,
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "subcat",
            "gcmlimit": str(SUBCAT_PAGE_LIMIT),
        },
        sleep=sleep,
    )
    данные: Any = json.loads(ответ.body.decode("utf-8"))
    страницы = (данные.get("query") or {}).get("pages") or {}
    усечено = bool((данные.get("continue") or {}).get("gcmcontinue"))
    return (
        tuple(
            str(страница.get("title", "")).removeprefix("Category:")
            for страница in страницы.values()
        ),
        усечено,
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

    Своего интервала обход не держит: интервал §3.2 (≤ 1 запрос / 2 с на домен) считает
    `DomainPacer` внутри транспорта — один на прогон, общий с поиском и скачиванием. Свой
    счётчик здесь обнулялся на каждом вызове, поэтому первый запрос вызова уходил без
    паузы, а в отгруженной конфигурации (`depth` не задан ни одному классу, значит 1)
    каждый вызов делает ровно один запрос — то есть пауза не срабатывала ни разу.
    `sleep` остаётся для пауз 2/4/8 с на кодах 429 и 503 внутри `request_with_backoff`.
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
    усечённых = 0

    очередь: deque[tuple[str, int]] = deque([(category, 0)])
    while очередь:
        родитель, уровень = очередь.popleft()
        if уровень >= min(depth, MAX_DEPTH):
            continue
        if запросов >= MAX_SUBCAT_REQUESTS:
            исчерпан = True
            break
        запросов += 1
        потомки, усечено = _подкатегории(transport, родитель, sleep)
        усечённых += int(усечено)
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

    return ExpandResult(
        tuple(итог), запросов, стоп_словом, исключением, исчерпан, не_своим, усечённых
    )
