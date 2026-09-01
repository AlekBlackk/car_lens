"""Решение «принадлежит ли эта подкатегория Commons нашему классу».

Живёт отдельным модулем, а не внутри `categories.py`, потому что решение требует и каталога
(кто ещё заявляет эту же категорию), и таблицы игл — а `categories.py` про классы не знает
по устройству своего модульного докстринга: обход получает готовый предикат и ничего не
решает сам. Отдельный модуль — способ дать `scripts/harvest.py` то, чем собрать предикат, не
протаскивая знание о классах в обход.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pipeline.harvest.aliases import AliasTable, alias_verdict, normalize_filename


def build_admit(
    class_id: str,
    alias_table: AliasTable,
    claimed: Mapping[str, frozenset[str]],
) -> Callable[[str], bool]:
    """Собрать предикат допуска подкатегории в обход `class_id`.

    Порядок проверок внутри предиката принципиален: сперва «заявлена другим классом», потом
    иглы. Заявленная другим классом полка отвергается, даже если иглы дали бы `own` —
    `Lada Niva Travel` ловится иглой `niva` класса `lada__niva_2121` (вердикт был бы `own`),
    но по `harvest_queries.yaml` это отдельный класс `lada__niva_travel`, и отвергает её
    именно пункт про заявителя, а не иглы. Свой `class_id` в `claimed` не мешает: множество
    заявителей уменьшается на него самого перед проверкой.
    """

    def допустить(имя: str) -> bool:
        нормальное = normalize_filename(имя)
        заявители = claimed.get(нормальное, frozenset()) - {class_id}
        if заявители:
            return False
        return alias_verdict(нормальное, class_id, alias_table).kind == "own"

    return допустить
