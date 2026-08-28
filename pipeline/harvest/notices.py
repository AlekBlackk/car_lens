"""Атрибуция авторов — обязательство Tier 2 решения Р-11.

Файл генерируется, а не ведётся руками: манифест и есть источник истины, а список из
двух тысяч строк, поддерживаемый вручную, разойдётся с ним на первой же правке.

Имя `NOTICES.md` взято из Р-11, но файл кладётся в `provenance/`, а не в корень: там
уже лежит `NOTICE.md` про сторонние компоненты (§15), и два файла в разницу одной буквы
— ловушка для читателя и для `git mv`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pipeline.harvest.manifest import HarvestRow

DEFAULT_NOTICES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "provenance" / "NOTICES.md"
)


def _раздел(заголовок: str, строки: Sequence[HarvestRow]) -> list[str]:
    if not строки:
        return []
    куски = [f"## {заголовок}", ""]
    for строка in sorted(строки, key=lambda с: (с.author, с.file)):
        куски.append(f"- {строка.author} — [{строка.file}]({строка.page_url}), {строка.license}")
    куски.append("")
    return куски


def render_notices(rows: Sequence[HarvestRow]) -> str:
    """Собрать текст атрибуции, разложив источники по тирам лицензий."""
    куски = [
        "# Атрибуция фотографий обучающей выборки",
        "",
        "Файл сгенерирован из `provenance/harvest_manifest.csv`. Правки вносятся в манифест,",
        "а не сюда.",
        "",
        f"Всего фотографий: {len(rows)}.",
        "",
    ]
    куски += _раздел(
        "Общественное достояние (атрибуция не требуется)",
        [с for с in rows if с.license_tier == 1],
    )
    куски += _раздел("Attribution (CC BY)", [с for с in rows if с.license_tier == 2])
    куски += _раздел("ShareAlike (CC BY-SA)", [с for с in rows if с.license_tier == 3])
    return "\n".join(куски)
