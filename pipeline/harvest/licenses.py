"""Тиры лицензий по правилу 5 решения Р-11.

Список закрытый и правится осознанно: формулировок свободных лицензий много, и
«похоже на CC-BY» — это не проверка, а догадка. Неизвестная строка означает отказ,
потому что цена ошибки здесь юридическая, а не метрическая.

Tier 3 (ShareAlike) вынесен за флаг и помечается в манифесте отдельно ровно затем,
чтобы риск снимался фильтром по колонке, а не пересбором выборки.
"""

from __future__ import annotations

import re
from enum import IntEnum


class LicenseTier(IntEnum):
    """Тир лицензии: чем меньше номер, тем меньше обязательств."""

    PUBLIC_DOMAIN = 1
    ATTRIBUTION = 2
    SHARE_ALIKE = 3


_ОБЩЕСТВЕННОЕ = {"cc0", "cc0 1.0", "public domain", "pd", "pdm", "pdm 1.0"}
_ВЕРСИИ = ("2.0", "3.0", "4.0")
_ПРОБЕЛЫ = re.compile(r"[\s_-]+")


def _нормализовать(raw: str) -> str:
    """Свести строку к виду «cc by sa 4.0»: регистр, дефисы и пробелы не значимы."""
    return _ПРОБЕЛЫ.sub(" ", raw.strip().lower())


def tier_of(raw: str) -> LicenseTier | None:
    """Тир лицензии или `None`, если она не разрешена решением Р-11."""
    строка = _нормализовать(raw)
    if not строка:
        return None
    if строка in _ОБЩЕСТВЕННОЕ:
        return LicenseTier.PUBLIC_DOMAIN
    # Проверка на NC и ND идёт первой: «cc by nc sa 4.0» содержит и «cc by», и «sa»,
    # и без явного запрета попал бы в разрешённые по совпадению подстроки.
    if " nc" in f" {строка}" or " nd" in f" {строка}":
        return None
    for версия in _ВЕРСИИ:
        if строка == f"cc by sa {версия}":
            return LicenseTier.SHARE_ALIKE
        if строка == f"cc by {версия}":
            return LicenseTier.ATTRIBUTION
    return None


def is_allowed(raw: str, allow_sa: bool) -> bool:
    """Годится ли лицензия при текущем режиме прогона."""
    тир = tier_of(raw)
    if тир is None:
        return False
    return тир is not LicenseTier.SHARE_ALIKE or allow_sa


def canonical_name(raw: str) -> str:
    """Каноническая запись лицензии для манифеста."""
    тир = tier_of(raw)
    строка = _нормализовать(raw)
    if тир is LicenseTier.PUBLIC_DOMAIN:
        return "CC0" if строка.startswith("cc0") else "PD"
    if тир is None:
        raise ValueError(f"лицензия {raw!r} не разрешена решением Р-11")
    версия = строка.rsplit(" ", 1)[1]
    return f"CC BY-SA {версия}" if тир is LicenseTier.SHARE_ALIKE else f"CC BY {версия}"
