"""Нормализация свободного текста модели в `model_id` (§3.3).

Порядок правил важен и совпадает со спецификацией: сначала чистка
регистра и диакритики, затем отсечение всего после токена объёма или
топлива, затем трим-уровни, затем выделение кузова в отдельное поле,
и только остаток идёт в fuzzy-сопоставление.

Кандидаты для сопоставления берутся **только среди моделей заявленной
марки**. Поиск по всему каталогу выдаёт правдоподобные, но неверные
сопоставления между марками.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

from pipeline.catalog import Catalog, brand_slug

DEFAULT_ALIASES_PATH = Path(__file__).resolve().parent.parent / "catalog" / "aliases.yaml"

BODY_STYLES = frozenset({"hatchback", "sedan", "wagon", "suv", "coupe", "van", "pickup"})

# Правило 2 §3.3: объём, топливо, привод. Всё начиная с такого токена — отбросить.
_ДВИГАТЕЛЬ_И_ПРИВОД = re.compile(
    r"\b("
    r"\d\.\d"
    r"|tdi|tsi|tfsi|hdi|cdi|dci|crdi|bluetec|bluehdi|ecoboost"
    r"|hybrid|phev|bev|e[- ]?tron|plug[- ]?in"
    r"|4matic|quattro|xdrive|awd|4wd|4x4|fwd|rwd"
    r"|\d{2,3}\s?(?:hv|kw|hp|ps|bhp)"
    r")\b"
)

# Правило 3 §3.3: трим-уровни.
_ТРИМ = re.compile(
    r"\b("
    r"gti|gtd|gte|r[- ]?line|amg|m[- ]?sport|s[- ]?line|sport|sportline"
    r"|comfortline|trendline|highline|style|elegance|avantgarde|ambition"
    r"|titanium|ghia|zetec|business|premium|executive|edition|limited"
    r")\b"
)

# Правило 4 §3.3: кузов сохраняется отдельным полем, но не отдельным классом.
_КУЗОВА: dict[str, str] = {
    "variant": "wagon",
    "kombi": "wagon",
    "estate": "wagon",
    "touring": "wagon",
    "avant": "wagon",
    "sportwagon": "wagon",
    "sw": "wagon",
    "sportback": "hatchback",
    "hatchback": "hatchback",
    "hb": "hatchback",
    "sedan": "sedan",
    "saloon": "sedan",
    "coupe": "coupe",
    "cabrio": "coupe",
    "cabriolet": "coupe",
    "convertible": "coupe",
    "roadster": "coupe",
    "suv": "suv",
    "crossover": "suv",
    "van": "van",
    "tourer": "van",
    "combi": "van",
    "pickup": "pickup",
    "double_cab": "pickup",
}

# Римские номера поколений: «Golf VII» — то же, что «Golf».
_ПОКОЛЕНИЕ = re.compile(r"\b(?:mk\s?\d{1,2}|[ivx]{1,5})\b")


@dataclass(frozen=True, slots=True)
class NormalizeResult:
    """Итог нормализации одной строки."""

    raw: str
    model_id: str | None
    body_style: str | None
    score: float
    reason: str


def load_aliases(path: Path) -> dict[str, str]:
    """Прочитать `aliases.yaml` в плоскую карту."""
    сырое: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    таблица = сырое.get("aliases") or {}
    return {str(k).strip().lower(): str(v) for k, v in таблица.items()}


def _снять_диакритику(текст: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", текст) if not unicodedata.combining(c))


def _подготовить(сырое: str) -> str:
    """Правило 1 §3.3: нижний регистр, диакритика, схлопывание пробелов."""
    текст = _снять_диакритику(сырое).lower()
    текст = re.sub(r"[_/]+", " ", текст)
    return re.sub(r"\s+", " ", текст).strip()


def _извлечь_кузов(токены: list[str]) -> tuple[list[str], str | None]:
    """Вынуть кузовной токен из строки, вернув остаток и тип кузова."""
    кузов: str | None = None
    остаток: list[str] = []
    for токен in токены:
        ключ = токен.strip("-")
        if кузов is None and ключ in _КУЗОВА:
            кузов = _КУЗОВА[ключ]
            continue
        остаток.append(токен)
    return остаток, кузов


def _ядро_модели(подготовленное: str) -> tuple[str, str | None]:
    """Свести подготовленную строку к ядру названия модели и кузову."""
    обрезано = _ДВИГАТЕЛЬ_И_ПРИВОД.split(подготовленное, maxsplit=1)[0]
    обрезано = _ТРИМ.sub(" ", обрезано)
    токены = [т for т in re.split(r"[\s,]+", обрезано) if т]
    токены, кузов = _извлечь_кузов(токены)
    токены = [т for т in токены if not _ПОКОЛЕНИЕ.fullmatch(т)]
    return " ".join(токены).strip(), кузов


def _ключ_сравнения(текст: str) -> str:
    """Свести к сравнимому виду: только буквы и цифры."""
    return re.sub(r"[^a-z0-9]+", "", _снять_диакритику(текст).lower())


def _варианты_ключа(подготовленное: str, ядро: str) -> Iterator[str]:
    """Кандидаты для поиска в карте алиасов, от длинного к короткому.

    Ключ алиаса обычно короче строки объявления: в карте лежит
    `c-klasse`, а в объявлении написано `C-Klasse 220 CDI`. Отсечение
    объёма и топлива убирает `cdi`, но оставляет `220`, поэтому одного
    сравнения с ядром недостаточно — нужны и его префиксы.
    """
    yield подготовленное
    if not ядро:
        return
    yield ядро
    токены = ядро.split()
    for n in range(len(токены) - 1, 0, -1):
        yield " ".join(токены[:n])


def _найти_алиас(aliases: Mapping[str, str], подготовленное: str, ядро: str) -> str | None:
    """Первое совпадение в карте алиасов, самый длинный ключ выигрывает."""
    for кандидат in _варианты_ключа(подготовленное, ядро):
        совпадение = aliases.get(кандидат)
        if совпадение:
            return совпадение
    return None


def normalize(
    raw: str,
    brand: str,
    catalog: Catalog,
    aliases: Mapping[str, str],
    threshold: int = 88,
) -> NormalizeResult:
    """Свободный текст модели → `model_id`, кузов и оценка совпадения."""
    подготовленное = _подготовить(raw)
    if not подготовленное:
        return NormalizeResult(raw, None, None, 0.0, "empty")

    слаг_марки = brand_slug(brand)
    свои = [m for m in catalog.models if brand_slug(m.brand) == слаг_марки]
    if not свои:
        return NormalizeResult(raw, None, None, 0.0, "unknown_brand")

    ядро, кузов = _ядро_модели(подготовленное)

    # Алиас принимается только если ведёт в модель заявленной марки: карта
    # общая на весь каталог, и ключ `3er` из объявления Volkswagen иначе утёк
    # бы в `bmw__3_series`. Чужой алиас игнорируется, и строка идёт в fuzzy
    # так же, как если бы совпадения в карте вовсе не нашлось.
    по_алиасу = _найти_алиас(aliases, подготовленное, ядро)
    if по_алиасу and по_алиасу in {m.model_id for m in свои}:
        return NormalizeResult(raw, по_алиасу, кузов, 100.0, "alias")

    if not ядро:
        return NormalizeResult(raw, None, кузов, 0.0, "empty_after_strip")

    кандидаты = {m.model_id: _ключ_сравнения(m.model_id.split("__", 1)[1]) for m in свои}
    попадание = process.extractOne(
        _ключ_сравнения(ядро),
        кандидаты,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    if попадание is None:
        return NormalizeResult(raw, None, кузов, 0.0, "below_threshold")

    _, оценка, model_id = попадание
    return NormalizeResult(raw, str(model_id), кузов, float(оценка), "fuzzy")
