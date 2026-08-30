"""Нормализация свободного текста модели в `model_id` (§3.3).

Порядок правил важен и совпадает со спецификацией: сначала чистка
регистра и диакритики, затем отсечение всего после токена объёма или
топлива, затем трим-уровни, затем выделение кузова в отдельное поле,
и только остаток идёт в fuzzy-сопоставление.

Кандидаты для сопоставления берутся **только среди моделей заявленной
марки**. Поиск по всему каталогу выдаёт правдоподобные, но неверные
сопоставления между марками.

Сама марка приводится к каноническому виду **до** `normalize()`, картой
`brand_aliases.yaml` (§6): кириллическое написание фильтр по слагу марки
не проходит, а карта моделей смотрит уже внутри отобранного, поэтому
внутри `normalize` починить это нечем.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

from pipeline.catalog import Catalog, CatalogError, brand_slug

DEFAULT_ALIASES_PATH = Path(__file__).resolve().parent.parent / "catalog" / "aliases.yaml"
DEFAULT_BRAND_ALIASES_PATH = (
    Path(__file__).resolve().parent.parent / "catalog" / "brand_aliases.yaml"
)

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
    "wagon": "wagon",
    "minivan": "van",
    "cab": "pickup",
}

# Римские номера поколений: «Golf VII» — то же, что «Golf».
_ПОКОЛЕНИЕ = re.compile(r"\b(?:mk\s?\d{1,2}|[ivx]{1,5})\b")

# Составные кузова американского рынка: «Extended Cab», «SuperCab», «Cab Plus»,
# «Double Cab». Схлопываются в один токен `cab` до разбиения на слова, иначе
# определение «extended» останется в ядре и утащит fuzzy-сопоставление в сторону.
# Пробел перед `cab` необязателен: «SuperCab» пишется слитно.
_КАБИНА = re.compile(
    r"\b(?:extended|regular|crew|quad|king|club|super|double)[- ]?cab\b|\bcab[- ]?plus\b"
)

# Модельный год в хвосте имени класса: «Audi A5 Coupe 2012».
_ГОД = re.compile(r"(?:19|20)\d{2}")


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
    return _карта_по_подготовленным_ключам(сырое.get("aliases") or {}, path)


def load_brand_aliases(path: Path) -> dict[str, str]:
    """Прочитать `brand_aliases.yaml` в плоскую карту марок (§6)."""
    сырое: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _карта_по_подготовленным_ключам(сырое.get("brands") or {}, path)


def _карта_по_подготовленным_ключам(таблица: Mapping[Any, Any], path: Path) -> dict[str, str]:
    """Ключи карты — через `_подготовить`, ту же подготовку, что и сырая строка.

    Одного `lower()` мало, и молчаливо: подготовка §3.3 снимает диакритику, и
    ключ «семёрка» или «Хёндэ» после `lower()` не совпал бы ни с одной
    подготовленной строкой объявления — мёртвая запись без всякого сигнала.
    Автор карты писать «семерка» не обязан, обе карты он ведёт руками.

    Обратная сторона — два ключа могут схлопнуться в один, и схлопывает их не
    YAML (для него «Хёндэ» и «Хендэ» разные), а это включение. Поэтому здесь
    падение, а не потеря последней записи.
    """
    карта: dict[str, str] = {}
    исходные: dict[str, str] = {}
    for сырой_ключ, значение in таблица.items():
        ключ = _подготовить(str(сырой_ключ))
        прежний = исходные.get(ключ)
        if прежний is not None:
            raise CatalogError(
                f"{path}: ключи {прежний!r} и {сырой_ключ!r} после подготовки совпадают ({ключ!r})"
            )
        исходные[ключ] = str(сырой_ключ)
        карта[ключ] = str(значение)
    return карта


def strip_diacritics(текст: str) -> str:
    """Снять диакритику: `Mégane` → `Megane`, `ё` → `е`, `й` → `и` (правило 1 §3.3).

    Публичная, а не приватная: той же подготовкой обязан пользоваться нормализатор имён
    файлов Commons (`pipeline/harvest/aliases.py`), иначе ключ таблицы игл и имя файла
    сворачивались бы по-разному и `Citroën` не нашёлся бы по игле `citroen`.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", текст) if not unicodedata.combining(c))


def _подготовить(сырое: str) -> str:
    """Правило 1 §3.3: нижний регистр, диакритика, схлопывание пробелов."""
    текст = strip_diacritics(сырое).lower()
    текст = re.sub(r"[_/]+", " ", текст)
    return re.sub(r"\s+", " ", текст).strip()


def _извлечь_кузов(токены: list[str]) -> tuple[list[str], str | None]:
    """Вынуть кузовные токены из строки, вернув остаток и тип кузова.

    Кузовных слов в строке бывает несколько: «E-Series Wagon Van»,
    «C/V Cargo Van Minivan», «Ram Pickup 3500 Crew Cab». Тип берётся по
    первому, но из ядра вычищаются все — правило 4 §3.3 требует, чтобы
    кузов был отдельным полем, а не частью класса. Оставленный второй
    токен уезжает в fuzzy и заводит собственный `model_id` вида
    `ford__e_series_van`.
    """
    кузов: str | None = None
    остаток: list[str] = []
    for токен in токены:
        ключ = токен.strip("-")
        if ключ in _КУЗОВА:
            if кузов is None:
                кузов = _КУЗОВА[ключ]
            continue
        остаток.append(токен)
    return остаток, кузов


def _снять_хвостовой_год(токены: list[str]) -> list[str]:
    """Убрать модельный год, если он последний и не единственный.

    «A5 Coupe 2012» → «a5». Guard `len > 1` держит единственный случай,
    который до него доходит, — «2008» от Peugeot: `model_core` зовут уже
    без марки (`normalize` получает её отдельным аргументом, генератор
    каталога скелета отрезает её сам), поэтому ядро такой модели состоит
    ровно из одного токена. Названия вроде «850» или «911» под правило не
    попадают вовсе: `_ГОД` знает только 19xx и 20xx.
    """
    if len(токены) > 1 and _ГОД.fullmatch(токены[-1]):
        return токены[:-1]
    return токены


def _ядро_модели(подготовленное: str) -> tuple[str, str | None]:
    """Свести подготовленную строку к ядру названия модели и кузову."""
    обрезано = _ДВИГАТЕЛЬ_И_ПРИВОД.split(подготовленное, maxsplit=1)[0]
    обрезано = _ТРИМ.sub(" ", обрезано)
    обрезано = _КАБИНА.sub(" cab ", обрезано)
    токены = [т for т in re.split(r"[\s,]+", обрезано) if т]
    токены, кузов = _извлечь_кузов(токены)
    токены = [т for т in токены if not _ПОКОЛЕНИЕ.fullmatch(т)]
    токены = _снять_хвостовой_год(токены)
    return " ".join(токены).strip(), кузов


def model_core(raw: str) -> tuple[str, str | None]:
    """Сырая строка модели → ядро названия и `body_style`.

    Публичная обёртка над внутренними правилами §3.3. Нужна генератору
    каталога скелета: тот обязан резать имена классов ровно теми же
    правилами, иначе каталог и нормализатор разойдутся и часть классов
    перестанет находиться.
    """
    return _ядро_модели(_подготовить(raw))


def _ключ_сравнения(текст: str) -> str:
    """Свести к сравнимому виду: только буквы и цифры."""
    return re.sub(r"[^a-z0-9]+", "", strip_diacritics(текст).lower())


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


def _первое_совпадение(карта: Mapping[str, str], кандидаты: Iterable[str]) -> str | None:
    """Первое непустое значение карты по кандидатам, в порядке их подачи.

    Порядок задаёт вызывающий, и он же решает, какой ключ считается лучшим:
    оба генератора кандидатов идут от длинного к короткому.
    """
    for кандидат in кандидаты:
        совпадение = карта.get(кандидат)
        if совпадение:
            return совпадение
    return None


def _найти_алиас(aliases: Mapping[str, str], подготовленное: str, ядро: str) -> str | None:
    """Первое совпадение в карте алиасов, самый длинный ключ выигрывает."""
    return _первое_совпадение(aliases, _варианты_ключа(подготовленное, ядро))


def _варианты_марки(подготовленное: str) -> Iterator[str]:
    """Кандидаты для поиска в карте марок, от длинного к короткому.

    В поле марки объявления попадает и номер модели: «Ваз 2107», «ВАЗ-2107».
    Такими сочетаниями карту не наполнить — их столько же, сколько моделей, —
    поэтому после полной строки пробуются её префиксы, и «ваз» находится.
    Составные марки от этого не страдают: «ленд ровер» пробуется целиком и
    только потом сокращается до «ленд».

    Дефис здесь разделитель наравне с пробелом, а в `_подготовить` — нет: та
    общая с разбором модели, где дефис несущий (`c-klasse`, `e-series`).
    Поэтому полная подготовленная строка идёт первым кандидатом: иначе ключ,
    записанный в карте с дефисом («мерседес-бенц»), перестал бы находиться.
    """
    yield подготовленное
    токены = re.split(r"[\s-]+", подготовленное)
    for n in range(len(токены), 0, -1):
        yield " ".join(токены[:n])


def canonical_brand(raw: str, aliases: Mapping[str, str]) -> str | None:
    """Сырое написание марки → каноническая марка каталога (§6).

    Промах в карте при непустом слаге отдаёт сырое имя: латинские марки
    каталога в карте не дублируются, `Volkswagen` доходит до каталога сам.

    `None` означает ровно одно — канонизировать нечем: слаг пуст, то есть
    от марки не осталось ни одного ASCII-символа, и с чем сравнивать
    каталог, неизвестно. Такую марку нельзя пускать в `normalize`: там она
    неотличима от любой другой кириллической.
    """
    по_карте = _первое_совпадение(aliases, _варианты_марки(_подготовить(raw)))
    if по_карте:
        return по_карте
    return raw if brand_slug(raw) else None


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
    # Пустой слаг — не «марки нет в каталоге», а «марку не провели через
    # `canonical_brand`»: у «Лада», «Москвич» и «УАЗ» слаг одинаково пуст, и
    # фильтр ниже ответил бы на все три одно и то же. Каталог пустых слагов не
    # содержит — `load_catalog` их отвергает, — так что общий `unknown_brand`
    # здесь просто соврал бы о причине, а чинятся эти два случая по-разному:
    # один каталогом, другой картой марок (§6).
    if not слаг_марки:
        return NormalizeResult(raw, None, None, 0.0, "unknown_brand_no_slug")

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
