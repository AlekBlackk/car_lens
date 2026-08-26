"""Тесты нормализации свободного текста модели (§3.3)."""

from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline.catalog import Catalog, CatalogError, brand_slug, load_catalog, member_index
from pipeline.normalize import (
    BODY_STYLES,
    DEFAULT_BRAND_ALIASES_PATH,
    canonical_brand,
    load_aliases,
    load_brand_aliases,
    model_core,
    normalize,
)

YAML = """
version: 1
brands:
  Volkswagen:
    models:
      golf: {display: Volkswagen Golf}
      passat: {display: Volkswagen Passat}
  Audi:
    models:
      a4: {display: Audi A4}
      a6: {display: Audi A6}
  Mercedes-Benz:
    models:
      c_class: {display: Mercedes-Benz C-Class}
  Citroen:
    models:
      c3: {display: Citroën C3}
"""


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    путь = tmp_path / "canonical.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return load_catalog(путь)


@pytest.mark.parametrize(
    "сырое",
    ["Golf", "Golf Variant", "GOLF 1.6 TDI", "golf gti", "  Golf   VII  "],
)
def test_варианты_из_спецификации_сходятся_в_один_id(каталог: Catalog, сырое: str) -> None:
    итог = normalize(сырое, "Volkswagen", каталог, {})
    assert итог.model_id == "volkswagen__golf"


def test_кузовной_суффикс_сохраняется_отдельным_полем(каталог: Catalog) -> None:
    итог = normalize("Golf Variant", "Volkswagen", каталог, {})
    assert итог.model_id == "volkswagen__golf"
    assert итог.body_style == "wagon"
    assert итог.reason == "fuzzy"
    assert итог.score >= 88


def test_кузовной_суффикс_не_создаёт_отдельный_класс(каталог: Catalog) -> None:
    обычный = normalize("Golf", "Volkswagen", каталог, {})
    универсал = normalize("Golf Kombi", "Volkswagen", каталог, {})
    assert обычный.model_id == "volkswagen__golf"
    assert универсал.model_id == "volkswagen__golf"


def test_все_кузова_из_словаря_допустимы_по_схеме() -> None:
    assert {
        "hatchback",
        "sedan",
        "wagon",
        "suv",
        "coupe",
        "van",
        "pickup",
    } == BODY_STYLES


@pytest.mark.parametrize(
    ("сырое", "ожидание"),
    [
        ("A4 2.0 TDI quattro", "audi__a4"),
        ("A6 Avant 3.0 TDI", "audi__a6"),
        ("A4 S-line", "audi__a4"),
    ],
)
def test_отсекает_объём_топливо_привод_и_трим(каталог: Catalog, сырое: str, ожидание: str) -> None:
    assert normalize(сырое, "Audi", каталог, {}).model_id == ожидание


def test_диакритика_снимается(каталог: Catalog) -> None:
    assert normalize("C3 Élégance", "Citroën", каталог, {}).model_id == "citroen__c3"


def test_немецкое_написание_класса_ловится_алиасом(каталог: Catalog) -> None:
    """`C-Klasse` фонетически далёк от `c_class`, fuzzy его не возьмёт.

    Это ровно тот случай, ради которого §3.3 держит карту алиасов:
    отсечение объёма и топлива работает как обычно, а последний шаг
    делает явная запись в `aliases.yaml`.
    """
    итог = normalize(
        "C-Klasse 220 CDI", "Mercedes-Benz", каталог, {"c-klasse": "mercedes_benz__c_class"}
    )
    assert итог.model_id == "mercedes_benz__c_class"
    assert итог.reason == "alias"


def test_алиас_имеет_приоритет_над_fuzzy(каталог: Catalog) -> None:
    итог = normalize("Kolmonen", "Volkswagen", каталог, {"kolmonen": "volkswagen__passat"})
    assert итог.model_id == "volkswagen__passat"
    assert итог.reason == "alias"


def test_алиас_чужой_марки_не_применяется(каталог: Catalog) -> None:
    """Карта алиасов общая на весь каталог, а марка в объявлении задана.

    `3er` в объявлении Volkswagen не должен превратиться в `bmw__3_series`:
    это та же межмарочная утечка, от которой fuzzy защищён фильтром
    кандидатов по марке. Чужой алиас отбрасывается, строка идёт в fuzzy
    и там не набирает порога.
    """
    итог = normalize("3er", "Volkswagen", каталог, {"3er": "bmw__3_series"})
    assert итог.model_id is None
    assert итог.reason == "below_threshold"


def test_ниже_порога_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("Kolmonen", "Volkswagen", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "below_threshold"


def test_неизвестная_марка_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("Golf", "Lada", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "unknown_brand"


def test_пустая_строка_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("   ", "Volkswagen", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "empty"


def test_строка_из_одних_отсекаемых_токенов_не_сопоставляется(каталог: Catalog) -> None:
    итог = normalize("1.6 TDI", "Volkswagen", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "empty_after_strip"


def test_сырое_значение_всегда_сохраняется(каталог: Catalog) -> None:
    assert normalize("GOLF 1.6 TDI", "Volkswagen", каталог, {}).raw == "GOLF 1.6 TDI"


def test_кандидат_ограничен_моделями_своей_марки(каталог: Catalog) -> None:
    """A4 не должен утечь в Volkswagen, даже если строка похожа."""
    итог = normalize("A4", "Volkswagen", каталог, {})
    assert итог.model_id is None


def test_хвостовой_год_отбрасывается(каталог: Catalog) -> None:
    """Имя класса Stanford — «Audi A5 Coupe 2012», год к классу отношения не имеет."""
    итог = normalize("A4 Sedan 2012", "Audi", каталог, {})
    assert итог.model_id == "audi__a4"
    assert итог.body_style == "sedan"


@pytest.fixture
def марки() -> dict[str, str]:
    """Настоящая карта марок проекта: она и есть данные, а не подставка."""
    return load_brand_aliases(DEFAULT_BRAND_ALIASES_PATH)


def test_кириллические_марки_неразличимы_по_слагу_и_разводятся_картой(
    марки: dict[str, str],
) -> None:
    """§6: у «Лада», «Москвич» и «УАЗ» слаг не просто пуст — он одинаков.

    Три разные марки схлопываются в одну, и любая из них получает отказ
    ещё до карты моделей: `normalize` фильтрует каталог по слагу марки, а
    `aliases.yaml` смотрит уже внутри отобранного. Починить это записью в
    карту моделей невозможно — развести марки может только карта марок,
    применённая до нормализации.
    """
    assert brand_slug("Лада") == brand_slug("Москвич") == brand_slug("УАЗ") == ""
    assert {canonical_brand(м, марки) for м in ("Лада", "Москвич", "УАЗ")} == {
        "Lada",
        "Moskvich",
        "UAZ",
    }


@pytest.mark.parametrize(
    ("сырое", "ожидание"),
    [
        ("Лада", "Lada"),
        ("ВАЗ", "Lada"),
        ("LADA", "Lada"),
        ("Ваз 2107", "Lada"),
        ("Москвич", "Moskvich"),
        ("УАЗ", "UAZ"),
        ("ГАЗ", "GAZ"),
        ("Хендай", "Hyundai"),
        ("Хёндэ", "Hyundai"),
        ("Мерседес", "Mercedes-Benz"),
        ("Ленд Ровер", "Land Rover"),
    ],
)
def test_карта_марок_приводит_написание_к_каталожному(
    марки: dict[str, str], сырое: str, ожидание: str
) -> None:
    """Транслитерации в правилах §3.3 нет: «Хёндэ» → `Hyundai` не выводится.

    «Ваз 2107» в поле марки разбирается префиксом, а не своей записью:
    сочетаний марки с номером модели столько же, сколько моделей, и руками
    их не перечислить.
    """
    assert canonical_brand(сырое, марки) == ожидание


@pytest.mark.parametrize(
    ("сырое", "ожидание"),
    [
        ("ВАЗ-2107", "Lada"),
        ("УАЗ-2206", "UAZ"),
        ("Мерседес-Бенц", "Mercedes-Benz"),
        ("Ленд Ровер", "Land Rover"),
        ("Mercedes-Benz", "Mercedes-Benz"),
    ],
)
def test_дефис_в_поле_марки_разбирается_как_пробел(
    марки: dict[str, str], сырое: str, ожидание: str
) -> None:
    """Дефисная форма «ВАЗ-2107» в объявлениях не реже пробельной, а промах на
    ней тихий: слаг у неё непустой («2107»), то есть марка вернулась бы сырой и
    `normalize` ответил бы `unknown_brand` вместо промаха карты марок.

    Режет варианты `_варианты_марки`, а не `_подготовить`: та общая с разбором
    модели, и дефис там несущий (`c-klasse`, `e-series`). Первым кандидатом
    остаётся подготовленная строка целиком, иначе ключ, записанный в карте с
    дефисом («мерседес-бенц»), перестал бы находиться.
    """
    assert canonical_brand(сырое, марки) == ожидание


def test_марка_вне_карты_с_непустым_слагом_возвращается_как_есть(марки: dict[str, str]) -> None:
    """Иначе карту пришлось бы наполнять всеми сорока марками каталога."""
    assert canonical_brand("Volkswagen", марки) == "Volkswagen"


def test_марка_вне_карты_с_пустым_слагом_даёт_none(марки: dict[str, str]) -> None:
    """`None` означает ровно «канонизировать нечем»: от марки не осталось
    ни одного ASCII-символа, и сравнивать с каталогом нечего.
    """
    assert canonical_brand("Тарантас", марки) is None


def test_канонические_имена_карты_дают_непустой_слаг(марки: dict[str, str]) -> None:
    """Из значения собирается слаг `model_id`, поэтому кириллица в нём завела
    бы ровно ту дыру, которую карта и закрывает.
    """
    assert марки
    assert [имя for имя in марки.values() if not brand_slug(имя)] == []


def test_ключ_карты_марок_готовится_как_и_сырая_строка(tmp_path: Path) -> None:
    """Иначе запись с «ё» не нашлась бы никогда: подготовка §3.3 снимает с «ё»
    диакритику, и ключ из файла обязан пройти ту же чистку.
    """
    путь = tmp_path / "brand_aliases.yaml"
    путь.write_text("version: 1\nbrands:\n  Хёндэ: Hyundai\n", encoding="utf-8")
    карта = load_brand_aliases(путь)
    assert canonical_brand("ХЁНДЭ", карта) == "Hyundai"
    assert canonical_brand("Хендэ", карта) == "Hyundai"


def test_пустой_слаг_марки_даёт_отдельную_причину(каталог: Catalog) -> None:
    """«Лада» не отсутствует в каталоге — она до каталога не доехала.

    Общий `unknown_brand` слил бы два разных случая: «такой марки нет» и
    «марку не провели через `canonical_brand`». Первый чинится каталогом,
    второй — картой марок, и по отчёту их надо различать.
    """
    итог = normalize("Жигули", "Лада", каталог, {})
    assert итог.model_id is None
    assert итог.reason == "unknown_brand_no_slug"


YAML_РФ = """
version: 1
brands:
  Lada:
    models:
      "2107": {display: Лада 2107}
      "2110": {display: Лада 2110}
      largus: {display: Лада Ларгус}
  UAZ:
    models:
      "2206": {display: УАЗ-2206}
      patriot: {display: УАЗ Патриот}
  Renault:
    models:
      logan_mcv: {display: Renault Logan MCV}
groups:
  grp__largus_logan:
    members: [lada__largus, renault__logan_mcv]
    display: Лада Ларгус / Renault Logan MCV
"""

# Ключи — строки после подготовки §3.3: «семёрка» доезжает до карты как
# «семерка», потому что диакритика с «ё» снимается. Карту из файла готовит сам
# `load_aliases`, а инлайновая приходит в `normalize` как есть, минуя загрузчик.
АЛИАСЫ_РФ = {
    "жигули": "lada__2107",
    "семерка": "lada__2107",
    "2107": "lada__2107",
    "ваз-2107": "lada__2107",
    "буханка": "uaz__2206",
    "патриот": "uaz__patriot",
    "ларгус": "lada__largus",
}


@pytest.fixture
def каталог_рф(tmp_path: Path) -> Catalog:
    путь = tmp_path / "canonical.yaml"
    путь.write_text(YAML_РФ, encoding="utf-8")
    return load_catalog(путь)


@pytest.mark.parametrize(
    ("сырое", "марка", "ожидание"),
    [
        ("Жигули", "Lada", "lada__2107"),
        ("семёрка", "Lada", "lada__2107"),
        ("Буханка", "UAZ", "uaz__2206"),
        ("Патриот", "UAZ", "uaz__patriot"),
    ],
)
def test_разговорное_имя_берётся_только_алиасом(
    каталог_рф: Catalog, сырое: str, марка: str, ожидание: str
) -> None:
    """Fuzzy этих строк не видит вовсе: `_ключ_сравнения` оставляет только
    `[a-z0-9]`, и от «Жигули» не остаётся ни одного символа для сравнения.
    """
    итог = normalize(сырое, марка, каталог_рф, АЛИАСЫ_РФ)
    assert итог.model_id == ожидание
    assert итог.reason == "alias"
    assert normalize(сырое, марка, каталог_рф, {}).model_id is None


@pytest.mark.parametrize("сырое", ["Семёрка", "семерка", "СЕМЁРКА"])
def test_ключ_карты_моделей_готовится_как_и_сырая_строка(
    tmp_path: Path, каталог_рф: Catalog, сырое: str
) -> None:
    """Оба загрузчика готовят ключ одинаково, подготовкой §3.3.

    Иначе «семёрка» в `aliases.yaml` стала бы мёртвой записью без всякого
    сигнала: `_подготовить` снимает с «ё» диакритику, и строка объявления
    доезжает до карты как «семерка», а ключ после одного `lower()` остаётся с
    «ё» и не совпадает ни с чем. Автор карты не обязан знать про подготовку —
    он пишет марку и модель так, как они пишутся.
    """
    путь = tmp_path / "aliases.yaml"
    путь.write_text("version: 1\naliases:\n  семёрка: lada__2107\n", encoding="utf-8")
    итог = normalize(сырое, "Lada", каталог_рф, load_aliases(путь))
    assert итог.model_id == "lada__2107"
    assert итог.reason == "alias"


@pytest.mark.parametrize(
    ("секция", "загрузчик", "имя"),
    [
        ("aliases", load_aliases, "aliases.yaml"),
        ("brands", load_brand_aliases, "brand_aliases.yaml"),
    ],
)
def test_ключи_совпадающие_после_подготовки_отвергаются(
    tmp_path: Path,
    секция: str,
    загрузчик: Callable[[Path], dict[str, str]],
    имя: str,
) -> None:
    """Пара «Хёндэ»/«Хендэ» — два ключа для YAML и один после подготовки.

    Линтер на дубли ключей такую пару не поймает: схлопывает её не YAML, а
    словарное включение загрузчика, и молча — вторая запись затирает первую.
    Обе карты заведены под ручной рост, поэтому это падение, а не потеря, и
    в сообщении оба исходных написания: по подготовленной форме автор свою
    строку в файле не найдёт.
    """
    путь = tmp_path / имя
    путь.write_text(
        f"version: 1\n{секция}:\n  Хёндэ: Hyundai\n  Хендэ: Hyundai\n", encoding="utf-8"
    )
    with pytest.raises(CatalogError, match="Хёндэ") as ошибка:
        загрузчик(путь)
    assert "Хендэ" in str(ошибка.value)
    assert имя in str(ошибка.value)


@pytest.mark.parametrize("сырое", ["2107", "ВАЗ-2107"])
def test_числовая_модель_доезжает_до_карты_целиком(каталог_рф: Catalog, сырое: str) -> None:
    """Отсечка по `\\d\\.\\d` на «2107» не срабатывает, и за модельный год
    четыре цифры тоже не принимаются: `_ГОД` знает только 19xx и 20xx.

    Строка доезжает до карты целиком — и решает именно карта: правилам §3.3
    в номере модели разбирать нечего, а «ВАЗ-2107» несёт ещё и кириллический
    префикс марки, от которого `_ключ_сравнения` не оставит ничего.
    """
    итог = normalize(сырое, "Lada", каталог_рф, АЛИАСЫ_РФ)
    assert итог.model_id == "lada__2107"
    assert итог.reason == "alias"


def test_чужой_номер_модели_не_притягивается_к_соседнему(каталог_рф: Catalog) -> None:
    """«2108» в каталоге нет, а «2107» и «2110» есть: на голых числах fuzzy
    обязан отказать, а не выбрать ближайшее — разница в один символ здесь
    означает другую модель.
    """
    итог = normalize("2108", "Lada", каталог_рф, АЛИАСЫ_РФ)
    assert итог.model_id is None
    assert итог.reason == "below_threshold"


def test_ребейдж_даёт_разные_model_id(каталог_рф: Catalog) -> None:
    """«Ларгус» и «Logan MCV» — одна машина под двумя шильдиками, но два
    разных `model_id`: `normalize` возвращает модель, а не класс.

    Сойтись они могут только в `class_idx`, и лишь потому, что группа
    объявлена секцией `groups:`; без этой записи их было бы два класса.
    """
    ларгус = normalize("Ларгус", "Lada", каталог_рф, АЛИАСЫ_РФ)
    логан = normalize("Logan MCV", "Renault", каталог_рф, АЛИАСЫ_РФ)
    assert ларгус.model_id == "lada__largus"
    assert логан.model_id == "renault__logan_mcv"
    assert ларгус.model_id != логан.model_id
    класс = member_index(каталог_рф)
    assert класс["lada__largus"] == класс["renault__logan_mcv"] == "grp__largus_logan"


def test_русское_объявление_проходит_две_карты_подряд(
    каталог_рф: Catalog, марки: dict[str, str]
) -> None:
    """Порядок §6: сначала марка, потом модель. В обратном порядке «ВАЗ» дал бы
    пустой слаг, и до карты моделей строка не дошла бы вовсе.
    """
    марка = canonical_brand("ВАЗ", марки)
    assert марка is not None
    итог = normalize("Жигули", марка, каталог_рф, АЛИАСЫ_РФ)
    assert (марка, итог.model_id) == ("Lada", "lada__2107")


def test_год_остаётся_если_он_единственный_токен() -> None:
    """«Peugeot 2008» — название модели, а не год выпуска.

    Правило снимает год только с хвоста и только если в ядре есть что-то
    ещё. `model_core` всегда зовут без марки, поэтому ядро такой модели —
    один токен, и он остаётся. Иначе нормализатор съел бы половину
    модельного ряда Peugeot.
    """
    ядро, _ = model_core("2008")
    assert ядро == "2008"


def test_четырёхзначный_номер_модели_за_год_не_принимается() -> None:
    """«1500» и «850» — номера моделей: `_ГОД` знает только 19xx и 20xx."""
    assert model_core("Silverado 1500 2012")[0] == "silverado 1500"
    assert model_core("850")[0] == "850"


def test_год_в_середине_не_трогается() -> None:
    """Снимается только хвостовой год: в середине он часть строки, а не суффикс."""
    ядро, _ = model_core("Ram 1500 2012 Laramie")
    assert ядро == "ram 1500 2012 laramie"


@pytest.mark.parametrize(
    ("сырое", "кузов"),
    [
        ("Silverado 1500 Extended Cab 2012", "pickup"),
        ("Silverado 1500 Regular Cab 2012", "pickup"),
        ("F-150 SuperCab 2012", "pickup"),
        ("Ram 1500 Crew Cab 2012", "pickup"),
        ("Frontier King Cab 2012", "pickup"),
        ("Dakota Club Cab 2012", "pickup"),
        ("Caravan Minivan 2012", "van"),
        ("Outback Wagon 2012", "wagon"),
        ("Ranger Cab Plus 2011", "pickup"),
        ("Hilux Double Cab 2.4", "pickup"),
    ],
)
def test_американские_кузова_распознаются(сырое: str, кузов: str) -> None:
    _, найденный = model_core(сырое)
    assert найденный == кузов


def test_кузов_из_кабины_не_остаётся_в_ядре() -> None:
    """Иначе «extended» уедет в fuzzy и утащит сопоставление в сторону."""
    ядро, кузов = model_core("Silverado 1500 Extended Cab 2012")
    assert ядро == "silverado 1500"
    assert кузов == "pickup"


@pytest.mark.parametrize(
    ("сырое", "ядро", "кузов"),
    [
        ("E-Series Wagon Van 2012", "e-series", "wagon"),
        ("Ram Pickup 3500 Crew Cab 2010", "ram 3500", "pickup"),
        ("C/V Cargo Van Minivan 2012", "c cargo", "van"),
        ("Elantra Touring Hatchback 2012", "elantra", "wagon"),
    ],
)
def test_второй_кузовной_токен_тоже_вычищается(сырое: str, ядро: str, кузов: str) -> None:
    """Правило 4 §3.3: кузов — отдельное поле, а не часть имени класса.

    Имена классов Stanford называют кузов дважды («E-Series Wagon Van»,
    «C/V Cargo Van Minivan»), и оставленный второй токен завёл бы
    отдельный `model_id` вида `ford__e_series_van`.
    """
    assert model_core(сырое) == (ядро, кузов)


def test_model_core_публичен_и_повторяет_внутреннее_правило() -> None:
    """Генератор каталога скелета обязан использовать ровно эти правила.

    Если он начнёт разбирать имена своей регуляркой, каталог и
    нормализатор разойдутся, и часть классов перестанет находиться.
    """
    assert model_core("Golf Variant 1.6 TDI") == ("golf", "wagon")
