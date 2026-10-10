"""Стражи состава `catalog/canonical.yaml`.

Здесь проверяется не загрузчик — его проверяет `test_catalog.py` на временных
фикстурах, — а сам состав каталога: сколько в нём классов, сколько марок, все
ли марки карты `brand_aliases.yaml` в нём есть и не схлопнулись ли раньше
времени кандидаты §5.2 дизайна.

Пороги живут в тесте, а не в комментарии к YAML. Комментарий не падает: состав
правят под конкретную задачу, и «ровно 260» без прогона уползает в 259 или 262
незаметно, а приёмка MVP §5.2 спеки считается именно по классам.
"""

import pytest

from pipeline.catalog import (
    DEFAULT_CATALOG_PATH,
    Catalog,
    load_catalog,
    member_index,
)
from pipeline.normalize import (
    DEFAULT_ALIASES_PATH,
    DEFAULT_BRAND_ALIASES_PATH,
    load_aliases,
    load_brand_aliases,
)

ЧИСЛО_КЛАССОВ = 260
МИНИМУМ_МАРОК = 40
ПРЕФИКС_ГРУППЫ = "grp__"

# §5.1 дизайна: пять пар, различающихся только шильдиком. Больше групп в
# каталоге нет — различать остальное сеть обязана.
ПОДТВЕРЖДЁННЫЕ_ГРУППЫ = {
    "grp__jac_js4": ("jac__js4", "moskvich__3"),
    "grp__coolray": ("belgee__x50", "geely__coolray"),
    "grp__atlas_pro": ("belgee__x70", "geely__atlas_pro"),
    "grp__solaris_hs": ("hyundai__solaris", "solaris__hs"),
    "grp__tiggo7": ("chery__tiggo_7_pro", "xcite__x_cross_7"),
}

# §5.2 дизайна: похожи, но различимы. Схлопывает их confusion-замер на M3, а не
# впечатление от фотографий, поэтому до замера каждый — сам себе класс.
КАНДИДАТЫ_НА_СХЛОПЫВАНИЕ = (
    "renault__duster",
    "nissan__terrano",
    "chevrolet__niva",
    "lada__niva_travel",
    "lada__largus",
    "renault__logan_mcv",
    "datsun__on_do",
    "lada__granta",
)


@pytest.fixture(scope="module")
def каталог() -> Catalog:
    return load_catalog(DEFAULT_CATALOG_PATH)


def test_ровно_260_классов(каталог: Catalog) -> None:
    """Каталог 0.9: 150 классов 0.5, 42 из MAD-Cars (`bench/catalog-192-preregistration.md`),
    16 названий с ≥ 85 машин (`bench/catalog-208-preregistration.md`), 28 с ≥ 60 машин
    (`bench/catalog-236-preregistration.md`) и 24 с ≥ 50 машин
    (`bench/catalog-260-preregistration.md`).

    Приёмка считается по классам, и их число — потолок: лишний класс не «немного
    точнее», а данные, отобранные у остальных. Решение о расширении принимает
    владелец проекта; правка числа без новой предрегистрации — уход от него.
    """
    assert len(каталог.classes) == ЧИСЛО_КЛАССОВ


def test_не_менее_40_марок(каталог: Catalog) -> None:
    """§10 спеки, порог M0. Считается по моделям: марка, живущая только
    участником платформенной группы, из каталога никуда не девается."""
    assert len(каталог.brands()) >= МИНИМУМ_МАРОК


def test_у_каждого_класса_непустой_display(каталог: Catalog) -> None:
    """`display` — единственное, что видит пользователь: `model_id` уходит в
    файлы и в JSON, а на экран идёт эта строка."""
    пустые = [к.class_id for к in каталог.classes if not к.display.strip()]
    assert пустые == []


def test_class_id_групп_с_префиксом_а_одиночек_без(каталог: Catalog) -> None:
    """Префикс закреплён за группами: без него `class_id` группы совпал бы с
    `model_id` одиночки, и два класса заняли бы один индекс."""
    for к in каталог.classes:
        составной = len(к.members) > 1
        assert к.class_id.startswith(ПРЕФИКС_ГРУППЫ) == составной, к.class_id


def test_у_каждой_группы_не_менее_двух_участников(каталог: Catalog) -> None:
    группы = [к for к in каталог.classes if к.class_id.startswith(ПРЕФИКС_ГРУППЫ)]
    assert группы, "в каталоге нет ни одной платформенной группы"
    assert all(len(к.members) >= 2 for к in группы)


def test_группы_каталога_это_ровно_пять_подтверждённых_5_1(каталог: Catalog) -> None:
    """Схлопывать до обучения можно только полное совпадение кузова (§5.1).

    Состав каждой группы проверяется поимённо: подмена участника прошла бы
    мимо проверки «групп ровно пять», а стоила бы целого класса.
    """
    группы = {
        к.class_id: tuple(sorted(к.members))
        for к in каталог.classes
        if к.class_id.startswith(ПРЕФИКС_ГРУППЫ)
    }
    assert группы == ПОДТВЕРЖДЁННЫЕ_ГРУППЫ


@pytest.mark.parametrize("model_id", КАНДИДАТЫ_НА_СХЛОПЫВАНИЕ)
def test_кандидат_5_2_остаётся_отдельным_классом(каталог: Catalog, model_id: str) -> None:
    """§5.2: Duster/Terrano, Chevrolet Niva / Niva Travel, Largus / Logan MCV,
    on-DO / Granta живут врозь до confusion-замера на M3.

    Утверждается именно «каждый сам себе класс», а не «групп ровно пять»:
    иначе схлопывание проехало бы молча внутри уже существующей группы.
    """
    участие = member_index(каталог)
    assert участие[model_id] == model_id


def test_каждая_каноническая_марка_карты_есть_в_каталоге() -> None:
    """Значение `brand_aliases.yaml` — обещание, что такая марка в каталоге есть.

    Если её нет, `canonical_brand` вернёт имя, по которому `normalize`
    отфильтрует пустой список и ответит `unknown_brand`: объявление
    потеряется, а карта будет выглядеть исправной. Чинится это составом
    каталога, а не правкой карты (§6).
    """
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    марки_каталога = set(каталог.brands())
    из_карты = set(load_brand_aliases(DEFAULT_BRAND_ALIASES_PATH).values())
    assert sorted(из_карты - марки_каталога) == []


def test_каждое_значение_aliases_существующий_model_id() -> None:
    """Алиас, ведущий в несуществующий `model_id`, — мёртвая запись без сигнала:
    `normalize` отбросит её как «чужую марку» и молча уйдёт в fuzzy."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    известные = {m.model_id for m in каталог.models}
    из_карты = set(load_aliases(DEFAULT_ALIASES_PATH).values())
    assert sorted(из_карты - известные) == []
