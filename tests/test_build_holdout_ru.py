"""Сборка holdout_ru и обучающего пула из манифеста харвеста."""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.manifest import HarvestRow
from pipeline.holdout import listing_id_of
from pipeline.schema import validate_frame
from pipeline.split import CrossClassPhashPair
from scripts.build_holdout_ru import (
    пары_в_отчёт,
    строки_в_кадр,
    строки_предупреждения,
    сформировать_отчёт_сплита,
)
from tests.test_holdout import _класс


def _единицы(строки: Sequence[HarvestRow]) -> dict[str, str]:
    return {строка.file: listing_id_of(строка) for строка in строки}


def test_кадр_проходит_схему_13_1() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки = _класс("lada__2107", 5)
    кадр = строки_в_кадр(строки, каталог, _единицы(строки))
    validate_frame(кадр)


def test_источник_помечен_commons() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки = _класс("lada__2107", 3)
    кадр = строки_в_кадр(строки, каталог, _единицы(строки))
    assert set(кадр["source"]) == {"commons"}


def test_listing_id_склеен_из_класса_и_автора() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки = _класс("lada__2107", 2)
    кадр = строки_в_кадр(строки, каталог, _единицы(строки))
    assert all(lid.startswith("commons:lada__2107:") for lid in кадр["listing_id"])


def test_группа_размечена_первым_участником() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    группа = next(к for к in каталог.classes if к.class_id.startswith("grp__"))
    строки = _класс(группа.class_id, 2)
    кадр = строки_в_кадр(строки, каталог, _единицы(строки))
    assert set(кадр["model_id"]) == {группа.members[0]}


def test_пустой_список_строк_даёт_валидный_пустой_кадр() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    кадр = строки_в_кадр([], каталог, {})
    validate_frame(кадр)
    assert кадр.empty


def test_ключ_дроблёной_единицы_попадает_в_listing_id() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки = _класс("lada__granta", 3)
    единицы = {строка.file: f"{listing_id_of(строка)}#7" for строка in строки}
    кадр = строки_в_кадр(строки, каталог, единицы)
    assert all(lid.endswith("#7") for lid in кадр["listing_id"])


def test_сформировать_отчёт_сплита_содержит_дроблёные_и_короткие_классы() -> None:
    from pipeline.holdout import HoldoutSelection

    выбор = HoldoutSelection(
        files=frozenset(["a.jpg", "b.jpg"]),
        short_classes=("lifan__x60",),
        refined_classes=("lada__granta", "bmw__x5"),
        units=(("a.jpg", "u1"), ("b.jpg", "u2")),
    )
    отчёт = сформировать_отчёт_сплита(holdout_frames=2, pool_frames=10, selection=выбор)
    assert отчёт["holdout_frames"] == 2
    assert отчёт["pool_frames"] == 10
    assert отчёт["refined_classes"] == ["lada__granta", "bmw__x5"]
    assert отчёт["short_classes"] == ["lifan__x60"]


def test_предупреждение_называет_классы_именами_а_не_индексами() -> None:
    """Индекс класса человеку ничего не говорит: действовать он будет по `class_id`."""
    пары = [CrossClassPhashPair(distance=6, phash_a=11, phash_b=22, class_a=0, class_b=1)]

    строки = строки_предупреждения(пары, {0: "mazda__cx5", 1: "mitsubishi__pajero"})

    assert строки[0].startswith("ВНИМАНИЕ:")
    assert "mazda__cx5" in строки[1]
    assert "mitsubishi__pajero" in строки[1]
    assert "6" in строки[1]


def test_без_межклассовых_пар_печатается_ноль() -> None:
    """Молчание не отличить от «проверка не выполнялась»: соседние блоки печатают нули."""
    строки = строки_предупреждения([], {})
    assert len(строки) == 1
    assert строки[0].endswith("0")


def test_пары_в_отчёт_подставляют_имена_классов() -> None:
    пары = [CrossClassPhashPair(distance=6, phash_a=11, phash_b=22, class_a=0, class_b=1)]

    записи = пары_в_отчёт(пары, {0: "mazda__cx5", 1: "mitsubishi__pajero"})

    assert записи == [
        {
            "class_a": "mazda__cx5",
            "class_b": "mitsubishi__pajero",
            "distance": 6,
            "phash_a": 11,
            "phash_b": 22,
        }
    ]


def test_пары_в_отчёт_без_имени_оставляют_индекс() -> None:
    """Отображение неполно только при поломке каталога, но падать на этом нельзя."""
    записи = пары_в_отчёт(
        [CrossClassPhashPair(distance=0, phash_a=1, phash_b=1, class_a=7, class_b=9)], {}
    )
    assert записи[0]["class_a"] == "7"
    assert записи[0]["class_b"] == "9"


def test_отчёт_сплита_несёт_межклассовые_пары() -> None:
    """Единственный новый сигнал правки обязан лежать в сайдкаре, а не в прокрутке."""
    from pipeline.holdout import HoldoutSelection

    выбор = HoldoutSelection(
        files=frozenset(["a.jpg"]), short_classes=(), refined_classes=(), units=(("a.jpg", "u1"),)
    )
    записи = [{"class_a": "mazda__cx5", "class_b": "mitsubishi__pajero", "distance": 6}]

    отчёт = сформировать_отчёт_сплита(
        holdout_frames=1, pool_frames=2, selection=выбор, cross_class_pairs=записи
    )

    assert отчёт["cross_class_close_phash"] == записи
