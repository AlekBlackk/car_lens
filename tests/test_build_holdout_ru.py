"""Сборка holdout_ru и обучающего пула из манифеста харвеста."""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.harvest.manifest import HarvestRow
from pipeline.holdout import listing_id_of
from pipeline.schema import validate_frame
from scripts.build_holdout_ru import строки_в_кадр, сформировать_отчёт_сплита
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
