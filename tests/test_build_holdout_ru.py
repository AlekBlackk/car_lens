"""Сборка holdout_ru и обучающего пула из манифеста харвеста."""

from __future__ import annotations

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.schema import validate_frame
from scripts.build_holdout_ru import строки_в_кадр
from tests.test_holdout import _класс


def test_кадр_проходит_схему_13_1() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    кадр = строки_в_кадр(_класс("lada__2107", 5), каталог)
    validate_frame(кадр)


def test_источник_помечен_commons() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    кадр = строки_в_кадр(_класс("lada__2107", 3), каталог)
    assert set(кадр["source"]) == {"commons"}


def test_listing_id_склеен_из_класса_и_автора() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    кадр = строки_в_кадр(_класс("lada__2107", 2), каталог)
    assert all(lid.startswith("commons:lada__2107:") for lid in кадр["listing_id"])


def test_группа_размечена_первым_участником() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    группа = next(к for к in каталог.classes if к.class_id.startswith("grp__"))
    кадр = строки_в_кадр(_класс(группа.class_id, 2), каталог)
    assert set(кадр["model_id"]) == {группа.members[0]}


def test_пустой_список_строк_даёт_валидный_пустой_кадр() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    кадр = строки_в_кадр([], каталог)
    validate_frame(кадр)
    assert кадр.empty
