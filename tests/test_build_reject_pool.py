"""Пул отказных прототипов: не пересекается с замером ни моделью, ни картинкой."""

from __future__ import annotations

import pytest

from scripts.build_reject_pool import (
    check_disjoint,
    check_off_catalog,
    coco_files_of,
    mad_models_of,
)

ЗАМЕР = [
    "off_catalog_street/mad_cars/bmw__x3",
    "truck_bus/coco/000001.jpg",
    "detector_fp/coco/000002.jpg",
]


def test_модели_и_картинки_разбираются_из_listing_id_и_origin() -> None:
    assert mad_models_of(ЗАМЕР) == {"bmw__x3"}
    assert coco_files_of(ЗАМЕР) == {"000001.jpg", "000002.jpg"}
    assert mad_models_of(["mad_cars/audi__q5"]) == {"audi__q5"}
    assert coco_files_of(["coco/000009.jpg"]) == {"000009.jpg"}


def test_пул_без_пересечения_проходит() -> None:
    check_disjoint(["mad_cars/audi__q5", "coco/000009.jpg"], ЗАМЕР)


def test_общая_модель_роняет_с_именем() -> None:
    with pytest.raises(ValueError, match="bmw__x3"):
        check_disjoint(["off_catalog_street/mad_cars/bmw__x3"], ЗАМЕР)


def test_общая_картинка_роняет_даже_из_другого_вида() -> None:
    # Картинка дала замеру кроп detector_fp, а пулу — truck_bus: соседние кропы одного кадра.
    with pytest.raises(ValueError, match=r"000002\.jpg"):
        check_disjoint(["truck_bus/coco/000002.jpg"], ЗАМЕР)


ТОЧНЫЕ = {("bmw", "x5")}
СПОРНЫЕ = frozenset({("bmw", "x6")})
ИГЛЫ = {"audi__a4": ("a4",)}


def test_чужая_модель_проходит_фильтры() -> None:
    check_off_catalog(["audi__q5"], ТОЧНЫЕ, СПОРНЫЕ, ИГЛЫ)


@pytest.mark.parametrize(
    ("папка", "причина"),
    [("bmw__x5", "exact"), ("bmw__x6", "disputed"), ("audi__a4", "needle")],
)
def test_модель_каталога_в_пуле_роняет_с_причиной(папка: str, причина: str) -> None:
    with pytest.raises(ValueError, match=rf"{папка} \({причина}"):
        check_off_catalog(["audi__q5", папка], ТОЧНЫЕ, СПОРНЫЕ, ИГЛЫ)


def test_папка_без_разделителя_роняет() -> None:
    with pytest.raises(ValueError, match=r"bmw-x3 \(bad_folder\)"):
        check_off_catalog(["bmw-x3"], ТОЧНЫЕ, СПОРНЫЕ, ИГЛЫ)
