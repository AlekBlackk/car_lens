from __future__ import annotations

import pytest

from pipeline.reject_bank import FOREIGN, NON_CAR, bank_kind, fold_of, model_key


@pytest.mark.parametrize(
    ("listing_id", "ожидание"),
    [
        ("off_catalog/autoevolution/2003_Opel_Signum", "opel|signum"),
        ("off_catalog_street/mad_cars/opel__signum", "opel|signum"),
        ("off_catalog/autoevolution/VOLVO_760_19821990", "volvo|760"),
        ("off_catalog/autoevolution/ALFA_ROMEO_155_19921998", "alfa|155"),
        ("off_catalog_street/mad_cars/alfa_romeo__155", "alfa|155"),
        ("off_catalog_street/mad_cars/acura__rdx", "acura|rdx"),
        ("off_catalog/autoevolution/ACURA_RDX_20132015", "acura|rdx"),
    ],
)
def test_ключ_модели_един_для_autoevolution_и_mad(listing_id: str, ожидание: str) -> None:
    assert model_key(listing_id) == ожидание


def test_не_машина_получает_ключом_весь_listing_id() -> None:
    assert model_key("truck_bus/coco/000001.jpg") == "truck_bus/coco/000001.jpg"


@pytest.mark.parametrize(
    ("listing_id", "вид"),
    [
        ("off_catalog/autoevolution/2003_Opel_Signum", FOREIGN),
        ("off_catalog_street/mad_cars/acura__rdx", FOREIGN),
        ("truck_bus/coco/000001.jpg", NON_CAR),
        ("detector_fp/coco/000002.jpg", NON_CAR),
    ],
)
def test_вид_кадра(listing_id: str, вид: str) -> None:
    assert bank_kind(listing_id) == вид


def test_фолд_стабилен_и_в_границах() -> None:
    фолды = [fold_of(f"бренд|модель{номер}", 20261008, 5) for номер in range(200)]
    assert set(фолды) == {0, 1, 2, 3, 4}
    assert fold_of("opel|signum", 20261008, 5) == fold_of("opel|signum", 20261008, 5)
    assert fold_of("opel|signum", 1, 5) in range(5)
