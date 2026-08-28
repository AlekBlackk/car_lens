import pytest

from pipeline.harvest.licenses import LicenseTier, canonical_name, is_allowed, tier_of


@pytest.mark.parametrize(
    ("сырое", "ожидание"),
    [
        ("CC0", LicenseTier.PUBLIC_DOMAIN),
        ("cc0 1.0", LicenseTier.PUBLIC_DOMAIN),
        ("Public domain", LicenseTier.PUBLIC_DOMAIN),
        ("PDM 1.0", LicenseTier.PUBLIC_DOMAIN),
        ("CC BY 4.0", LicenseTier.ATTRIBUTION),
        ("cc-by-3.0", LicenseTier.ATTRIBUTION),
        ("CC BY-SA 2.0", LicenseTier.SHARE_ALIKE),
    ],
)
def test_разрешённые_тиры(сырое: str, ожидание: LicenseTier) -> None:
    assert tier_of(сырое) is ожидание


@pytest.mark.parametrize(
    "сырое",
    ["CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-NC-SA 3.0", "GFDL", "fair use", ""],
)
def test_запрещённые_и_неизвестные(сырое: str) -> None:
    """Неизвестная формулировка — отказ, а не повод расширить список на ходу."""
    assert tier_of(сырое) is None


def test_share_alike_только_под_флагом() -> None:
    assert is_allowed("CC BY-SA 4.0", allow_sa=True)
    assert not is_allowed("CC BY-SA 4.0", allow_sa=False)
    assert is_allowed("CC BY 4.0", allow_sa=False)


def test_каноническое_имя_для_манифеста() -> None:
    assert canonical_name("cc-by-sa-3.0") == "CC BY-SA 3.0"
    assert canonical_name("  CC0 1.0  ") == "CC0"
