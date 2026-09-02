import pytest

from pipeline.harvest.flickr import license_string, requested_licenses
from pipeline.harvest.licenses import LicenseTier, tier_of


@pytest.mark.parametrize(
    ("идентификатор", "ожидаемый_тир"),
    [
        ("9", LicenseTier.PUBLIC_DOMAIN),
        ("10", LicenseTier.PUBLIC_DOMAIN),
        ("4", LicenseTier.ATTRIBUTION),
        ("5", LicenseTier.SHARE_ALIKE),
    ],
)
def test_разрешённые_id_ложатся_в_известный_тир(
    идентификатор: str, ожидаемый_тир: LicenseTier
) -> None:
    """Строка адаптера обязана быть понятна `tier_of` без правок `licenses.py`."""
    строка = license_string(идентификатор)
    assert строка is not None
    assert tier_of(строка) is ожидаемый_тир


@pytest.mark.parametrize(
    "идентификатор", ["0", "1", "2", "3", "6", "7", "8", "", "  ", "42", "нет"]
)
def test_прочие_id_отвергаются(идентификатор: str) -> None:
    """`7` — заявление о неизвестности статуса, `8` — юрисдикция вне закрытого списка."""
    assert license_string(идентификатор) is None


def test_запрошенные_лицензии_без_sa_не_содержат_пятёрку() -> None:
    assert requested_licenses(allow_sa=False) == "9,10,4"


def test_запрошенные_лицензии_с_sa_содержат_пятёрку() -> None:
    assert requested_licenses(allow_sa=True) == "9,10,4,5"
