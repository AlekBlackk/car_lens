from pipeline.catalog import STANFORD_CATALOG_PATH, load_catalog
from spike.clip_zeroshot.run_stanford import stanford_names


def test_имена_берутся_из_display_и_латинские() -> None:
    """У Stanford `display` уже английский — ручная таблица здесь была бы дублем."""
    каталог = load_catalog(STANFORD_CATALOG_PATH)
    имена = stanford_names(каталог)
    assert len(имена) == len(каталог.classes) == 171
    assert all(имя.isascii() for имя in имена.values())
    assert имена["acura__integra_type_r"] == "Acura Integra Type R"
