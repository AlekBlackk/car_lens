"""Тесты загрузки и валидации каталога марок и моделей."""

from pathlib import Path

import pytest

from pipeline.catalog import CatalogError, load_catalog

ВАЛИДНЫЙ_YAML = """
version: 1
brands:
  Volkswagen:
    models:
      golf:
        display: Volkswagen Golf
      passat:
        display: Volkswagen Passat
  Toyota:
    models:
      aygo:
        display: Toyota Aygo
        platform_group: kei_trio
  Peugeot:
    models:
      "108":
        display: Peugeot 108
        platform_group: kei_trio
"""


def _записать(tmp_path: Path, текст: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    путь = tmp_path / "canonical.yaml"
    путь.write_text(текст, encoding="utf-8")
    return путь


def test_загружает_все_модели(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    assert len(каталог) == 4
    assert "volkswagen__golf" in каталог
    assert "peugeot__108" in каталог


def test_собирает_model_id_из_марки_и_ключа(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    модель = каталог.by_id("volkswagen__golf")
    assert модель.brand == "Volkswagen"
    assert модель.display == "Volkswagen Golf"
    assert модель.platform_group is None


def test_марки_отсортированы_и_уникальны(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    assert каталог.brands() == ("Peugeot", "Toyota", "Volkswagen")


def test_models_of_возвращает_только_свою_марку(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    ids = [м.model_id for м in каталог.models_of("Volkswagen")]
    assert ids == ["volkswagen__golf", "volkswagen__passat"]


def test_platform_group_сохраняется(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    assert каталог.by_id("toyota__aygo").platform_group == "kei_trio"


def test_by_id_на_неизвестном_идентификаторе_падает(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    with pytest.raises(KeyError):
        каталог.by_id("nissan__leaf")


def test_дубликат_model_id_отвергается(tmp_path: Path) -> None:
    # Две записи марки, различающиеся только регистром, дают один и тот же слаг,
    # а значит и один и тот же model_id — это и должно быть отвергнуто.
    текст = """
version: 1
brands:
  Volkswagen:
    models:
      golf:
        display: Volkswagen Golf
  VOLKSWAGEN:
    models:
      golf:
        display: Дубликат
"""
    with pytest.raises(CatalogError, match="дубликат"):
        load_catalog(_записать(tmp_path, текст))


def test_пустой_display_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Volkswagen:
    models:
      golf:
        display: "  "
"""
    with pytest.raises(CatalogError, match="display"):
        load_catalog(_записать(tmp_path, текст))


def test_platform_group_в_одиночестве_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo:
        display: Toyota Aygo
        platform_group: kei_trio
"""
    with pytest.raises(CatalogError, match="platform_group"):
        load_catalog(_записать(tmp_path, текст))


def test_неизвестная_версия_схемы_отвергается(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="version"):
        load_catalog(_записать(tmp_path, "version: 99\nbrands: {}\n"))


def test_реальный_каталог_проекта_валиден() -> None:
    from pipeline.catalog import DEFAULT_CATALOG_PATH

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    assert len(каталог.brands()) >= 40


def test_индекс_классов_идёт_по_порядку_model_id(tmp_path: Path) -> None:
    from pipeline.catalog import class_index

    индекс = class_index(load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML)))
    assert индекс == {
        "peugeot__108": 0,
        "toyota__aygo": 1,
        "volkswagen__golf": 2,
        "volkswagen__passat": 3,
    }


def test_индекс_не_зависит_от_порядка_в_yaml(tmp_path: Path) -> None:
    """Перестановка марок в файле не должна менять ни одного class_idx.

    Иначе правка YAML тихо перенумеровала бы классы, а обученные веса и
    `catalog_index.bin` §13.2 стали бы указывать не туда.
    """
    from pipeline.catalog import class_index

    перевёрнутый = """
version: 1
brands:
  Peugeot:
    models:
      "108": {display: Peugeot 108, platform_group: kei_trio}
  Toyota:
    models:
      aygo: {display: Toyota Aygo, platform_group: kei_trio}
  Volkswagen:
    models:
      passat: {display: Volkswagen Passat}
      golf: {display: Volkswagen Golf}
"""
    прямой = class_index(load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML)))
    обратный = class_index(load_catalog(_записать(tmp_path / "два", перевёрнутый)))
    assert прямой == обратный
