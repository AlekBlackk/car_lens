"""Тесты загрузки и валидации каталога марок и моделей."""

from pathlib import Path

import pytest

from pipeline.catalog import CatalogError, class_index, load_catalog, member_index

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
  Peugeot:
    models:
      "108":
        display: Peugeot 108
groups:
  grp__kei_duo:
    members: [toyota__aygo, peugeot__108]
    display: "Toyota Aygo / Peugeot 108"
"""

ГРУППА_ОДНОЙ_МАРКИ = """
version: 1
brands:
  Volkswagen:
    models:
      golf: {display: Volkswagen Golf}
      passat: {display: Volkswagen Passat}
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
  Peugeot:
    models:
      "108": {display: Peugeot 108}
groups:
  grp__vw_duo:
    members: [volkswagen__golf, volkswagen__passat]
    display: "Volkswagen Golf / Passat"
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


def test_участник_помечен_идентификатором_своей_группы(tmp_path: Path) -> None:
    """`platform_group` производно от секции `groups:`, а не читается с модели."""
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    assert каталог.by_id("toyota__aygo").platform_group == "grp__kei_duo"
    assert каталог.by_id("peugeot__108").platform_group == "grp__kei_duo"


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


def test_platform_group_на_модели_отвергается(tmp_path: Path) -> None:
    """Снятое поле нельзя игнорировать молча: группа тогда потерялась бы."""
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo:
        display: Toyota Aygo
        platform_group: kei_trio
"""
    with pytest.raises(CatalogError, match="groups"):
        load_catalog(_записать(tmp_path, текст))


def test_группа_из_одного_участника_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
groups:
  grp__kei_trio:
    members: [toyota__aygo]
    display: Toyota Aygo
"""
    with pytest.raises(CatalogError, match="двух участников"):
        load_catalog(_записать(tmp_path, текст))


def test_неизвестная_версия_схемы_отвергается(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="version"):
        load_catalog(_записать(tmp_path, "version: 99\nbrands: {}\n"))


def test_реальный_каталог_проекта_валиден() -> None:
    from pipeline.catalog import DEFAULT_CATALOG_PATH

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    assert len(каталог.brands()) >= 40
    # Миграция kei_trio из полей моделей в секцию `groups:` — состав группы прежний.
    assert member_index(каталог)["toyota__aygo"] == "grp__kei_trio"
    assert member_index(каталог)["citroen__c1"] == "grp__kei_trio"


def test_группа_даёт_один_класс_на_двух_моделей(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    assert len(каталог.classes) == 3
    assert len(каталог.classes) < len(каталог)
    группа = next(к for к in каталог.classes if к.class_id == "grp__kei_duo")
    assert группа.members == ("toyota__aygo", "peugeot__108")
    assert группа.display == "Toyota Aygo / Peugeot 108"


def test_одиночная_модель_становится_классом_сама(tmp_path: Path) -> None:
    """Частный случай группы из одного участника — тот же тип, тот же код (§13.2)."""
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    класс = next(к for к in каталог.classes if к.class_id == "volkswagen__golf")
    assert класс.members == ("volkswagen__golf",)
    assert класс.brand == "Volkswagen"
    assert класс.display == "Volkswagen Golf"


def test_member_index_ведёт_обоих_участников_в_один_класс(tmp_path: Path) -> None:
    индекс = member_index(load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML)))
    assert индекс == {
        "toyota__aygo": "grp__kei_duo",
        "peugeot__108": "grp__kei_duo",
        "volkswagen__golf": "volkswagen__golf",
        "volkswagen__passat": "volkswagen__passat",
    }


def test_индекс_нумерует_классы_а_не_модели(tmp_path: Path) -> None:
    индекс = class_index(load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML)))
    assert индекс == {"grp__kei_duo": 0, "volkswagen__golf": 1, "volkswagen__passat": 2}


def test_группа_встаёт_на_место_первого_участника(tmp_path: Path) -> None:
    """Класс нумеруется по минимальному `model_id` участников, а не по `class_id`.

    Сортировка по `class_id` собрала бы все `grp__` в начало списка и
    перенумеровала бы весь каталог при вводе первой же группы — обученный
    чекпоинт перестал бы соответствовать `catalog_index.bin` §13.2.
    """
    индекс = class_index(load_catalog(_записать(tmp_path, ГРУППА_ОДНОЙ_МАРКИ)))
    assert индекс == {"peugeot__108": 0, "toyota__aygo": 1, "grp__vw_duo": 2}


def test_индекс_не_зависит_от_порядка_в_yaml(tmp_path: Path) -> None:
    """Перестановка записей в файле не должна менять ни одного class_idx.

    Иначе правка YAML тихо перенумеровала бы классы, а обученные веса и
    `catalog_index.bin` §13.2 стали бы указывать не туда.
    """
    перевёрнутый = """
version: 1
groups:
  grp__kei_duo:
    members: [toyota__aygo, peugeot__108]
    display: "Toyota Aygo / Peugeot 108"
brands:
  Peugeot:
    models:
      "108": {display: Peugeot 108}
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
  Volkswagen:
    models:
      passat: {display: Volkswagen Passat}
      golf: {display: Volkswagen Golf}
"""
    прямой = class_index(load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML)))
    обратный = class_index(load_catalog(_записать(tmp_path / "два", перевёрнутый)))
    assert прямой == обратный


def test_индексы_классов_плотные() -> None:
    """Дыра в нумерации дала бы пустой слот в `catalog_index.bin` §13.2 и выход
    головы ArcFace, который никогда не активируется.
    """
    from pipeline.catalog import DEFAULT_CATALOG_PATH

    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    индекс = class_index(каталог)
    assert len(индекс) == len(каталог.classes)
    assert sorted(индекс.values()) == list(range(len(каталог.classes)))


def test_brand_группы_пуст_при_разных_марках(tmp_path: Path) -> None:
    """§13.2: показать одну марку из двух было бы угадыванием."""
    каталог = load_catalog(_записать(tmp_path, ВАЛИДНЫЙ_YAML))
    группа = next(к for к in каталог.classes if к.class_id == "grp__kei_duo")
    assert группа.brand is None


def test_brand_группы_общий_при_одной_марке(tmp_path: Path) -> None:
    каталог = load_catalog(_записать(tmp_path, ГРУППА_ОДНОЙ_МАРКИ))
    группа = next(к for к in каталог.classes if к.class_id == "grp__vw_duo")
    assert группа.brand == "Volkswagen"


def test_class_id_группы_без_префикса_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
  Peugeot:
    models:
      "108": {display: Peugeot 108}
groups:
  kei_trio:
    members: [toyota__aygo, peugeot__108]
    display: Toyota Aygo / Peugeot 108
"""
    with pytest.raises(CatalogError, match="grp__"):
        load_catalog(_записать(tmp_path, текст))


def test_model_id_с_префиксом_группы_отвергается(tmp_path: Path) -> None:
    """Иначе `class_id` одиночной модели совпал бы с `class_id` группы."""
    текст = """
version: 1
brands:
  Grp:
    models:
      kei_trio: {display: Grp Kei Trio}
"""
    with pytest.raises(CatalogError, match="grp__"):
        load_catalog(_записать(tmp_path, текст))


def test_участник_в_двух_группах_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
  Peugeot:
    models:
      "108": {display: Peugeot 108}
  Citroen:
    models:
      c1: {display: Citroën C1}
groups:
  grp__kei_duo:
    members: [toyota__aygo, peugeot__108]
    display: Toyota Aygo / Peugeot 108
  grp__kei_alt:
    members: [toyota__aygo, citroen__c1]
    display: Toyota Aygo / Citroën C1
"""
    with pytest.raises(CatalogError, match="повторно"):
        load_catalog(_записать(tmp_path, текст))


def test_неизвестный_участник_группы_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
groups:
  grp__kei_duo:
    members: [toyota__aygo, peugeot__108]
    display: Toyota Aygo / Peugeot 108
"""
    with pytest.raises(CatalogError, match="peugeot__108"):
        load_catalog(_записать(tmp_path, текст))


def test_пустой_display_группы_отвергается(tmp_path: Path) -> None:
    текст = """
version: 1
brands:
  Toyota:
    models:
      aygo: {display: Toyota Aygo}
  Peugeot:
    models:
      "108": {display: Peugeot 108}
groups:
  grp__kei_duo:
    members: [toyota__aygo, peugeot__108]
    display: "  "
"""
    with pytest.raises(CatalogError, match="display"):
        load_catalog(_записать(tmp_path, текст))


def test_индексация_stanford_не_изменилась() -> None:
    """171 класс скелета M-1 обязан сохранить прежние индексы: иначе
    `catalog_index.bin` и веса в `bench/` перестают соответствовать друг другу.
    """
    from pipeline.catalog import STANFORD_CATALOG_PATH

    каталог = load_catalog(STANFORD_CATALOG_PATH)
    assert len(каталог.classes) == 171
    assert [к.class_id for к in каталог.classes] == [м.model_id for м in каталог.models]
    assert каталог.classes[0].class_id == "acura__integra_type_r"
    assert class_index(каталог)["volvo__xc90"] == 170
