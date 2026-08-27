from pathlib import Path

import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from spike.clip_zeroshot.labels import (
    DEFAULT_LATIN_NAMES,
    PROMPT_TEMPLATE,
    load_latin_names,
    prompt_for,
)


def test_шаблон_промпта_дословно_из_спеки() -> None:
    assert PROMPT_TEMPLATE == "a photo of a {name}, a car"
    assert prompt_for("Lada Granta") == "a photo of a Lada Granta, a car"


def test_таблица_покрывает_все_150_классов() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES)
    непокрытые = sorted({к.class_id for к in каталог.classes} - set(имена))
    assert непокрытые == [], f"нет латинского имени у: {непокрытые[:5]}"
    лишние = sorted(set(имена) - {к.class_id for к in каталог.classes})
    assert лишние == [], f"имена для несуществующих классов: {лишние[:5]}"


def test_все_имена_ascii() -> None:
    """Кириллица в промпте — шум: энкодер обучен на английских подписях (§5.3)."""
    имена = load_latin_names(DEFAULT_LATIN_NAMES)
    нелатинские = sorted(c for c, имя in имена.items() if not имя.isascii())
    assert нелатинские == [], f"не-ASCII имена: {нелатинские[:5]}"


def test_имена_не_повторяются() -> None:
    """Два класса с одним именем неразличимы по построению — это дыра, а не мелочь."""
    имена = load_latin_names(DEFAULT_LATIN_NAMES)
    повторы = {имя for имя in имена.values() if list(имена.values()).count(имя) > 1}
    assert повторы == set(), f"повторяющиеся имена: {sorted(повторы)[:5]}"


def test_группа_несёт_оба_имени() -> None:
    """У §5.1 группы кузов один, а шильдиков два — текст обязан назвать оба."""
    имена = load_latin_names(DEFAULT_LATIN_NAMES)
    assert имена["grp__coolray"] == "Geely Coolray or Belgee X50"
    assert имена["grp__jac_js4"] == "Moskvich 3 or JAC JS4"


def test_пустое_имя_отвергается(tmp_path: Path) -> None:
    путь = tmp_path / "latin_names.yaml"
    путь.write_text("version: 1\nnames:\n  lada__granta: ''\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пустое"):
        load_latin_names(путь)
