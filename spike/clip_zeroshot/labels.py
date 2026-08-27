"""Латинские имена классов и фиксированный шаблон промпта (§5.3 спеки).

Таблица ведётся руками и иначе не выйдет: `display` каталога у отечественных классов
кириллический, а слаг `model_id` даёт «Cs35 Plus» вместо «CS35 Plus». Полнота таблицы
и её ASCII-чистота держатся тестом, а не вниманием автора.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_LATIN_NAMES = Path(__file__).resolve().parent / "latin_names.yaml"
PROMPT_TEMPLATE = "a photo of a {name}, a car"


def prompt_for(имя: str) -> str:
    """Единственный шаблон спайка. Подбор промптов запрещён спекой §5.3."""
    return PROMPT_TEMPLATE.format(name=имя)


def load_latin_names(path: Path) -> dict[str, str]:
    """Прочитать `class_id` → латинское имя."""
    сырое: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    таблица = сырое.get("names") or {}
    if not isinstance(таблица, dict):
        raise ValueError(f"{path}: names должно быть отображением")
    результат: dict[str, str] = {}
    for class_id, имя in таблица.items():
        текст = str(имя or "").strip()
        if not текст:
            raise ValueError(f"{path}: пустое имя у {class_id}")
        результат[str(class_id)] = текст
    return результат
