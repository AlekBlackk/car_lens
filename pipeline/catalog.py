"""Каталог марок и моделей: загрузка и валидация `canonical.yaml`.

`model_id` собирается из марки и ключа модели и нигде не хранится
отдельным полем. Отдельное поле рассинхронизируется — та же причина,
по которой §13.2 не хранит индекс класса рядом с классом.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog" / "canonical.yaml"

SUPPORTED_VERSION = 1

_MODEL_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*__[a-z0-9]+(?:_[a-z0-9]+)*$")


class CatalogError(ValueError):
    """Каталог не проходит валидацию."""


@dataclass(frozen=True, slots=True)
class CarModel:
    """Одна модель автомобиля."""

    model_id: str
    brand: str
    display: str
    platform_group: str | None


@dataclass(frozen=True, slots=True)
class Catalog:
    """Неизменяемый набор моделей, отсортированный по `model_id`."""

    models: tuple[CarModel, ...]

    def __len__(self) -> int:
        return len(self.models)

    def __contains__(self, model_id: object) -> bool:
        return any(m.model_id == model_id for m in self.models)

    def by_id(self, model_id: str) -> CarModel:
        """Модель по идентификатору. `KeyError`, если такой нет."""
        for m in self.models:
            if m.model_id == model_id:
                return m
        raise KeyError(model_id)

    def brands(self) -> tuple[str, ...]:
        """Отсортированный кортеж уникальных марок."""
        return tuple(sorted({m.brand for m in self.models}))

    def models_of(self, brand: str) -> tuple[CarModel, ...]:
        """Модели одной марки, в порядке `model_id`."""
        return tuple(m for m in self.models if m.brand == brand)


def brand_slug(brand: str) -> str:
    """Каноническая марка → слаг для `model_id`.

    `Mercedes-Benz` → `mercedes_benz`, `Land Rover` → `land_rover`,
    `Citroën` → `citroen`. Диакритика снимается, чтобы идентификатор
    оставался ASCII: он попадает в имена файлов и в JSON на устройстве.
    """
    без_диакритики = "".join(
        c for c in unicodedata.normalize("NFKD", brand) if not unicodedata.combining(c)
    )
    очищено = re.sub(r"[^a-z0-9]+", "_", без_диакритики.lower())
    return очищено.strip("_")


def load_catalog(path: Path) -> Catalog:
    """Прочитать и провалидировать `canonical.yaml`."""
    сырое = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(сырое, dict):
        raise CatalogError(f"{path}: корень должен быть отображением")

    версия = сырое.get("version")
    if версия != SUPPORTED_VERSION:
        raise CatalogError(f"{path}: version={версия!r}, поддерживается только {SUPPORTED_VERSION}")

    марки = сырое.get("brands") or {}
    if not isinstance(марки, dict):
        raise CatalogError(f"{path}: brands должно быть отображением")

    модели = tuple(sorted(_разобрать_марки(path, марки), key=lambda m: m.model_id))
    _проверить_уникальность(path, модели)
    _проверить_группы_платформ(path, модели)
    return Catalog(models=модели)


def _разобрать_марки(path: Path, марки: dict[str, Any]) -> list[CarModel]:
    результат: list[CarModel] = []
    for марка, тело in марки.items():
        слаг = brand_slug(str(марка))
        if not слаг:
            raise CatalogError(f"{path}: марка {марка!r} даёт пустой слаг")
        подмодели = (тело or {}).get("models") or {}
        if not isinstance(подмодели, dict):
            raise CatalogError(f"{path}: {марка}.models должно быть отображением")
        for ключ, поля in подмодели.items():
            результат.append(_разобрать_модель(path, марка, слаг, str(ключ), поля or {}))
    return результат


def _разобрать_модель(
    path: Path, марка: str, слаг: str, ключ: str, поля: dict[str, Any]
) -> CarModel:
    model_id = f"{слаг}__{brand_slug(ключ)}"
    if not _MODEL_ID_RE.match(model_id):
        raise CatalogError(f"{path}: {model_id!r} не подходит под формат brand__model")

    display = str(поля.get("display", "")).strip()
    if not display:
        raise CatalogError(f"{path}: у {model_id} пустой display")

    группа = поля.get("platform_group")
    return CarModel(
        model_id=model_id,
        brand=str(марка),
        display=display,
        platform_group=str(группа) if группа else None,
    )


def _проверить_уникальность(path: Path, модели: tuple[CarModel, ...]) -> None:
    счёт = Counter(m.model_id for m in модели)
    повторы = sorted(i for i, n in счёт.items() if n > 1)
    if повторы:
        raise CatalogError(f"{path}: дубликат model_id: {', '.join(повторы)}")


def _проверить_группы_платформ(path: Path, модели: tuple[CarModel, ...]) -> None:
    группы: dict[str, list[str]] = defaultdict(list)
    for m in модели:
        if m.platform_group:
            группы[m.platform_group].append(m.model_id)
    одиночки = sorted(g for g, участники in группы.items() if len(участники) < 2)
    if одиночки:
        raise CatalogError(
            f"{path}: platform_group из одного участника бессмысленна: {', '.join(одиночки)}"
        )
