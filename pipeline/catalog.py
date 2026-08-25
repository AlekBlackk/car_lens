"""Каталог марок и моделей: загрузка и валидация `canonical.yaml`.

`model_id` собирается из марки и ключа модели и нигде не хранится
отдельным полем. Отдельное поле рассинхронизируется — та же причина,
по которой §13.2 не хранит индекс класса рядом с классом.

Единица индексации — класс, а не модель (§13.2): модели, отличающиеся
только шильдиком, занимают один слот в `catalog_index.bin` и один выход
головы ArcFace. Состав платформенных групп объявлен секцией `groups:`, а
не полем модели: поле задавало группу неявно и не оставляло места для её
`display`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog" / "canonical.yaml"
STANFORD_CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog" / "stanford.yaml"

SUPPORTED_VERSION = 1

_MODEL_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*__[a-z0-9]+(?:_[a-z0-9]+)*$")
_GROUP_ID_PREFIX = "grp__"


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
class CarClass:
    """Класс обучения: один слот в `catalog_index.bin`, один выход головы (§13.2).

    Одиночная модель — частный случай группы из одного участника и
    описывается тем же типом. Индекс класса не хранится: он равен позиции
    в `Catalog.classes`. `brand` пуст у разнобрендовой группы.
    """

    class_id: str
    members: tuple[str, ...]
    brand: str | None
    display: str


@dataclass(frozen=True, slots=True)
class Catalog:
    """Модели в порядке `model_id` и классы в порядке своих индексов."""

    models: tuple[CarModel, ...]
    classes: tuple[CarClass, ...]

    def __len__(self) -> int:
        """Число моделей. Классов бывает меньше — платформенная группа их объединяет."""
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

    сырые_группы = сырое.get("groups") or {}
    if not isinstance(сырые_группы, dict):
        raise CatalogError(f"{path}: groups должно быть отображением")

    модели = tuple(sorted(_разобрать_марки(path, марки), key=lambda m: m.model_id))
    _проверить_уникальность(path, модели)
    классы = _собрать_классы(path, модели, сырые_группы)
    return Catalog(models=_проставить_группы(модели, классы), classes=классы)


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
    # Иначе `class_id` одиночной модели совпал бы с `class_id` группы, и два
    # класса схлопнулись бы в один индекс.
    if model_id.startswith(_GROUP_ID_PREFIX):
        raise CatalogError(
            f"{path}: {model_id!r}: префикс {_GROUP_ID_PREFIX!r} закреплён за группами"
        )

    display = str(поля.get("display", "")).strip()
    if not display:
        raise CatalogError(f"{path}: у {model_id} пустой display")

    if "platform_group" in поля:
        raise CatalogError(
            f"{path}: у {model_id} снятый ключ platform_group: "
            "состав групп объявляется секцией groups: (§13.2)"
        )

    return CarModel(model_id=model_id, brand=str(марка), display=display, platform_group=None)


def _проверить_уникальность(path: Path, модели: tuple[CarModel, ...]) -> None:
    счёт = Counter(m.model_id for m in модели)
    повторы = sorted(i for i, n in счёт.items() if n > 1)
    if повторы:
        raise CatalogError(f"{path}: дубликат model_id: {', '.join(повторы)}")


def _собрать_классы(
    path: Path, модели: tuple[CarModel, ...], сырые_группы: dict[str, Any]
) -> tuple[CarClass, ...]:
    по_id = {m.model_id: m for m in модели}
    группы = [
        _разобрать_группу(path, str(class_id), тело or {}, по_id)
        for class_id, тело in сырые_группы.items()
    ]
    _проверить_участников(path, группы)
    в_группах = {model_id for г in группы for model_id in г.members}
    одиночки = [
        CarClass(class_id=m.model_id, members=(m.model_id,), brand=m.brand, display=m.display)
        for m in модели
        if m.model_id not in в_группах
    ]
    # Позиция класса и есть его индекс, поэтому порядок обязан быть определён
    # однозначно и не зависеть от порядка записей в YAML. Ключ — минимальный
    # `model_id` среди участников, а не `class_id`: по `class_id` все `grp__`
    # собрались бы в начало списка, и первая же группа перенумеровала бы весь
    # каталог, оставив обученный чекпоинт без соответствия ему.
    return tuple(sorted(группы + одиночки, key=lambda к: min(к.members)))


def _разобрать_группу(
    path: Path, class_id: str, тело: dict[str, Any], по_id: dict[str, CarModel]
) -> CarClass:
    if not class_id.startswith(_GROUP_ID_PREFIX):
        raise CatalogError(
            f"{path}: class_id группы {class_id!r} обязан начинаться с {_GROUP_ID_PREFIX!r}"
        )

    участники = тело.get("members") or []
    if not isinstance(участники, list):
        raise CatalogError(f"{path}: {class_id}.members должно быть списком model_id")
    члены = tuple(str(u) for u in участники)
    if len(члены) < 2:
        raise CatalogError(f"{path}: у группы {class_id} меньше двух участников: объединять нечего")

    неизвестные = [model_id for model_id in члены if model_id not in по_id]
    if неизвестные:
        raise CatalogError(
            f"{path}: в группе {class_id} нет таких моделей: {', '.join(неизвестные)}"
        )

    display = str(тело.get("display", "")).strip()
    if not display:
        raise CatalogError(f"{path}: у группы {class_id} пустой display")

    марки = {по_id[model_id].brand for model_id in члены}
    # §13.2: у разнобрендовой группы бренда нет. Показать одну марку из двух
    # было бы угадыванием, и brand-only §6.4 в таком случае бренд не показывает.
    return CarClass(
        class_id=class_id,
        members=члены,
        brand=next(iter(марки)) if len(марки) == 1 else None,
        display=display,
    )


def _проверить_участников(path: Path, группы: list[CarClass]) -> None:
    счёт = Counter(model_id for г in группы for model_id in г.members)
    повторы = sorted(model_id for model_id, n in счёт.items() if n > 1)
    if повторы:
        raise CatalogError(f"{path}: участник указан в группах повторно: {', '.join(повторы)}")


def _проставить_группы(
    модели: tuple[CarModel, ...], классы: tuple[CarClass, ...]
) -> tuple[CarModel, ...]:
    """Проставить моделям `platform_group` по составу классов.

    Поле производно от секции `groups:` и с модели не читается: состав,
    записанный в двух местах, рассинхронизируется. У класса из одного
    участника группы нет — это сама модель.
    """
    группа = {
        model_id: класс.class_id
        for класс in классы
        if len(класс.members) > 1
        for model_id in класс.members
    }
    return tuple(replace(m, platform_group=группа.get(m.model_id)) for m in модели)


def class_index(catalog: Catalog) -> dict[str, int]:
    """`class_id` → индекс класса, равный позиции в `catalog.classes`.

    Индекс вычисляется, а не хранится, — по той же причине, по которой не
    хранится `model_id`: поле рядом со значением рассинхронизируется.
    Порядок классов задан `load_catalog`, поэтому результат не зависит от
    порядка записей в YAML, и правка файла не перенумерует классы молча.
    Индексы плотные, `0..N-1`: дыра дала бы пустой слот в
    `catalog_index.bin` §13.2 и выход головы, который не активируется.
    """
    return {к.class_id: i for i, к in enumerate(catalog.classes)}


def member_index(catalog: Catalog) -> dict[str, str]:
    """`model_id` → `class_id`: обратное отображение из `members[]` (§13.2).

    Применяется при ingest, а не при нормализации: в датасете хранится
    исходный `model_id`, чтобы группы можно было пересобрать, не
    пересобирая данные.
    """
    return {model_id: к.class_id for к in catalog.classes for model_id in к.members}
