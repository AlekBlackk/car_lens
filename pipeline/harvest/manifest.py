"""Юридический след выборки: по строке на каждую принятую фотографию.

Файл отслеживается git и живёт вне каталога данных: он обязан пережить чистку
`~/carlens-data/`, потому что доказывает право на использование каждого кадра.

`source_item_id` здесь не украшение, а защита от утечки. Один автор часто заливает
серию из пяти ракурсов одного экземпляра машины; случайный сплит разнесёт её между
`train` и `holdout_ru`, и метрика покажет то, чего нет, — ровно та ошибка, от которой
§3.5 спеки защищается сплитом по `listing_id`.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pipeline.harvest.licenses import is_allowed, tier_of

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent / "provenance" / "harvest_manifest.csv"
)

_ТЕГИ = re.compile(r"<[^>]+>")
ОБЯЗАТЕЛЬНЫЕ = (
    "file",
    "class_id",
    "source",
    "source_item_id",
    "page_url",
    "file_url",
    "author",
    "license",
    "checked",
    "basis",
    "sha256",
)


class ManifestError(ValueError):
    """Манифест не годится: выборку нельзя пускать в данные."""


@dataclass(frozen=True, slots=True)
class HarvestRow:
    """Одна принятая фотография с полным провенансом и числами конвейера.

    Геометрия бокса главной машины хранится в пикселях исходного кадра, а не кропа: кроп
    — это уже вырезанный бокс, и внутри него координаты вырождаются в рамку самого файла.
    §13.1 просит рамку в системе координат исходной фотографии.
    """

    file: str
    class_id: str
    source: str
    source_item_id: str
    page_url: str
    file_url: str
    author: str
    license: str
    license_tier: int
    checked: str
    basis: str
    sha256: str
    phash: int
    clip_rank: int
    clip_margin: float
    plates: int
    faces: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    bbox_area_frac: float
    det_conf: float
    audit: str


COLUMNS = tuple(поле.name for поле in fields(HarvestRow))


def strip_html(value: str) -> str:
    """Снять разметку с поля автора: манифест читает человек, а не браузер."""
    return _ТЕГИ.sub("", value).strip()


def append_row(path: Path, row: HarvestRow) -> None:
    """Дописать строку, создав файл с шапкой при первом вызове."""
    path.parent.mkdir(parents=True, exist_ok=True)
    новый = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(COLUMNS))
        if новый:
            писатель.writeheader()
        писатель.writerow(asdict(row))


def write_manifest(path: Path, rows: Sequence[HarvestRow]) -> None:
    """Перезаписать манифест целиком. Атомарно — во временный файл и rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    временный = path.with_suffix(path.suffix + ".tmp")
    with временный.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(COLUMNS))
        писатель.writeheader()
        for строка in rows:
            писатель.writerow(asdict(строка))
    временный.replace(path)



def load_manifest(path: Path) -> list[HarvestRow]:
    """Прочитать манифест целиком, восстановив типы чисел."""
    строки = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    return [
        HarvestRow(
            file=с["file"],
            class_id=с["class_id"],
            source=с["source"],
            source_item_id=с["source_item_id"],
            page_url=с["page_url"],
            file_url=с["file_url"],
            author=с["author"],
            license=с["license"],
            license_tier=int(с["license_tier"]),
            checked=с["checked"],
            basis=с["basis"],
            sha256=с["sha256"],
            phash=int(с["phash"]),
            clip_rank=int(с["clip_rank"]),
            clip_margin=float(с["clip_margin"]),
            plates=int(с["plates"]),
            faces=int(с["faces"]),
            bbox_x=float(с["bbox_x"]),
            bbox_y=float(с["bbox_y"]),
            bbox_w=float(с["bbox_w"]),
            bbox_h=float(с["bbox_h"]),
            bbox_area_frac=float(с["bbox_area_frac"]),
            det_conf=float(с["det_conf"]),
            audit=с["audit"],
        )
        for с in строки
    ]


def validate(rows: Sequence[HarvestRow], allow_sa: bool) -> None:
    """Проверить весь манифест. Это гейт H4, а не советы."""
    видели: set[str] = set()
    for номер, строка in enumerate(rows, start=2):
        for поле in ОБЯЗАТЕЛЬНЫЕ:
            if not str(getattr(строка, поле)).strip():
                raise ManifestError(f"строка {номер}: пустое поле {поле}")
        if not is_allowed(строка.license, allow_sa=True):
            raise ManifestError(f"строка {номер}: лицензия {строка.license!r} не разрешена Р-11")
        if not is_allowed(строка.license, allow_sa=allow_sa):
            raise ManifestError(
                f"строка {номер}: {строка.license} требует флага --allow-sa"
            )
        тир = tier_of(строка.license)
        assert тир is not None  # проверено выше
        if int(тир) != строка.license_tier:
            raise ManifestError(
                f"строка {номер}: тир {строка.license_tier} не совпадает с лицензией"
            )
        if строка.sha256 in видели:
            raise ManifestError(f"строка {номер}: повтор sha256 {строка.sha256}")
        видели.add(строка.sha256)
        отметка = строка.audit
        if отметка not in {"", "ok", "conflict"} and not отметка.startswith("drop:"):
            raise ManifestError(f"строка {номер}: непонятная отметка audit {отметка!r}")


def holdout_candidates(rows: Sequence[HarvestRow]) -> list[HarvestRow]:
    """Кадры, годные в `holdout_ru`: только прошедшие ручной аудит (правило 2 Р-11)."""
    return [строка for строка in rows if строка.audit == "ok"]
