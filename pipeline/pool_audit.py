"""Сплошной ручной аудит приватности обучающего пула.

Спека: `docs/superpowers/specs/2026-09-18-pool-privacy-audit-design.md`. Модуль — единый
источник правды для галереи, пилота и приёмки: таксономия вердиктов и клавиши, реестр,
правило исключения и предикат `full_pool_audited`. Клавиши живут здесь, а не в JS
галереи: страница получает их отсюда, и тест маппинга проверяет ровно то, что видит
человек.

Свидетель приёмки живёт вне автомата продвижения (`redaction_calibration/state.py`): туда
`full_pool_audited` попадает только из `END_TO_END_REPORTED`, а профиль не продвинут.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import astuple, dataclass
from pathlib import Path

from pipeline.plate_leak import LEAK_WITH_HISTORICAL

PENDING = "pending"
PLATE_VERDICTS: tuple[str, ...] = (
    "leak",
    "leak_hist",
    "clean_partial",
    "clean_unreadable",
    "clean_filled",
    "clean_no_plate",
)
FACE_VERDICTS: tuple[str, ...] = ("leak", "clean")

# Номер — на цифрах в порядке `PLATE_VERDICTS`, лицо — на буквах: у двух осей нет общей
# клавиши, и промах пальцем не перепишет вердикт чужой оси.
PLATE_KEYS: Mapping[str, str] = {str(i): v for i, v in enumerate(PLATE_VERDICTS, start=1)}
FACE_KEYS: Mapping[str, str] = {"f": "leak", "g": "clean"}

REGISTRY_COLUMNS: tuple[str, ...] = (
    "photo_id",
    "session",
    "plate_verdict",
    "face_verdict",
    "crop_sha256",
)
SESSION_SIZE = 300


class PoolAuditError(ValueError):
    """Реестр аудита противоречит сам себе, таксономии или пулу."""


@dataclass(frozen=True, slots=True)
class RegistryRow:
    """Вердикты человека по одному кадру пула.

    `crop_sha256` — sha256 байтов кропа, который человек видел в момент вердикта. Не имя
    файла `<sha256>.jpg`: имя — хеш оригинала, и сравнение с ним ничего бы не проверяло.
    """

    photo_id: str
    session: int
    plate_verdict: str
    face_verdict: str
    crop_sha256: str

    def __post_init__(self) -> None:
        if not self.photo_id:
            raise PoolAuditError("пустой photo_id")
        if self.plate_verdict not in (PENDING, *PLATE_VERDICTS):
            raise PoolAuditError(
                f"{self.photo_id}: неизвестный plate_verdict {self.plate_verdict!r}"
            )
        if self.face_verdict not in (PENDING, *FACE_VERDICTS):
            raise PoolAuditError(f"{self.photo_id}: неизвестный face_verdict {self.face_verdict!r}")

    @property
    def pending(self) -> bool:
        """Непросмотренная ось не равна `clean`: такой кадр блокирует приёмку."""
        return PENDING in (self.plate_verdict, self.face_verdict)

    @property
    def excluded(self) -> bool:
        """Любая утечка — номер, исторический номер или лицо — выводит кадр из обучения."""
        return self.plate_verdict in LEAK_WITH_HISTORICAL or self.face_verdict == "leak"


def read_registry(path: Path) -> list[RegistryRow]:
    """Прочитать реестр; шапка обязана совпасть с `REGISTRY_COLUMNS` целиком."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = tuple(reader.fieldnames or ())
        if header != REGISTRY_COLUMNS:
            raise PoolAuditError(f"{path}: шапка {header}, ожидалась {REGISTRY_COLUMNS}")
        return [
            RegistryRow(
                photo_id=row["photo_id"],
                session=int(row["session"]),
                plate_verdict=row["plate_verdict"],
                face_verdict=row["face_verdict"],
                crop_sha256=row["crop_sha256"],
            )
            for row in reader
        ]


def write_registry(path: Path, rows: Iterable[RegistryRow]) -> None:
    """Записать реестр в UTF-8 с LF — ту же форму отдаёт экспорт галереи."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(REGISTRY_COLUMNS)
        for row in rows:
            writer.writerow(astuple(row))


def assign_sessions(photo_ids: Iterable[str], size: int = SESSION_SIZE) -> dict[str, int]:
    """Разложить кадры по сеансам по `size` в порядке `photo_id`.

    Сеанс — только группировка работы, не состояние. Порядок по `photo_id` (sha256
    оригинала) не зависит от порядка строк parquet и перемешивает кадры относительно
    номеров 300-выборки: в пилоте утечки не стоят на знакомых местах.
    """
    ordered = sorted(photo_ids)
    if len(set(ordered)) != len(ordered):
        raise PoolAuditError("photo_id встречается дважды")
    return {photo_id: index // size + 1 for index, photo_id in enumerate(ordered)}


def file_sha256(path: Path) -> str:
    """sha256 байтов файла — основание проверки протухших вердиктов."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
