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

import pandas as pd

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


def index_registry(rows: Iterable[RegistryRow]) -> dict[str, RegistryRow]:
    """Реестр по `photo_id`; повтор — ошибка, а не «последний выигрывает»."""
    indexed: dict[str, RegistryRow] = {}
    for row in rows:
        if row.photo_id in indexed:
            raise PoolAuditError(f"photo_id дважды в реестре: {row.photo_id}")
        indexed[row.photo_id] = row
    return indexed


@dataclass(frozen=True, slots=True)
class RegistryCheck:
    """Условия 1–3 предиката приёмки; пустой кортеж — условие выполнено."""

    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    pending_ids: tuple[str, ...]
    stale_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing_ids or self.extra_ids or self.pending_ids or self.stale_ids)


def check_registry(
    rows: Iterable[RegistryRow],
    *,
    pool_ids: Iterable[str],
    current_crop_sha256: Mapping[str, str | None],
) -> RegistryCheck:
    """Проверить условия 1–3: реестр покрывает ровно пул, без `pending`, без протухших.

    `current_crop_sha256` — sha256 байтов кропа на диске сейчас; `None` значит, что кропа
    нет, и вердикт по нему действующим не считается.
    """
    indexed = index_registry(rows)
    pool = set(pool_ids)
    present = pool & indexed.keys()
    pending = [i for i in present if indexed[i].pending]
    stale = [i for i in present if current_crop_sha256.get(i) != indexed[i].crop_sha256]
    return RegistryCheck(
        missing_ids=tuple(sorted(pool - indexed.keys())),
        extra_ids=tuple(sorted(indexed.keys() - pool)),
        pending_ids=tuple(sorted(pending)),
        stale_ids=tuple(sorted(stale)),
    )


def excluded_ids(rows: Iterable[RegistryRow]) -> frozenset[str]:
    """Кадры с любой утечкой. Вычисляются из вердиктов, отдельной колонкой не хранятся."""
    return frozenset(row.photo_id for row in rows if row.excluded)


def audited_pool(pool: pd.DataFrame, rows: Iterable[RegistryRow]) -> pd.DataFrame:
    """Пул минус кадры с утечкой — вход обучения вместо неизменного `pool.parquet`."""
    drop = excluded_ids(rows)
    return pool.loc[~pool["photo_id"].isin(drop)].reset_index(drop=True)


def audited_mismatch(
    rows: Iterable[RegistryRow], *, pool_ids: Iterable[str], audited_ids: Iterable[str]
) -> tuple[str, ...]:
    """Условие 4: `photo_id`, где `pool-audited` расходится с «пул минус утечки»."""
    expected = set(pool_ids) - excluded_ids(rows)
    return tuple(sorted(expected ^ set(audited_ids)))


# Лица 300-выборки, найденные мимоходом при замере номеров (`bench/plate-leak-bound.md` §8).
# Нижняя оценка, а не полная разметка: лицо сверх этого списка ошибкой пилота не считается.
KNOWN_FACE_LEAK_NUMBERS: tuple[int, ...] = (32, 36, 68, 102, 120, 179, 196, 218, 231, 254)


@dataclass(frozen=True, slots=True)
class PilotTruth:
    """Известные утечки 300-выборки: номера по колонке `вердикт`, лица по списку §8."""

    plate_ids: frozenset[str]
    face_ids: frozenset[str]


def pilot_truth(sample: pd.DataFrame) -> PilotTruth:
    """Эталон пилота из `bench/plate-leak-sample.csv` (колонки `номер`, `sha256`, `вердикт`)."""
    plates = sample.loc[sample["вердикт"].isin(LEAK_WITH_HISTORICAL), "sha256"]
    faces = sample.loc[sample["номер"].isin(KNOWN_FACE_LEAK_NUMBERS), "sha256"]
    if len(faces) != len(KNOWN_FACE_LEAK_NUMBERS):
        raise PoolAuditError(
            f"в выборке {len(faces)} кадров лиц из {len(KNOWN_FACE_LEAK_NUMBERS)} известных"
        )
    return PilotTruth(frozenset(map(str, plates)), frozenset(map(str, faces)))


@dataclass(frozen=True, slots=True)
class PilotScore:
    """Итог пилота против эталона; `passed` — гейт выполнимости формата."""

    plate_misses: tuple[str, ...]
    face_misses: tuple[str, ...]
    pending_ids: tuple[str, ...]
    plate_expected: int
    face_expected: int

    @property
    def passed(self) -> bool:
        """Пропуск хоть одного известного номера или лица — формат провален."""
        return not (self.plate_misses or self.face_misses or self.pending_ids)


def score_pilot(rows: Iterable[RegistryRow], truth: PilotTruth) -> PilotScore:
    """Сверить выгрузку пилота с эталоном. Кадр эталона без строки — промах."""
    indexed = index_registry(rows)
    plate_misses = [
        i
        for i in truth.plate_ids
        if i not in indexed or indexed[i].plate_verdict not in LEAK_WITH_HISTORICAL
    ]
    face_misses = [
        i for i in truth.face_ids if i not in indexed or indexed[i].face_verdict != "leak"
    ]
    return PilotScore(
        plate_misses=tuple(sorted(plate_misses)),
        face_misses=tuple(sorted(face_misses)),
        pending_ids=tuple(sorted(i for i, row in indexed.items() if row.pending)),
        plate_expected=len(truth.plate_ids),
        face_expected=len(truth.face_ids),
    )
