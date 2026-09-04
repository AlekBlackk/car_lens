"""Карта `pageid → sha256` для кандидатов, отбитых редакцией.

В `reject_log.csv` шесть колонок, и `sha256` среди них нет, а кэш оригиналов адресуется
именно им. Карты не существует больше нигде: манифест хранит её только для принятых
строк, `state.json.done_items` — голые метки `commons:<pageid>`. Без этого файла замер
редакции не находит ни одного оригинала.

Скрипт берёт из сети **только метаданные**: `iiprop=sha1`, ни одной картинки. Граница
«здесь сеть, дальше офлайн» проходит по этому файлу и потому проверяема.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from pipeline.harvest.commons import API_URL, Transport, request_with_backoff
from pipeline.harvest.rejects import Reject, load_rejects, unique_by_reason
from pipeline.paths import harvest_dir

BATCH = 50


@dataclass(frozen=True, slots=True)
class IndexRow:
    """Одна строка карты. Пустой `sha256` — оригинала в кэше нет."""

    class_id: str
    source_item_id: str
    sha256: str


INDEX_COLUMNS = tuple(поле.name for поле in fields(IndexRow))


def карта_кэша(root: Path, source: str) -> dict[str, str]:
    """`sha1 содержимого → sha256` по всему кэшу оригиналов.

    Строится один раз за прогон: имя файла в кэше — уже готовый `sha256`, считать надо
    только SHA-1. Пересчёт на каждом батче API означал бы дюжину проходов по 25 ГБ.
    """
    карта: dict[str, str] = {}
    for путь in sorted((root / "raw" / source).rglob("*.jpg")):
        карта[hashlib.sha1(путь.read_bytes()).hexdigest()] = путь.stem
    return карта


def _батчи(значения: Sequence[str], размер: int) -> Iterator[Sequence[str]]:
    for начало in range(0, len(значения), размер):
        yield значения[начало : начало + размер]


def запросить_sha1(
    transport: Transport,
    pageids: Sequence[str],
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """`pageid → sha1` батчами по пятьдесят. Страница без `sha1` в карту не попадает."""
    карта: dict[str, str] = {}
    for батч in _батчи(pageids, BATCH):
        ответ = request_with_backoff(
            transport,
            API_URL,
            {
                "action": "query",
                "format": "json",
                "pageids": "|".join(батч),
                "prop": "imageinfo",
                "iiprop": "sha1",
            },
            sleep=sleep,
        )
        данные: Any = json.loads(ответ.body.decode("utf-8"))
        for ключ, страница in ((данные.get("query") or {}).get("pages") or {}).items():
            сведения = (страница.get("imageinfo") or [{}])[0]
            if сведения.get("sha1"):
                карта[str(ключ)] = str(сведения["sha1"])
    return карта


def build_index(
    rejects: Sequence[Reject],
    кэш: dict[str, str],
    sha1_по_странице: dict[str, str],
) -> list[IndexRow]:
    """Склеить карту. Ненайденный оригинал даёт строку с пустым `sha256`, а не пропуск."""
    строки: list[IndexRow] = []
    for отказ in rejects:
        sha1 = sha1_по_странице.get(отказ.source_item_id, "")
        строки.append(IndexRow(отказ.class_id, отказ.source_item_id, кэш.get(sha1, "")))
    return строки


def записать_индекс(path: Path, строки: Sequence[IndexRow]) -> None:
    """Перезаписать карту целиком: она производная, дописывать нечего."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(INDEX_COLUMNS))
        писатель.writeheader()
        for строка in строки:
            писатель.writerow(asdict(строка))


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="карта pageid→sha256 для отбитых кандидатов")
    парсер.add_argument("--data-root", type=Path, default=None)
    парсер.add_argument("--reason", default="redaction_failed")
    парсер.add_argument("--source", default="commons")
    парсер.add_argument("--contact", required=True, help="контакт для User-Agent (§3.2)")
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать карту и напечатать, скольким кандидатам оригинал не нашёлся."""
    from pipeline.harvest.backends import UrllibTransport

    аргументы = build_parser().parse_args(argv)
    корень = аргументы.data_root if аргументы.data_root is not None else harvest_dir()

    кандидаты = unique_by_reason(load_rejects(корень / "reject_log.csv"), аргументы.reason)
    print(f"кандидатов с причиной {аргументы.reason}: {len(кандидаты)}")

    кэш = карта_кэша(корень, аргументы.source)
    print(f"файлов в кэше: {len(кэш)}")

    транспорт = UrllibTransport(аргументы.contact)
    sha1 = запросить_sha1(транспорт, [кандидат.source_item_id for кандидат in кандидаты])
    строки = build_index(кандидаты, кэш, sha1)

    путь = корень / "reject_index.csv"
    записать_индекс(путь, строки)
    без_оригинала = sum(1 for строка in строки if not строка.sha256)
    print(f"карта: {путь} — {len(строки)} строк, без оригинала {без_оригинала}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
