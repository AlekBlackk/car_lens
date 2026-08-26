"""Манифест весов спайка: входное условие замера, а не отчётный побочный продукт.

Лицензия проверяется на конкретные веса и ревизию, а не на репозиторий: MIT у кода
`open_clip` ничего не говорит о лицензии LAION-чекпоинта, который в этот код грузится.
Поэтому пустая `revision` — ошибка загрузки, а не мелочь оформления (§4 спеки).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS_MANIFEST = Path(__file__).resolve().parent / "models_manifest.csv"

# `weights_sha256` заполняется скачиванием (Task 2), поэтому пустым он быть вправе.
ОБЯЗАТЕЛЬНЫЕ = ("name", "repo_id", "revision", "url", "license", "checked")


class ManifestError(ValueError):
    """Манифест неполон: замер по нему недействителен."""


@dataclass(frozen=True, slots=True)
class ModelPin:
    """Один кандидат линейки, приколоченный к ревизии."""

    name: str
    repo_id: str
    revision: str
    url: str
    license: str
    checked: str
    weights_sha256: str


def load_models_manifest(path: Path) -> list[ModelPin]:
    """Прочитать манифест весов и провалидировать полноту."""
    строки = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not строки:
        raise ManifestError(f"{path}: манифест пуст")

    пины: list[ModelPin] = []
    имена: set[str] = set()
    for номер, строка in enumerate(строки, start=2):
        for поле in ОБЯЗАТЕЛЬНЫЕ:
            if not (строка.get(поле) or "").strip():
                raise ManifestError(f"{path}:{номер}: пустое поле {поле}")
        имя = строка["name"].strip()
        if имя in имена:
            raise ManifestError(f"{path}:{номер}: повтор имени {имя!r}")
        имена.add(имя)
        пины.append(
            ModelPin(
                name=имя,
                repo_id=строка["repo_id"].strip(),
                revision=строка["revision"].strip(),
                url=строка["url"].strip(),
                license=строка["license"].strip(),
                checked=строка["checked"].strip(),
                weights_sha256=(строка.get("weights_sha256") or "").strip(),
            )
        )
    return пины
