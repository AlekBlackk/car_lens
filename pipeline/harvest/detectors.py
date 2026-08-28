"""Пины весов детекторов редакции — входное условие прогона.

Правило то же, что у `models_manifest.csv` на спайке Р-10: лицензия проверяется на
конкретные веса, а не на репозиторий, и приколачивается SHA256 файла. Прогон при
неполном манифесте недействителен — иначе «номера замыты» держится на честном слове.

Формат строго ONNX: он читается штатным `cv2.dnn` и пинуется одним SHA256, тогда как
IR — это пара `.xml` + `.bin`, у которой можно подменить половину. Файл, полученный
конвертацией, обязан нести `derived_from` с исходным SHA256 и дословной командой.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DETECTORS_MANIFEST = Path(__file__).resolve().parent / "detectors_manifest.csv"

REQUIRED_DETECTORS = ("plate", "face")
ALLOWED_LICENSES = frozenset({"mit", "apache-2.0"})
ОБЯЗАТЕЛЬНЫЕ_ПОЛЯ = ("name", "format", "url", "revision", "license", "checked", "sha256")


class DetectorManifestError(ValueError):
    """Манифест детекторов не годится: прогон недействителен."""


@dataclass(frozen=True, slots=True)
class DetectorPin:
    """Один детектор с приколоченной ревизией и лицензией весов."""

    name: str
    format: str
    url: str
    revision: str
    license: str
    checked: str
    sha256: str
    derived_from: str


def load_detectors_manifest(path: Path) -> dict[str, DetectorPin]:
    """Прочитать манифест и провалидировать его целиком."""
    строки = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    пины: dict[str, DetectorPin] = {}
    for номер, строка in enumerate(строки, start=2):
        for поле in ОБЯЗАТЕЛЬНЫЕ_ПОЛЯ:
            if not (строка.get(поле) or "").strip():
                raise DetectorManifestError(f"{path}:{номер}: пустое поле {поле}")
        имя = строка["name"].strip()
        формат = строка["format"].strip().lower()
        if формат != "onnx":
            raise DetectorManifestError(f"{path}:{номер}: формат {формат!r}, допустим только onnx")
        лицензия = строка["license"].strip().lower()
        if лицензия not in ALLOWED_LICENSES:
            raise DetectorManifestError(f"{path}:{номер}: лицензия {лицензия!r} не свободная")
        sha = строка["sha256"].strip().lower()
        if len(sha) != 64:
            raise DetectorManifestError(f"{path}:{номер}: sha256 должен быть 64 символа")
        derived = (строка.get("derived_from") or "").strip()
        url = строка["url"].strip()
        url_lower = url.lower()
        # Файл, чьё имя не совпадает с концом URL, получен конвертацией: без строки
        # происхождения доказать, каким файлом замывали, невозможно.
        if not url.endswith(Path(url).name):  # pragma: no cover
            raise DetectorManifestError(f"{path}:{номер}: URL без имени файла")
        if not derived and (not url_lower.endswith(".onnx") or "-converted" in url_lower):
            raise DetectorManifestError(
                f"{path}:{номер}: нужен derived_from для производного/не-ONNX источника"
            )
        if имя in пины:
            raise DetectorManifestError(f"{path}:{номер}: повтор детектора {имя!r}")
        пины[имя] = DetectorPin(
            name=имя,
            format=формат,
            url=url,
            revision=строка["revision"].strip(),
            license=лицензия,
            checked=строка["checked"].strip(),
            sha256=sha,
            derived_from=derived,
        )

    отсутствуют = [имя for имя in REQUIRED_DETECTORS if имя not in пины]
    if отсутствуют:
        raise DetectorManifestError(
            f"{path}: нет детекторов {', '.join(отсутствуют)} — правило 1 Р-11 необеспечимо"
        )
    return пины


def verify_weights(pins: dict[str, DetectorPin], models_dir: Path) -> None:
    """Сверить SHA256 скачанных файлов с манифестом."""
    for имя, пин in pins.items():
        файл = models_dir / f"{имя}.onnx"
        if not файл.exists():
            raise DetectorManifestError(f"{имя}: файл {файл} не найден")
        фактический = hashlib.sha256(файл.read_bytes()).hexdigest()
        if фактический != пин.sha256:
            raise DetectorManifestError(
                f"{имя}: sha256 {фактический} не совпадает с манифестом {пин.sha256}"
            )
