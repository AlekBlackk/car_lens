"""Скачивание весов строго по приколоченной ревизии и запись их SHA256 в манифест.

`huggingface_hub` импортируется внутри функции, а не на уровне модуля: тесты этого
модуля проверяют арифметику хеша и переписывание CSV, и тянуть ради них сетевую
библиотеку незачем — то же правило, по которому `training/build_index.py` держит torch
внутри функций (Р-15).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from spike.clip_zeroshot.manifest import ModelPin, load_models_manifest

СТОЛБЦЫ = ("name", "repo_id", "revision", "url", "license", "checked", "weights_sha256")


def sha256_of_file(path: Path) -> str:
    """sha256 файла в hex, потоково — веса бывают в сотни мегабайт."""
    хеш = hashlib.sha256()
    with path.open("rb") as поток:
        for блок in iter(lambda: поток.read(1 << 20), b""):
            хеш.update(блок)
    return хеш.hexdigest()


def write_weights_sha256(path: Path, значения: dict[str, str]) -> None:
    """Проставить SHA256 по имени кандидата, остальные столбцы оставив как есть."""
    строки = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    for строка in строки:
        имя = строка["name"].strip()
        if имя in значения:
            строка["weights_sha256"] = значения[имя]
    with path.open("w", encoding="utf-8", newline="") as поток:
        писатель = csv.DictWriter(поток, fieldnames=list(СТОЛБЦЫ))
        писатель.writeheader()
        писатель.writerows(строки)


def download(пин: ModelPin, кэш: Path) -> Path:
    """Скачать снапшот репозитория ровно на приколоченной ревизии."""
    from huggingface_hub import snapshot_download

    каталог = snapshot_download(
        repo_id=пин.repo_id,
        revision=пин.revision,
        cache_dir=str(кэш),
    )
    return Path(каталог)


def main() -> None:
    """Скачать все четыре кандидата и записать SHA256 их весовых файлов в манифест."""
    from spike.clip_zeroshot.manifest import DEFAULT_MODELS_MANIFEST

    кэш = Path.home() / "carlens-data" / "spike_models"
    кэш.mkdir(parents=True, exist_ok=True)
    значения: dict[str, str] = {}
    for пин in load_models_manifest(DEFAULT_MODELS_MANIFEST):
        каталог = download(пин, кэш)
        веса = sorted(
            [*каталог.glob("*.safetensors"), *каталог.glob("*.bin"), *каталог.glob("*.pt")]
        )
        if not веса:
            raise FileNotFoundError(f"{пин.name}: в снапшоте {каталог} нет файлов весов")
        значения[пин.name] = ";".join(f"{ф.name}={sha256_of_file(ф)}" for ф in веса)
        print(f"{пин.name}: {каталог}")
    write_weights_sha256(DEFAULT_MODELS_MANIFEST, значения)


if __name__ == "__main__":
    main()
