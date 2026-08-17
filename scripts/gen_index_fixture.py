"""Генератор фикстуры индекса для теста читалки на Node.

Фикстура синтетическая и маленькая (2 класса × 4 × 256 float32), поэтому лежит в
git: Node-тест не может собрать индекс сам — для этого нужны веса и torch, которых
в его окружении нет. Значения детерминированы сидом, `tests/test_index_fixture.py`
следит, чтобы файл в дереве не разъехался с генератором.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.catalog import CarModel
from pipeline.index_format import meta_json, write_index

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "index"

SEED = 1337
CLASSES = 2
CENTROIDS_PER_CLASS = 4
EMBEDDING_DIM = 256

_MODELS = (
    CarModel(model_id="acme__one", brand="Acme", display="Acme One", platform_group=None),
    CarModel(model_id="acme__two", brand="Acme", display="Acme Two", platform_group="p2"),
)


def build_fixture(directory: Path) -> None:
    """Записать пару файлов фикстуры в `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    генератор = np.random.default_rng(SEED)
    сырые = генератор.normal(size=(CLASSES, CENTROIDS_PER_CLASS, EMBEDDING_DIM))
    нормы = np.linalg.norm(сырые, axis=2, keepdims=True)
    центроиды = (сырые / нормы).astype(np.float32)
    мета = meta_json(
        _MODELS,
        embedding_dim=EMBEDDING_DIM,
        centroids_per_class=CENTROIDS_PER_CLASS,
        model_weights_sha256="00" * 32,
        generated_at="2026-08-17T00:00:00Z",
    )
    (directory / "catalog_meta.json").write_bytes(мета)
    записанная = (directory / "catalog_meta.json").read_bytes()
    write_index(directory / "catalog_index.bin", центроиды, meta_bytes=записанная)


def main() -> None:
    build_fixture(FIXTURE_DIR)
    print(f"фикстура записана в {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
