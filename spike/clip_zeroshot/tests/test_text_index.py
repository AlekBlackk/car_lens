import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.index_format import read_index, verify_meta
from spike.clip_zeroshot.labels import DEFAULT_LATIN_NAMES, PROMPT_TEMPLATE, load_latin_names
from spike.clip_zeroshot.manifest import ModelPin
from spike.clip_zeroshot.text_index import TextIndexError, build_text_index

ПИН = ModelPin(
    name="tinyclip8m",
    repo_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
    revision="a" * 40,
    url="https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
    license="mit",
    checked="2026-08-26",
    weights_sha256="model.safetensors=" + "b" * 64,
)


def _кодировщик(размерность: int = 8):
    """Детерминированный фейк: эмбеддинг зависит только от текста промпта."""

    def encode(промпты: list[str]) -> np.ndarray:
        вектора = np.zeros((len(промпты), размерность), dtype=np.float32)
        for строка, промпт in enumerate(промпты):
            цифра = int(hashlib.sha256(промпт.encode()).hexdigest()[:8], 16)
            генератор = np.random.default_rng(цифра)
            вектора[строка] = генератор.standard_normal(размерность, dtype=np.float32)
        return вектора

    return encode


def test_пара_согласована_и_читается(tmp_path: Path) -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES)

    путь = build_text_index(
        каталог, имена, _кодировщик(), ПИН, "clip", tmp_path, "2026-08-26T00:00:00Z"
    )

    заголовок, центроиды = read_index(путь)
    мета = (tmp_path / "catalog_meta.json").read_bytes()
    verify_meta(заголовок, мета)
    assert заголовок.num_classes == 150
    assert заголовок.centroids_per_class == 1
    assert центроиды.shape == (150, 1, 8)


def test_вектора_нормированы(tmp_path: Path) -> None:
    """kNN на устройстве считает голый косинус: ненормированный вектор ломает шкалу."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES)

    путь = build_text_index(
        каталог, имена, _кодировщик(), ПИН, "clip", tmp_path, "2026-08-26T00:00:00Z"
    )

    _, центроиды = read_index(путь)
    нормы = np.linalg.norm(центроиды.reshape(-1, 8), axis=1)
    assert np.allclose(нормы, 1.0, atol=1e-5)


def test_мета_хранит_текстовую_сторону(tmp_path: Path) -> None:
    """Без этих полей два индекса, собранных в разные дни, неотличимы (§5.3)."""
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES)

    build_text_index(каталог, имена, _кодировщик(), ПИН, "clip", tmp_path, "2026-08-26T00:00:00Z")

    документ = json.loads((tmp_path / "catalog_meta.json").read_text(encoding="utf-8"))
    источник = документ["text_source"]
    assert источник["repo_id"] == ПИН.repo_id
    assert источник["revision"] == ПИН.revision
    assert источник["tokenizer"] == "clip"
    assert источник["prompt_template"] == PROMPT_TEMPLATE
    assert источник["labels"]["lada__granta"] == имена["lada__granta"]
    assert документ["classes"][0]["class_id"] == каталог.classes[0].class_id


def test_недостающее_имя_роняет_сборку(tmp_path: Path) -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = dict(load_latin_names(DEFAULT_LATIN_NAMES))
    имена.pop("lada__granta")

    with pytest.raises(TextIndexError, match="lada__granta"):
        build_text_index(
            каталог, имена, _кодировщик(), ПИН, "clip", tmp_path, "2026-08-26T00:00:00Z"
        )


def test_кодировщик_получает_промпты_в_порядке_классов(tmp_path: Path) -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    имена = load_latin_names(DEFAULT_LATIN_NAMES)
    увиденное: list[str] = []

    def encode(промпты: list[str]) -> np.ndarray:
        увиденное.extend(промпты)
        return _кодировщик()(промпты)

    build_text_index(каталог, имена, encode, ПИН, "clip", tmp_path, "2026-08-26T00:00:00Z")

    assert len(увиденное) == 150
    первый = каталог.classes[0].class_id
    assert увиденное[0] == f"a photo of a {имена[первый]}, a car"
