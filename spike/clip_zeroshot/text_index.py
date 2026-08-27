"""Индекс §13.2, у которого опорный вектор класса — текстовый эмбеддинг его имени.

Формат файла и арифметика поиска не меняются: меняется происхождение векторов и их
число на класс — один вместо четырёх. Поэтому `write_index` и `meta_json` берутся из
`pipeline/index_format.py` как есть, а текстовая сторона дописывается в мету отдельным
блоком `text_source`: `CatalogIndex.kt` разбирает мету с `ignoreUnknownKeys = true`,
поэтому лишний ключ его не роняет, а без этого блока индекс невоспроизводим.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pipeline.catalog import Catalog
from pipeline.index_format import meta_json, write_index
from spike.clip_zeroshot.labels import PROMPT_TEMPLATE, prompt_for
from spike.clip_zeroshot.manifest import ModelPin

CENTROIDS_PER_CLASS = 1


class TextIndexError(ValueError):
    """Индекс собрать нельзя."""


def build_text_index(
    catalog: Catalog,
    имена: dict[str, str],
    encode: Callable[[list[str]], np.ndarray],
    пин: ModelPin,
    tokenizer: str,
    out_dir: Path,
    generated_at: str,
) -> Path:
    """Собрать пару `catalog_index.bin` + `catalog_meta.json` из текстовых эмбеддингов."""
    пропущенные = [к.class_id for к in catalog.classes if к.class_id not in имена]
    if пропущенные:
        raise TextIndexError(f"нет латинского имени у: {', '.join(пропущенные[:5])}")

    промпты = [prompt_for(имена[к.class_id]) for к in catalog.classes]
    вектора = np.asarray(encode(промпты), dtype=np.float32)
    if вектора.shape[0] != len(catalog.classes):
        raise TextIndexError(
            f"кодировщик вернул {вектора.shape[0]} векторов на {len(catalog.classes)} классов"
        )

    нормы = np.linalg.norm(вектора, axis=1, keepdims=True)
    if bool(np.any(нормы == 0)):
        raise TextIndexError("нулевой текстовый эмбеддинг: по нему не построить косинус")
    центроиды = (вектора / нормы).astype(np.float32)[:, None, :]

    out_dir.mkdir(parents=True, exist_ok=True)
    документ = json.loads(
        meta_json(
            catalog.classes,
            embedding_dim=int(центроиды.shape[2]),
            centroids_per_class=CENTROIDS_PER_CLASS,
            model_weights_sha256=пин.weights_sha256,
            generated_at=generated_at,
        )
    )
    документ["text_source"] = {
        "repo_id": пин.repo_id,
        "revision": пин.revision,
        "license": пин.license,
        "tokenizer": tokenizer,
        "prompt_template": PROMPT_TEMPLATE,
        "labels": {к.class_id: имена[к.class_id] for к in catalog.classes},
    }
    # Сериализация повторяет форму `meta_json` дословно: другой `indent` дал бы другой
    # sha256 при том же содержимом, и заголовок разошёлся бы с файлом.
    meta_bytes = json.dumps(документ, ensure_ascii=False, indent=2).encode("utf-8")

    путь_меты = out_dir / "catalog_meta.json"
    путь_меты.write_bytes(meta_bytes)
    путь_индекса = out_dir / "catalog_index.bin"
    write_index(путь_индекса, центроиды, meta_bytes=путь_меты.read_bytes())
    return путь_индекса
