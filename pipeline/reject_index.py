"""Отказные прототипы: `reject_index.bin` и `reject_meta.json` (спека 2026-09-28 §2).

Тело — тот же формат CLIX §13.2, что у `catalog_index.bin`, мета — в схеме
`catalog_meta.json` (`class_id`, `members`, `brand`, `display`): на устройстве оба файла
разбирает один `CatalogIndex.parse`, второго разборщика формата нет. Вид прототипа и
происхождение пула едут лишними полями — `kotlinx` их пропускает.

k-means передаётся параметром, как в `pipeline/centroids.py`: sklearn стоит только в venv
обучения и экспорта, а раскладка проверяется в CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from pipeline.centroids import centroids_for_class
from pipeline.index_format import FORMAT_VERSION, read_index, verify_meta, write_index

REJECT_KINDS = ("off_catalog_street", "truck_bus")
CENTROIDS_PER_KIND = 32
CENTROIDS_PER_CLASS = 4
REJECT_INDEX_NAME = "reject_index.bin"
REJECT_META_NAME = "reject_meta.json"


class RejectIndexError(ValueError):
    """Прототипы нельзя собрать или им нельзя верить."""


def build_reject_centroids(
    embeddings_by_kind: Mapping[str, np.ndarray],
    *,
    kmeans: Callable[[np.ndarray, int], np.ndarray],
    centroids_per_kind: int = CENTROIDS_PER_KIND,
) -> tuple[np.ndarray, list[str]]:
    """Центроиды `(классы, 4, dim)` и вид каждого класса; виды — в порядке `REJECT_KINDS`.

    Кадров меньше `centroids_per_kind` — падение, а не добивка повтором, как у классов
    каталога: K зафиксирован до замера, и молча меньше кластеров значило бы мерить не то.
    """
    if centroids_per_kind % CENTROIDS_PER_CLASS:
        raise RejectIndexError(f"K={centroids_per_kind} не делится на {CENTROIDS_PER_CLASS}")
    классов = centroids_per_kind // CENTROIDS_PER_CLASS
    части: list[np.ndarray] = []
    виды: list[str] = []
    for вид in REJECT_KINDS:
        эмбеддинги = embeddings_by_kind.get(вид)
        кадров = 0 if эмбеддинги is None else len(эмбеддинги)
        if эмбеддинги is None or кадров < centroids_per_kind:
            raise RejectIndexError(f"вид {вид}: кадров {кадров}, нужно ≥ {centroids_per_kind}")
        центры = centroids_for_class(
            эмбеддинги, centroids_per_class=centroids_per_kind, kmeans=kmeans
        )
        части.append(центры.reshape(классов, CENTROIDS_PER_CLASS, центры.shape[1]))
        виды += [вид] * классов
    return np.concatenate(части).astype(np.float32), виды


def _мета(
    kinds: Sequence[str],
    *,
    embedding_dim: int,
    classifier_sha256: str,
    pool_sha256: str,
    frames: Mapping[str, int],
    generated_at: str,
) -> bytes:
    документ = {
        "format_version": FORMAT_VERSION,
        "generated_at": generated_at,
        "classifier_sha256": classifier_sha256,
        "pool_sha256": pool_sha256,
        "embedding_dim": embedding_dim,
        "centroids_per_class": CENTROIDS_PER_CLASS,
        "frames": dict(frames),
        "classes": [
            {
                "class_id": f"reject__{вид}__{номер:02d}",
                "members": [],
                "brand": None,
                "display": "",
                "kind": вид,
            }
            for номер, вид in enumerate(kinds)
        ],
    }
    return json.dumps(документ, ensure_ascii=False, indent=2).encode("utf-8")


def write_reject_index(
    out_dir: Path,
    centroids: np.ndarray,
    kinds: Sequence[str],
    *,
    classifier_sha256: str,
    pool_sha256: str,
    frames: Mapping[str, int],
    generated_at: str,
) -> Path:
    """Записать оба файла; sha меты уходит в заголовок индекса, как у каталога."""
    out_dir.mkdir(parents=True, exist_ok=True)
    мета = _мета(
        kinds,
        embedding_dim=int(centroids.shape[2]),
        classifier_sha256=classifier_sha256,
        pool_sha256=pool_sha256,
        frames=frames,
        generated_at=generated_at,
    )
    (out_dir / REJECT_META_NAME).write_bytes(мета)
    write_index(out_dir / REJECT_INDEX_NAME, centroids, meta_bytes=мета)
    return out_dir


def read_reject_index(directory: Path, classifier_sha256: str) -> np.ndarray:
    """Центроиды прототипов, если их собрал тот же классификатор, что меряется.

    Прототипы от другой модели живут в чужом пространстве эмбеддингов: сходства с ними
    правдоподобны и бессмысленны, поэтому несовпадение — падение, а не предупреждение.
    """
    заголовок, центроиды = read_index(directory / REJECT_INDEX_NAME)
    байты = (directory / REJECT_META_NAME).read_bytes()
    verify_meta(заголовок, байты)
    собран = json.loads(байты).get("classifier_sha256")
    if собран != classifier_sha256:
        raise RejectIndexError(
            f"прототипы собраны классификатором {собран}, а меряется {classifier_sha256}"
        )
    return центроиды
