"""Сверка того, что модель и индекс происходят из одних весов (§4.6 дизайна M-1.4)."""

from __future__ import annotations

from typing import Any

import pytest

from export.provenance import ProvenanceError, check_provenance

ЧЕКПОИНТ_SHA = "aa" * 32
INT8_SHA = "bb" * 32

МАНИФЕСТ: dict[str, Any] = {
    "checkpoint_sha256": ЧЕКПОИНТ_SHA,
    "weights": "raw",
    "embedding_dim": 256,
    "preprocess_version": 1,
}
МЕТА: dict[str, Any] = {"model_weights_sha256": ЧЕКПОИНТ_SHA, "embedding_dim": 256}
ОТЧЁТ_ИНДЕКСА: dict[str, Any] = {"weights": "raw", "embedding_dim": 256}


def test_fp32_индекс_из_того_же_чекпоинта_проходит() -> None:
    check_provenance(
        export_manifest=МАНИФЕСТ,
        catalog_meta=МЕТА,
        index_report=ОТЧЁТ_ИНДЕКСА,
        index_manifest=None,
        int8_sha256=INT8_SHA,
    )


def test_чужой_чекпоинт_отвергается() -> None:
    with pytest.raises(ProvenanceError, match="sha256"):
        check_provenance(
            export_manifest=МАНИФЕСТ,
            catalog_meta={**МЕТА, "model_weights_sha256": "cc" * 32},
            index_report=ОТЧЁТ_ИНДЕКСА,
            index_manifest=None,
            int8_sha256=INT8_SHA,
        )


def test_ema_против_raw_отвергается() -> None:
    """Один файл best.pt содержит двое весов: sha файла их не различает."""
    with pytest.raises(ProvenanceError, match="ema"):
        check_provenance(
            export_manifest={**МАНИФЕСТ, "weights": "ema"},
            catalog_meta=МЕТА,
            index_report=ОТЧЁТ_ИНДЕКСА,
            index_manifest=None,
            int8_sha256=INT8_SHA,
        )


def test_отсутствие_отчёта_индекса_отвергается() -> None:
    """Молчаливое допущение `raw` — ровно та ошибка, которую веха обязана ловить."""
    with pytest.raises(ProvenanceError, match="на каких весах"):
        check_provenance(
            export_manifest=МАНИФЕСТ,
            catalog_meta=МЕТА,
            index_report=None,
            index_manifest=None,
            int8_sha256=INT8_SHA,
        )


def test_разная_размерность_эмбеддинга_отвергается() -> None:
    with pytest.raises(ProvenanceError, match="embedding_dim"):
        check_provenance(
            export_manifest=МАНИФЕСТ,
            catalog_meta={**МЕТА, "embedding_dim": 128},
            index_report=ОТЧЁТ_ИНДЕКСА,
            index_manifest=None,
            int8_sha256=INT8_SHA,
        )


def test_int8_индекс_сверяется_с_файлом_модели() -> None:
    check_provenance(
        export_manifest=МАНИФЕСТ,
        catalog_meta={**МЕТА, "model_weights_sha256": INT8_SHA},
        index_report=None,
        index_manifest={
            "source": "tflite",
            "source_model_sha256": INT8_SHA,
            "checkpoint_sha256": ЧЕКПОИНТ_SHA,
            "preprocess_version": 1,
            "weights": "raw",
        },
        int8_sha256=INT8_SHA,
    )


def test_int8_индекс_от_другой_модели_отвергается() -> None:
    with pytest.raises(ProvenanceError, match="source_model_sha256"):
        check_provenance(
            export_manifest=МАНИФЕСТ,
            catalog_meta={**МЕТА, "model_weights_sha256": "dd" * 32},
            index_report=None,
            index_manifest={
                "source": "tflite",
                "source_model_sha256": "dd" * 32,
                "checkpoint_sha256": ЧЕКПОИНТ_SHA,
                "preprocess_version": 1,
                "weights": "raw",
            },
            int8_sha256=INT8_SHA,
        )


def test_разная_версия_препроцессинга_отвергается() -> None:
    with pytest.raises(ProvenanceError, match="preprocess_version"):
        check_provenance(
            export_manifest=МАНИФЕСТ,
            catalog_meta={**МЕТА, "model_weights_sha256": INT8_SHA},
            index_report=None,
            index_manifest={
                "source": "tflite",
                "source_model_sha256": INT8_SHA,
                "checkpoint_sha256": ЧЕКПОИНТ_SHA,
                "preprocess_version": 2,
                "weights": "raw",
            },
            int8_sha256=INT8_SHA,
        )
