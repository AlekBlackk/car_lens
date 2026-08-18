"""Сверка происхождения модели и индекса (§4.6 дизайна M-1.4).

Индекс ищет по центроидам, посчитанным какой-то моделью. Если экспортирована
другая — метрики паритета мерят не квантизацию, а рассинхрон, причём выглядит это
как «INT8 просел». Проверка стоит перед всеми замерами и роняет прогон.

Одного sha чекпоинта мало: `best.pt` содержит и `model_state`, и `ema_state`, то
есть двое разных весов при одном хеше файла. Поэтому режим весов сверяется
отдельно.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ProvenanceError(ValueError):
    """Модель и индекс происходят из разных весов."""


def check_provenance(
    *,
    export_manifest: Mapping[str, Any],
    catalog_meta: Mapping[str, Any],
    index_report: Mapping[str, Any] | None,
    index_manifest: Mapping[str, Any] | None,
    int8_sha256: str,
) -> None:
    """Проверить, что экспортированная модель и индекс — из одних весов.

    Ветвление по `index_manifest`: его наличие означает индекс, собранный на
    выходах квантованной модели (§4.7 дизайна), и тогда `model_weights_sha256`
    меты — это хеш `.tflite`, а не чекпоинта.
    """
    размерность_модели = int(export_manifest["embedding_dim"])
    размерность_индекса = int(catalog_meta["embedding_dim"])
    if размерность_модели != размерность_индекса:
        raise ProvenanceError(
            f"embedding_dim модели {размерность_модели} против {размерность_индекса} у индекса"
        )

    if index_manifest is None:
        _проверить_fp32_индекс(export_manifest, catalog_meta, index_report)
        return
    _проверить_int8_индекс(export_manifest, catalog_meta, index_manifest, int8_sha256)


def _проверить_fp32_индекс(
    export_manifest: Mapping[str, Any],
    catalog_meta: Mapping[str, Any],
    index_report: Mapping[str, Any] | None,
) -> None:
    """Индекс собран самим PyTorch: мета несёт sha чекпоинта."""
    ожидаемый = str(export_manifest["checkpoint_sha256"])
    фактический = str(catalog_meta["model_weights_sha256"])
    if ожидаемый != фактический:
        raise ProvenanceError(
            f"sha256 чекпоинта {ожидаемый} против {фактический} в catalog_meta.json"
        )
    if index_report is None:
        raise ProvenanceError(
            "рядом с индексом нет report.json: неизвестно, на каких весах он собран "
            "(raw или ema) — пересоберите индекс либо укажите другой каталог"
        )
    веса_модели = str(export_manifest["weights"])
    веса_индекса = str(index_report["weights"])
    if веса_модели != веса_индекса:
        raise ProvenanceError(
            f"модель экспортирована с весами {веса_модели}, индекс собран на {веса_индекса}: "
            "один файл чекпоинта содержит двое весов, хеш их не различает"
        )


def _проверить_int8_индекс(
    export_manifest: Mapping[str, Any],
    catalog_meta: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    int8_sha256: str,
) -> None:
    """Индекс собран на выходах `.tflite`: мета несёт sha того файла."""
    источник = str(index_manifest["source_model_sha256"])
    if источник != int8_sha256:
        raise ProvenanceError(
            f"index_manifest.source_model_sha256 {источник} против {int8_sha256} "
            "у classifier_int8.tflite"
        )
    хеш_меты = str(catalog_meta["model_weights_sha256"])
    if хеш_меты != int8_sha256:
        raise ProvenanceError(f"model_weights_sha256 меты {хеш_меты} против {int8_sha256} у модели")
    ожидаемый = str(export_manifest["checkpoint_sha256"])
    фактический = str(index_manifest["checkpoint_sha256"])
    if ожидаемый != фактический:
        raise ProvenanceError(
            f"sha256 чекпоинта {ожидаемый} против {фактический} в index_manifest.json"
        )
    версия_модели = int(export_manifest["preprocess_version"])
    версия_индекса = int(index_manifest["preprocess_version"])
    if версия_модели != версия_индекса:
        raise ProvenanceError(
            f"preprocess_version модели {версия_модели} против {версия_индекса} у индекса"
        )
