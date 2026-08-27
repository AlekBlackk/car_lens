import numpy as np

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from spike.clip_zeroshot.commons_classes import (
    COMMONS_CLASSES,
    CONTROL,
    CORE_G4,
    REBADGE_PAIRS,
)
from spike.clip_zeroshot.evaluate import accuracy_by_class, topk_hits


def test_состав_commons_ровно_25_и_существует_в_каталоге() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    известные = {к.class_id for к in каталог.classes}
    assert len(COMMONS_CLASSES) == 25
    assert len(set(COMMONS_CLASSES)) == 25
    assert set(COMMONS_CLASSES) <= известные


def test_срезы_вложены_в_состав() -> None:
    assert set(CONTROL) <= set(COMMONS_CLASSES)
    assert set(CORE_G4) <= set(COMMONS_CLASSES)
    assert len(CONTROL) == 3
    assert len(CORE_G4) == 4
    for левый, правый in REBADGE_PAIRS:
        assert левый in COMMONS_CLASSES, левый
        assert правый in COMMONS_CLASSES, правый


def test_topk_ловит_попадание_и_промах() -> None:
    центроиды = np.eye(4, dtype=np.float32)[:, None, :]
    эмбеддинги = np.array([[1, 0, 0, 0], [0, 0.9, 0.1, 0]], dtype=np.float32)
    метки = np.array([0, 2])

    assert topk_hits(эмбеддинги, центроиды, метки, k=1).tolist() == [True, False]
    assert topk_hits(эмбеддинги, центроиды, метки, k=2).tolist() == [True, True]


def test_accuracy_by_class_считает_по_классам() -> None:
    попадания = np.array([True, False, True, True])
    метки = np.array([0, 0, 1, 1])
    доли = accuracy_by_class(попадания, метки, классов=2)
    assert доли == {0: 0.5, 1: 1.0}


def test_класс_без_фото_в_доли_не_попадает() -> None:
    """Деление на ноль дало бы NaN, который тихо просочится в отчёт как «0.0»."""
    попадания = np.array([True])
    метки = np.array([1])
    assert accuracy_by_class(попадания, метки, классов=3) == {1: 1.0}
