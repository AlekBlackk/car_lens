"""Чистая часть сборки few-shot индекса: раскладка кропов и полнота покрытия."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from pipeline.catalog import DEFAULT_CATALOG_PATH, class_index, load_catalog
from pipeline.holdout import listing_id_of
from scripts.build_fewshot_index import (
    ENCODER_NAME,
    ENCODER_REPO,
    ENCODER_REVISION,
    ENCODER_WEIGHTS_SHA256,
    BuildFewShotError,
    build_parser,
    check_full_coverage,
    crops_by_class,
)
from scripts.build_holdout_ru import строки_в_кадр
from tests.test_holdout import _класс

МАНИФЕСТ_СПАЙКА = Path("spike/clip_zeroshot/models_manifest.csv")


def _кадр(*классы: tuple[str, int]) -> pd.DataFrame:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    строки = [с for имя, сколько in классы for с in _класс(имя, сколько)]
    единицы = {с.file: listing_id_of(с) for с in строки}
    return строки_в_кадр(строки, каталог, единицы)


def test_кропы_группируются_по_индексу_класса() -> None:
    кадр = _кадр(("lada__2107", 3), ("kia__rio", 2))
    группы = crops_by_class(кадр, Path("/кропы"))
    номера = [номер for номер, _ in группы]
    assert номера == sorted(номера)
    сколько = {номер: len(пути) for номер, пути in группы}
    ожидание = кадр.groupby("class_idx").size().to_dict()
    assert сколько == {int(к): int(з) for к, з in ожидание.items()}  # type: ignore[call-overload]


def test_путь_собирается_из_корня_и_crop_path() -> None:
    кадр = _кадр(("lada__2107", 1))
    группы = crops_by_class(кадр, Path("/кропы"))
    _, пути = группы[0]
    assert пути == [Path("/кропы") / кадр["crop_path"].iloc[0]]


def test_пустой_сплит_отвергается() -> None:
    from pipeline.schema import empty_frame

    with pytest.raises(BuildFewShotError, match="сплит пуст"):
        crops_by_class(empty_frame(), Path("/кропы"))


def test_строка_без_class_idx_отвергается() -> None:
    кадр = _кадр(("lada__2107", 2))
    # Колонка подменяется целиком: присваивание `None` в int64-колонку pandas 2.3
    # встречает предупреждением о будущем понижении типа, и тест ловил бы его, а не дефект.
    кадр["class_idx"] = [None, *кадр["class_idx"].tolist()[1:]]
    with pytest.raises(BuildFewShotError, match="без class_idx"):
        crops_by_class(кадр, Path("/кропы"))


def test_неполное_покрытие_называет_недостающие_классы() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    индекс = class_index(каталог)
    по_индексу = {значение: ключ for ключ, значение in индекс.items()}
    группы = crops_by_class(_кадр(("lada__2107", 2)), Path("/кропы"))
    with pytest.raises(BuildFewShotError, match="без кадров"):
        check_full_coverage(группы, len(индекс), по_индексу)


def test_полное_покрытие_молчит() -> None:
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    индекс = class_index(каталог)
    по_индексу = {значение: ключ for ключ, значение in индекс.items()}
    кадр = _кадр(*((класс.class_id, 1) for класс in каталог.classes))
    check_full_coverage(crops_by_class(кадр, Path("/кропы")), len(индекс), по_индексу)


def test_пин_энкодера_совпадает_с_манифестом_спайка() -> None:
    """Копия пина обязана сходиться с манифестом Р-10, иначе мерят не те веса."""
    строки = list(csv.DictReader(МАНИФЕСТ_СПАЙКА.read_text(encoding="utf-8").splitlines()))
    пин = next(с for с in строки if с["name"] == ENCODER_NAME)
    assert пин["repo_id"] == ENCODER_REPO
    assert пин["revision"] == ENCODER_REVISION
    хеши = dict(часть.split("=", 1) for часть in пин["weights_sha256"].split(";"))
    assert хеши["open_clip_pytorch_model.bin"] == ENCODER_WEIGHTS_SHA256


def test_покрытие_с_allow_missing_возвращает_дефицитные_классы() -> None:
    """Дефицит эталона — измеренное состояние (16 классов на 2026-08-31), а не поломка.

    Названия обязаны выйти наружу: их печатает сборка и по ним живёт задача C.
    """
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    индекс = class_index(каталог)
    по_индексу = {значение: ключ for ключ, значение in индекс.items()}
    группы = crops_by_class(_кадр(("lada__2107", 2)), Path("/кропы"))
    дефицит = check_full_coverage(группы, len(индекс), по_индексу, allow_missing=True)
    assert "lada__2107" not in дефицит
    assert len(дефицит) == len(индекс) - 1


def test_дефицит_разрешается_только_флагом() -> None:
    """По умолчанию сборка обязана падать: молчаливый пустой слот — тот самый дефект."""
    assert build_parser().parse_args([]).allow_missing_classes is False
    assert build_parser().parse_args(["--allow-missing-classes"]).allow_missing_classes is True
