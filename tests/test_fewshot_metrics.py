"""Метрики few-shot замера: top-1, top-3 и поклассовый top-3 (Р-11)."""

from __future__ import annotations

import pytest

from pipeline.fewshot.metrics import FEWSHOT_TOP_K, aggregate_fewshot

# Четыре кадра: попадание top-1, попадание только в top-3, полный промах и попадание
# третьим кандидатом. Разобрано вручную, чтобы ожидаемые числа не зависели от кода.
РАНЖИРОВАНИЕ = [[0, 1, 2], [1, 0, 2], [2, 3, 4], [4, 5, 2]]
МЕТКИ = [0, 0, 1, 2]
ИМЕНА = {n: f"класс{n}" for n in range(6)}


def test_top1_и_top3_считаются_по_разным_срезам() -> None:
    метрики = aggregate_fewshot(РАНЖИРОВАНИЕ, МЕТКИ, idx_to_class_id=ИМЕНА)
    assert метрики.top1 == pytest.approx(0.25)
    assert метрики.top3 == pytest.approx(0.75)


def test_поклассовый_top3_и_опора() -> None:
    метрики = aggregate_fewshot(РАНЖИРОВАНИЕ, МЕТКИ, idx_to_class_id=ИМЕНА)
    assert метрики.per_class_top3 == {"класс0": 1.0, "класс1": 0.0, "класс2": 1.0}
    assert метрики.per_class_support == {"класс0": 2, "класс1": 1, "класс2": 1}


def test_класса_без_кадров_в_отчёте_нет() -> None:
    """Классу без эталона нельзя приписать ни попадание, ни промах."""
    метрики = aggregate_fewshot(РАНЖИРОВАНИЕ, МЕТКИ, idx_to_class_id=ИМЕНА)
    assert "класс5" not in метрики.per_class_top3


def test_короткое_ранжирование_отвергается() -> None:
    with pytest.raises(ValueError, match=f"top-{FEWSHOT_TOP_K}"):
        aggregate_fewshot([[0, 1]], [0], idx_to_class_id=ИМЕНА)


def test_несовпадение_длин_отвергается() -> None:
    with pytest.raises(ValueError, match="несовпадение длин"):
        aggregate_fewshot(РАНЖИРОВАНИЕ, МЕТКИ[:2], idx_to_class_id=ИМЕНА)


def test_пустая_выборка_отвергается() -> None:
    with pytest.raises(ValueError, match="пустая выборка"):
        aggregate_fewshot([], [], idx_to_class_id=ИМЕНА)


def test_неинъективная_карта_имён_отвергается() -> None:
    with pytest.raises(ValueError, match="неинъективен"):
        aggregate_fewshot(РАНЖИРОВАНИЕ, МЕТКИ, idx_to_class_id={n: "один" for n in range(6)})


def test_метка_вне_каталога_ронит_оценку() -> None:
    with pytest.raises(KeyError):
        aggregate_fewshot([[0, 1, 2]], [99], idx_to_class_id=ИМЕНА)
