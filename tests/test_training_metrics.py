"""Тесты метрик закрытого множества (M-1.2, §4.4 дизайна)."""

import pytest

from training.metrics import aggregate, brand_of

ИНДЕКС = {0: "acura__rl", 1: "acura__tl", 2: "bmw__x5"}


def test_марка_из_идентификатора_модели() -> None:
    assert brand_of("rolls_royce__phantom_drophead") == "rolls_royce"


def test_идеальные_предсказания() -> None:
    метрики = aggregate([[0, 1, 2], [2, 1, 0]], [0, 2], idx_to_model_id=ИНДЕКС)
    assert метрики.top1_model == pytest.approx(1.0)
    assert метрики.top5_model == pytest.approx(1.0)
    assert метрики.top1_brand == pytest.approx(1.0)


def test_ошибка_внутри_марки_портит_модель_но_не_марку() -> None:
    # Ответ acura__tl вместо acura__rl: модель неверна, марка верна.
    метрики = aggregate([[1, 0, 2]], [0], idx_to_model_id=ИНДЕКС)
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top1_brand == pytest.approx(1.0)
    assert метрики.top5_model == pytest.approx(1.0)


def test_top5_считается_по_всему_списку() -> None:
    метрики = aggregate([[2, 1, 0]], [0], idx_to_model_id=ИНДЕКС)
    assert метрики.top1_model == pytest.approx(0.0)
    assert метрики.top5_model == pytest.approx(1.0)


def test_recall_по_классам() -> None:
    метрики = aggregate(
        [[0, 1, 2], [1, 0, 2], [2, 0, 1]],
        [0, 0, 2],
        idx_to_model_id=ИНДЕКС,
    )
    assert метрики.per_class_recall["acura__rl"] == pytest.approx(0.5)
    assert метрики.per_class_recall["bmw__x5"] == pytest.approx(1.0)
    # Класс, которого нет в выборке, в recall не попадает: делить не на что.
    assert "acura__tl" not in метрики.per_class_recall


def test_пустая_выборка_отвергается() -> None:
    with pytest.raises(ValueError, match="пустая выборка"):
        aggregate([], [], idx_to_model_id=ИНДЕКС)


def test_несовпадение_длин_отвергается() -> None:
    with pytest.raises(ValueError, match="длин"):
        aggregate([[0, 1, 2]], [0, 1], idx_to_model_id=ИНДЕКС)


def test_неизвестный_индекс_отвергается() -> None:
    with pytest.raises(KeyError):
        aggregate([[7, 1, 2]], [7], idx_to_model_id=ИНДЕКС)
