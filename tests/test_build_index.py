"""Тесты сборки индекса. Чистые части проверяются без torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.build_index import BuildIndexError, check_class_index, parse_embedder, stack_centroids


def _нормировать(матрица: np.ndarray) -> np.ndarray:
    нормы = np.linalg.norm(матрица, axis=1, keepdims=True)
    return np.asarray(матрица / нормы, dtype=np.float32)


def _kmeans(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
    return эмбеддинги[:k].copy()


def test_сверка_нумерации_пропускает_совпадающие() -> None:
    check_class_index({"a__b": 0, "c__d": 1}, {"a__b": 0, "c__d": 1})


def test_сверка_ловит_сдвиг_нумерации() -> None:
    """Каталог пополнился, веса обучены под старую нумерацию — подписи разъедутся."""
    with pytest.raises(BuildIndexError, match="нумерация классов"):
        check_class_index({"a__b": 0, "c__d": 1}, {"a__b": 0, "b__x": 1, "c__d": 2})


def test_сверка_ловит_другой_состав() -> None:
    with pytest.raises(BuildIndexError, match="нумерация классов"):
        check_class_index({"a__b": 0}, {"z__z": 0})


def test_stack_центроиды_по_классам() -> None:
    эмбеддинги = _нормировать(np.random.default_rng(0).normal(size=(30, 8)))
    метки = np.array([0] * 10 + [1] * 10 + [2] * 10)
    результат = stack_centroids(
        эмбеддинги, метки, num_classes=3, centroids_per_class=4, kmeans=_kmeans
    )
    assert результат.shape == (3, 4, 8)
    # Первый центроид класса 1 — первый эмбеддинг этого класса (заглушка берёт первые k).
    np.testing.assert_allclose(результат[1, 0], эмбеддинги[10], rtol=1e-6)


def test_stack_падает_на_классе_без_эмбеддингов() -> None:
    эмбеддинги = _нормировать(np.random.default_rng(1).normal(size=(10, 8)))
    метки = np.zeros(10, dtype=int)
    with pytest.raises(BuildIndexError, match="класс 1"):
        stack_centroids(эмбеддинги, метки, num_classes=2, centroids_per_class=4, kmeans=_kmeans)


def test_stack_падает_на_метке_вне_диапазона() -> None:
    эмбеддинги = _нормировать(np.random.default_rng(2).normal(size=(10, 8)))
    метки = np.array([0] * 9 + [5])
    with pytest.raises(BuildIndexError, match="метка 5"):
        stack_centroids(эмбеддинги, метки, num_classes=2, centroids_per_class=4, kmeans=_kmeans)


def test_collect_embeddings_нормализует_и_сохраняет_порядок() -> None:
    # Модульный `importorskip` скипнул бы весь файл, а шесть тестов выше обязаны
    # проходить в CI, где группы `train` нет. Отсюда пропуск внутри теста, а `torch`
    # импортируется отдельной строкой: `torch` из `importorskip` — переменная, и по
    # ней mypy не разбирает аннотации `torch.Tensor`.
    pytest.importorskip("torch")

    import torch
    from torch import Tensor, nn

    from training.build_index import collect_embeddings

    class ФальшиваяМодель(nn.Module):  # type: ignore[misc]
        def forward(self, картинки: Tensor) -> Tensor:
            # Ненормированный эмбеддинг: длина заведомо не единичная (Р-18).
            return картинки.flatten(1)[:, :4] * 5.0

    батчи = [
        (torch.ones(2, 1, 2, 2), torch.tensor([0, 1])),
        (torch.full((1, 1, 2, 2), 2.0), torch.tensor([1])),
    ]
    эмбеддинги, метки = collect_embeddings(ФальшиваяМодель(), батчи, device=torch.device("cpu"))
    assert эмбеддинги.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(эмбеддинги, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(метки, np.array([0, 1, 1]))


def test_эмбеддер_по_умолчанию_torch() -> None:
    вид, путь = parse_embedder("torch")
    assert вид == "torch"
    assert путь is None


def test_эмбеддер_tflite_разбирает_путь() -> None:
    вид, путь = parse_embedder("tflite:/tmp/classifier_int8.tflite")
    assert вид == "tflite"
    assert путь == Path("/tmp/classifier_int8.tflite")


def test_неизвестный_эмбеддер_падает() -> None:
    # Не `match="эмбеддер"`: это слово есть в обоих сообщениях `parse_embedder`, и тест
    # прошёл бы на чужой ветке отказа. Регэксп обязан различать.
    with pytest.raises(BuildIndexError, match="неизвестный эмбеддер"):
        parse_embedder("onnx:/tmp/model.onnx")


def test_tflite_без_пути_падает() -> None:
    # В брифе стоит `match="путь"`, но заданный там же текст ошибки — «без пути к
    # модели»: слова «путь» в нём нет, а переписать текст под регэксп значило бы
    # испортить падеж. Проверяется тот же смысл — сообщение про отсутствующий путь.
    with pytest.raises(BuildIndexError, match="пути"):
        parse_embedder("tflite:")
