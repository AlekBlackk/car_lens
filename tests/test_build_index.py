"""Тесты сборки индекса. Чистые части проверяются без torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.catalog import STANFORD_CATALOG_PATH, Catalog, class_index, load_catalog
from training.build_index import BuildIndexError, check_class_index, parse_embedder, stack_centroids


def _нормировать(матрица: np.ndarray) -> np.ndarray:
    нормы = np.linalg.norm(матрица, axis=1, keepdims=True)
    return np.asarray(матрица / нормы, dtype=np.float32)


def _kmeans(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
    return эмбеддинги[:k].copy()


def _индекс_как_в_чекпоинте_m1(каталог: Catalog) -> dict[str, int]:
    """Воспроизвести `class_index` старой формы: `model_id` → idx по порядку моделей.

    Ровно это лежит в `class_index` чекпоинтов M-1 — до того, как единицей
    индексации стал класс. Сам чекпоинт в тест не положить: он весит сотни
    мегабайт, а сверку интересует только эта карта.
    """
    return {m.model_id: i for i, m in enumerate(каталог.models)}


def test_сверка_нумерации_пропускает_совпадающие() -> None:
    check_class_index({"a__b": 0, "c__d": 1}, {"a__b": 0, "c__d": 1})


def test_сверка_ловит_сдвиг_нумерации() -> None:
    """Каталог пополнился, веса обучены под старую нумерацию — подписи разъедутся."""
    with pytest.raises(BuildIndexError, match="нумерация классов"):
        check_class_index({"a__b": 0, "c__d": 1}, {"a__b": 0, "b__x": 1, "c__d": 2})


def test_сверка_ловит_другой_состав() -> None:
    with pytest.raises(BuildIndexError, match="нумерация классов"):
        check_class_index({"a__b": 0}, {"z__z": 0})


def test_чекпоинт_m1_сходится_с_каталогом_stanford() -> None:
    """Веса M-1 обязаны продолжать воспроизводиться после смены единицы индексации.

    В чекпоинте лежит `class_index` старой формы (`model_id` → idx), а
    сверка получает новый (`class_id` → idx). На Stanford групп нет,
    поэтому `class_id` каждого класса равен `model_id` его единственного
    участника, и карты совпадают один в один. Разойдись они — сборка
    индекса на весах M-1 упала бы, а `bench/` перестал бы быть
    воспроизводимым.
    """
    каталог = load_catalog(STANFORD_CATALOG_PATH)
    из_чекпоинта = _индекс_как_в_чекпоинте_m1(каталог)
    assert len(из_чекпоинта) == 171
    check_class_index(из_чекпоинта, class_index(каталог))


def test_чекпоинт_без_групп_расходится_с_каталогом_с_группой(tmp_path: Path) -> None:
    """Обратная сторона совместимости: ввод группы обязан ронять сверку.

    Иначе чекпоинт с 171 выходом молча собрал бы индекс на каталоге со
    170 классами, и подписи сдвинулись бы начиная с первой группы.
    """
    путь = tmp_path / "с_группой.yaml"
    путь.write_text(
        """
version: 1
brands:
  Geely:
    models:
      coolray: {display: Geely Coolray}
  Belgee:
    models:
      x50: {display: Belgee X50}
groups:
  grp__coolray:
    members: [geely__coolray, belgee__x50]
    display: "Geely Coolray / Belgee X50"
""",
        encoding="utf-8",
    )
    каталог = load_catalog(путь)
    with pytest.raises(BuildIndexError, match="нумерация классов"):
        check_class_index(_индекс_как_в_чекпоинте_m1(каталог), class_index(каталог))


def test_stack_центроиды_по_классам() -> None:
    эмбеддинги = _нормировать(np.random.default_rng(0).normal(size=(30, 8)))
    метки = np.array([0] * 10 + [1] * 10 + [2] * 10)
    результат = stack_centroids(
        эмбеддинги, метки, num_classes=3, centroids_per_class=4, kmeans=_kmeans
    )
    assert результат.shape == (3, 4, 8)
    # Первый центроид класса 1 — первый эмбеддинг этого класса (заглушка берёт первые k).
    np.testing.assert_allclose(результат[1, 0], эмбеддинги[10], rtol=1e-6)


def test_центроиды_группы_считаются_по_всем_участникам() -> None:
    """Фотографии обоих участников группы дают один центроидный набор (§13.2).

    Разделение по `model_id` дало бы два набора на один слот индекса, а
    класс — это то, что различает сеть. Проверяется через k-means-заглушку,
    которая записывает, сколько эмбеддингов ей отдали: `class_idx` группы
    обязан получить сумму обоих участников, а не долю одного.
    """
    эмбеддинги = _нормировать(np.random.default_rng(3).normal(size=(14, 8)))
    # Кадр данных §13.1: у участников группы разные `model_id`, но один `class_idx`.
    # Здесь это метки 0 (одиночная модель) и 1 (группа из двух участников).
    метки = np.array([0] * 4 + [1] * 6 + [1] * 4)
    отданных: list[int] = []

    def kmeans_со_счётчиком(свои: np.ndarray, k: int) -> np.ndarray:
        отданных.append(len(свои))
        return свои[:k].copy()

    результат = stack_centroids(
        эмбеддинги, метки, num_classes=2, centroids_per_class=4, kmeans=kmeans_со_счётчиком
    )
    assert результат.shape == (2, 4, 8)
    assert отданных == [4, 10]
    # Первый центроид группы — первый эмбеддинг её первого участника, а не
    # отдельный набор на каждого.
    np.testing.assert_allclose(результат[1, 0], эмбеддинги[4], rtol=1e-6)


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


def test_stack_с_allow_empty_оставляет_нули_классу_без_эмбеддингов() -> None:
    """Нули — договорённая метка «эталона нет» (§13.2), а не забытый слот.

    Читалки обязаны исключать такой класс из ранжирования: `pipeline.knn`,
    `app/src/vision/catalogIndex.ts`, `CatalogIndex.kt`.
    """
    эмбеддинги = _нормировать(np.random.default_rng(4).normal(size=(10, 8)))
    метки = np.zeros(10, dtype=int)
    результат = stack_centroids(
        эмбеддинги,
        метки,
        num_classes=2,
        centroids_per_class=4,
        kmeans=_kmeans,
        allow_empty=True,
    )
    assert результат.shape == (2, 4, 8)
    assert np.all(результат[1] == 0.0)
    assert np.any(результат[0] != 0.0)
