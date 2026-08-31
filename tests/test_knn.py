"""Тесты поиска по центроидам. Модуль без torch, поэтому тест идёт в CI."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.knn import ranked_by_centroids, top5_by_centroids


def test_top5_ставит_свой_класс_первым() -> None:
    центроиды = np.zeros((6, 4, 3), dtype=np.float32)
    for класс in range(6):
        база = np.zeros(3, dtype=np.float32)
        база[класс % 3] = 1.0
        # Слегка разные центроиды одного класса, все около своей оси.
        for k in range(4):
            вектор = база + 0.01 * k
            центроиды[класс, k] = вектор / np.linalg.norm(вектор)
    эмбеддинги = центроиды[2, 0][None, :]
    результат = top5_by_centroids(эмбеддинги, центроиды)
    assert len(результат) == 1
    assert результат[0][0] == 2
    assert len(результат[0]) == 5
    assert len(set(результат[0])) == 5


def test_каждая_строка_даёт_разные_классы() -> None:
    """Пять слотов top-5 — это пять классов, а не пять центроидов одного класса."""
    генератор = np.random.default_rng(0)
    сырые = генератор.normal(size=(8, 4, 16))
    центроиды = (сырые / np.linalg.norm(сырые, axis=2, keepdims=True)).astype(np.float32)
    эмбеддинги = центроиды[:, 0, :]
    результат = top5_by_centroids(эмбеддинги, центроиды)
    for строка in результат:
        assert len(set(строка)) == 5


def test_модуль_не_тянет_torch() -> None:
    """Тот самый дефект §3.2 дизайна: файл теста ронял сбор в CI из-за импорта torch."""
    import subprocess
    import sys
    from pathlib import Path

    import pipeline.knn

    # Проверка идёт в подпроцессе, а не здесь: в обучающей venv torch установлен, и к
    # этому тесту его уже импортировали соседние тесты. `sys.modules` текущего процесса
    # поэтому не доказал бы ничего — чистый импорт виден только в чистом процессе.
    программа = (
        "import sys\n"
        "import pipeline.knn\n"
        "подтянутые = {имя.split('.')[0] for имя in sys.modules if имя.startswith('torch')}\n"
        "assert not подтянутые, sorted(подтянутые)\n"
    )
    результат = subprocess.run(
        [sys.executable, "-c", программа],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert результат.returncode == 0, результат.stderr
    assert not any(имя.startswith("torch") for имя in vars(pipeline.knn))


def test_ranked_отдаёт_запрошенное_число_классов() -> None:
    генератор = np.random.default_rng(1)
    сырые = генератор.normal(size=(9, 4, 16))
    центроиды = (сырые / np.linalg.norm(сырые, axis=2, keepdims=True)).astype(np.float32)
    эмбеддинги = центроиды[:, 0, :]
    for k in (1, 3, 5, 9):
        результат = ranked_by_centroids(эмбеддинги, центроиды, k)
        assert all(len(строка) == k for строка in результат)
        assert all(len(set(строка)) == k for строка in результат)


def test_ranked_согласован_с_top5() -> None:
    """Обёртка обязана быть обёрткой, а не второй реализацией."""
    генератор = np.random.default_rng(2)
    сырые = генератор.normal(size=(7, 4, 8))
    центроиды = (сырые / np.linalg.norm(сырые, axis=2, keepdims=True)).astype(np.float32)
    эмбеддинги = центроиды[:, 1, :]
    assert top5_by_centroids(эмбеддинги, центроиды) == ranked_by_centroids(эмбеддинги, центроиды, 5)


def test_ranked_отвергает_k_больше_числа_классов() -> None:
    центроиды = np.zeros((3, 2, 4), dtype=np.float32)
    центроиды[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="k=4"):
        ranked_by_centroids(np.zeros((1, 4), dtype=np.float32), центроиды, 4)


def test_ranked_отвергает_нулевой_k() -> None:
    центроиды = np.zeros((3, 2, 4), dtype=np.float32)
    центроиды[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="k=0"):
        ranked_by_centroids(np.zeros((1, 4), dtype=np.float32), центроиды, 0)


def test_класс_без_эталона_не_вытесняет_живой() -> None:
    """Нулевые центроиды — метка «эталонов нет»; косинус к ним ровно 0.

    Ноль бьёт любое отрицательное сходство, поэтому на кадре не из домена мёртвый класс
    занимал top-1 у класса, у которого эталон есть. В `catalog_index.bin` таких классов
    16 после чистки чужих моделей 2026-08-31.
    """
    центроиды = np.zeros((3, 2, 2), dtype=np.float32)
    центроиды[0, :, 0] = 1.0
    центроиды[1, :, 1] = 1.0
    # Запрос смотрит против обеих живых осей: сходства -0.707 и -0.707 против нуля мёртвого.
    эмбеддинги = np.array([[-0.7071, -0.7071]], dtype=np.float32)
    результат = ranked_by_centroids(эмбеддинги, центроиды, 2)
    assert 2 not in результат[0], "мёртвый класс встал впереди живого"
    assert sorted(результат[0]) == [0, 1]


def test_мёртвый_класс_уходит_в_хвост_а_не_исчезает() -> None:
    """Контракт «k разных классов» сильнее: слот заполняется, но последним."""
    центроиды = np.zeros((3, 2, 2), dtype=np.float32)
    центроиды[0, :, 0] = 1.0
    центроиды[1, :, 1] = 1.0
    эмбеддинги = np.array([[-0.7071, -0.7071]], dtype=np.float32)
    результат = ranked_by_centroids(эмбеддинги, центроиды, 3)
    assert результат[0][2] == 2


def test_индекс_без_живых_классов_отвергается() -> None:
    """Все классы мёртвые — сортировать нечего, и молчаливый класс 0 хуже отказа."""
    with pytest.raises(ValueError, match="ни одного класса с эталоном"):
        ranked_by_centroids(
            np.zeros((1, 4), dtype=np.float32), np.zeros((3, 2, 4), dtype=np.float32), 2
        )
