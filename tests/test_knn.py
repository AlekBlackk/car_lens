"""Тесты поиска по центроидам. Модуль без torch, поэтому тест идёт в CI."""

from __future__ import annotations

import numpy as np

from pipeline.knn import top5_by_centroids


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
