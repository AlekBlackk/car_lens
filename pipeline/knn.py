"""Поиск ближайших классов по центроидам индекса (§13.2).

Модуль не импортирует torch: одной и той же функцией меряется индекс на M-1.3
(`scripts/eval_index.py`) и паритет квантованной модели на M-1.4
(`export/verify_parity.py`). Вторая копия сравнивала бы заодно две реализации
сравнения, а джоб `python` в CI обходится группой `dev`, где torch нет.
"""

from __future__ import annotations

import numpy as np

from training.metrics import TOP_K


def _class_scores(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Сходство с каждым классом — максимум по его центроидам; мёртвые классы — `-inf`."""
    классов, _, размерность = centroids.shape
    живые = np.any(centroids != 0.0, axis=(1, 2))
    if not bool(живые.any()):
        raise ValueError("в индексе ни одного класса с эталоном: все центроиды нулевые")
    плоские = centroids.reshape(-1, размерность)
    сходства = embeddings @ плоские.T
    по_классам = сходства.reshape(len(embeddings), классов, -1).max(axis=2)
    return np.where(живые, по_классам, -np.inf)


def ranked_by_centroids(
    embeddings: np.ndarray, centroids: np.ndarray, k: int = TOP_K
) -> list[list[int]]:
    """Индексы `k` ближайших классов по максимуму косинуса к их центроидам.

    `centroids` — `(классы, центроиды, измерения)`, всё нормализовано, поэтому косинус —
    скалярное произведение. Максимум по центроидам берётся до сортировки: результат — это
    `k` разных классов, а не `k` ближайших центроидов, которые могут принадлежать одному.

    `k` проверяется явно. `argsort`-срез длиннее числа классов молча вернул бы короткие
    строки, а вызывающий (метрика) считал бы по ним попадание в top-k, которого нет.

    Класс, у которого нет эталонных кадров, записан в индексе нулями (§13.2 SPEC). Косинус
    к нулевому центроиду равен ровно нулю, и ноль бьёт любое отрицательное сходство живого
    класса: на кадре не из домена мёртвый класс занимал top-1. Такие классы уводятся в
    хвост через `-inf`, а не выбрасываются, потому что контракт «`k` разных классов»
    сильнее: пусть слот займёт класс без эталона, но последним.
    """
    классов = centroids.shape[0]
    if not 1 <= k <= классов:
        raise ValueError(f"k={k} вне диапазона 1..{классов}")
    порядок = np.argsort(-_class_scores(embeddings, centroids), axis=1)[:, :k]
    return [[int(класс) for класс in строка] for строка in порядок]


def top2_with_sims(
    embeddings: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Два лучших класса и их сходства — то, что телефон подаёт в решение §6.4.

    Сходство — тот же максимум по центроидам, что у `CatalogIndex.knn` на устройстве:
    подобранный офлайн порог обязан сравниваться с тем же числом, что в `decide.ts`.
    """
    if centroids.shape[0] < 2:
        raise ValueError("для разрыва top-1/top-2 нужно не меньше двух классов")
    оценки = _class_scores(embeddings, centroids)
    порядок = np.argsort(-оценки, axis=1)[:, :2]
    строки = np.arange(len(embeddings))
    top1, top2 = порядок[:, 0], порядок[:, 1]
    return (
        top1.astype(np.int64),
        оценки[строки, top1].astype(np.float64),
        top2.astype(np.int64),
        оценки[строки, top2].astype(np.float64),
    )


def max_sim(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Лучшее сходство с любым центроидом индекса — `r` отказных прототипов.

    Классы здесь не различаются: прототипу неважно, какой из отказных «классов» ближе, важно
    лишь, насколько. То же число считает `CatalogIndex.maxSim` на устройстве.
    """
    плоские = centroids.reshape(-1, centroids.shape[-1])
    return np.asarray((embeddings @ плоские.T).max(axis=1), dtype=np.float64)


def top5_by_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> list[list[int]]:
    """top-5 классов: имя, на которое опираются M-1.3 и сверка паритета M-1.4 (§13.2)."""
    return ranked_by_centroids(embeddings, centroids, TOP_K)
