"""Сборка индекса эталонов из чекпоинта обучения (§13.2, веха M-1.3).

Нумерация классов берётся из чекпоинта и лишь сверяется с каталогом (Р-21):
пересчёт по каталогу детерминирован относительно его содержимого, а не весов, и
одна новая модель в YAML сдвинула бы половину индексов при тех же весах.

Препроцессинг — тот же, что у запроса на устройстве: `Resize` и нормализация, без
аугментаций и без TTA (Р-23).

torch и sklearn импортируются внутри функций, а не на уровне модуля (Р-15). Чистая
часть сборки — сверка нумерации классов и раскладка центроидов по классам — арифметика
на numpy, и она обязана проверяться в CI, где стоит только группа `dev`. Импорт torch
сверху делал бы `from training.build_index import stack_centroids` невыполнимым без
группы `train`, то есть ронял бы весь прогон pytest на этапе сбора.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.catalog import STANFORD_CATALOG_PATH, class_index, load_catalog
from pipeline.centroids import centroids_for_class
from pipeline.index_format import meta_json, write_index
from pipeline.paths import crops_dir, splits_dir

if TYPE_CHECKING:
    import torch
    from torch import Tensor

CENTROIDS_PER_CLASS = 4


class BuildIndexError(RuntimeError):
    """Сборка индекса невозможна."""


def check_class_index(
    checkpoint_index: Mapping[str, int], catalog_index: Mapping[str, int]
) -> None:
    """Сверить нумерацию классов чекпоинта с каталогом (Р-21)."""
    if dict(checkpoint_index) != dict(catalog_index):
        лишние = set(catalog_index) - set(checkpoint_index)
        пропавшие = set(checkpoint_index) - set(catalog_index)
        raise BuildIndexError(
            "нумерация классов чекпоинта разошлась с каталогом: "
            f"{len(checkpoint_index)} против {len(catalog_index)} классов, "
            f"нет в чекпоинте {sorted(лишние)[:5]}, нет в каталоге {sorted(пропавшие)[:5]}"
        )


def stack_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    centroids_per_class: int,
    kmeans: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Центроиды всех классов: `(num_classes, centroids_per_class, dim)`."""
    вне_диапазона = [метка for метка in np.unique(labels) if not 0 <= int(метка) < num_classes]
    if вне_диапазона:
        raise BuildIndexError(f"метка {int(вне_диапазона[0])} вне диапазона 0..{num_classes - 1}")
    сложенные = np.zeros((num_classes, centroids_per_class, embeddings.shape[1]), dtype=np.float32)
    for класс in range(num_classes):
        свои = embeddings[labels == класс]
        if len(свои) == 0:
            raise BuildIndexError(f"класс {класс} без эталонов в выборке")
        сложенные[класс] = centroids_for_class(
            свои, centroids_per_class=centroids_per_class, kmeans=kmeans
        )
    return сложенные


def collect_embeddings(
    model: torch.nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Эмбеддинги и метки всей выборки. Нормализация делается здесь, один раз (Р-18)."""
    import torch

    model.eval()
    куски: list[np.ndarray] = []
    метки: list[np.ndarray] = []
    with torch.no_grad():
        for картинки, цели in loader:
            эмбеддинг = model(картинки.to(device, non_blocking=True)).float()
            эмбеддинг = torch.nn.functional.normalize(эмбеддинг)
            куски.append(эмбеддинг.cpu().numpy().astype(np.float32))
            метки.append(np.asarray(цели.cpu(), dtype=np.int64))
    if not куски:
        raise BuildIndexError("выборка пуста: нечего складывать в индекс")
    return np.concatenate(куски), np.concatenate(метки)


def sha256_of_file(path: Path) -> str:
    """sha256 файла в hex."""
    хеш = hashlib.sha256()
    with path.open("rb") as поток:
        for блок in iter(lambda: поток.read(1 << 20), b""):
            хеш.update(блок)
    return хеш.hexdigest()


def _kmeans_factory(seed: int) -> Callable[[np.ndarray, int], np.ndarray]:
    """k-means с фиксированным сидом: две сборки на одних весах дают один файл."""
    from sklearn.cluster import KMeans

    def kmeans(эмбеддинги: np.ndarray, k: int) -> np.ndarray:
        модель = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        модель.fit(эмбеддинги)
        return np.asarray(модель.cluster_centers_, dtype=np.float32)

    return kmeans


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Сборка catalog_index.bin по §13.2")
    парсер.add_argument("--checkpoint", type=Path, required=True)
    парсер.add_argument("--catalog", type=Path, default=STANFORD_CATALOG_PATH)
    парсер.add_argument("--split", default="train")
    парсер.add_argument("--weights", choices=("raw", "ema"), default="raw")
    парсер.add_argument("--device", default="cuda")
    парсер.add_argument("--out", type=Path, default=None)
    return парсер.parse_args()


def main() -> None:
    import torch

    from training.dataset import build_transforms, make_loader
    from training.model import create_model
    from training.samples import read_samples

    аргументы = _parse_args()
    чекпоинт: dict[str, Any] = torch.load(аргументы.checkpoint, map_location="cpu")
    каталог = load_catalog(аргументы.catalog)
    индекс: dict[str, int] = чекпоинт["class_index"]
    check_class_index(индекс, class_index(каталог))

    конфиг = чекпоинт["config"]
    модель, _ = create_model(
        конфиг["backbone"],
        num_classes=len(индекс),
        embedding_dim=конфиг["embedding_dim"],
        pretrained=False,
    )
    ключ = "model_state" if аргументы.weights == "raw" else "ema_state"
    модель.load_state_dict(чекпоинт[ключ])
    устройство = torch.device(аргументы.device)
    модель.to(устройство)

    загрузчик = make_loader(
        read_samples(
            splits_dir() / f"{аргументы.split}.parquet",
            crops_dir() / "stanford",
            num_classes=len(индекс),
            skip_multi_car=True,
        ),
        build_transforms(
            чекпоинт["img_size"],
            train=False,
            mean=tuple(чекпоинт["mean"]),
            std=tuple(чекпоинт["std"]),
        ),
        batch_size=конфиг["batch_size"],
        shuffle=False,
        num_workers=конфиг["num_workers"],
    )
    эмбеддинги, метки = collect_embeddings(модель, загрузчик, устройство)
    центроиды = stack_centroids(
        эмбеддинги,
        метки,
        num_classes=len(индекс),
        centroids_per_class=CENTROIDS_PER_CLASS,
        kmeans=_kmeans_factory(int(конфиг["seed"])),
    )

    каталог_вывода = аргументы.out or аргументы.checkpoint.parent / "index"
    каталог_вывода.mkdir(parents=True, exist_ok=True)
    по_индексу = {значение: ключ_ for ключ_, значение in индекс.items()}
    модели = tuple(каталог.by_id(по_индексу[i]) for i in range(len(индекс)))
    мета = meta_json(
        модели,
        embedding_dim=центроиды.shape[2],
        centroids_per_class=CENTROIDS_PER_CLASS,
        model_weights_sha256=sha256_of_file(аргументы.checkpoint),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    путь_меты = каталог_вывода / "catalog_meta.json"
    путь_меты.write_bytes(мета)
    путь_индекса = каталог_вывода / "catalog_index.bin"
    заголовок = write_index(путь_индекса, центроиды, meta_bytes=путь_меты.read_bytes())

    размеры = np.bincount(метки, minlength=len(индекс))
    отчёт = {
        "checkpoint": str(аргументы.checkpoint),
        "weights": аргументы.weights,
        "split": аргументы.split,
        "num_classes": заголовок.num_classes,
        "centroids_per_class": заголовок.centroids_per_class,
        "embedding_dim": заголовок.embedding_dim,
        "embeddings_per_class": {
            "min": int(размеры.min()),
            "median": int(np.median(размеры)),
            "max": int(размеры.max()),
        },
        "index_bytes": путь_индекса.stat().st_size,
        "index_sha256": sha256_of_file(путь_индекса),
        "meta_sha256": заголовок.meta_sha256.hex(),
        "git_sha": чекпоинт.get("git_sha"),
    }
    (каталог_вывода / "report.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(отчёт, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
