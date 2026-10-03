"""Сплиты каталога 192: перенумерация, пул и проверки (`bench/catalog-192-preregistration.md`).

Каталог вырос с 150 до 192 классов, а `class_idx` — позиция класса в отсортированном каталоге
(`pipeline.catalog.class_index`): новые классы сдвигают старые. Истина в сплитах — `model_id`,
поэтому всё здесь выводит индекс заново, а не сдвигает старый.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from pipeline.catalog import Catalog, class_index, member_index
from pipeline.clean import DEFAULT_PHASH_HAMMING
from pipeline.hashing import bucket_of, close_pairs, hamming
from pipeline.ingest.external import merge_external
from pipeline.schema import COLUMNS


class Cat192Error(ValueError):
    """Вход сборки сплитов каталога 192 не сходится."""


def remap_class_idx(frame: pd.DataFrame, catalog: Catalog) -> pd.DataFrame:
    """`class_idx` по `model_id` каталога; у негативов (без `model_id`) остаётся как был."""
    индекс = class_index(catalog)
    класс_модели = member_index(catalog)
    результат = frame.copy()
    есть = frame["model_id"].notna()
    неизвестные = sorted(set(frame.loc[есть, "model_id"]) - set(класс_модели))
    if неизвестные:
        raise Cat192Error(f"model_id вне каталога: {', '.join(неизвестные[:5])}")
    результат["class_idx"] = результат["class_idx"].astype("Int32")
    результат.loc[есть, "class_idx"] = frame.loc[есть, "model_id"].map(
        lambda модель: индекс[класс_модели[модель]]
    )
    return результат


def split_source_frames(
    external: pd.DataFrame, new: pd.DataFrame, source: str = "mad_cars"
) -> dict[str, pd.DataFrame]:
    """Внешние кадры пула по источникам, кадры источника `source` пересобраны по новым классам.

    Классы, у которых есть новые кадры этого источника, теряют его старые кадры: выборка
    тянется заново (для MAD-Cars — из объединения названий). Остальные источники и остальные
    классы идут как были.
    """
    пересобираемые = set(new["class_idx"])
    старые = external[external["source"] == source]
    оставленные = старые[~старые["class_idx"].isin(пересобираемые)]
    кадры = {
        имя: external[external["source"] == имя]
        for имя in sorted(set(external["source"]) - {source})
    }
    кадры[source] = pd.concat([оставленные, new], ignore_index=True)
    return кадры


@dataclass(frozen=True, slots=True)
class PoolResult:
    """Внешняя часть пула и сверка с пулом 0.5."""

    external: pd.DataFrame
    rejected: dict[str, int]
    # Классы, чьи кадры изменились против пула 0.5: ровно пересобранные, и только они.
    changed_classes: frozenset[int]


def build_external_pool(
    old_external: pd.DataFrame,
    new: pd.DataFrame,
    catalog: Catalog,
    *,
    max_per_class: int,
    own_caps: Mapping[str, int],
    seed: int,
    source: str = "mad_cars",
) -> PoolResult:
    """Склейка внешних источников тем же `merge_external`, что у пула 0.5.

    Кадры старых классов уже прошли дедуп и потолок, поэтому вторая склейка их не меняет —
    это проверяется, а не предполагается: изменившийся класс вне пересобираемых роняет сборку.
    """
    старое = remap_class_idx(old_external, catalog)
    новое = remap_class_idx(new, catalog)
    итог, отказы = merge_external(
        split_source_frames(старое, новое, source),
        max_per_class=max_per_class,
        seed=seed,
        own_caps=own_caps,
    )
    пересобираемые = set(новое["class_idx"])
    до = старое.groupby("class_idx")["photo_id"].agg(frozenset)
    после = итог.groupby("class_idx")["photo_id"].agg(frozenset)
    изменились = {
        int(класс)
        for класс in set(до.index) | set(после.index)
        if до.get(класс, frozenset()) != после.get(класс, frozenset())
    }
    лишние = изменились - пересобираемые
    if лишние:
        raise Cat192Error(f"склейка изменила классы вне пересобираемых: {sorted(лишние)[:10]}")
    return PoolResult(итог, отказы, frozenset(изменились))


def pick_one_frame_per_car(frames: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Один кадр на машину: с наименьшим `bucket_of(photo_id, seed)` среди переживших приём."""
    ключи = frames["photo_id"].map(lambda photo_id: bucket_of(str(photo_id), seed))
    return (
        frames.assign(_ключ=ключи)
        .sort_values(["listing_id", "_ключ"], kind="stable")
        .drop_duplicates("listing_id", keep="first")[list(COLUMNS)]
        .reset_index(drop=True)
    )


def leaking_cars(
    eval_frames: pd.DataFrame, train_hashes: Sequence[int], *, hamming: int = DEFAULT_PHASH_HAMMING
) -> frozenset[str]:
    """Машины проверки, у которых хоть один кадр близок по pHash к кадру обучения.

    Без оглядки на класс, как утечка в `holdout_ru`: проверка, похожая на обучение, меряет
    запоминание. Машина выбывает целиком — остаток её кадров уже не независим.
    """
    граница = len(eval_frames)
    хеши = [*(int(хеш) for хеш in eval_frames["phash"]), *train_hashes]
    листинги = list(eval_frames["listing_id"])
    утекли: set[str] = set()
    for i, j in close_pairs(хеши, hamming):
        if i < граница <= j:
            утекли.add(str(листинги[i]))
    return frozenset(утекли)


def split_groups(
    group_classes: Mapping[str, str], *, salt: str, min_groups: int = 4, share: int = 4
) -> frozenset[str]:
    """Отложенные группы: у класса с `min_groups` группами и больше — каждая `share`-я.

    Порядок внутри класса — по `sha256(dir_path + salt)`, от порядка словаря результат не зависит.
    Отложенная группа — другое поколение или кузов, а не другие кадры того же набора: проверка
    по ней жёсткая и сознательно такая (`bench/catalog-192-studio-preregistration.md`).
    """
    по_классам: defaultdict[str, list[str]] = defaultdict(list)
    for папка, класс in group_classes.items():
        по_классам[класс].append(папка)
    отложены: set[str] = set()
    for папки in по_классам.values():
        if len(папки) < min_groups:
            continue
        упорядоченные = sorted(
            папки, key=lambda папка: hashlib.sha256((папка + salt).encode()).hexdigest()
        )
        отложены.update(упорядоченные[: len(папки) // share])
    return frozenset(отложены)


def without_close_frames(
    frames: pd.DataFrame, reference_hashes: Sequence[int], *, hamming: int = DEFAULT_PHASH_HAMMING
) -> pd.DataFrame:
    """Кадры, у которых нет близкого по pHash в `reference_hashes` (без оглядки на класс)."""
    граница = len(frames)
    хеши = [*(int(хеш) for хеш in frames["phash"]), *reference_hashes]
    близкие = {i for i, j in close_pairs(хеши, hamming) if i < граница <= j}
    return frames.iloc[[i for i in range(граница) if i not in близкие]].reset_index(drop=True)


def without_close_twins(frames: pd.DataFrame, *, max_distance: int, seed: int) -> pd.DataFrame:
    """Из близких по pHash кадров одного класса остаётся кадр с меньшим `bucket_of(photo_id, seed)`.

    Перебор попарный: `close_pairs` отвергает порог от восьми бит, а проверяемый набор мал.
    """
    порядок = sorted(
        range(len(frames)), key=lambda i: bucket_of(str(frames["photo_id"].iloc[i]), seed)
    )
    хеши = [int(хеш) for хеш in frames["phash"]]
    классы = list(frames["class_idx"])
    оставлены: list[int] = []
    for i in порядок:
        if all(
            классы[i] != классы[j] or hamming(хеши[i], хеши[j]) > max_distance for j in оставлены
        ):
            оставлены.append(i)
    return frames.iloc[sorted(оставлены)].reset_index(drop=True)


def cap_per_class(frames: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    """Не более `cap` кадров на класс, по наименьшему `bucket_of(photo_id, seed)`."""
    ключи = frames["photo_id"].map(lambda photo_id: bucket_of(str(photo_id), seed))
    упорядочено = frames.assign(_ключ=ключи).sort_values(["class_idx", "_ключ"], kind="stable")
    оставить = упорядочено.groupby("class_idx", sort=False).cumcount() < cap
    return упорядочено.loc[оставить, list(COLUMNS)].reset_index(drop=True)
