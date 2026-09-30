"""Сплиты каталога 192: перенумерация, пул и проверки (`bench/catalog-192-preregistration.md`).

Каталог вырос с 150 до 192 классов, а `class_idx` — позиция класса в отсортированном каталоге
(`pipeline.catalog.class_index`): новые классы сдвигают старые. Истина в сплитах — `model_id`,
поэтому всё здесь выводит индекс заново, а не сдвигает старый.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from pipeline.catalog import Catalog, class_index, member_index
from pipeline.clean import DEFAULT_PHASH_HAMMING
from pipeline.hashing import bucket_of, close_pairs
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


def split_source_frames(external: pd.DataFrame, new_mad: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Внешние кадры пула 0.5 по источникам, кадры MAD-Cars пересобраны по новым классам.

    Классы, у которых есть новые кадры MAD-Cars (новые и четыре преемника), теряют старые
    кадры этого источника: их выборка тянется заново из объединения названий. Остальные
    источники и остальные классы идут как были.
    """
    пересобираемые = set(new_mad["class_idx"])
    старые_mad = external[external["source"] == "mad_cars"]
    оставленные = старые_mad[~старые_mad["class_idx"].isin(пересобираемые)]
    кадры = {
        источник: external[external["source"] == источник]
        for источник in sorted(set(external["source"]) - {"mad_cars"})
    }
    кадры["mad_cars"] = pd.concat([оставленные, new_mad], ignore_index=True)
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
    new_mad: pd.DataFrame,
    catalog: Catalog,
    *,
    max_per_class: int,
    own_caps: Mapping[str, int],
    seed: int,
) -> PoolResult:
    """Склейка внешних источников тем же `merge_external`, что у пула 0.5.

    Кадры старых классов уже прошли дедуп и потолок, поэтому вторая склейка их не меняет —
    это проверяется, а не предполагается: изменившийся класс вне пересобираемых роняет сборку.
    """
    старое = remap_class_idx(old_external, catalog)
    новое = remap_class_idx(new_mad, catalog)
    итог, отказы = merge_external(
        split_source_frames(старое, новое),
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
