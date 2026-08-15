"""Раскладка по сплитам и проверка инвариантов §13.1.

Ключ сплита — `listing_id`, и это не деталь реализации, а самое важное
решение раздела §3.5. Одно объявление даёт 10–20 фотографий одной
физической машины; при сплите по фото та же машина попадает и в train, и
в val, валидация показывает 97 процентов, а на улице выходит 60.

Инвариант о минимуме объявлений на класс параметризован, но значение по
умолчанию — 300, как требует §3.3. Скелет M-1 передаёт меньшее число
явно, и отдельный тест следит, чтобы умолчание не поехало.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import SupportsInt, cast

import pandas as pd

from pipeline.hashing import close_pairs, hamming

# `hamming` переехал в `pipeline.hashing`, но остаётся импортируемым отсюда:
# он часть публичного интерфейса `split`, объявленного планом M-1a.
__all__ = [
    "DEFAULT_MIN_LISTINGS_PER_CLASS",
    "DEFAULT_PHASH_HAMMING",
    "SPLIT_NAMES",
    "InvariantError",
    "SplitRatios",
    "check_invariants",
    "hamming",
    "split_by_listing",
]

DEFAULT_MIN_LISTINGS_PER_CLASS = 300
DEFAULT_PHASH_HAMMING = 6

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class InvariantError(ValueError):
    """Нарушен один из инвариантов §13.1."""


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Доли сплитов. Сумма обязана равняться единице."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        сумма = self.train + self.val + self.test
        if abs(сумма - 1.0) > 1e-9:
            raise ValueError(f"сумма долей должна быть 1.0, получено {сумма}")


def _корзина(listing_id: str, seed: int) -> float:
    """Устойчивое псевдослучайное число из идентификатора объявления.

    Хеш вместо генератора случайных чисел: раскладка обязана зависеть
    только от `listing_id` и `seed`, но не от порядка строк во входном
    кадре и не от версии numpy.
    """
    сырое = hashlib.sha256(f"{seed}:{listing_id}".encode()).digest()
    return int.from_bytes(сырое[:8], "big") / float(1 << 64)


def split_by_listing(
    df: pd.DataFrame,
    ratios: SplitRatios | None = None,
    seed: int = 1337,
) -> dict[str, pd.DataFrame]:
    """Разложить записи по сплитам, стратифицируя по `class_idx`.

    Кадры с `is_multi_car` в `train` не попадают: разметка на них
    неоднозначна. В `val` и `test` они остаются — на улице такой кадр
    встречается постоянно.
    """
    доли = ratios or SplitRatios()
    корзины: dict[str, list[pd.DataFrame]] = {имя: [] for имя in SPLIT_NAMES}

    for _, группа in df.groupby("class_idx", dropna=False, sort=True):
        объявления = sorted(set(группа["listing_id"]))
        упорядоченные = sorted(объявления, key=lambda lid: _корзина(lid, seed))
        n = len(упорядоченные)
        граница_train = round(n * доли.train)
        граница_val = граница_train + round(n * доли.val)

        назначение = {
            **{lid: "train" for lid in упорядоченные[:граница_train]},
            **{lid: "val" for lid in упорядоченные[граница_train:граница_val]},
            **{lid: "test" for lid in упорядоченные[граница_val:]},
        }
        колонка = группа["listing_id"].map(назначение)
        for имя in SPLIT_NAMES:
            корзины[имя].append(группа[колонка == имя])

    сплиты = {
        имя: pd.concat(куски, ignore_index=True) if куски else df.iloc[:0].copy()
        for имя, куски in корзины.items()
    }
    return _выбросить_многомашинные_из_train(сплиты)


def _выбросить_многомашинные_из_train(
    сплиты: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Убрать из `train` кадры с двумя сопоставимыми машинами.

    Именно выбросить, а не перенести в `test`. Перенос строк ломает
    инвариант 1: у объявления часть фотографий может быть с одной
    машиной, часть с двумя, и после переноса его `listing_id` окажется
    сразу в двух сплитах — то есть возникнет ровно та утечка, которую
    инвариант и должен ловить.

    В `val` и `test` такие кадры остаются нетронутыми: на улице кадр с
    двумя машинами встречается постоянно, и оценка обязана его
    учитывать.
    """
    train = сплиты["train"]
    флаг = train["is_multi_car"].astype(bool)
    if bool(флаг.any()):
        сплиты["train"] = train[~флаг].reset_index(drop=True)
    return сплиты


def check_invariants(
    splits: Mapping[str, pd.DataFrame],
    min_listings_per_class: int = DEFAULT_MIN_LISTINGS_PER_CLASS,
    phash_hamming: int = DEFAULT_PHASH_HAMMING,
) -> None:
    """Проверить четыре инварианта §13.1. Первое нарушение бросает исключение.

    Инвариант 2 (`class_idx` ⟺ `model_id` ⟺ `brand`) здесь не проверяется:
    он живёт в `schema.validate_frame`, потому что относится к одной
    строке, а не к раскладке по сплитам.
    """
    _инвариант_1_нет_утечки_объявлений(splits)
    _инвариант_3_минимум_объявлений(splits, min_listings_per_class)
    _инвариант_4_согласованность_классов(splits)
    _инвариант_5_нет_близких_phash(splits, phash_hamming)


def _инвариант_1_нет_утечки_объявлений(splits: Mapping[str, pd.DataFrame]) -> None:
    имена = sorted(splits)
    for i, первый in enumerate(имена):
        for второй in имена[i + 1 :]:
            общие = set(splits[первый]["listing_id"]) & set(splits[второй]["listing_id"])
            if общие:
                примеры = sorted(общие)[:5]
                raise InvariantError(
                    f"утечка: listing_id встречается и в {первый}, и в {второй}: {примеры}"
                )


def _инвариант_3_минимум_объявлений(splits: Mapping[str, pd.DataFrame], минимум: int) -> None:
    train = splits.get("train")
    if train is None or train.empty:
        return
    известные = train[train["class_idx"].fillna(-1).astype(int) >= 0]
    покрытие = известные.groupby("class_idx")["listing_id"].nunique()
    бедные = покрытие[покрытие < минимум]
    if not бедные.empty:
        # `cast`: ключ индекса pandas типизирован как `Hashable`, но здесь это
        # `class_idx`, по которому шла группировка, то есть заведомо число.
        худшие = ", ".join(
            f"{int(cast(SupportsInt, i))}:{int(n)}" for i, n in бедные.head(5).items()
        )
        raise InvariantError(f"классы с менее чем {минимум} объявлениями в train: {худшие}")


def _инвариант_4_согласованность_классов(splits: Mapping[str, pd.DataFrame]) -> None:
    отображение: dict[int, str] = {}
    for имя, d in sorted(splits.items()):
        известные = d[d["class_idx"].fillna(-1).astype(int) >= 0]
        for idx, model_id in zip(известные["class_idx"], известные["model_id"], strict=True):
            ключ = int(idx)
            известное = отображение.get(ключ)
            if известное is None:
                отображение[ключ] = str(model_id)
            elif известное != str(model_id):
                raise InvariantError(
                    f"class_idx {ключ} — это {известное}, но в {имя} он {model_id}"
                )


def _инвариант_5_нет_близких_phash(splits: Mapping[str, pd.DataFrame], предел: int) -> None:
    """Ни один pHash не встречается в двух сплитах с расстоянием ≤ предела.

    Сравниваются только пары, попавшие в общую корзину `close_pairs`, —
    полный перебор на шестнадцати тысячах фотографий стоит минуты.

    Значения сначала дедуплицируются, и это не украшательство. Один и тот
    же `phash` у нескольких кадров — обычное дело: в объявлении десяток
    почти одинаковых фотографий. Обход по строкам превращает k повторов в
    C(k, 2) пар внутри одного сплита, и все они до единой будут отброшены
    проверкой имён — работа впустую, квадратичная по числу повторов. На
    вырожденной колонке (скажем, `phash` посчитался с ошибкой и стал нулём
    везде) это десятки миллионов пар вместо мгновенной внятной ошибки. По
    уникальным значениям такой работы нет вовсе.

    Повтор одного значения внутри одного сплита разрешён: одинаковые
    кадры в train утечкой не являются, утечка — это одно и то же фото в
    двух разных сплитах.
    """
    сплиты_значения: dict[int, set[str]] = defaultdict(set)
    for имя, d in sorted(splits.items()):
        for v in d["phash"]:
            сплиты_значения[int(v)].add(имя)

    значения = sorted(сплиты_значения)
    for a in значения:
        где = сплиты_значения[a]
        if len(где) > 1:
            первый, второй = sorted(где)[:2]
            raise InvariantError(f"одинаковый phash {a} в сплитах {первый} и {второй}")

    for i, j in close_pairs(значения, предел):
        a, b = значения[i], значения[j]
        # После проверки выше у каждого значения ровно один сплит.
        (имя_a,) = сплиты_значения[a]
        (имя_b,) = сплиты_значения[b]
        if имя_a != имя_b:
            raise InvariantError(f"близкие phash {a} и {b} в разных сплитах ({имя_a} и {имя_b})")
