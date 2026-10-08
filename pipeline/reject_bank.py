"""Банк отказных прототипов 0.8.2: ключ модели, фолд и вид кадра.

Правила — `bench/reject-v2-preregistration.md`.

Банк собирается из трёх доменов (студийные чужие машины autoevolution, уличные MAD, не-машины
COCO). Честная оценка идёт перекрёстно **по моделям**: кадр оценивается прототипами, построенными
без его модели. Одна и та же машина в autoevolution (`2003_Opel_Signum`) и в MAD
(`opel__signum`) должна получить один ключ, иначе она попала бы и в пул, и в оценку.
"""

from __future__ import annotations

import hashlib
import re

NON_CAR_KINDS = ("truck_bus", "detector_fp")
FOREIGN = "foreign_car"
NON_CAR = "non_car"

# Год в имени галереи autoevolution: `2003`, а у поколений — `19821990`.
_ГОД = re.compile(r"(19|20)\d{2}(\d{4})?")


def listing_kind(listing_id: str) -> str:
    """Первый компонент `listing_id`: `off_catalog`, `off_catalog_street`, `truck_bus`…"""
    return listing_id.split("/", 1)[0]


def bank_kind(listing_id: str) -> str:
    """Вид прототипа: не-машина (грузовик, автобус, ложный бокс детектора) или чужая машина."""
    return NON_CAR if listing_kind(listing_id) in NON_CAR_KINDS else FOREIGN


def model_key(listing_id: str) -> str:
    """Ключ модели: первый и последний токены имени без годов.

    `off_catalog/autoevolution/2003_Opel_Signum` и `off_catalog_street/mad_cars/opel__signum`
    дают `opel|signum`. Многословные модели (`land_cruiser_prado`) сводятся к первому и
    последнему слову — это грубо, но ошибка в сторону слияния разных моделей делает проверку
    строже, а не мягче. Не-машины ключом получают весь `listing_id` (одна картинка — один ключ).
    """
    if bank_kind(listing_id) == NON_CAR:
        return listing_id
    имя = listing_id.rsplit("/", 1)[-1]
    токены = [т for т in re.findall(r"[a-z0-9]+", имя.lower()) if not _ГОД.fullmatch(т)]
    return f"{токены[0]}|{токены[-1]}" if len(токены) >= 2 else "|".join(токены)


def fold_of(key: str, seed: int, folds: int) -> int:
    """Фолд ключа: `md5(seed:ключ) % folds` — не зависит от порядка кадров и от версии питона."""
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) % folds
