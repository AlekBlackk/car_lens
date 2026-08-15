"""Тесты раскладки по сплитам и инвариантов §13.1."""

import hashlib

import pandas as pd
import pytest

from pipeline.schema import COLUMNS
from pipeline.split import (
    DEFAULT_MIN_LISTINGS_PER_CLASS,
    InvariantError,
    SplitRatios,
    check_invariants,
    hamming,
    split_by_listing,
)


def сделать_кадр(
    классов: int = 3, объявлений_на_класс: int = 50, фото_на_объявление: int = 4
) -> pd.DataFrame:
    строки: list[dict[str, object]] = []
    n = 0
    for cls in range(классов):
        for номер_объявления in range(объявлений_на_класс):
            listing = f"l-{cls}-{номер_объявления}"
            for _ in range(фото_на_объявление):
                n += 1
                хеш = int.from_bytes(hashlib.sha256(f"p-{n}".encode()).digest()[:8], "big")
                строки.append(
                    {
                        "photo_id": f"p-{n}",
                        "listing_id": listing,
                        "source": "nettiauto",
                        "crop_path": f"{n}.jpg",
                        "brand": "Volkswagen",
                        "model_id": f"volkswagen__m{cls}",
                        "class_idx": cls,
                        "body_style": None,
                        "year": 2018,
                        "bbox": [0.0, 0.0, 100.0, 100.0],
                        "bbox_area_frac": 0.5,
                        "det_conf": 0.9,
                        "phash": хеш >> 1,
                        "raw_model_string": f"M{cls}",
                        "is_multi_car": False,
                    }
                )
    return pd.DataFrame(строки, columns=list(COLUMNS))


def test_порог_по_умолчанию_равен_тремстам() -> None:
    """Послабление скелета не должно доживать до продакшна."""
    assert DEFAULT_MIN_LISTINGS_PER_CLASS == 300


def test_объявление_целиком_попадает_в_один_сплит() -> None:
    сплиты = split_by_listing(сделать_кадр())
    множества = {имя: set(d["listing_id"]) for имя, d in сплиты.items()}
    assert множества["train"] & множества["val"] == set()
    assert множества["train"] & множества["test"] == set()
    assert множества["val"] & множества["test"] == set()


def test_ни_одна_строка_не_потеряна_без_многомашинных() -> None:
    исходный = сделать_кадр()
    сплиты = split_by_listing(исходный)
    assert sum(len(d) for d in сплиты.values()) == len(исходный)


def test_пропорции_примерно_соблюдены() -> None:
    сплиты = split_by_listing(сделать_кадр(классов=4, объявлений_на_класс=100))
    всего = sum(len(d) for d in сплиты.values())
    assert 0.75 < len(сплиты["train"]) / всего < 0.85


def test_каждый_класс_присутствует_во_всех_сплитах() -> None:
    сплиты = split_by_listing(сделать_кадр())
    for имя, d in сплиты.items():
        assert set(d["class_idx"]) == {0, 1, 2}, f"класс потерян в {имя}"


def test_сплит_воспроизводим_по_seed() -> None:
    первый = split_by_listing(сделать_кадр(), seed=7)
    второй = split_by_listing(сделать_кадр(), seed=7)
    assert set(первый["train"]["listing_id"]) == set(второй["train"]["listing_id"])


def test_разные_seed_дают_разную_раскладку() -> None:
    первый = split_by_listing(сделать_кадр(), seed=1)
    второй = split_by_listing(сделать_кадр(), seed=2)
    assert set(первый["val"]["listing_id"]) != set(второй["val"]["listing_id"])


def test_многомашинные_кадры_выброшены_из_train() -> None:
    кадр = сделать_кадр()
    кадр.loc[кадр.index[:40], "is_multi_car"] = True
    сплиты = split_by_listing(кадр)
    assert not bool(сплиты["train"]["is_multi_car"].any())


def test_многомашинные_кадры_остаются_в_val_и_test() -> None:
    """Выбрасываются только из train: оценка обязана видеть уличную реальность."""
    кадр = сделать_кадр()
    кадр["is_multi_car"] = True
    сплиты = split_by_listing(кадр)
    assert len(сплиты["train"]) == 0
    assert len(сплиты["val"]) > 0
    assert len(сплиты["test"]) > 0
    assert bool(сплиты["val"]["is_multi_car"].all())
    assert bool(сплиты["test"]["is_multi_car"].all())


def test_выбрасывание_многомашинных_не_создаёт_утечку() -> None:
    """Регрессия: перенос строк между сплитами ломал инвариант 1."""
    кадр = сделать_кадр(объявлений_на_класс=60)
    кадр.loc[кадр.index[::3], "is_multi_car"] = True
    сплиты = split_by_listing(кадр)
    check_invariants(сплиты, min_listings_per_class=1)


def test_инварианты_проходят_на_честном_сплите() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    check_invariants(сплиты, min_listings_per_class=40)


def test_утечка_по_listing_id_ловится() -> None:
    """Главный тест плана: сплит по фото вместо объявления обязан падать."""
    кадр = сделать_кадр()
    перемешано = кадр.sample(frac=1.0, random_state=0).reset_index(drop=True)
    граница = int(len(перемешано) * 0.8)
    протекающие = {
        "train": перемешано.iloc[:граница].reset_index(drop=True),
        "val": перемешано.iloc[граница:].reset_index(drop=True),
        "test": перемешано.iloc[:0].reset_index(drop=True),
    }
    with pytest.raises(InvariantError, match="listing_id"):
        check_invariants(протекающие, min_listings_per_class=1)


def test_класс_с_недобором_объявлений_ловится() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=10))
    with pytest.raises(InvariantError, match="300"):
        check_invariants(сплиты)


def test_рассогласование_class_idx_и_model_id_между_сплитами_ловится() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    сплиты["val"] = сплиты["val"].copy()
    сплиты["val"].loc[сплиты["val"].index[0], "model_id"] = "volkswagen__чужая"
    with pytest.raises(InvariantError, match="class_idx"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_близкие_phash_в_разных_сплитах_ловятся() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    значение = int(сплиты["train"].iloc[0]["phash"])
    сплиты["val"] = сплиты["val"].copy()
    сплиты["val"].loc[сплиты["val"].index[0], "phash"] = значение ^ 0b11
    with pytest.raises(InvariantError, match="phash"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_хэмминг_считает_биты() -> None:
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1010, 0b1011) == 1
    assert hamming(0, -1) == 64


def test_пропорции_суммируются_в_единицу() -> None:
    with pytest.raises(ValueError, match="сумма"):
        SplitRatios(train=0.8, val=0.3, test=0.1)
