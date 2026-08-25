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


def test_один_model_id_под_двумя_class_idx_ловится() -> None:
    """Сломанная проекция §13.1: одна модель размечена двумя разными классами.

    Так выглядит кадр, собранный на двух версиях каталога: часть строк
    указывает на прежний индекс, и обучение получает одну машину под двумя
    метками. Инвариант проверяет именно это направление; обратное — много
    `model_id` под одним `class_idx` — у платформенной группы норма.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    сплиты["val"] = сплиты["val"].copy()
    строка = сплиты["val"].index[0]
    # `model_id` остаётся тем же, что и в train, а `class_idx` — чужой.
    # Считается от текущего, а не пишется литералом: литерал мог бы совпасть
    # с настоящим классом строки, и тест прошёл бы, ничего не сломав.
    чужой = (int(сплиты["val"].loc[строка, "class_idx"]) + 1) % 3
    сплиты["val"].loc[строка, "class_idx"] = чужой
    with pytest.raises(InvariantError, match="class_idx"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_участники_группы_под_одним_class_idx_разрешены() -> None:
    """Много `model_id` под одним `class_idx` — это платформенная группа §13.2.

    До переворота инвариант проверял `class_idx → model_id` и падал на первом
    же участнике группы, попавшем в другой сплит, то есть запрещал ровно то,
    ради чего группы введены.
    """
    кадр = сделать_кадр(объявлений_на_класс=60)
    # Класс 0 — группа из двух участников разных марок: половина объявлений
    # размечена одним участником, половина другим. Раскладка идёт по
    # `listing_id`, поэтому участники неизбежно разъезжаются по сплитам.
    объявления = sorted(кадр.loc[кадр["class_idx"] == 0, "listing_id"].unique())
    маска = кадр["listing_id"].isin(set(объявления[::2]))
    кадр.loc[маска, "model_id"] = "peugeot__m0"
    кадр.loc[маска, "brand"] = "Peugeot"

    сплиты = split_by_listing(кадр)
    участники = {имя: set(d.loc[d["class_idx"] == 0, "model_id"]) for имя, d in сплиты.items()}
    # Без этой проверки тест был бы пустым: если оба участника осели в одном
    # сплите, согласованность между сплитами не проверялась бы вовсе.
    оба = {"volkswagen__m0", "peugeot__m0"}
    assert участники == {"train": оба, "val": оба, "test": оба}

    check_invariants(сплиты, min_listings_per_class=40)


def test_близкие_phash_в_разных_сплитах_ловятся() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    значение = int(сплиты["train"].iloc[0]["phash"])
    сплиты["val"] = сплиты["val"].copy()
    сплиты["val"].loc[сплиты["val"].index[0], "phash"] = значение ^ 0b11
    with pytest.raises(InvariantError, match="phash"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_одинаковый_phash_внутри_одного_сплита_разрешён() -> None:
    """Утечка — это одно фото в двух сплитах, а не десяток похожих кадров в train.

    В кадре все `phash` различны (выведены из sha256), поэтому правило
    «повтор внутри одного сплита допустим» не проверялось нигде: правку,
    начинающую отвергать такие повторы, CI пропустил бы молча.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    сплиты["train"] = сплиты["train"].copy()
    # Значение берём из самого train: новых межсплитовых совпадений не
    # появляется, так что проверяется ровно повтор внутри сплита.
    сплиты["train"]["phash"] = int(сплиты["train"].iloc[0]["phash"])
    assert len(сплиты["train"]) > 1
    check_invariants(сплиты, min_listings_per_class=40)


def test_хэмминг_считает_биты() -> None:
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1010, 0b1011) == 1
    assert hamming(0, -1) == 64


def test_пропорции_суммируются_в_единицу() -> None:
    with pytest.raises(ValueError, match="сумма"):
        SplitRatios(train=0.8, val=0.3, test=0.1)
