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
    cross_class_close_phash,
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


def индекс_строки_класса(кадр: pd.DataFrame, класс: int) -> int:
    """Индекс первой строки класса: тест не должен зависеть от порядка склейки сплита."""
    свои = кадр.index[кадр["class_idx"] == класс]
    assert len(свои) > 0, f"класса {класс} нет в сплите"
    return int(свои[0])


def подсадить_phash(сплиты: dict[str, pd.DataFrame], сплит: str, класс: int, значение: int) -> None:
    """Записать `значение` в `phash` первой строки класса внутри сплита."""
    сплиты[сплит] = сплиты[сплит].copy()
    сплиты[сплит].loc[индекс_строки_класса(сплиты[сплит], класс), "phash"] = значение


def взять_phash(сплиты: dict[str, pd.DataFrame], сплит: str, класс: int) -> int:
    """`phash` первой строки класса. Через `to_numpy`, а не индексацию кадра: скаляр из
    `DataFrame.loc` mypy видит как union из пятнадцати типов схемы."""
    кадр = сплиты[сплит]
    свои = кадр.loc[кадр["class_idx"] == класс, "phash"].to_numpy()
    assert len(свои) > 0, f"класса {класс} нет в сплите {сплит}"
    return int(свои[0])


def test_близкие_phash_разных_классов_не_утечка() -> None:
    """Одна машина в двух классах не лежит, поэтому близость хешей там — не утечка.

    Автор, снимающий разные модели в одной композиции, даёт близкие pHash у
    разных классов. Класс-безразличное сравнение отдавало таким парам право
    решать раскладку сплитов и валило сборку `holdout_ru` там, где утечки нет.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "val", 1, взять_phash(сплиты, "train", 0) ^ 0b11)
    check_invariants(сплиты, min_listings_per_class=40)


def test_близкие_phash_одного_класса_в_разных_сплитах_ловятся() -> None:
    """То же построение, но класс один — это утечка, и она обязана падать."""
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "val", 0, взять_phash(сплиты, "train", 0) ^ 0b11)
    with pytest.raises(InvariantError, match="phash"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_одинаковый_phash_разных_классов_не_утечка() -> None:
    """Расстояние 0 между классами — подозрение на ошибку метки, а не утечка."""
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "val", 1, взять_phash(сплиты, "train", 0))
    check_invariants(сплиты, min_listings_per_class=40)


def test_одинаковый_phash_одного_класса_в_разных_сплитах_ловятся() -> None:
    """Ветвь расстояния 0 внутри класса: до этого теста её не проверял никто."""
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "val", 0, взять_phash(сплиты, "train", 0))
    with pytest.raises(InvariantError, match="phash"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_межклассовая_близость_возвращается_парой() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    значение = взять_phash(сплиты, "train", 0)
    подсадить_phash(сплиты, "val", 1, значение ^ 0b11)

    пары = cross_class_close_phash(сплиты)

    assert len(пары) == 1
    пара = пары[0]
    assert пара.distance == 2
    assert (пара.class_a, пара.class_b) == (0, 1)
    assert {пара.phash_a, пара.phash_b} == {значение, значение ^ 0b11}


def test_межклассовое_совпадение_даёт_нулевое_расстояние() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "val", 1, взять_phash(сплиты, "train", 0))

    пары = cross_class_close_phash(сплиты)

    assert [(п.distance, п.class_a, п.class_b) for п in пары] == [(0, 0, 1)]


def test_честный_сплит_не_даёт_межклассовых_пар() -> None:
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    assert cross_class_close_phash(сплиты) == []


def test_межклассовая_близость_внутри_одного_сплита_тоже_видна() -> None:
    """Ошибка метки не зависит от раскладки: пара обязана находиться и внутри train."""
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "train", 1, взять_phash(сплиты, "train", 0) ^ 0b1)

    пары = cross_class_close_phash(сплиты)

    assert [(п.distance, п.class_a, п.class_b) for п in пары] == [(1, 0, 1)]


def test_пара_держит_сторону_вместе() -> None:
    """`phash_a` обязан принадлежать `class_a`, иначе печать уводит к чужому кадру.

    Классы в паре упорядочиваются; если phash упорядочить отдельно от класса,
    часть строк вывода укажет на другую сторону пары — а вся ценность пары в
    том, чтобы человек нашёл по ней конкретный кадр.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    # Старшему классу — меньшее значение: при раздельной сортировке разъедутся.
    подсадить_phash(сплиты, "train", 2, 0b1000)
    подсадить_phash(сплиты, "val", 1, 0b1011)

    (пара,) = cross_class_close_phash(сплиты)

    assert (пара.class_a, пара.phash_a) == (1, 0b1011)
    assert (пара.class_b, пара.phash_b) == (2, 0b1000)


def test_коллизия_в_старшем_классе_тоже_ловится() -> None:
    """Цикл обязан пройти все классы: проверка только первого прошла бы весь набор."""
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "val", 2, взять_phash(сплиты, "train", 2) ^ 0b11)
    with pytest.raises(InvariantError, match="phash"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_строки_без_класса_сравниваются_между_собой() -> None:
    """Пустой `class_idx` и `-1` — один и тот же «класса нет» §13.1.

    Негативы обязаны сравниваться между собой, а не выпадать из проверки:
    иначе одно фото под двумя способами записи «класса нет» разъедется по
    сплитам молча. Инвариант 4 таким строкам не мешает — он смотрит только
    строки с `class_idx >= 0`.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    значение = взять_phash(сплиты, "train", 0)
    for сплит, класс, метка in (("train", 0, float("nan")), ("val", 1, -1.0)):
        сплиты[сплит] = сплиты[сплит].copy()
        # Колонка приводится к float заранее: именно так `read_split` отдаёт
        # nullable int32 из parquet, и запись NaN в целочисленную колонку
        # pandas ругает предупреждением о несовместимом типе.
        сплиты[сплит]["class_idx"] = сплиты[сплит]["class_idx"].astype(float)
        сплиты[сплит].loc[индекс_строки_класса(сплиты[сплит], класс), "class_idx"] = метка
    подсадить_phash(сплиты, "val", -1, значение)

    with pytest.raises(InvariantError, match="phash"):
        check_invariants(сплиты, min_listings_per_class=40)


def test_межклассовая_пара_ровно_на_пороге_попадает_а_за_порогом_нет() -> None:
    """Порог предупреждения обязан совпадать с порогом инварианта.

    Все три настоящие межклассовые пары манифеста стоят ровно на расстоянии 6:
    сужение порога на единицу гасит предупреждение целиком, и без этого теста
    такая правка прошла бы зелёной.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "train", 0, 0)

    подсадить_phash(сплиты, "val", 1, 0b111111)
    assert [п.distance for п in cross_class_close_phash(сплиты)] == [6]

    подсадить_phash(сплиты, "val", 1, 0b1111111)
    assert cross_class_close_phash(сплиты) == []


def test_внутриклассовая_близость_в_список_не_попадает() -> None:
    """Фильтр по классу: без него предупреждение зовёт ошибкой метки серию автора.

    Инвариант такую пару ловит сам, и это его дело; предупреждение о метках
    обязано молчать, иначе человек получает ту же ложную тревогу, от которой
    правка уводит.
    """
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "train", 0, 0)
    подсадить_phash(сплиты, "val", 0, 0b11)

    assert cross_class_close_phash(сплиты) == []


def test_пары_упорядочены_расстоянием() -> None:
    """Совпадение хеша при разных метках — сильнейший признак, оно идёт первым."""
    сплиты = split_by_listing(сделать_кадр(объявлений_на_класс=60))
    подсадить_phash(сплиты, "train", 0, 0)
    подсадить_phash(сплиты, "val", 1, 0b11)
    подсадить_phash(сплиты, "train", 2, 0)

    пары = cross_class_close_phash(сплиты)

    assert [(п.distance, п.class_a, п.class_b) for п in пары] == [(0, 0, 2), (2, 0, 1), (2, 1, 2)]


def test_хэмминг_считает_биты() -> None:
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1010, 0b1011) == 1
    assert hamming(0, -1) == 64


def test_пропорции_суммируются_в_единицу() -> None:
    with pytest.raises(ValueError, match="сумма"):
        SplitRatios(train=0.8, val=0.3, test=0.1)
