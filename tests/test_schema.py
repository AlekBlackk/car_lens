"""Тесты схемы данных §13.1."""

from pathlib import Path

import pandas as pd
import pytest

from pipeline.schema import (
    BODY_STYLES,
    COLUMNS,
    SOURCES,
    SchemaError,
    empty_frame,
    read_split,
    validate_frame,
    write_split,
)


def строка(**переопределения: object) -> dict[str, object]:
    основа: dict[str, object] = {
        "photo_id": "0f8e4f70-aadc-4f70-9f8e-000000000001",
        "listing_id": "stanford_000001",
        "source": "street",
        "crop_path": "stanford/000001.jpg",
        "brand": "Volkswagen",
        "model_id": "volkswagen__golf",
        "class_idx": 0,
        "body_style": "hatchback",
        "year": 2015,
        "bbox": [10.0, 20.0, 300.0, 200.0],
        "bbox_area_frac": 0.42,
        "det_conf": 0.91,
        "phash": 1234567890123456789,
        "raw_model_string": "Golf 1.6 TDI",
        "is_multi_car": False,
    }
    основа.update(переопределения)
    return основа


def кадр(*строки: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(строки), columns=list(COLUMNS))


def test_схема_содержит_ровно_пятнадцать_колонок() -> None:
    assert len(COLUMNS) == 15


def test_порядок_колонок_совпадает_со_спецификацией() -> None:
    assert COLUMNS == (
        "photo_id",
        "listing_id",
        "source",
        "crop_path",
        "brand",
        "model_id",
        "class_idx",
        "body_style",
        "year",
        "bbox",
        "bbox_area_frac",
        "det_conf",
        "phash",
        "raw_model_string",
        "is_multi_car",
    )


def test_допустимые_источники() -> None:
    # Ожидание слева: ruff (SIM300) считает `SOURCES` константой и требует
    # держать её справа от сравнения. Смысл проверки от порядка не меняется.
    assert {
        "nettiauto",
        "tori",
        "autoscout_fi",
        "autoscout_de",
        "mobile_de",
        "street",
        "stanford",
        "negatives",
    } == SOURCES


def test_допустимые_кузова() -> None:
    # Те же семь значений намеренно продублированы в `pipeline/normalize.py`
    # (§3.3 и §13.1 — разные разделы спеки). Расхождение копий ловит этот тест
    # вместе с парным ему в `tests/test_normalize.py`, поэтому список здесь
    # выписан дословно, а не импортирован.
    assert {"hatchback", "sedan", "wagon", "suv", "coupe", "van", "pickup"} == BODY_STYLES


def test_валидный_кадр_проходит() -> None:
    validate_frame(кадр(строка()))


def test_пустой_кадр_проходит() -> None:
    validate_frame(empty_frame())


def test_отсутствие_колонки_отвергается() -> None:
    плохой = кадр(строка()).drop(columns=["phash"])
    with pytest.raises(SchemaError, match="phash"):
        validate_frame(плохой)


def test_лишняя_колонка_отвергается() -> None:
    плохой = кадр(строка())
    плохой["color"] = "red"
    with pytest.raises(SchemaError, match="color"):
        validate_frame(плохой)


def test_дубликат_photo_id_отвергается() -> None:
    плохой = кадр(строка(), строка())
    with pytest.raises(SchemaError, match="photo_id"):
        validate_frame(плохой)


def test_неизвестный_источник_отвергается() -> None:
    with pytest.raises(SchemaError, match="source"):
        validate_frame(кадр(строка(source="avito")))


def test_негатив_без_модели_и_марки_проходит() -> None:
    validate_frame(
        кадр(
            строка(
                source="negatives",
                brand=None,
                model_id=None,
                class_idx=-1,
                body_style=None,
                raw_model_string=None,
            )
        )
    )


def test_инвариант_2_класс_без_модели_отвергается() -> None:
    with pytest.raises(SchemaError, match="class_idx"):
        validate_frame(кадр(строка(class_idx=5, model_id=None)))


def test_инвариант_2_модель_без_класса_отвергается() -> None:
    with pytest.raises(SchemaError, match="class_idx"):
        validate_frame(кадр(строка(class_idx=-1, model_id="volkswagen__golf")))


def test_инвариант_2_модель_без_марки_отвергается() -> None:
    with pytest.raises(SchemaError, match="brand"):
        validate_frame(кадр(строка(brand=None)))


def test_bbox_не_из_четырёх_чисел_отвергается() -> None:
    with pytest.raises(SchemaError, match="bbox"):
        validate_frame(кадр(строка(bbox=[1.0, 2.0, 3.0])))
    # Пропущенное поле приходит скалярным NaN: тоже SchemaError, а не TypeError.
    with pytest.raises(SchemaError, match="bbox"):
        validate_frame(кадр(строка(bbox=float("nan"))))


def test_недопустимый_кузов_отвергается() -> None:
    with pytest.raises(SchemaError, match="body_style"):
        validate_frame(кадр(строка(body_style="limousine")))


def test_запись_и_чтение_сохраняют_данные(tmp_path: Path) -> None:
    исходный = кадр(строка(), строка(photo_id="x-2", listing_id="l-2", phash=42))
    путь = tmp_path / "train.parquet"
    write_split(исходный, путь)
    прочитанный = read_split(путь)
    assert list(прочитанный.columns) == list(COLUMNS)
    assert len(прочитанный) == 2
    assert [int(v) for v in прочитанный["phash"]] == [1234567890123456789, 42]


def test_запись_и_чтение_пустого_сплита(tmp_path: Path) -> None:
    # Пустой сплит реалистичен (`negatives` без строк), а колонки `empty_frame`
    # имеют dtype object: проверяем, что pyarrow приводит их к int32, float32,
    # bool и list<float32>[4] по схеме, а не падает.
    путь = tmp_path / "negatives.parquet"
    write_split(empty_frame(), путь)
    прочитанный = read_split(путь)
    assert list(прочитанный.columns) == list(COLUMNS)
    assert прочитанный.empty


def test_запись_невалидного_кадра_падает(tmp_path: Path) -> None:
    путь = tmp_path / "bad.parquet"
    with pytest.raises(SchemaError):
        write_split(кадр(строка(source="avito")), путь)
    # Валидация идёт до mkdir и записи, поэтому обрубка файла остаться не может.
    assert not путь.exists()


def test_источник_stanford_принимается() -> None:
    """Скелет M-1 пишет свои строки с этим источником (Р-1)."""
    validate_frame(кадр(строка(source="stanford")))
