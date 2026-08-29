"""Схема данных §13.1 и запись parquet.

Единый формат для `train`, `val`, `test`, `street_holdout`, `negatives`.
Схема жёсткая: лишняя колонка — такая же ошибка, как отсутствующая.
Пишется здесь, читается обучением и оценкой. Любое расширение — сначала
правка §13.1, потом код.

`phash` хранится как int64 с той же битовой картиной, что и uint64:
у parquet нет беззнакового 64-битного целого, совместимого со всеми
читателями, а расстояние Хэмминга считается по битам и знак ему
безразличен.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = frozenset(
    {
        "nettiauto",
        "tori",
        "autoscout_fi",
        "autoscout_de",
        "mobile_de",
        "street",
        "stanford",
        # Выборка открытого провенанса Р-11 (Wikimedia Commons). Отделена от `street`
        # намеренно: `street_holdout` §3.5 меряет улицу, `holdout_ru` — каталог.
        "commons",
        "negatives",
    }
)

BODY_STYLES = frozenset({"hatchback", "sedan", "wagon", "suv", "coupe", "van", "pickup"})

PHOTO_SCHEMA = pa.schema(
    [
        pa.field("photo_id", pa.string(), nullable=False),
        pa.field("listing_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("crop_path", pa.string(), nullable=False),
        pa.field("brand", pa.string(), nullable=True),
        pa.field("model_id", pa.string(), nullable=True),
        pa.field("class_idx", pa.int32(), nullable=True),
        pa.field("body_style", pa.string(), nullable=True),
        pa.field("year", pa.int16(), nullable=True),
        pa.field("bbox", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("bbox_area_frac", pa.float32(), nullable=False),
        pa.field("det_conf", pa.float32(), nullable=False),
        pa.field("phash", pa.int64(), nullable=False),
        pa.field("raw_model_string", pa.string(), nullable=True),
        pa.field("is_multi_car", pa.bool_(), nullable=False),
    ]
)

COLUMNS: tuple[str, ...] = tuple(PHOTO_SCHEMA.names)


class SchemaError(ValueError):
    """Кадр данных не соответствует §13.1."""


def empty_frame() -> pd.DataFrame:
    """Пустой кадр с правильными колонками."""
    return pd.DataFrame({имя: pd.Series(dtype="object") for имя in COLUMNS})


def validate_frame(df: pd.DataFrame) -> None:
    """Проверить кадр против §13.1. Бросает `SchemaError` на первом нарушении."""
    _проверить_колонки(df)
    if df.empty:
        return
    _проверить_ключи(df)
    _проверить_перечисления(df)
    _проверить_инвариант_2(df)
    _проверить_bbox(df)


def _проверить_колонки(df: pd.DataFrame) -> None:
    факт = set(df.columns)
    ожидание = set(COLUMNS)
    нет = sorted(ожидание - факт)
    лишние = sorted(факт - ожидание)
    if нет:
        raise SchemaError(f"нет колонок: {', '.join(нет)}")
    if лишние:
        raise SchemaError(f"лишние колонки: {', '.join(лишние)}")


def _проверить_ключи(df: pd.DataFrame) -> None:
    повторы = df["photo_id"][df["photo_id"].duplicated()].tolist()
    if повторы:
        raise SchemaError(f"дубликат photo_id: {повторы[:5]}")


def _проверить_перечисления(df: pd.DataFrame) -> None:
    чужие = sorted(set(df["source"]) - SOURCES)
    if чужие:
        raise SchemaError(f"неизвестный source: {', '.join(map(str, чужие))}")

    кузова = df["body_style"].dropna()
    чужие_кузова = sorted(set(кузова) - BODY_STYLES)
    if чужие_кузова:
        raise SchemaError(f"неизвестный body_style: {', '.join(map(str, чужие_кузова))}")


def _проверить_инвариант_2(df: pd.DataFrame) -> None:
    """`class_idx >= 0` ⟺ `model_id` есть ⟺ `brand` есть ⟺ не негатив."""
    есть_класс = df["class_idx"].fillna(-1).astype(int) >= 0
    есть_модель = df["model_id"].notna()
    есть_марка = df["brand"].notna()

    рассогласование_класса = есть_класс != есть_модель
    if bool(рассогласование_класса.any()):
        плохие = df.loc[рассогласование_класса, "photo_id"].tolist()
        raise SchemaError(f"class_idx и model_id рассогласованы: {плохие[:5]}")

    рассогласование_марки = есть_модель != есть_марка
    if bool(рассогласование_марки.any()):
        плохие = df.loc[рассогласование_марки, "photo_id"].tolist()
        raise SchemaError(f"model_id и brand рассогласованы: {плохие[:5]}")

    негативы = df["source"] == "negatives"
    if bool((негативы & есть_класс).any()):
        raise SchemaError("у негатива не может быть class_idx >= 0")


def _проверить_bbox(df: pd.DataFrame) -> None:
    """Длина `bbox` равна четырём.

    Неитерируемое значение — тоже нарушение схемы, а не `TypeError` наружу:
    `pd.DataFrame` из записей без поля `bbox` подставляет скалярный `NaN`,
    и валидатор обязан ответить на него `SchemaError`, как обещает
    докстринг `validate_frame`.
    """
    плохие: list[object] = []
    for pid, box in zip(df["photo_id"], df["bbox"], strict=True):
        try:
            подходит = len(list(box)) == 4
        except TypeError:
            подходит = False
        if not подходит:
            плохие.append(pid)
    if плохие:
        raise SchemaError(f"bbox должен быть из четырёх чисел: {плохие[:5]}")


def write_split(df: pd.DataFrame, path: Path) -> None:
    """Проверить и записать сплит в parquet."""
    validate_frame(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    таблица = pa.Table.from_pandas(df[list(COLUMNS)], schema=PHOTO_SCHEMA, preserve_index=False)
    pq.write_table(таблица, path, compression="zstd")


def read_split(path: Path) -> pd.DataFrame:
    """Прочитать сплит и проверить его против схемы."""
    # Аннотация обязательна: у pyarrow нет типов, `to_pandas()` для mypy — `Any`.
    df: pd.DataFrame = pq.read_table(path, schema=PHOTO_SCHEMA).to_pandas()
    validate_frame(df)
    return df
