"""Тесты оркестратора сборки кадра данных из Stanford Cars."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.catalog import Catalog, load_catalog
from pipeline.clean import phash_of
from pipeline.crop import Detection
from pipeline.ingest.build import build_frame, iou
from pipeline.ingest.stanford import StanfordPhoto
from pipeline.schema import COLUMNS, validate_frame

YAML = """
version: 1
brands:
  Audi:
    models:
      a5: {display: Audi A5}
  BMW:
    models:
      x5: {display: BMW X5}
"""


class ФейковыйДетектор:
    """Отдаёт заранее заданные боксы, ничего не считая."""

    def __init__(self, боксы: list[Detection]) -> None:
        self._боксы = боксы

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self._боксы)


@pytest.fixture
def каталог(tmp_path: Path) -> Catalog:
    путь = tmp_path / "stanford.yaml"
    путь.write_text(YAML, encoding="utf-8")
    return load_catalog(путь)


def сделать_фото(tmp_path: Path, имя: str, класс: str, оттенок: int = 0) -> StanfordPhoto:
    """Синтетическое фото: серый фон с квадратом обратного тона.

    Различать снимки только параметром `оттенок`, не трогая геометрию,
    недостаточно: перцептивный хеш строится на DCT и к чисто линейному
    масштабированию амплитуды не чувствителен — на несжатом массиве
    фон=0/квадрат=255 и фон=90/квадрат=165 дают побитово одинаковый
    phash (проверено вручную вне пайплайна). Раньше это маскировалось
    тем, что `cv2.imwrite` пишет JPEG, и нелинейное квантование
    DCT-блоков всё же разводило хеши после записи/чтения с диска — но
    это побочный эффект конкретного кодека, а не осознанный дизайн
    фикстуры, и он был бы хрупким к смене библиотеки JPEG. Поэтому при
    ненулевом `оттенок` квадрат ещё и сдвигается — различие теперь
    геометрическое и не зависит от кодека. Порог
    `DEFAULT_PHASH_HAMMING` в `pipeline/clean.py` этой правкой не
    трогается.
    """
    каталог_картинок = tmp_path / "images"
    каталог_картинок.mkdir(parents=True, exist_ok=True)
    путь = каталог_картинок / имя
    поле = np.full((100, 100, 3), оттенок, dtype=np.uint8)
    if оттенок == 0:
        поле[20:80, 20:80] = 255 - оттенок
    else:
        поле[10:60, 30:90] = 255 - оттенок
    cv2.imwrite(str(путь), поле)
    return StanfordPhoto(
        path=путь, class_name=класс, manual_box=(20, 20, 80, 80), official_split="train"
    )


ХОРОШИЙ_БОКС = [Detection(x=20.0, y=20.0, w=60.0, h=60.0, conf=0.95, cls=2)]


def test_кадр_проходит_валидацию_схемы(tmp_path: Path, каталог: Catalog) -> None:
    """Главная проверка задачи: результат оркестратора — валидный §13.1."""
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    validate_frame(итог.frame)
    assert list(итог.frame.columns) == list(COLUMNS)


def test_колонки_заполнены_ожидаемым(tmp_path: Path, каталог: Catalog) -> None:
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    кропы = tmp_path / "crops"
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, кропы)
    строка = итог.frame.iloc[0]
    assert строка["source"] == "stanford"
    assert строка["model_id"] == "audi__a5"
    assert строка["brand"] == "Audi"
    assert строка["body_style"] == "coupe"
    assert строка["year"] == 2012
    assert строка["raw_model_string"] == "Audi A5 Coupe 2012"
    assert строка["class_idx"] == 0
    assert bool(строка["is_multi_car"]) is False
    # Закрепляет Р-11: хеш считается по записанному на диск кропу, а не по
    # исходному кадру. Заглушка `phash = 0` в build_frame ничем, кроме этой
    # проверки, не поймана — validate_frame значение phash не проверяет.
    записанный_кроп = cv2.imread(str(кропы / строка["crop_path"]))
    assert записанный_кроп is not None
    assert строка["phash"] == phash_of(записанный_кроп)


def test_listing_id_равен_photo_id(tmp_path: Path, каталог: Catalog) -> None:
    """У Stanford нет объявлений: каждое фото само себе объявление (Р-9)."""
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    строка = итог.frame.iloc[0]
    assert строка["listing_id"] == строка["photo_id"]


def test_кроп_записан_на_диск(tmp_path: Path, каталог: Catalog) -> None:
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    кропы = tmp_path / "crops"
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, кропы)
    assert (кропы / итог.frame.iloc[0]["crop_path"]).is_file()


def test_фото_без_машины_отбраковано(tmp_path: Path, каталог: Catalog) -> None:
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    итог = build_frame(фото, ФейковыйДетектор([]), каталог, {}, tmp_path / "crops")
    assert итог.frame.empty
    assert итог.stats.rejected["no_vehicle"] == 1


def test_слишком_мелкая_машина_отбракована(tmp_path: Path, каталог: Catalog) -> None:
    мелкий = [Detection(x=0.0, y=0.0, w=10.0, h=10.0, conf=0.95, cls=2)]
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    итог = build_frame(фото, ФейковыйДетектор(мелкий), каталог, {}, tmp_path / "crops")
    assert итог.frame.empty
    assert итог.stats.rejected["too_small"] == 1


def test_многомашинный_кадр_помечен(tmp_path: Path, каталог: Catalog) -> None:
    двое = [
        Detection(x=20.0, y=20.0, w=60.0, h=60.0, conf=0.95, cls=2),
        Detection(x=0.0, y=0.0, w=50.0, h=50.0, conf=0.9, cls=2),
    ]
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    итог = build_frame(фото, ФейковыйДетектор(двое), каталог, {}, tmp_path / "crops")
    assert bool(итог.frame.iloc[0]["is_multi_car"]) is True


def test_несопоставленный_класс_не_попадает_в_кадр(tmp_path: Path, каталог: Catalog) -> None:
    фото = [сделать_фото(tmp_path, "00001.jpg", "Plymouth Neon Coupe 1999")]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    assert итог.frame.empty
    assert итог.stats.unmapped == 1


def test_дубли_убраны(tmp_path: Path, каталог: Catalog) -> None:
    фото = [
        сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012"),
        сделать_фото(tmp_path, "00002.jpg", "Audi A5 Coupe 2012"),
    ]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    assert len(итог.frame) == 1
    assert итог.stats.deduped == 1


def test_кропы_дублей_удаляются_с_диска(tmp_path: Path, каталог: Catalog) -> None:
    """Кроп дубля пишется на диск раньше, чем известен его хеш (Minor 4).

    Без явного удаления в `crops_root` копится сирота, на которую не
    ссылается ни одна строка кадра — на полном Stanford это тысячи
    лишних файлов.
    """
    фото = [
        сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012"),
        сделать_фото(tmp_path, "00002.jpg", "Audi A5 Coupe 2012"),
    ]
    кропы = tmp_path / "crops"
    build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, кропы)
    assert len(list(кропы.iterdir())) == 1


def test_счётчик_увиденного(tmp_path: Path, каталог: Catalog) -> None:
    фото = [
        сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012"),
        сделать_фото(tmp_path, "00002.jpg", "BMW X5 SUV 2012", оттенок=90),
    ]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    assert итог.stats.seen == 2
    assert len(итог.frame) == 2


def test_iou_считается(tmp_path: Path, каталог: Catalog) -> None:
    фото = [сделать_фото(tmp_path, "00001.jpg", "Audi A5 Coupe 2012")]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    assert итог.stats.iou_values == (1.0,)


def test_пустой_вход_даёт_пустой_кадр(tmp_path: Path, каталог: Catalog) -> None:
    итог = build_frame([], ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    validate_frame(итог.frame)
    assert итог.frame.empty


def test_битый_файл_отбракован_как_unreadable(tmp_path: Path, каталог: Catalog) -> None:
    каталог_картинок = tmp_path / "images"
    каталог_картинок.mkdir(parents=True, exist_ok=True)
    путь = каталог_картинок / "битый.jpg"
    путь.write_bytes(b"not an image")
    фото = [
        StanfordPhoto(
            path=путь, class_name="Audi A5 Coupe 2012", manual_box=None, official_split="train"
        )
    ]
    итог = build_frame(фото, ФейковыйДетектор(ХОРОШИЙ_БОКС), каталог, {}, tmp_path / "crops")
    assert итог.frame.empty
    assert итог.stats.rejected["unreadable"] == 1


@pytest.mark.parametrize(
    ("a", "b", "ожидание"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        ((0, 0, 10, 10), (10, 10, 20, 20), 0.0),
        ((0, 0, 10, 10), (0, 0, 10, 5), 0.5),
    ],
)
def test_iou_на_известных_случаях(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], ожидание: float
) -> None:
    assert iou(a, b) == pytest.approx(ожидание)
