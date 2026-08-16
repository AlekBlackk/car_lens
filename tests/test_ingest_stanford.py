"""Тесты чтения раскладки Stanford Cars с диска."""

from pathlib import Path

import numpy as np
import pytest

from pipeline.ingest.stanford import IngestError, StanfordPhoto, read_annotations, year_of

ЗАГОЛОВОК = "image,x1,y1,x2,y2,class_name,official_split"


def разложить(корень: Path, строки: list[str], картинок: int = 2) -> Path:
    """Создать дерево датасета: картинки 8×8 и annotations.csv."""
    import cv2

    (корень / "images").mkdir(parents=True, exist_ok=True)
    for н in range(1, картинок + 1):
        cv2.imwrite(str(корень / "images" / f"{н:05d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    (корень / "annotations.csv").write_text(
        "\n".join([ЗАГОЛОВОК, *строки]) + "\n", encoding="utf-8"
    )
    return корень


def test_читает_все_строки(tmp_path: Path) -> None:
    разложить(
        tmp_path,
        [
            "00001.jpg,10,20,300,200,Audi A5 Coupe 2012,train",
            "00002.jpg,5,5,100,100,BMW X5 SUV 2012,test",
        ],
    )
    фото = read_annotations(tmp_path)
    assert len(фото) == 2
    assert фото[0].class_name == "Audi A5 Coupe 2012"
    assert фото[1].official_split == "test"


def test_путь_указывает_на_существующий_файл(tmp_path: Path) -> None:
    разложить(tmp_path, ["00001.jpg,10,20,300,200,Audi A5 Coupe 2012,train"], картинок=1)
    фото = read_annotations(tmp_path)
    assert фото[0].path.is_file()
    assert фото[0].path.name == "00001.jpg"


def test_ручной_бокс_разбирается(tmp_path: Path) -> None:
    разложить(tmp_path, ["00001.jpg,10,20,300,200,Audi A5 Coupe 2012,train"], картинок=1)
    assert read_annotations(tmp_path)[0].manual_box == (10, 20, 300, 200)


def test_пустой_бокс_даёт_none(tmp_path: Path) -> None:
    """Зеркало могло не отдать ручную разметку — это допустимо (Р-8)."""
    разложить(tmp_path, ["00001.jpg,,,,,Audi A5 Coupe 2012,train"], картинок=1)
    assert read_annotations(tmp_path)[0].manual_box is None


def test_частичный_бокс_отвергается(tmp_path: Path) -> None:
    """Три координаты из четырёх — испорченный источник, а не «почти данные»."""
    разложить(tmp_path, ["00001.jpg,10,20,300,,Audi A5 Coupe 2012,train"], картинок=1)
    with pytest.raises(IngestError, match="bbox"):
        read_annotations(tmp_path)


def test_отсутствующая_картинка_отвергается(tmp_path: Path) -> None:
    разложить(tmp_path, ["09999.jpg,10,20,300,200,Audi A5 Coupe 2012,train"], картинок=1)
    with pytest.raises(IngestError, match="09999"):
        read_annotations(tmp_path)


def test_нет_annotations_csv(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir(parents=True)
    with pytest.raises(IngestError, match=r"annotations\.csv"):
        read_annotations(tmp_path)


def test_чужая_колонка_отвергается(tmp_path: Path) -> None:
    """Контракт закрытый: источник, отдавший лишнее, отдал не то, что мы ждём."""
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "annotations.csv").write_text(
        "image,x1,y1,x2,y2,class_name,official_split,color\n", encoding="utf-8"
    )
    with pytest.raises(IngestError, match="color"):
        read_annotations(tmp_path)


def test_пустая_разметка_даёт_пустой_список(tmp_path: Path) -> None:
    разложить(tmp_path, [], картинок=0)
    assert read_annotations(tmp_path) == []


@pytest.mark.parametrize(
    ("имя", "год"),
    [
        ("Audi A5 Coupe 2012", 2012),
        ("AM General Hummer SUV 2000", 2000),
        ("Volvo 240 Sedan 1993", 1993),
        ("Что-то без года", None),
    ],
)
def test_год_берётся_из_хвоста_имени(имя: str, год: int | None) -> None:
    assert year_of(имя) == год


def test_запись_неизменяема(tmp_path: Path) -> None:
    разложить(tmp_path, ["00001.jpg,10,20,300,200,Audi A5 Coupe 2012,train"], картинок=1)
    фото = read_annotations(tmp_path)[0]
    with pytest.raises(AttributeError):
        фото.class_name = "другое"  # type: ignore[misc]


def test_тип_записи_экспортируется() -> None:
    assert StanfordPhoto.__name__ == "StanfordPhoto"


def test_контракт_совпадает_с_качалкой() -> None:
    """Список колонок продублирован в двух местах — копии обязаны совпадать.

    `pipeline` не зависит от `scripts` по построению, поэтому вместо
    общей константы здесь стоит тест.
    """
    from pipeline.ingest.stanford import ANNOTATION_COLUMNS as У_INGEST
    from scripts.fetch_stanford import ANNOTATION_COLUMNS as У_КАЧАЛКИ

    assert У_INGEST == У_КАЧАЛКИ
