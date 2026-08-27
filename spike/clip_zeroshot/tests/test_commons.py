# spike/clip_zeroshot/tests/test_commons.py
from pathlib import Path

import pytest

from spike.clip_zeroshot.commons import CommonsError, load_commons_manifest

ШАПКА = "file,class_id,url,author,license,checked,basis\n"
СТРОКА = (
    "lada_granta_01.jpg,lada__granta,https://commons.wikimedia.org/wiki/File:X.jpg,"
    "Ivan Petrov,CC BY-SA 4.0,2026-08-26,подпись файла на Commons\n"
)


def _записать(tmp_path: Path, тело: str) -> Path:
    путь = tmp_path / "commons_manifest.csv"
    путь.write_text(ШАПКА + тело, encoding="utf-8")
    return путь


def test_полная_строка_читается(tmp_path: Path) -> None:
    фото = load_commons_manifest(_записать(tmp_path, СТРОКА))
    assert len(фото) == 1
    assert фото[0].class_id == "lada__granta"
    assert фото[0].license == "CC BY-SA 4.0"


def test_класс_вне_состава_отвергается(tmp_path: Path) -> None:
    """Подменять класс при недоборе фотографий запрещено спекой §5.2."""
    строка = СТРОКА.replace("lada__granta", "lada__vesta")
    with pytest.raises(CommonsError, match="lada__vesta"):
        load_commons_manifest(_записать(tmp_path, строка))


def test_несвободная_лицензия_отвергается(tmp_path: Path) -> None:
    строка = СТРОКА.replace("CC BY-SA 4.0", "CC BY-NC 4.0")
    with pytest.raises(CommonsError, match="лицензия"):
        load_commons_manifest(_записать(tmp_path, строка))


def test_пустое_основание_разметки_отвергается(tmp_path: Path) -> None:
    """«Почему это Гранта» — часть замера, а не украшение: без него метка ничем не подпёрта."""
    строка = СТРОКА.replace("подпись файла на Commons", "")
    with pytest.raises(CommonsError, match="basis"):
        load_commons_manifest(_записать(tmp_path, строка))


def test_повтор_файла_отвергается(tmp_path: Path) -> None:
    with pytest.raises(CommonsError, match="повтор"):
        load_commons_manifest(_записать(tmp_path, СТРОКА + СТРОКА))


def test_больше_восьми_на_класс_отвергается(tmp_path: Path) -> None:
    """Перекос по классу тихо утяжелил бы его вклад в долю G3."""
    тело = "".join(
        СТРОКА.replace("lada_granta_01.jpg", f"lada_granta_{номер:02d}.jpg")
        for номер in range(1, 10)
    )
    with pytest.raises(CommonsError, match="больше 8"):
        load_commons_manifest(_записать(tmp_path, тело))
