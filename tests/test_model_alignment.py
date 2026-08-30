from pipeline.harvest.alignment import auto_drops, review_queue, triage
from pipeline.harvest.manifest import HarvestRow

_ТАБЛИЦА = {
    "lexus__lx": ("lx",),
    "toyota__land_cruiser": ("land cruiser",),
    "lada__samara": ("2108", "2109"),
    "lada__samara_2": ("2113", "2114"),
}


def _строка(файл: str, класс: str, имя_файла: str, базис: str) -> HarvestRow:
    основа = файл.removesuffix(".jpg")
    return HarvestRow(
        file=файл,
        class_id=класс,
        source="commons",
        source_item_id=основа,
        page_url=f"https://commons.wikimedia.org/wiki/File:{имя_файла}",
        file_url="https://upload.wikimedia.org/x.jpg",
        author="автор",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis=базис,
        sha256=основа,
        phash=1,
        clip_rank=1,
        clip_margin=0.2,
        plates=0,
        faces=0,
        bbox_x=0.0,
        bbox_y=0.0,
        bbox_w=100.0,
        bbox_h=80.0,
        bbox_area_frac=0.5,
        det_conf=0.9,
        audit="ok",
    )


def test_категории_триаж_не_трогает() -> None:
    строки = [_строка("a.jpg", "lexus__lx", "IMG_0412.jpg", "category:Lexus LX")]
    assert triage(строки, _ТАБЛИЦА) == []


def test_чужой_класс_ровно_один_уходит_в_автодроп() -> None:
    строки = [_строка("a.jpg", "lexus__lx", "1982_Toyota_Land_Cruiser_BJ41.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert [в.kind for в in вердикты] == ["foreign"]
    assert [в.file for в in auto_drops(вердикты)] == ["a.jpg"]
    assert review_queue(вердикты) == []


def test_молчащее_имя_уходит_человеку() -> None:
    строки = [_строка("b.jpg", "lexus__lx", "IMG_0412.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert [в.kind for в in вердикты] == ["unknown"]
    assert auto_drops(вердикты) == []
    assert [в.file for в in review_queue(вердикты)] == ["b.jpg"]


def test_два_чужих_класса_уходят_человеку() -> None:
    """Иглы спорят между собой — вердикт за человека выдавать нельзя."""
    строки = [_строка("c.jpg", "lexus__lx", "Lada_2108_and_2113_meet.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert вердикты[0].foreign == ("lada__samara", "lada__samara_2")
    assert auto_drops(вердикты) == []
    assert [в.file for в in review_queue(вердикты)] == ["c.jpg"]


def test_своё_имя_не_попадает_ни_в_один_список() -> None:
    строки = [_строка("d.jpg", "lexus__lx", "2019_Lexus_LX570.jpg", "search:Lexus LX")]
    вердикты = triage(строки, _ТАБЛИЦА)
    assert [в.kind for в in вердикты] == ["own"]
    assert auto_drops(вердикты) == []
    assert review_queue(вердикты) == []
