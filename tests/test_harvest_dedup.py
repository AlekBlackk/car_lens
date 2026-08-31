from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.manifest import HarvestRow


def _строка(**правки: object) -> HarvestRow:
    поля: dict[str, object] = {
        "file": "a.jpg",
        "class_id": "lada__granta",
        "source": "commons",
        "source_item_id": "1",
        "page_url": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "file_url": "https://upload.wikimedia.org/X.jpg",
        "author": "A",
        "license": "CC0",
        "license_tier": 1,
        "checked": "2026-08-28",
        "basis": "category:Lada Granta",
        "sha256": "a" * 64,
        "phash": 0,
        "clip_rank": 1,
        "clip_margin": 0.0,
        "plates": 0,
        "faces": 0,
        "bbox_x": 12.0,
        "bbox_y": 34.0,
        "bbox_w": 500.0,
        "bbox_h": 300.0,
        "bbox_area_frac": 0.42,
        "det_conf": 0.91,
        "audit": "",
    }
    поля.update(правки)
    return HarvestRow(**поля)  # type: ignore[arg-type]


def test_новый_хеш_проходит() -> None:
    индекс = DedupIndex()
    assert индекс.check(0b1010, "lada__granta", "category:Lada Granta").status == "new"


def test_близкий_хеш_в_том_же_классе_это_дубль() -> None:
    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg", "category:Lada Granta")
    вердикт = индекс.check(0b11, "lada__granta", "category:Lada Granta")
    assert вердикт.status == "duplicate"
    assert вердикт.other == "a.jpg"


def test_далёкий_хеш_в_том_же_классе_проходит() -> None:
    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg", "category:Lada Granta")
    assert индекс.check(0b1111111, "lada__granta", "category:Lada Granta").status == "new"


def test_порог_совпадает_с_очисткой_конвейера() -> None:
    from pipeline.clean import DEFAULT_PHASH_HAMMING

    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg", "category:Lada Granta")
    на_пороге = (1 << DEFAULT_PHASH_HAMMING) - 1
    за_порогом = (1 << (DEFAULT_PHASH_HAMMING + 1)) - 1
    assert индекс.check(на_пороге, "lada__granta", "category:Lada Granta").status == "duplicate"
    assert индекс.check(за_порогом, "lada__granta", "category:Lada Granta").status == "new"


def test_категорийный_вытесняет_поисковый_в_чужом_классе() -> None:
    индекс = DedupIndex()
    индекс.add(0, "daewoo__lanos", "a.jpg", "search:Daewoo Lanos")

    вердикт = индекс.check(0, "chevrolet__lanos", "category:Chevrolet Lanos")

    assert (вердикт.status, вердикт.other) == ("supersede", "a.jpg")


def test_два_категорийных_спорят_и_уходят_человеку() -> None:
    индекс = DedupIndex()
    индекс.add(0, "daewoo__lanos", "a.jpg", "category:Daewoo Lanos")

    вердикт = индекс.check(0, "chevrolet__lanos", "category:Chevrolet Lanos")

    assert (вердикт.status, вердикт.other) == ("conflict", "a.jpg")


def test_поисковый_не_вытесняет_категорийный() -> None:
    индекс = DedupIndex()
    индекс.add(0, "daewoo__lanos", "a.jpg", "category:Daewoo Lanos")

    assert индекс.check(0, "chevrolet__lanos", "search:lanos").status == "conflict"


def test_свой_класс_остаётся_дублем() -> None:
    индекс = DedupIndex()
    индекс.add(0, "kia__rio", "a.jpg", "category:Kia Rio")

    assert индекс.check(0, "kia__rio", "category:Kia Rio").status == "duplicate"


def test_засев_из_манифеста_видит_прошлые_прогоны() -> None:
    строка = _строка(file="a.jpg", class_id="daewoo__lanos", phash=0, basis="search:lanos")

    индекс = DedupIndex.from_manifest([строка])

    assert индекс.check(0, "chevrolet__lanos", "category:Chevrolet Lanos").status == "supersede"
