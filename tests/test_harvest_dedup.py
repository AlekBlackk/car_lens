from pipeline.harvest.dedup import DedupIndex


def test_новый_хеш_проходит() -> None:
    индекс = DedupIndex()
    assert индекс.check(0b1010, "lada__granta").status == "new"


def test_близкий_хеш_в_том_же_классе_это_дубль() -> None:
    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg")
    вердикт = индекс.check(0b11, "lada__granta")
    assert вердикт.status == "duplicate"
    assert вердикт.other == "a.jpg"


def test_далёкий_хеш_в_том_же_классе_проходит() -> None:
    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg")
    assert индекс.check(0b1111111, "lada__granta").status == "new"


def test_тот_же_снимок_в_двух_классах_это_конфликт() -> None:
    """Одна фотография в двух классах — улика ошибки разметки, а не дубль."""
    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg")
    вердикт = индекс.check(0b11, "uaz__patriot")
    assert вердикт.status == "conflict"
    assert вердикт.other == "a.jpg"


def test_порог_совпадает_с_очисткой_конвейера() -> None:
    from pipeline.clean import DEFAULT_PHASH_HAMMING

    индекс = DedupIndex()
    индекс.add(0, "lada__granta", "a.jpg")
    на_пороге = (1 << DEFAULT_PHASH_HAMMING) - 1
    за_порогом = (1 << (DEFAULT_PHASH_HAMMING + 1)) - 1
    assert индекс.check(на_пороге, "lada__granta").status == "duplicate"
    assert индекс.check(за_порогом, "lada__granta").status == "new"
