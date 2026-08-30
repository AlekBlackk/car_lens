"""Отбор отложенного эталона: правило 2 Р-11 и §3.5."""

from __future__ import annotations

from pipeline.harvest.manifest import HarvestRow
from pipeline.holdout import HOLDOUT_MIN, listing_id_of, select_holdout


def _строка(файл: str, класс: str, автор: str, вердикт: str = "ok") -> HarvestRow:
    основа = файл.removesuffix(".jpg")
    return HarvestRow(
        file=файл,
        class_id=класс,
        source="commons",
        source_item_id=основа,
        page_url="https://commons.wikimedia.org/wiki/File:x",
        file_url="https://upload.wikimedia.org/x.jpg",
        author=автор,
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis="category:X",
        sha256=основа,
        phash=abs(hash(основа)) % (1 << 60),
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
        audit=вердикт,
    )


def _класс(класс: str, сколько: int, автор_на_кадр: bool = True) -> list[HarvestRow]:
    return [
        _строка(f"{класс}{n}.jpg", класс, f"автор{n if автор_на_кадр else 0}")
        for n in range(сколько)
    ]


def test_набирает_десять_из_тридцати() -> None:
    выбор = select_holdout(_класс("lada__2107", 30))
    assert len(выбор.files) == 10
    assert выбор.short_classes == ()


def test_серия_одного_автора_не_разрывается() -> None:
    серия = [_строка(f"s{n}.jpg", "lada__granta", "общий") for n in range(6)]
    строки = серия + _класс("lada__granta", 20)
    имена_серии = {с.file for с in серия}
    попало = имена_серии & select_holdout(строки).files
    assert попало in (имена_серии, set()), "серия автора разорвана между сплитами"


def test_кадр_без_вердикта_в_эталон_не_идёт() -> None:
    строки = [
        _строка(f"a{n}.jpg", "uaz__patriot", f"автор{n}", "ok" if n % 2 else "")
        for n in range(30)
    ]
    без_вердикта = {с.file for с in строки if not с.audit}
    выбор = select_holdout(строки)
    assert not (выбор.files & без_вердикта)


def test_вердикт_conflict_в_эталон_не_идёт() -> None:
    строки = [_строка(f"c{n}.jpg", "kia__rio", f"автор{n}", "conflict") for n in range(30)]
    выбор = select_holdout(строки)
    assert выбор.files == frozenset()
    assert выбор.short_classes == ("kia__rio",)


def test_класс_короче_минимума_объявлен_дефицитным() -> None:
    выбор = select_holdout(_класс("lifan__x60", HOLDOUT_MIN - 1))
    assert выбор.files == frozenset()
    assert выбор.short_classes == ("lifan__x60",)


def test_отбор_детерминирован() -> None:
    строки = _класс("toyota__camry", 30)
    assert select_holdout(строки).files == select_holdout(строки).files


def test_listing_id_склеивает_автора_с_классом() -> None:
    assert listing_id_of(_строка("a.jpg", "kia__rio", "Иван")) == "commons:kia__rio:Иван"


def test_группа_со_смешанными_вердиктами_не_разрывается_и_в_эталон_не_идёт() -> None:
    смешанная = [
        _строка("m1.jpg", "lada__2107", "автор_смешанный", "ok"),
        _строка("m2.jpg", "lada__2107", "автор_смешанный", "ok"),
        _строка("m3.jpg", "lada__2107", "автор_смешанный", "conflict"),
    ]
    строки = смешанная + _класс("lada__2107", 20)
    выбор = select_holdout(строки)
    assert not (выбор.files & {с.file for с in смешанная})
