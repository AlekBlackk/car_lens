"""Отбор отложенного эталона: правило 2 Р-11 и §3.5."""

from __future__ import annotations

from dataclasses import replace

from pipeline.harvest.manifest import HarvestRow
from pipeline.holdout import (
    HOLDOUT_MIN,
    POOL_RESERVE,
    classes_short_of_holdout,
    listing_id_of,
    phash_components,
    select_holdout,
)
from pipeline.split import DEFAULT_PHASH_HAMMING


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
        _строка(f"a{n}.jpg", "uaz__patriot", f"автор{n}", "ok" if n % 2 else "") for n in range(30)
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


def test_дефицит_считается_по_каталогу_а_не_по_манифесту() -> None:
    """Класс без строк манифеста дефицитен: `short_classes` о нём не знает вовсе.

    Именно этим `G_C1` занижался в самом тяжёлом случае — класс, потерявший провенанс
    целиком, выпадал из знаменателя, а не попадал в дефицит.
    """
    строки = _класс("lada__2107", 30) + _класс("lifan__x60", HOLDOUT_MIN - 1)
    выбор = select_holdout(строки)

    дефицит = classes_short_of_holdout(
        выбор, строки, ("lada__2107", "lifan__x60", "chevrolet__lanos")
    )

    assert выбор.short_classes == ("lifan__x60",)
    assert дефицит == ("chevrolet__lanos", "lifan__x60")


def test_дефицит_не_считает_классы_вне_каталога() -> None:
    """Знаменатель — каталог: класс манифеста, которого в каталоге нет, гейт не касается."""
    строки = _класс("lifan__x60", HOLDOUT_MIN - 1)

    assert classes_short_of_holdout(select_holdout(строки), строки, ()) == ()


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


def test_серия_дробится_когда_класс_иначе_без_эталона() -> None:
    """Один автор, 30 визуально разных кадров: класс обязан получить эталон."""
    серия = [_строка(f"g{n}.jpg", "lada__granta", "галерейщик") for n in range(30)]
    выбор = select_holdout(серия)
    assert выбор.short_classes == ()
    assert len(выбор.files) == 10
    assert выбор.refined_classes == ("lada__granta",)


def test_дробление_не_включается_когда_грубый_ключ_справился() -> None:
    выбор = select_holdout(_класс("kia__rio", 30))
    assert выбор.refined_classes == ()
    assert all("#" not in ключ for _, ключ in выбор.units)


def test_conflict_внутри_дроблёной_серии_остаётся_в_пуле() -> None:
    серия = [
        _строка(f"x{n}.jpg", "bmw__x5", "галерейщик", "conflict" if n == 0 else "ok")
        for n in range(30)
    ]
    выбор = select_holdout(серия)
    assert выбор.refined_classes == ("bmw__x5",)
    assert "x0.jpg" not in выбор.files
    assert len(выбор.files) == 10


def test_близкие_по_phash_кадры_не_расходятся_по_сплитам() -> None:
    """Компонента связности неделима даже при дроблении серии."""
    близкие = [_строка(f"n{n}.jpg", "lada__kalina", "галерейщик") for n in range(2)]
    близкие = [replace(близкие[0], phash=1024), replace(близкие[1], phash=1025)]
    остальные = [_строка(f"d{n}.jpg", "lada__kalina", "галерейщик") for n in range(28)]
    выбор = select_holdout(близкие + остальные)
    попало = {с.file for с in близкие} & выбор.files
    assert попало in ({"n0.jpg", "n1.jpg"}, set())


def test_класс_оставляет_пулу_резерв_под_центроиды() -> None:
    """Двенадцать кадров: эталон не имеет права забрать больше восьми."""
    выбор = select_holdout(_класс("uaz__patriot", 12))
    assert len(выбор.files) == 12 - POOL_RESERVE


def test_класс_без_запаса_на_резерв_объявлен_дефицитным() -> None:
    выбор = select_holdout(_класс("peugeot__308", HOLDOUT_MIN + POOL_RESERVE - 1))
    assert выбор.files == frozenset()
    assert выбор.short_classes == ("peugeot__308",)


def test_компоненты_склеивают_близкие_и_разделяют_далёкие() -> None:
    строки = [
        replace(_строка("a.jpg", "kia__rio", "автор"), phash=0b1010),
        replace(_строка("b.jpg", "kia__rio", "автор"), phash=0b1011),
        replace(_строка("c.jpg", "kia__rio", "автор"), phash=~0b1010),
    ]
    номера = phash_components(строки, DEFAULT_PHASH_HAMMING)
    assert номера["a.jpg"] == номера["b.jpg"]
    assert номера["c.jpg"] != номера["a.jpg"]


def test_дроблёный_отбор_не_зависит_от_порядка_строк() -> None:
    """Переderivация манифеста тасует строки — эталон дробимого класса тасоваться не смеет."""
    серия = [_строка(f"g{n}.jpg", "lada__granta", "галерейщик") for n in range(30)]
    прямой = select_holdout(серия)
    обратный = select_holdout(list(reversed(серия)))
    assert прямой.refined_classes == ("lada__granta",), "класс обязан пройти дробление"
    assert прямой.files == обратный.files
    assert прямой.units == обратный.units
