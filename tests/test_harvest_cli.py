"""CLI харвеста: порядок слияния вердиктов, выбор источника и подача запросов.

`scripts/harvest.py::main` до этого не был покрыт ничем, а именно в нём живёт решение о
том, может ли автомат перебить человека (§8 дизайна C1). Прогон берётся в режиме
`--report-only`: он проходит весь путь вердиктов, но не требует ни сети, ни весов CLIP.
Все пути в теле `main` — модульные константы, поэтому тест подменяет их на `tmp_path`:
иначе он переписал бы настоящий манифест провенанса.

Вторая половина файла — задача C2: `build_source`, `source_queries`, умолчание `--source` и
отказной путь `main` по незаданному ключу. Ни сети, ни весов и здесь не нужно: отказ по
ключу стоит выше всего дорогого, а состав запроса проверяется на подставном транспорте —
у одного теста он доказывает, что до сети дело не дошло вовсе, у другого записывает
параметры единственного вызова.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from pipeline.harvest.audit import load_applied
from pipeline.harvest.commons import HttpResponse
from pipeline.harvest.flickr import requested_licenses
from pipeline.harvest.manifest import HarvestRow, load_manifest, write_manifest
from pipeline.harvest.queries import DEFAULT_QUERIES_PATH, ClassQueries, load_queries
from pipeline.harvest.sources import SourceQuery
from scripts.harvest import ОтсутствуетКлюч, build_parser, build_source, main, source_queries


def _строка(file: str, class_id: str = "kia__rio", audit: str = "") -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id=file.removesuffix(".jpg"),
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="A",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis="category:Kia Rio",
        sha256=file.removesuffix(".jpg").rjust(64, "0"),
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=0,
        faces=0,
        bbox_x=12.0,
        bbox_y=34.0,
        bbox_w=500.0,
        bbox_h=300.0,
        bbox_area_frac=0.42,
        det_conf=0.91,
        audit=audit,
    )


@pytest.fixture
def стенд(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Подменить все пути `main` на временные и вернуть их."""
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path / "data"))
    пути = {
        "манифест": tmp_path / "harvest_manifest.csv",
        "журнал": tmp_path / "dedup_verdicts.csv",
        "отчёт": tmp_path / "harvest-m0.md",
        "атрибуция": tmp_path / "NOTICES.md",
        "вердикты": tmp_path / "data" / "harvest" / "verdicts.dedup.csv",
    }
    monkeypatch.setattr("scripts.harvest.DEFAULT_MANIFEST_PATH", пути["манифест"])
    monkeypatch.setattr("scripts.harvest.DEFAULT_APPLIED_PATH", пути["журнал"])
    monkeypatch.setattr("scripts.harvest.REPORT_PATH", пути["отчёт"])
    monkeypatch.setattr("scripts.harvest.DEFAULT_NOTICES_PATH", пути["атрибуция"])
    пути["вердикты"].parent.mkdir(parents=True, exist_ok=True)
    return пути


def test_автовердикт_не_сносит_строку_с_отметкой_человека(стенд: dict[str, Path]) -> None:
    """Вытеснение проверенной строки — это удаление провенанса решением автомата."""
    write_manifest(стенд["манифест"], [_строка("a.jpg", audit="ok"), _строка("b.jpg", audit="")])
    стенд["вердикты"].write_text(
        "file,verdict\na.jpg,drop:superseded\nb.jpg,drop:superseded\n", encoding="utf-8"
    )

    main(["--contact", "mail@example.com", "--report-only"])

    отметки = {с.file: с.audit for с in load_manifest(стенд["манифест"])}
    assert отметки == {"a.jpg": "conflict"}, "проверенная строка цела, непроверенная снесена"


def test_вердикт_человека_побеждает_автовердикт(стенд: dict[str, Path]) -> None:
    """Порядок слияния: файл дедупа ранний, `--audit` поздний, поздний побеждает."""
    write_manifest(стенд["манифест"], [_строка("a.jpg", audit="")])
    стенд["вердикты"].write_text("file,verdict\na.jpg,drop:superseded\n", encoding="utf-8")
    ручной = стенд["манифест"].parent / "verdicts.csv"
    ручной.write_text("file,verdict\na.jpg,ok\n", encoding="utf-8")

    main(["--contact", "mail@example.com", "--report-only", "--audit", str(ручной)])

    отметки = {с.file: с.audit for с in load_manifest(стенд["манифест"])}
    assert отметки == {"a.jpg": "ok"}, "человек возвращает кадр, снесённый автоматом"


def test_применённый_автовердикт_попадает_в_журнал_и_усекается(стенд: dict[str, Path]) -> None:
    """След удаления обязан остаться в git, а сам вердикт — не примениться второй раз."""
    write_manifest(
        стенд["манифест"],
        [_строка("a.jpg", class_id="mitsubishi__pajero"), _строка("b.jpg", audit="ok")],
    )
    стенд["вердикты"].write_text(
        "file,verdict\na.jpg,drop:superseded\nb.jpg,drop:superseded\n", encoding="utf-8"
    )

    main(["--contact", "mail@example.com", "--report-only"])

    журнал = {з.file: (з.class_id, з.verdict) for з in load_applied(стенд["журнал"])}
    assert журнал["a.jpg"] == ("mitsubishi__pajero", "drop:superseded")
    assert журнал["b.jpg"][1] == "conflict", "в журнал идёт применённый вердикт, а не исходный"
    assert стенд["вердикты"].read_text(encoding="utf-8") == ""


def test_второй_прогон_не_сносит_принятый_заново_кадр(стенд: dict[str, Path]) -> None:
    """Цикл потери: прополка возвращает кадр в кандидаты, а старый вердикт ждёт его снова."""
    write_manifest(стенд["манифест"], [_строка("a.jpg"), _строка("b.jpg")])
    стенд["вердикты"].write_text("file,verdict\na.jpg,drop:superseded\n", encoding="utf-8")
    main(["--contact", "mail@example.com", "--report-only"])
    assert [с.file for с in load_manifest(стенд["манифест"])] == ["b.jpg"]

    # Кадр принят заново под тем же детерминированным именем `{sha}.jpg`.
    write_manifest(стенд["манифест"], [_строка("a.jpg"), _строка("b.jpg")])
    main(["--contact", "mail@example.com", "--report-only"])

    assert [с.file for с in load_manifest(стенд["манифест"])] == ["a.jpg", "b.jpg"]


class ПустойТранспорт:
    def get(self, url: str, params: object) -> HttpResponse:  # pragma: no cover
        raise AssertionError("сеть не должна трогаться при выборе источника")


def test_commons_не_требует_ключа() -> None:
    источник = build_source("commons", ПустойТранспорт(), {}, allow_sa=False)
    assert источник.name == "commons"


def test_flickr_берёт_ключ_из_окружения() -> None:
    источник = build_source("flickr", ПустойТранспорт(), {"FLICKR_API_KEY": "К"}, allow_sa=False)
    assert источник.name == "flickr"


@pytest.mark.parametrize("окружение", [{}, {"FLICKR_API_KEY": ""}, {"FLICKR_API_KEY": "   "}])
def test_без_ключа_flickr_падает_до_сети(окружение: dict[str, str]) -> None:
    """Транспорт бросил бы AssertionError, если бы до него дошло."""
    with pytest.raises(ОтсутствуетКлюч, match="FLICKR_API_KEY"):
        build_source("flickr", ПустойТранспорт(), окружение, allow_sa=False)


class ЗаписанныйТранспорт:
    """Подставной транспорт: отдаёт заготовленный ответ и записывает параметры запроса.

    Своя копия, а не импорт из `tests/test_harvest_flickr.py`: тесты не берут друг у друга
    внутреннюю оснастку, иначе правка фикстуры чужого файла роняет этот.
    """

    def __init__(self, ответ: HttpResponse) -> None:
        self.ответ = ответ
        self.вызовы: list[dict[str, str]] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.вызовы.append(dict(params))
        return self.ответ


def test_ключ_и_allow_sa_доезжают_до_источника() -> None:
    """Жёсткий `allow_sa=False` вместо флага прогона потерял бы все CC BY-SA молча.

    Проверяется состав запроса, а не приватные атрибуты источника (приём
    `tests/test_harvest_flickr.py:114-119`). Без флага код `5` у API не запрашивается вовсе
    (`flickr.py:58`), и потеря выглядела бы как «Flickr не даёт SA» — то есть как вывод §13 о
    потолке открытых источников, сделанный по нашему собственному дефекту.
    """
    выдача = {"stat": "ok", "photos": {"page": 1, "pages": 1, "photo": []}}
    транспорт = ЗаписанныйТранспорт(HttpResponse(200, json.dumps(выдача).encode("utf-8")))

    источник = build_source("flickr", транспорт, {"FLICKR_API_KEY": "КЛЮЧ-42"}, allow_sa=True)
    источник.search(SourceQuery("chevrolet__lanos", None, "Ланос", 2, None))

    assert транспорт.вызовы[0]["api_key"] == "КЛЮЧ-42"
    assert транспорт.вызовы[0]["license"] == requested_licenses(allow_sa=True)


def _таблица(flickr_terms: tuple[str, ...]) -> dict[str, ClassQueries]:
    """Таблица из одного класса: подмена решается на класс, а не на таблицу целиком."""
    return {
        "chevrolet__lanos": ClassQueries(
            "chevrolet__lanos",
            ("Chevrolet Lanos",),
            ("Chevrolet Lanos",),
            flickr_terms=flickr_terms,
        )
    }


def test_commons_оставляет_таблицу_как_есть() -> None:
    """Подмена — только фликровская ветка (§11.1 спеки): коммонсовский прогон её не видит."""
    итог = source_queries(_таблица(("Ланос",)), "commons")

    assert итог["chevrolet__lanos"].categories == ("Chevrolet Lanos",)
    assert итог["chevrolet__lanos"].terms == ("Chevrolet Lanos",)


def test_flickr_подставляет_свои_строки_и_снимает_категории() -> None:
    """Без подмены Flickr спрашивали бы коммонсовскими `terms` — и по чужому дереву категорий."""
    итог = source_queries(_таблица(("Ланос", "ZAZ Lanos")), "flickr")

    assert итог["chevrolet__lanos"].terms == ("Ланос", "ZAZ Lanos")
    assert итог["chevrolet__lanos"].categories == ()


def test_класс_без_flickr_terms_остаётся_без_запросов() -> None:
    """Пустое поле — заявленное «из Flickr этот класс не добирается», а не потеря."""
    итог = source_queries(_таблица(()), "flickr")

    assert итог["chevrolet__lanos"].terms == ()
    assert итог["chevrolet__lanos"].categories == ()


def test_живой_каталог_даёт_запросы_обоим_классам_задачи() -> None:
    """Единственная автоматическая связь каталога с прогоном: поле заполнено и доезжает."""
    итог = source_queries(load_queries(DEFAULT_QUERIES_PATH), "flickr")

    assert итог["chevrolet__lanos"].terms, "класс задачи остался без фликровских запросов"
    assert итог["lifan__x60"].terms, "класс задачи остался без фликровских запросов"


def test_умолчание_источника_коммонс() -> None:
    """Прогон без флага обязан идти в Commons: §11 открывает внешнего донора только явно.

    Мутация `default="flickr"` — или снятое `default=` вовсе, тогда в фликровскую ветку
    уводит `None != "commons"` — иначе проходит весь набор молча. Цена: обычный
    `--contact X` на машине с экспортированным `FLICKR_API_KEY` ушёл бы во Flickr по всем
    150 классам с обнулёнными категориями и перезаписал бы отчёт нулями.
    """
    базовые = ["--contact", "m@e.com"]

    assert build_parser().parse_args(базовые).source == "commons"
    assert build_parser().parse_args([*базовые, "--source", "flickr"]).source == "flickr"


def test_flickr_без_ключа_даёт_код_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Не дубль теста про `build_source`: тот проверяет функцию, этот — порядок внутри `main`.

    Держит четыре вещи разом: флаг существует, `аргументы.source` доезжает до `build_source`,
    `except` отдаёт код 2 — и отказ стоит выше `import open_clip`. Последнее и есть главное:
    в CI установлена только группа `dev`, а `open_clip` живёт в `train`, поэтому опущенный
    ниже импорта fail-fast уронит этот тест `ModuleNotFoundError`, а не расхождением кода
    возврата. На машине разработчика, где группа `train` стоит, тот же порядок ловится только
    замером времени, то есть руками — здесь его держит тест.

    Фикстура `стенд` не нужна: до `return 2` `main` читает каталог и путь данных, в сеть не
    ходит и на диск не пишет. Ключ снимается из окружения, чтобы тест не зависел от машины.
    """
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)

    assert main(["--contact", "m@e.com", "--source", "flickr"]) == 2
