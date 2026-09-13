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
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pipeline.crop import Detector
from pipeline.harvest.audit import load_applied
from pipeline.harvest.commons import HttpResponse
from pipeline.harvest.detectors import DetectorPin
from pipeline.harvest.flickr import requested_licenses
from pipeline.harvest.manifest import HarvestRow, load_manifest, write_manifest
from pipeline.harvest.queries import DEFAULT_QUERIES_PATH, ClassQueries, load_queries
from pipeline.harvest.redaction_profile import CURRENT_REDACTION_PROFILE, RedactionProfile
from pipeline.harvest.run import CategoryExpander
from pipeline.harvest.sources import HarvestSource, SourceQuery
from scripts.harvest import (
    ОтсутствуетКлюч,
    build_deps,
    build_parser,
    build_source,
    main,
    source_queries,
)


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


def test_вердикт_человека_тоже_попадает_в_журнал(стенд: dict[str, Path]) -> None:
    """§12.1: без записи `drop:plate` человека найденный дефект исчезает на следующем прогоне.

    Журнал вёлся только по каналу дедупа, и строка, выброшенная рукой, не оставляла следа
    нигде: в манифесте её больше нет, очередь `--audit` не усекается, но и не читается
    отчётом. «Найдено/применено» §12.1 держится именно на этой записи.
    """
    write_manifest(стенд["манифест"], [_строка("a.jpg"), _строка("b.jpg")])
    ручной = стенд["манифест"].parent / "verdicts.csv"
    ручной.write_text("file,verdict\na.jpg,drop:plate\n", encoding="utf-8")

    main(["--contact", "mail@example.com", "--report-only", "--audit", str(ручной)])

    журнал = {з.file: (з.requested, з.verdict) for з in load_applied(стенд["журнал"])}
    assert журнал == {"a.jpg": ("drop:plate", "drop:plate")}
    assert [с.file for с in load_manifest(стенд["манифест"])] == ["b.jpg"]


def test_история_грязных_переживает_следующий_прогон(
    стенд: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Следующий прогон строки уже не видит, а «найдено 1, применено 1» обязано остаться."""
    write_manifest(стенд["манифест"], [_строка("a.jpg"), _строка("b.jpg", audit="ok")])
    ручной = стенд["манифест"].parent / "verdicts.csv"
    ручной.write_text("file,verdict\na.jpg,drop:plate\n", encoding="utf-8")
    main(["--contact", "mail@example.com", "--report-only", "--audit", str(ручной)])
    capsys.readouterr()

    main(["--contact", "mail@example.com", "--report-only"])

    assert "найдено 1, применено 1, осталось 0" in capsys.readouterr().out


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
    уводит `None != "commons"` — иначе проходит весь набор молча. Цена одна и та же: обычный
    `--contact X` на машине с экспортированным `FLICKR_API_KEY` ушёл бы во Flickr по всем
    150 классам и перезаписал бы отчёт нулями. Обнуление категорий добавляет к этому только
    первая мутация: при снятом `default=` таблица уходит в прогон как есть (`None != "flickr"`
    в `source_queries`), и категорийные запросы отвергает уже `_параметры` адаптера.
    """
    базовые = ["--contact", "m@e.com"]

    assert build_parser().parse_args(базовые).source == "commons"
    assert build_parser().parse_args([*базовые, "--source", "flickr"]).source == "flickr"


def test_flickr_без_ключа_даёт_код_2(
    стенд: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Не дубль теста про `build_source`: тот проверяет функцию, этот — порядок внутри `main`.

    Держит четыре вещи разом: флаг существует, `аргументы.source` доезжает до `build_source`,
    `except` отдаёт код 2 — и отказ стоит выше `import open_clip`. Последнее и есть главное:
    в CI установлена только группа `dev`, а `open_clip` живёт в `train`, поэтому опущенный
    ниже импорта fail-fast уронит этот тест `ModuleNotFoundError`, а не расхождением кода
    возврата. На машине разработчика, где группа `train` стоит, тот же порядок ловится только
    замером времени, то есть руками — здесь его держит тест.

    Фикстура `стенд` взята не потому, что тест что-то пишет: пока порядок цел, до путей дело
    не доходит вовсе. Она нужна на случай, когда порядок сломан — тогда на машине с группой
    `train` `main` пойдёт дальше и допишет отслеживаемые git `provenance/harvest_manifest.csv`,
    `provenance/NOTICES.md` и `bench/harvest-m0.md`, да ещё и по сети. Сторож не должен портить
    репозиторий в тот момент, когда ловит регрессию. Ключ снимается из окружения, чтобы тест
    не зависел от машины.
    """
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)

    assert main(["--contact", "m@e.com", "--source", "flickr"]) == 2


def _пины() -> dict[str, DetectorPin]:
    def пин(имя: str, формат: str) -> DetectorPin:
        return DetectorPin(
            name=имя,
            format=формат,
            url=f"https://example.invalid/{имя}.{формат}",
            revision="v1",
            license="mit",
            checked="2026-09-07",
            sha256="1" * 64,
            derived_from="",
        )

    return {"plate": пин("plate", "onnx"), "plate_hires": пин("plate_hires", "pt")}


def test_build_deps_берёт_детекторы_номера_из_единой_фабрики(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Боевой прогон обязан собирать номера тем же профилем, что пересборка и замер.

    Собственный блок конструирования здесь и был третьей копией (§3.1 дизайна
    рекалибровки): пока он жив, правка порога в фабрике боевой харвест не догоняет.

    `open_clip` живёт в группе `train`, поэтому подменяется заглушкой в `sys.modules`, а
    текстовая матрица — функцией: доказывается маршрут детекторов, а не работа CLIP.
    """
    заглушка = types.SimpleNamespace(
        create_model_and_transforms=lambda *_, **__: (object(), None, object()),
        get_tokenizer=lambda *_: object(),
    )
    monkeypatch.setitem(sys.modules, "open_clip", cast(types.ModuleType, заглушка))
    пины = _пины()
    monkeypatch.setattr("scripts.harvest.load_detectors_manifest", lambda _: пины)
    monkeypatch.setattr("scripts.harvest.verify_weights", lambda *_: None)
    monkeypatch.setattr("scripts.harvest.build_text_matrix", lambda *_: np.zeros((1, 2)))
    monkeypatch.setattr("scripts.harvest.YoloDetector", lambda *_, **__: object())
    monkeypatch.setattr("scripts.harvest.YuNetFaceDetector", lambda *_, **__: object())
    номера = (cast(Detector, object()), cast(Detector, object()))
    вызовы: list[tuple[RedactionProfile, Path, Mapping[str, DetectorPin]]] = []

    def фабрика(
        profile: RedactionProfile, *, models_dir: Path, pins: Mapping[str, DetectorPin]
    ) -> tuple[Detector, Detector]:
        вызовы.append((profile, models_dir, pins))
        return номера

    monkeypatch.setattr("scripts.harvest.build_plate_detectors", фабрика)

    зависимости = build_deps(
        cast(HarvestSource, object()),
        cast(CategoryExpander, object()),
        root=tmp_path,
        class_order=("kia__rio",),
        latin_names={"kia__rio": "Kia Rio"},
        alias_table={},
        manifest_path=tmp_path / "manifest.csv",
        clip_model="ViT-B-32",
        clip_weights="laion2b_s34b_b79k",
        allow_sa=False,
        checked="2026-09-07",
    )

    assert вызовы == [(CURRENT_REDACTION_PROFILE, tmp_path / "models", пины)]
    assert зависимости.plate_detectors == номера
