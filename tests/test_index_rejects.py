import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from pipeline.harvest.cache import cache_path
from pipeline.harvest.commons import HttpResponse
from pipeline.harvest.rejects import Reject
from scripts.index_rejects import (
    IndexRow,
    build_index,
    запросить_sha1,
    карта_кэша,
    отобрать_кандидатов,
)


class Транспорт:
    """Подставной транспорт: отдаёт заготовленные тела, запоминает параметры запросов."""

    def __init__(self, тела: list[bytes]) -> None:
        self.тела = list(тела)
        self.запросы: list[Mapping[str, str]] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.запросы.append(dict(params))
        return HttpResponse(200, self.тела.pop(0))


def _отказ(pageid: str, reason: str = "redaction_failed") -> Reject:
    return Reject(
        class_id="lifan__x60",
        basis="category:Lifan X60",
        source="commons",
        source_item_id=pageid,
        reason=reason,
        checked="2026-09-01",
    )


def test_отбираются_только_уникальные_кандидаты_нужной_причины() -> None:
    журнал = [_отказ("1"), _отказ("1"), _отказ("2"), _отказ("3", "multi_car")]

    отобранные = отобрать_кандидатов(журнал, "redaction_failed")

    assert [отказ.source_item_id for отказ in отобранные] == ["1", "2"]


def test_карта_кэша_ключуется_sha1_содержимого(tmp_path: Path) -> None:
    данные = b"\xff\xd8jpeg-ish"
    sha256 = hashlib.sha256(данные).hexdigest()
    путь = cache_path(tmp_path, "commons", sha256, "jpg")
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_bytes(данные)

    карта = карта_кэша(tmp_path, "commons")

    assert карта == {hashlib.sha1(данные).hexdigest(): sha256}


def test_sha1_спрашивается_батчами_по_пятьдесят() -> None:
    страницы = {str(номер): {"imageinfo": [{"sha1": f"s{номер}"}]} for номер in range(60)}
    первый = json.dumps({"query": {"pages": dict(list(страницы.items())[:50])}})
    второй = json.dumps({"query": {"pages": dict(list(страницы.items())[50:])}})
    транспорт = Транспорт([первый.encode(), второй.encode()])

    карта = запросить_sha1(транспорт, [str(номер) for номер in range(60)], sleep=lambda _: None)

    assert len(транспорт.запросы) == 2
    assert транспорт.запросы[0]["pageids"].count("|") == 49
    assert карта["59"] == "s59"


def test_кандидат_без_оригинала_в_кэше_попадает_в_карту_с_пустым_sha256() -> None:
    строки = build_index([_отказ("1"), _отказ("2")], {"aa": "bb"}, {"1": "aa"})

    assert строки == [
        IndexRow("lifan__x60", "1", "bb"),
        IndexRow("lifan__x60", "2", ""),
    ]
