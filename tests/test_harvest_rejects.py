# tests/test_harvest_rejects.py
from pathlib import Path

from pipeline.harvest.rejects import (
    Reject,
    append_reject_row,
    load_rejects,
    reason_counts,
    unique_by_reason,
)


def _отказ(pageid: str, reason: str) -> Reject:
    return Reject(
        class_id="lifan__x60",
        basis="category:Lifan X60",
        source="commons",
        source_item_id=pageid,
        reason=reason,
        checked="2026-09-01",
    )


def test_журнал_дописывается_с_шапкой(tmp_path: Path) -> None:
    путь = tmp_path / "reject_log.csv"

    append_reject_row(
        путь,
        Reject(
            class_id="kia__rio",
            basis="category:Kia Rio",
            source="commons",
            source_item_id="1",
            reason="http",
            checked="2026-08-31",
        ),
    )
    append_reject_row(
        путь,
        Reject(
            class_id="kia__rio",
            basis="search:Kia Rio",
            source="commons",
            source_item_id="2",
            reason="too_small_file",
            checked="2026-08-31",
        ),
    )

    строки = load_rejects(путь)
    assert [с.reason for с in строки] == ["http", "too_small_file"]
    assert путь.read_text(encoding="utf-8").splitlines()[0].startswith("class_id,")


def test_счётчик_причин_складывает_прогоны(tmp_path: Path) -> None:
    путь = tmp_path / "reject_log.csv"
    for причина in ("http", "http", "not_a_car_photo"):
        append_reject_row(
            путь,
            Reject(
                class_id="kia__rio",
                basis="search:x",
                source="commons",
                source_item_id=причина,
                reason=причина,
                checked="2026-08-31",
            ),
        )

    assert reason_counts(load_rejects(путь)) == {"http": 2, "not_a_car_photo": 1}


def test_запись_в_обрезанный_журнал_не_теряет_строку(tmp_path: Path) -> None:
    """Ротации у журнала нет, и укорачивают его `> reject_log.csv`: файл есть, шапки нет.

    Без проверки размера первая же дописанная строка стала бы шапкой, а `load_rejects` вернул
    бы пустой список без единой ошибки — ровно та непродиагностируемая пустота, против которой
    журнал и заведён.
    """
    путь = tmp_path / "reject_log.csv"
    путь.touch()

    append_reject_row(
        путь,
        Reject(
            class_id="kia__rio",
            basis="category:Kia Rio",
            source="commons",
            source_item_id="1",
            reason="http",
            checked="2026-08-31",
        ),
    )

    assert [с.reason for с in load_rejects(путь)] == ["http"]
    assert путь.read_text(encoding="utf-8").splitlines()[0].startswith("class_id,")


def test_отсутствующий_журнал_читается_пустым(tmp_path: Path) -> None:
    """До первого отказа файла нет, и это не ошибка чтения."""
    assert load_rejects(tmp_path / "reject_log.csv") == []


def test_unique_by_reason_снимает_повторы_и_держит_порядок_первой_встречи() -> None:
    журнал = [
        _отказ("111", "redaction_failed"),
        _отказ("222", "multi_car"),
        _отказ("111", "redaction_failed"),
        _отказ("333", "redaction_failed"),
    ]
    отобранные = unique_by_reason(журнал, "redaction_failed")
    assert [отказ.source_item_id for отказ in отобранные] == ["111", "333"]
