# tests/test_harvest_rejects.py
from pathlib import Path

from pipeline.harvest.rejects import Reject, append_reject_row, load_rejects, reason_counts


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
