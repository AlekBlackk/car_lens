"""Тесты генератора отчёта о перекосе источника (scripts/build_skew_report)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.class_purity import LABEL_CAN_THIN, LABEL_COVERAGE_HOLE, read_drop_list
from scripts.build_skew_report import (
    DEFAULT_SKEW_LIST,
    END_MARKER,
    START_MARKER,
    ClassRow,
    build_rows,
    fill_table,
    main,
    render_table,
)
from tests.test_build_pool_clean import _make_drop_csv, _make_synthetic_pool

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pool() -> pd.DataFrame:
    # honda__cr_v: 5 кадров, 4 в перекосе → останется 1 (дыра при пороге 2)
    # kia__rio: 4 кадра, 1 в перекосе → останется 3 (можно проредить при пороге 2)
    return pd.DataFrame(
        {
            "photo_id": ["c1", "c2", "c3", "c4", "c5", "r1", "r2", "r3", "r4"],
            "model_id": [
                "honda__cr_v",
                "honda__cr_v",
                "honda__cr_v",
                "honda__cr_v",
                "honda__cr_v",
                "kia__rio",
                "kia__rio",
                "kia__rio",
                "kia__rio",
            ],
        }
    )


def test_build_rows_считает_всего_перекос_остаток_и_ярлык() -> None:
    skew_ids = ["c1", "c2", "c3", "c4", "r1"]
    rows = build_rows(_pool(), skew_ids, min_street=2)

    by_model = {r.model_id: r for r in rows}
    assert by_model["honda__cr_v"] == ClassRow(
        model_id="honda__cr_v",
        total=5,
        skew=4,
        remaining=1,
        label=LABEL_COVERAGE_HOLE,
        borderline=True,
    )
    assert by_model["kia__rio"] == ClassRow(
        model_id="kia__rio",
        total=4,
        skew=1,
        remaining=3,
        label=LABEL_CAN_THIN,
        borderline=True,
    )


def test_build_rows_сортирует_по_убыванию_перекоса() -> None:
    skew_ids = ["c1", "c2", "c3", "c4", "r1"]
    rows = build_rows(_pool(), skew_ids, min_street=2)
    assert [r.model_id for r in rows] == ["honda__cr_v", "kia__rio"]


def test_render_table_содержит_шапку_и_строку() -> None:
    rows = [
        ClassRow(
            "honda__cr_v", total=5, skew=4, remaining=1, label=LABEL_COVERAGE_HOLE, borderline=True
        ),
        ClassRow(
            "kia__rio", total=30, skew=2, remaining=28, label=LABEL_CAN_THIN, borderline=False
        ),
    ]
    table = render_table(rows)
    assert "| класс | всего | перекос | останется | ярлык | у порога |" in table
    assert "| honda__cr_v | 5 | 4 | 1 | дыра_покрытия | да |" in table
    assert "| kia__rio | 30 | 2 | 28 | можно_проредить |  |" in table


def test_fill_table_заменяет_только_между_маркерами() -> None:
    text = (
        "# Отчёт\n\nРучной раздел выше.\n\n"
        f"{START_MARKER}\nстарая таблица\n{END_MARKER}\n\nРучной раздел ниже.\n"
    )
    result = fill_table(text, "НОВАЯ ТАБЛИЦА")

    assert "Ручной раздел выше." in result
    assert "Ручной раздел ниже." in result
    assert "старая таблица" not in result
    assert "НОВАЯ ТАБЛИЦА" in result
    # Маркеры остаются на месте для следующего запуска
    assert START_MARKER in result
    assert END_MARKER in result


def test_fill_table_без_маркеров_ошибка() -> None:
    with pytest.raises(ValueError, match=r"(?i)маркер"):
        fill_table("# Отчёт без маркеров\n", "таблица")


def test_build_rows_помечает_классы_у_порога() -> None:
    # 15 кадров при пороге 10: остаток 13 и 7 — дальше двух кадров от порога, 12 и 8 — в пределах
    pool = pd.DataFrame({"photo_id": [f"p{i}" for i in range(15)], "model_id": ["m"] * 15})
    for n_skew, expected in [(2, False), (3, True), (7, True), (8, False)]:
        (row,) = build_rows(pool, [f"p{i}" for i in range(n_skew)], min_street=10)
        assert row.borderline is expected, f"остаток {row.remaining}"


def test_main_вписывает_таблицу_и_не_трогает_рукописное(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool_path = tmp_path / "pool.parquet"
    skew_path = tmp_path / "skew.csv"
    report_path = tmp_path / "report.md"
    _make_synthetic_pool(pool_path, [("a1", "audi"), ("a2", "audi"), ("a3", "audi"), ("b1", "bmw")])
    _make_drop_csv(skew_path, [("a1", "audi", 2), ("a2", "audi", 2), ("x9", "audi", 2)])
    report_path.write_text(
        f"Рукописное.\n{START_MARKER}\nстарое\n{END_MARKER}\nХвост.\n", encoding="utf-8"
    )

    ret = main(
        [
            "--split",
            str(pool_path),
            "--skew-list",
            str(skew_path),
            "--min-street",
            "1",
            "--out",
            str(report_path),
        ]
    )

    assert ret == 0
    text = report_path.read_text(encoding="utf-8")
    assert text.startswith("Рукописное.\n")
    assert text.endswith("Хвост.\n")
    assert "| audi | 3 | 2 | 1 | можно_проредить | да |" in text
    assert "bmw" not in text
    # Кадр, которого нет в пуле, не считается, но о нём сказано
    assert "нет в пуле (в счёт не идут): 1" in capsys.readouterr().out


def test_реальный_список_перекоса_читается() -> None:
    real_csv = REPO_ROOT / DEFAULT_SKEW_LIST
    if not real_csv.exists():
        pytest.skip("нет bench/source-skew-candidates.csv")
    rows = read_drop_list(real_csv)
    assert rows
    assert {r.category for r in rows} == {"source_skew"}
    assert {r.tier for r in rows} == {2}
