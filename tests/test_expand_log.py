from pathlib import Path

from pipeline.harvest.expand_log import (
    EXPAND_LOG_COLUMNS,
    ExpandRow,
    append_expand_row,
    expand_totals,
    load_expand_rows,
)


def _строка(**правки: object) -> ExpandRow:
    поля: dict[str, object] = {
        "class_id": "lada__niva_2131",
        "category": "Long-wheelbase Lada Nivas",
        "categories": 1,
        "requests": 1,
        "rejected_stopword": 0,
        "rejected_exclude": 0,
        "rejected_not_own": 3,
        "budget_exhausted": False,
        "checked": "2026-09-01",
    }
    поля.update(правки)
    return ExpandRow(**поля)  # type: ignore[arg-type]


def test_первая_запись_создаёт_файл_с_шапкой(tmp_path: Path) -> None:
    журнал = tmp_path / "expand_log.csv"

    append_expand_row(журнал, _строка())

    строки = журнал.read_text(encoding="utf-8").splitlines()
    assert строки[0] == ",".join(EXPAND_LOG_COLUMNS)
    assert len(строки) == 2


def test_повторная_запись_дописывает_без_повторной_шапки(tmp_path: Path) -> None:
    журнал = tmp_path / "expand_log.csv"

    append_expand_row(журнал, _строка())
    append_expand_row(журнал, _строка(class_id="uaz__patriot"))

    строки = журнал.read_text(encoding="utf-8").splitlines()
    assert len(строки) == 3
    assert строки[0] == ",".join(EXPAND_LOG_COLUMNS)


def test_файл_нулевой_длины_считается_файлом_без_шапки(tmp_path: Path) -> None:
    """Идиома `rejects.append_reject_row`: журнал можно укоротить через `> файл`, и без
    проверки размера первая дописанная строка молча стала бы шапкой."""
    журнал = tmp_path / "expand_log.csv"
    журнал.touch()

    append_expand_row(журнал, _строка())

    строки = журнал.read_text(encoding="utf-8").splitlines()
    assert строки[0] == ",".join(EXPAND_LOG_COLUMNS)
    assert len(строки) == 2


def test_отсутствующий_файл_даёт_пустой_журнал(tmp_path: Path) -> None:
    assert load_expand_rows(tmp_path / "нет.csv") == []


def test_булево_переживает_запись_и_чтение(tmp_path: Path) -> None:
    """Наивный разбор (`bool(строка[...])` — на непустой строке всегда `True`, или
    инвертированный) прошёл бы молча: обе строки нужны, чтобы поймать оба случая."""
    журнал = tmp_path / "expand_log.csv"

    append_expand_row(журнал, _строка(class_id="исчерпан", budget_exhausted=True))
    append_expand_row(журнал, _строка(class_id="не_исчерпан", budget_exhausted=False))

    прочитанное = load_expand_rows(журнал)

    assert [(с.class_id, с.budget_exhausted) for с in прочитанное] == [
        ("исчерпан", True),
        ("не_исчерпан", False),
    ]


def test_журнал_без_колонки_усечения_читается_нулём(tmp_path: Path) -> None:
    """Журналы прогонов до M6 колонки не имеют, а отчёт обязан собираться и на них."""
    журнал = tmp_path / "expand_log.csv"
    старые_колонки = [к for к in EXPAND_LOG_COLUMNS if к != "truncated"]
    журнал.write_text(
        ",".join(старые_колонки) + "\nkia__rio,Kia Rio,2,1,0,0,3,False,2026-09-01\n",
        encoding="utf-8",
    )

    (строка,) = load_expand_rows(журнал)

    assert строка.truncated == 0
    assert (строка.class_id, строка.rejected_not_own) == ("kia__rio", 3)


def test_totals_на_пустом_журнале_отдаёт_все_ключи_нулями() -> None:
    """Контракт тотальный: вызывающему (`scripts/report_c1.py`) не нужен `.get(..., 0)`."""
    assert expand_totals([]) == {
        "requests": 0,
        "rejected_stopword": 0,
        "rejected_exclude": 0,
        "rejected_not_own": 0,
        "budget_exhausted": 0,
        "truncated": 0,
    }


def test_totals_суммирует_счётчики_по_всем_строкам() -> None:
    строки = [
        _строка(
            requests=1, rejected_stopword=0, rejected_exclude=0, rejected_not_own=3, truncated=1
        ),
        _строка(
            requests=2, rejected_stopword=1, rejected_exclude=1, rejected_not_own=0, truncated=1
        ),
    ]

    итоги = expand_totals(строки)

    assert итоги == {
        "requests": 3,
        "rejected_stopword": 1,
        "rejected_exclude": 1,
        "rejected_not_own": 3,
        "budget_exhausted": 0,
        "truncated": 2,
    }
