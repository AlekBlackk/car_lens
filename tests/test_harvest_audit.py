# tests/test_harvest_audit.py
from pathlib import Path

import pytest

from pipeline.harvest.audit import (
    APPLIED_COLUMNS,
    AppliedVerdict,
    AuditError,
    append_applied,
    apply_audit,
    audit_counts,
    audit_gap,
    counts_with_verdicts,
    demote_over_human,
    describe_applied,
    load_applied,
    load_audit,
    load_audit_events,
    merge_verdicts,
    required_audit_files,
    suppress_applied,
)
from pipeline.harvest.manifest import HarvestRow


def _строка(
    file: str,
    *,
    plates: int = 0,
    faces: int = 0,
    class_id: str = "kia__rio",
    audit: str = "",
) -> HarvestRow:
    return HarvestRow(
        file=file,
        class_id=class_id,
        source="commons",
        source_item_id="1",
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="A",
        license="CC0",
        license_tier=1,
        checked="2026-08-29",
        basis="category:X",
        sha256=file.split(".")[0].rjust(64, "0"),
        phash=1,
        clip_rank=1,
        clip_margin=0.0,
        plates=plates,
        faces=faces,
        bbox_x=12.0,
        bbox_y=34.0,
        bbox_w=500.0,
        bbox_h=300.0,
        bbox_area_frac=0.42,
        det_conf=0.91,
        audit=audit,
    )


def _применённый(
    file: str,
    *,
    class_id: str = "kia__rio",
    basis: str = "category:X",
    verdict: str = "conflict",
    applied: str = "2026-09-02",
    requested: str = "",
) -> AppliedVerdict:
    return AppliedVerdict(
        file=file,
        class_id=class_id,
        basis=basis,
        verdict=verdict,
        applied=applied,
        requested=requested,
    )


def test_вердикты_читаются(tmp_path: Path) -> None:
    p = tmp_path / "verdicts.csv"
    p.write_text("file,verdict\na.jpg,ok\nb.jpg,drop:plate\nc.jpg,conflict\n", encoding="utf-8")
    assert load_audit(p) == {"a.jpg": "ok", "b.jpg": "drop:plate", "c.jpg": "conflict"}


def test_неизвестный_вердикт_отвергается(tmp_path: Path) -> None:
    p = tmp_path / "verdicts.csv"
    p.write_text("file,verdict\na.jpg,maybe\n", encoding="utf-8")
    with pytest.raises(AuditError, match="maybe"):
        load_audit(p)


def test_события_читаются_построчно_а_словарь_схлопывает(tmp_path: Path) -> None:
    """У файла две роли: очередь (итог по файлу) и след прогона (число событий, §9)."""
    p = tmp_path / "verdicts.csv"
    p.write_text(
        "file,verdict\na.jpg,conflict\na.jpg,drop:superseded\nb.jpg,conflict\n", encoding="utf-8"
    )

    assert load_audit_events(p) == [
        ("a.jpg", "conflict"),
        ("a.jpg", "drop:superseded"),
        ("b.jpg", "conflict"),
    ]
    assert load_audit(p) == {"a.jpg": "drop:superseded", "b.jpg": "conflict"}


def test_apply_проставляет_отметку_и_выкидывает_dirty() -> None:
    строки = [_строка("a.jpg"), _строка("b.jpg", plates=1), _строка("c.jpg")]
    итог = apply_audit(строки, {"a.jpg": "ok", "b.jpg": "drop:plate"})
    файлы = {с.file: с.audit for с in итог}
    assert "b.jpg" not in файлы, "кадр с читаемым номером удаляется из манифеста"
    assert файлы["a.jpg"] == "ok"
    assert файлы["c.jpg"] == ""


def test_счётчики_h3_считают_dirty_до_удаления() -> None:
    строки = [
        _строка("a.jpg", audit="ok"),
        _строка("b.jpg", audit="drop:plate"),
        _строка("c.jpg", audit="drop:face"),
        _строка("d.jpg", audit="drop:wrong_model"),
        _строка("e.jpg", audit=""),
    ]
    total, dirty = audit_counts(строки)
    assert total == 4
    assert dirty == 2, "drop:wrong_model — не нарушение приватности, в dirty не идёт"


def test_спор_не_считается_проверенным() -> None:
    """`conflict` — «человек посмотрел и не решил»: открытая задача не идёт в счёт
    закрытого охвата H3, в отличие от решённого `drop:*` рядом."""
    строки = [
        _строка("a.jpg", audit="ok"),
        _строка("b.jpg", audit="conflict"),
        _строка("c.jpg", audit="drop:wrong_model"),
        _строка("d.jpg", audit=""),
    ]

    assert audit_counts(строки) == (2, 0)


def test_числа_аудита_считаются_и_без_новых_вердиктов() -> None:
    """Отметки живут в манифесте постоянно; пустой файл вердиктов — штатное состояние
    после усечения очереди, и оно не повод отчитаться нулём."""
    строки = [_строка("a.jpg", audit="ok"), _строка("b.jpg", audit="drop:plate")]

    assert counts_with_verdicts(строки, {}) == (2, 1)


def test_новые_вердикты_накладываются_поверх_манифеста() -> None:
    строки = [_строка("a.jpg", audit=""), _строка("b.jpg", audit="ok")]

    assert counts_with_verdicts(строки, {"a.jpg": "drop:face"}) == (2, 1)


def test_грязные_считаются_до_удаления() -> None:
    """`apply_audit` выкидывает `drop:*`, поэтому считать надо раньше него — иначе
    `audited_dirty` всегда ноль, и гейт H3 не сработает никогда."""
    строки = [_строка("a.jpg", audit="")]

    всего, грязных = counts_with_verdicts(строки, {"a.jpg": "drop:plate"})

    assert (всего, грязных) == (1, 1)


def test_спор_новым_вердиктом_отменяет_прежнюю_проверку() -> None:
    """Человек пересмотрел `ok` на `conflict` — счёт обязан это увидеть через наложение,
    а не остаться при прежней отметке манифеста."""
    строки = [_строка("a.jpg", audit="ok")]

    assert counts_with_verdicts(строки, {"a.jpg": "conflict"}) == (0, 0)


def test_обязательный_охват_аудита() -> None:
    строки = (
        [_строка(f"p{n}.jpg", plates=1) for n in range(5)]
        + [_строка(f"f{n}.jpg", faces=2) for n in range(5)]
        + [_строка(f"core{n}.jpg", class_id="lada__granta") for n in range(3)]
        + [_строка(f"clean{n}.jpg") for n in range(500)]
    )
    надо = required_audit_files(строки, seed=0)
    assert all(f"p{n}.jpg" in надо for n in range(5))
    assert all(f"f{n}.jpg" in надо for n in range(5))
    assert all(f"core{n}.jpg" in надо for n in range(3))
    assert sum(1 for f in надо if f.startswith("clean")) == 200


def test_пустой_аудит_даёт_полную_недостачу() -> None:
    """Ноль вердиктов — это непроверенный протокол, а не чистая выборка."""
    строки = [_строка(f"p{n}.jpg", plates=1) for n in range(3)]
    пропущено = audit_gap(строки, seed=0)
    assert пропущено == {"p0.jpg", "p1.jpg", "p2.jpg"}


def test_частичный_аудит_называет_непроверенные_кадры() -> None:
    строки = [_строка(f"p{n}.jpg", plates=1, audit="ok" if n < 2 else "") for n in range(4)]
    assert audit_gap(строки, seed=0) == {"p2.jpg", "p3.jpg"}


def test_спор_остаётся_в_недостаче_охвата() -> None:
    """Спор закрытым протоколом не считается: кадр покидает недостачу только решённым
    вердиктом, иначе отметка «не решил» тихо снимала бы блокировку H3."""
    строки = [
        _строка("p0.jpg", plates=1, audit="ok"),
        _строка("p1.jpg", plates=1, audit="conflict"),
        _строка("p2.jpg", plates=1, audit=""),
    ]

    assert audit_gap(строки, seed=0) == {"p1.jpg", "p2.jpg"}


def test_сплошной_аудит_недостачи_не_оставляет() -> None:
    строки = [_строка(f"p{n}.jpg", plates=1, audit="ok") for n in range(3)]
    assert audit_gap(строки, seed=0) == set()


def test_недостача_не_считает_лишние_кадры_вне_протокола() -> None:
    """Проверенный сверх протокола кадр недостачу не создаёт."""
    строки = [_строка("p0.jpg", plates=1, audit="ok")] + [
        _строка(f"clean{n}.jpg", audit="ok") for n in range(3)
    ]
    assert audit_gap(строки, seed=0) == set()


def test_поздний_вердикт_побеждает_ранний() -> None:
    итог = merge_verdicts({"a.jpg": "ok"}, {"a.jpg": "drop:plate", "b.jpg": "ok"})
    assert итог == {"a.jpg": "drop:plate", "b.jpg": "ok"}


def test_склейка_без_аргументов_пуста() -> None:
    assert merge_verdicts() == {}


def test_автовердикт_не_сносит_проверенную_человеком_строку() -> None:
    """Первое живое `supersede` снесло строку с `audit=ok`: автомат перебил человека."""
    строки = [_строка("a.jpg", audit="ok"), _строка("b.jpg", audit="")]

    итог = demote_over_human({"a.jpg": "drop:superseded", "b.jpg": "drop:superseded"}, строки)

    assert итог["a.jpg"] == "conflict", "спор с человеком уходит человеку, а не в удаление"
    assert итог["b.jpg"] == "drop:superseded", "строку без отметки человека дедуп решает сам"


def test_понижение_не_трогает_conflict_и_ok() -> None:
    строки = [_строка("a.jpg", audit="ok"), _строка("b.jpg", audit="ok")]

    итог = demote_over_human({"a.jpg": "conflict", "b.jpg": "ok"}, строки)

    assert итог == {"a.jpg": "conflict", "b.jpg": "ok"}


def test_понижение_считает_проверенной_только_отметку_ok() -> None:
    """`conflict` — это «человек не уверен», такая строка автомат не связывает."""
    строки = [_строка("a.jpg", audit="conflict")]

    assert demote_over_human({"a.jpg": "drop:superseded"}, строки) == {"a.jpg": "drop:superseded"}


def test_журнал_применённых_переживает_запись_и_чтение(tmp_path: Path) -> None:
    строки = [_строка("a.jpg", class_id="mitsubishi__pajero")]
    решения = {"a.jpg": "drop:superseded"}
    записи = describe_applied(решения, решения, строки, "2026-09-01")
    путь = tmp_path / "dedup_verdicts.csv"

    append_applied(путь, записи)
    append_applied(
        путь, describe_applied({"a.jpg": "conflict"}, {"a.jpg": "conflict"}, строки, "2026-09-02")
    )

    прочитано = load_applied(путь)
    assert [(з.file, з.verdict, з.applied) for з in прочитано] == [
        ("a.jpg", "drop:superseded", "2026-09-01"),
        ("a.jpg", "conflict", "2026-09-02"),
    ]
    assert прочитано[0].class_id == "mitsubishi__pajero"
    assert прочитано[0].basis == "category:X"


def test_журнал_помнит_чего_хотел_дедуп_а_не_только_применённое() -> None:
    """Понижение по человеку не должно стирать из следа сам факт вытеснения.

    Иначе счёт `supersede` §9 недосчитает: строка с `ok`, которую дедуп хотел вытеснить,
    попадёт в журнал как `conflict`, и «дедуп хотел вытеснить» из юридического следа исчезнет.
    """
    строки = [_строка(file="a.jpg", audit="ok")]
    решения = {"a.jpg": "drop:superseded"}

    применённые = demote_over_human(решения, строки)
    журнал = describe_applied(решения, применённые, строки, "2026-09-02")

    assert журнал[0].requested == "drop:superseded"
    assert журнал[0].verdict == "conflict"


def test_журнал_не_записывает_вердикт_на_отсутствующую_строку() -> None:
    """Вердикт на файл вне манифеста ничего не применил — врать о применении нечем."""
    решения = {"нет.jpg": "drop:superseded"}
    assert describe_applied(решения, решения, [_строка("a.jpg")], "2026-09-01") == []


def test_отсутствие_журнала_читается_пустым(tmp_path: Path) -> None:
    assert load_applied(tmp_path / "нет.csv") == []


def test_укороченный_журнал_получает_шапку_заново(tmp_path: Path) -> None:
    путь = tmp_path / "dedup_verdicts.csv"
    решения = {"a.jpg": "conflict"}
    записи = describe_applied(решения, решения, [_строка("a.jpg")], "2026-09-01")
    append_applied(путь, записи)

    путь.write_text("", encoding="utf-8")
    append_applied(путь, записи)

    assert load_applied(путь) == записи


def test_append_applied_не_плодит_дубли_по_файлу_и_вердикту(tmp_path: Path) -> None:
    """Повторная запись той же пары «файл + вердикт» не должна удвоить строку журнала."""
    путь = tmp_path / "dedup_verdicts.csv"
    решения = {"a.jpg": "drop:superseded"}
    записи = describe_applied(решения, решения, [_строка("a.jpg")], "2026-09-01")
    append_applied(путь, записи)

    append_applied(путь, записи)

    assert load_applied(путь) == записи


def test_уже_применённый_вердикт_не_применяется_повторно() -> None:
    """Заряженная очередь не должна переписывать отметки человека на следующем прогоне."""
    журнал = [_применённый(file="a.jpg", verdict="conflict")]

    остаток = suppress_applied({"a.jpg": "conflict", "b.jpg": "conflict"}, журнал)

    assert остаток == {"b.jpg": "conflict"}


def test_тот_же_файл_с_другим_вердиктом_применяется() -> None:
    """Совпадение по имени файла — не повод молчать: вердикт мог смениться."""
    журнал = [_применённый(file="a.jpg", verdict="conflict")]

    assert suppress_applied({"a.jpg": "drop:superseded"}, журнал) == {"a.jpg": "drop:superseded"}


def test_подавление_сверяется_с_сырым_запросом_старой_схемы() -> None:
    """Живой случай C1b: журнал хранит непонижённый вердикт как `verdict` (пустой `requested`).

    Строка записана до появления `demote_over_human` — тогда понижения не было вообще, и
    колонка `verdict` несёт исходный запрос дедупа как есть. `suppress_applied` вызывается
    ДО понижения (тем же сырым вердиктом из очереди), поэтому сверка обязана найти
    совпадение по `verdict` журнала напрямую, а не ждать `requested`, которого в старой
    записи нет.
    """
    журнал = [_применённый(file="a.jpg", verdict="drop:superseded", requested="")]

    assert suppress_applied({"a.jpg": "drop:superseded"}, журнал) == {}


def test_append_applied_переписывает_пятиколоночную_шапку(tmp_path: Path) -> None:
    """Реальный `provenance/dedup_verdicts.csv` лежит в git пятиколоночным (без `requested`).

    Дописывание новой строки не имеет права ни уронить старые значения, ни размазать новые
    шесть полей под старую пятиколоночную шапку. Проверяется на настоящем формате файла —
    самая дешёвая находка Critical 2, спрятанная тем, что все прочие тесты пишут в чистый
    `tmp_path` и никогда не встречают старую шапку.
    """
    путь = tmp_path / "dedup_verdicts.csv"
    путь.write_text(
        "file,class_id,basis,verdict,applied\n"
        "old.jpg,audi__q5,category:Audi Q5,conflict,2026-09-01\n",
        encoding="utf-8",
    )

    новая = _применённый(
        file="new.jpg",
        class_id="kia__rio",
        basis="category:X",
        verdict="drop:superseded",
        requested="drop:superseded",
    )
    append_applied(путь, [новая])

    прочитано = load_applied(путь)
    assert прочитано[0] == _применённый(
        file="old.jpg",
        class_id="audi__q5",
        basis="category:Audi Q5",
        verdict="conflict",
        applied="2026-09-01",
        requested="",
    )
    assert прочитано[1] == новая
    assert путь.read_text(encoding="utf-8").splitlines()[0] == ",".join(APPLIED_COLUMNS)
