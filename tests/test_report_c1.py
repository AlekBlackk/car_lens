from pathlib import Path

import pytest

from pipeline.harvest.audit import AppliedVerdict, append_applied
from pipeline.harvest.expand_log import ExpandRow, append_expand_row
from pipeline.harvest.manifest import HarvestRow, write_manifest
from pipeline.harvest.rejects import Reject, append_reject_row
from scripts.report_c1 import (
    G_C1_MAX_MISSING,
    C1Numbers,
    C2ScopeRow,
    check_c1,
    collect,
    main,
    render,
)


def _строка(**правки: object) -> HarvestRow:
    поля: dict[str, object] = {
        "file": "ab.jpg",
        "class_id": "lada__granta",
        "source": "commons",
        "source_item_id": "12345",
        "page_url": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "file_url": "https://upload.wikimedia.org/X.jpg",
        "author": "Иван Петров",
        "license": "CC BY 4.0",
        "license_tier": 2,
        "checked": "2026-08-28",
        "basis": "category:Lada Granta",
        "sha256": "ab" * 32,
        "phash": -12345,
        "clip_rank": 4,
        "clip_margin": 0.12,
        "plates": 1,
        "faces": 0,
        "bbox_x": 12.0,
        "bbox_y": 34.0,
        "bbox_w": 500.0,
        "bbox_h": 300.0,
        "bbox_area_frac": 0.42,
        "det_conf": 0.91,
        "audit": "",
    }
    поля.update(правки)
    return HarvestRow(**поля)  # type: ignore[arg-type]


def _отказ(**правки: object) -> Reject:
    поля: dict[str, object] = {
        "class_id": "kia__rio",
        "basis": "category:Kia Rio",
        "source": "commons",
        "source_item_id": "1",
        "reason": "over_redacted",
        "checked": "2026-08-31",
    }
    поля.update(правки)
    return Reject(**поля)  # type: ignore[arg-type]


def _обход(**правки: object) -> ExpandRow:
    поля: dict[str, object] = {
        "class_id": "lada__niva_2131",
        "category": "Long-wheelbase Lada Nivas",
        "categories": 1,
        "requests": 1,
        "rejected_stopword": 0,
        "rejected_exclude": 0,
        "rejected_not_own": 3,
        "budget_exhausted": False,
        "truncated": 0,
        "checked": "2026-09-01",
    }
    поля.update(правки)
    return ExpandRow(**поля)  # type: ignore[arg-type]


def _числа(**правки: object) -> C1Numbers:
    поля: dict[str, object] = {
        "rows": 10,
        "category_rows": 4,
        "category_share": 0.4,
        "classes_with_category": 3,
        "classes_without_holdout": 6,
        "catalog_classes": 150,
        "conflicts": 0,
        "supersedes": 0,
        "reasons": {},
        "expand_requests": 0,
        "expand_rejected_stopword": 0,
        "expand_rejected_exclude": 0,
        "expand_rejected_not_own": 0,
        "expand_budget_exhausted": 0,
        "expand_truncated": 0,
        "c2_scope": (),
    }
    поля.update(правки)
    return C1Numbers(**поля)  # type: ignore[arg-type]


def _каталог(tmp_path: Path, классов: int) -> Path:
    """Крошечный каталог: `lada__granta` из манифеста плюс добивка до нужного числа.

    Число классов каталога — знаменатель гейта, поэтому размер каталога задаёт вердикт:
    класс без строк манифеста дефицитен по определению.
    """
    имена = ["granta", *(f"m{номер}" for номер in range(1, классов))]
    модели = "".join(f"      {имя}:\n        display: Lada {имя}\n" for имя in имена)
    путь = tmp_path / "canonical.yaml"
    путь.write_text(f"version: 1\nbrands:\n  Lada:\n    models:\n{модели}", encoding="utf-8")
    return путь


def _подготовить(tmp_path: Path, классов_в_каталоге: int = 8) -> tuple[list[str], Path]:
    """Манифест, журнал отказов и вердикты дедупа в `tmp_path`; флаги путей и путь отчёта."""
    манифест = tmp_path / "harvest_manifest.csv"
    write_manifest(
        манифест,
        [
            _строка(basis="category:Kia Rio"),
            _строка(file="cd.jpg", sha256="cd" * 32, basis="search:Kia Rio"),
        ],
    )
    журнал = tmp_path / "reject_log.csv"
    append_reject_row(журнал, _отказ(reason="over_redacted"))
    append_reject_row(журнал, _отказ(reason="not_a_car_photo", source_item_id="2"))
    append_reject_row(журнал, _отказ(reason="not_a_car_photo", source_item_id="3"))
    вердикты = tmp_path / "verdicts.dedup.csv"
    вердикты.write_text(
        "file,verdict\na.jpg,conflict\nb.jpg,drop:superseded\nc.jpg,drop:superseded\n",
        encoding="utf-8",
    )
    журнал_обхода = tmp_path / "expand_log.csv"
    append_expand_row(журнал_обхода, _обход())
    append_expand_row(журнал_обхода, _обход(class_id="kia__rio", rejected_not_own=0))
    отчёт = tmp_path / "harvest-c1.md"
    флаги = [
        "--report",
        str(отчёт),
        "--manifest",
        str(манифест),
        "--catalog",
        str(_каталог(tmp_path, классов_в_каталоге)),
        "--reject-log",
        str(журнал),
        "--dedup-verdicts",
        str(вердикты),
        "--applied-verdicts",
        str(tmp_path / "dedup_verdicts.csv"),
        "--expand-log",
        str(журнал_обхода),
    ]
    return флаги, отчёт


def test_гейт_блокирует_по_классам_без_эталона() -> None:
    числа = collect(
        rows=[], rejects=[], verdict_events=(), catalog_classes=(), classes_without_holdout=16
    )

    прошёл, провалы = check_c1(числа)

    assert прошёл is False
    assert провалы == [f"G_C1: классов без эталона — 16 > {G_C1_MAX_MISSING}"]


def test_гейт_проходит_на_шести() -> None:
    числа = collect(
        rows=[], rejects=[], verdict_events=(), catalog_classes=(), classes_without_holdout=6
    )

    assert check_c1(числа) == (True, [])


def test_класс_без_строк_манифеста_считается_дефицитным() -> None:
    """Знаменатель гейта — каталог: класс, потерявший провенанс целиком, не исчезает.

    `select_holdout` группирует строки манифеста, поэтому в `short_classes` такой класс не
    попадает вовсе, и до этой правки дефицит занижался ровно в самом тяжёлом случае (на
    BASE 15 против настоящих 16).
    """
    числа = collect(
        rows=[_строка()],
        rejects=[],
        verdict_events=(),
        catalog_classes=("lada__granta", "chevrolet__lanos"),
    )

    assert числа.classes_without_holdout == 2
    assert [к.class_id for к in числа.c2_scope] == ["chevrolet__lanos", "lada__granta"]
    без_строк = next(к for к in числа.c2_scope if к.class_id == "chevrolet__lanos")
    assert (без_строк.accepted, без_строк.category_rows, без_строк.category_share) == (0, 0, 0.0)


def test_скоуп_c2_считает_принятое_и_долю_категорий() -> None:
    """Таблица скоупа C2 и вердикт гейта — одна величина: `len(c2_scope)`."""
    строки = [
        _строка(basis="category:Lada Granta"),
        _строка(file="cd.jpg", sha256="cd" * 32, basis="search:Lada Granta"),
        _строка(file="ef.jpg", sha256="ef" * 32, class_id="kia__rio", basis="search:Kia Rio"),
    ]

    числа = collect(rows=строки, rejects=[], verdict_events=(), catalog_classes=("lada__granta",))

    assert числа.classes_without_holdout == len(числа.c2_scope) == 1
    (гранта,) = числа.c2_scope
    assert (гранта.accepted, гранта.category_rows) == (2, 1)
    assert гранта.category_share == 0.5


def test_вердикты_считаются_событиями_а_не_файлами() -> None:
    """§9 просит поштучно: два вытеснения одного файла — два события, а не одно."""
    числа = collect(
        rows=[],
        rejects=[],
        verdict_events=["drop:superseded", "drop:superseded", "conflict"],
        catalog_classes=(),
        classes_without_holdout=0,
    )

    assert (числа.supersedes, числа.conflicts) == (2, 1)


def test_доля_категорийного_базиса_считается_но_не_блокирует() -> None:
    строки = [_строка(basis="category:Kia Rio"), _строка(basis="search:Kia Rio")]

    числа = collect(
        rows=строки, rejects=[], verdict_events=(), catalog_classes=(), classes_without_holdout=6
    )

    assert числа.category_rows == 1
    assert числа.category_share == 0.5
    assert check_c1(числа)[0] is True


def test_отчёт_на_красном_гейте_печатает_провал_и_вердикт() -> None:
    числа = _числа(classes_without_holdout=16)
    прошёл, провалы = check_c1(числа)

    текст = render(числа, прошёл, провалы, "abc1234", "2026-08-31")

    assert "- вердикт гейта: НЕ ПРОЙДЕН" in текст
    assert f"- G_C1: классов без эталона — 16 > {G_C1_MAX_MISSING}" in текст


def test_отчёт_на_зелёном_гейте_печатает_нет_вместо_провалов() -> None:
    числа = _числа(classes_without_holdout=6)
    прошёл, провалы = check_c1(числа)

    текст = render(числа, прошёл, провалы, "abc1234", "2026-08-31")

    assert "## Блокирующие провалы\n\nНет.\n" in текст
    assert "- вердикт гейта: пройден" in текст
    assert "НЕ ПРОЙДЕН" not in текст


def test_причины_отказа_идут_по_убыванию_числа_кандидатов() -> None:
    отказы = [
        _отказ(reason="alias_mismatch"),
        *(_отказ(reason="not_a_car_photo") for _ in range(3)),
        *(_отказ(reason="over_redacted") for _ in range(2)),
    ]
    числа = collect(
        rows=[], rejects=отказы, verdict_events=(), catalog_classes=(), classes_without_holdout=6
    )

    текст = render(числа, True, [], "abc1234", "2026-08-31")

    assert [с for с in текст.splitlines() if с.startswith("| `")] == [
        "| `not_a_car_photo` | 3 |",
        "| `over_redacted` | 2 |",
        "| `alias_mismatch` | 1 |",
    ]


def test_потери_обхода_суммируются_по_журналу() -> None:
    строки_обхода = [
        _обход(requests=1, rejected_stopword=0, rejected_exclude=0, rejected_not_own=3),
        _обход(requests=2, rejected_stopword=1, rejected_exclude=1, rejected_not_own=0),
    ]

    числа = collect(
        rows=[],
        rejects=[],
        verdict_events=(),
        catalog_classes=(),
        classes_without_holdout=6,
        expand_rows=строки_обхода,
    )

    assert числа.expand_requests == 3
    assert числа.expand_rejected_stopword == 1
    assert числа.expand_rejected_exclude == 1
    assert числа.expand_rejected_not_own == 3
    assert числа.expand_budget_exhausted == 0


def test_отсутствие_журнала_обхода_даёт_нули() -> None:
    числа = collect(
        rows=[], rejects=[], verdict_events=(), catalog_classes=(), classes_without_holdout=6
    )

    assert числа.expand_requests == 0
    assert числа.expand_rejected_not_own == 0


def test_отчёт_показывает_чем_срезан_обход() -> None:
    числа = _числа(expand_requests=5, expand_rejected_not_own=3, expand_budget_exhausted=1)
    прошёл, провалы = check_c1(числа)

    текст = render(числа, прошёл, провалы, "abc1234", "2026-08-31")

    assert "## Обход категорий: запросы и потери" in текст
    assert "| запросов к Commons | 5 |" in текст
    assert "| отвергнуто не своей полкой | 3 |" in текст
    assert "| категорий с исчерпанным бюджетом | 1 |" in текст
    assert "| запросов с усечённым списком подкатегорий | 0 |" in текст


def test_отчёт_показывает_скоуп_c2_по_классам() -> None:
    """Раздел собирается `render`, а не дописывается руками: `main` пишет отчёт целиком."""
    числа = _числа(
        classes_without_holdout=2,
        c2_scope=(
            C2ScopeRow(class_id="chery__amulet", accepted=30, category_rows=20),
            C2ScopeRow(class_id="chevrolet__lanos", accepted=0, category_rows=0),
        ),
    )

    текст = render(числа, False, [], "abc1234", "2026-08-31")

    assert "## Классы без эталона (скоуп задачи C2)" in текст
    assert "| `chery__amulet` | 30 | 20 | 66.7% |" in текст
    assert "| `chevrolet__lanos` | 0 | 0 | 0.0% |" in текст
    assert "молчание игл на поколенческих деревьях" in текст, (
        "второе предложение C2 — иглы, а не новые источники"
    )


def test_отчёт_без_дефицитных_классов_печатает_нет() -> None:
    текст = render(_числа(classes_without_holdout=0), True, [], "abc1234", "2026-08-31")

    assert "## Классы без эталона (скоуп задачи C2)\n\nКлассы, у которых" in текст
    assert текст.rstrip().endswith("Нет.")


def test_отчёт_оговаривает_пожизненность_таблиц() -> None:
    """Два отчёта одного прогона дают разные причины отказа: один дельта, другой пожизненный."""
    текст = render(_числа(), True, [], "abc1234", "2026-08-31")

    assert "Числа пожизненные: `reject_log.csv` копится между прогонами" in текст
    assert "`bench/harvest-m0.md` показан дельтой из `ClassOutcome`" in текст
    assert "Журнал обхода тоже пожизненный" in текст
    assert "считает **отказы**, а не потерянные категории" in текст


def test_main_на_красном_гейте_пишет_отчёт_и_возвращает_1(tmp_path: Path) -> None:
    """Число гейта считается само: восемь классов каталога, ни у одного нет эталона."""
    флаги, отчёт = _подготовить(tmp_path)

    код = main(флаги)

    assert код == 1
    текст = отчёт.read_text(encoding="utf-8")
    assert "- вердикт гейта: НЕ ПРОЙДЕН" in текст
    assert f"- G_C1: классов без эталона — 8 > {G_C1_MAX_MISSING}" in текст
    assert "| споров `conflict` | 0 | 1 |" in текст
    assert "| вытеснений `supersede` | 0 | 2 |" in текст
    assert "| `not_a_car_photo` | 2 |" in текст
    assert "| отвергнуто не своей полкой | 3 |" in текст
    assert "| `lada__granta` | 2 | 1 | 50.0% |" in текст


def test_main_на_зелёном_гейте_возвращает_0(tmp_path: Path) -> None:
    флаги, отчёт = _подготовить(tmp_path, классов_в_каталоге=6)

    код = main(флаги)

    assert код == 0
    текст = отчёт.read_text(encoding="utf-8")
    assert "- вердикт гейта: пройден" in текст
    assert "- строк в манифесте: 2" in текст
    assert "50.0%" in текст


def test_main_перекрытие_числа_гейта_предупреждает_о_расхождении(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Флаг оставлен для сверки со старым выводом `build_holdout_ru`, и расхождение слышно."""
    флаги, _ = _подготовить(tmp_path, классов_в_каталоге=6)

    код = main([*флаги, "--classes-without-holdout", "16"])

    assert код == 1
    assert "ВНИМАНИЕ: перекрытие 16 расходится с вычисленными 6" in capsys.readouterr().out


def test_main_без_файла_вердиктов_считает_их_пустыми(tmp_path: Path) -> None:
    флаги, отчёт = _подготовить(tmp_path, классов_в_каталоге=6)
    флаги[флаги.index("--dedup-verdicts") + 1] = str(tmp_path / "нет.csv")

    код = main(флаги)

    assert код == 0
    текст = отчёт.read_text(encoding="utf-8")
    assert "| споров `conflict` | 0 | 0 |" in текст
    assert "| вытеснений `supersede` | 0 | 0 |" in текст


def test_main_считает_вердикты_и_из_журнала_применённых(tmp_path: Path) -> None:
    """`verdicts.dedup.csv` усекается после применения, а событие прогона обязано остаться.

    До правки C1 отчёт читал только очередь, поэтому усечение обнулило бы оба числа в
    git-отслеживаемом отчёте, хотя вытеснение и спор в прогоне были.
    """
    флаги, отчёт = _подготовить(tmp_path, классов_в_каталоге=6)
    флаги[флаги.index("--dedup-verdicts") + 1] = str(tmp_path / "нет.csv")
    append_applied(
        tmp_path / "dedup_verdicts.csv",
        [
            AppliedVerdict(
                file="ab.jpg",
                class_id="lada__granta",
                basis="category:Kia Rio",
                verdict="drop:superseded",
                applied="2026-09-01",
            )
        ],
    )

    код = main(флаги)

    assert код == 0
    assert "| вытеснений `supersede` | 0 | 1 |" in отчёт.read_text(encoding="utf-8")


def test_main_не_двоит_вердикт_из_журнала_и_очереди(tmp_path: Path) -> None:
    """Неусечённая очередь не превращает одно событие в два: пара «файл + вердикт» одна."""
    флаги, отчёт = _подготовить(tmp_path, классов_в_каталоге=6)
    append_applied(
        tmp_path / "dedup_verdicts.csv",
        [
            AppliedVerdict(
                file="b.jpg",
                class_id="lada__granta",
                basis="category:Kia Rio",
                verdict="drop:superseded",
                applied="2026-09-01",
            )
        ],
    )

    код = main(флаги)

    assert код == 0
    # В очереди `b.jpg` и `c.jpg`, в журнале — только `b.jpg`: событий два, а не три.
    assert "| вытеснений `supersede` | 0 | 2 |" in отчёт.read_text(encoding="utf-8")


def test_main_без_журнала_обхода_считает_потери_нулями(tmp_path: Path) -> None:
    флаги, отчёт = _подготовить(tmp_path, классов_в_каталоге=6)
    флаги[флаги.index("--expand-log") + 1] = str(tmp_path / "нет.csv")

    код = main(флаги)

    assert код == 0
    текст = отчёт.read_text(encoding="utf-8")
    assert "| отвергнуто не своей полкой | 0 |" in текст
