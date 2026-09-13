# tests/test_harvest_gate.py
from dataclasses import replace

from pipeline.harvest.audit import AuditSummary
from pipeline.harvest.gate import H2_CORE_CLASSES, check_gate, render_report
from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.run import ClassOutcome


def _итог(class_id: str, принято: int) -> ClassOutcome:
    return ClassOutcome(class_id, seen=200, accepted=принято, rejects={"duplicate": 3})


def _строка(номер: int) -> HarvestRow:
    return HarvestRow(
        file=f"{номер}.jpg",
        class_id="lada__granta",
        source="commons",
        source_item_id=str(номер),
        page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        file_url="https://upload.wikimedia.org/X.jpg",
        author="Иван Петров",
        license="CC BY 4.0",
        license_tier=2,
        checked="2026-08-28",
        basis="category:Lada Granta",
        sha256=f"{номер:064d}",
        phash=номер,
        clip_rank=2,
        clip_margin=0.1,
        plates=1,
        faces=0,
        bbox_x=12.0,
        bbox_y=34.0,
        bbox_w=500.0,
        bbox_h=300.0,
        bbox_area_frac=0.42,
        det_conf=0.91,
        audit="ok",
    )


def _полный_прогон(добравших: int) -> list[ClassOutcome]:
    итоги = [_итог(f"класс_{n}", 15 if n < добравших else 4) for n in range(150 - 4)]
    итоги += [_итог(class_id, 15) for class_id in H2_CORE_CLASSES]
    return итоги


def _аудит(
    *,
    resolved_total: int = 100,
    dirty_found: tuple[str, ...] = (),
    dirty_applied: tuple[str, ...] = (),
    dirty_remaining: tuple[str, ...] = (),
    gap: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> AuditSummary:
    """Сводка аудита для гейта; умолчание — закрытый и чистый протокол."""
    return AuditSummary(
        resolved_total=resolved_total,
        dirty_found=dirty_found,
        dirty_applied=dirty_applied,
        dirty_remaining=dirty_remaining,
        gap=gap,
        conflicts=conflicts,
    )


def test_прогон_проходит_гейт() -> None:
    гейт = check_gate(_полный_прогон(120), [_строка(1)], _аудит(), allow_sa=False)
    assert гейт.passed
    assert гейт.blocking_failures == ()


def test_найденный_и_применённый_dirty_не_блокирует() -> None:
    """§12.1: найдено и применено — выполненная работа аудита, а не регресс редакции."""
    гейт = check_gate(
        _полный_прогон(120),
        [_строка(1)],
        _аудит(dirty_found=("b.jpg",), dirty_applied=("b.jpg",)),
        allow_sa=False,
    )

    assert гейт.passed
    assert гейт.blocking_failures == ()


def test_оставшийся_в_манифесте_dirty_блокирует() -> None:
    """H3 блокирующий: это нарушение правила 1 Р-11, а не низкое покрытие."""
    гейт = check_gate(
        _полный_прогон(120),
        [_строка(1)],
        _аудит(dirty_found=("b.jpg",), dirty_remaining=("b.jpg",)),
        allow_sa=False,
    )

    assert not гейт.passed
    причина = next(п for п in гейт.blocking_failures if "H3" in п)
    assert "b.jpg" in причина


def test_непроверенный_протокол_блокирует() -> None:
    """Гейт обязан различать «аудит чист» и «аудита не было»: 0 из 0 — не пройден."""
    гейт = check_gate(
        _полный_прогон(120),
        [_строка(1)],
        _аудит(resolved_total=0, gap=("1.jpg",)),
        allow_sa=False,
    )
    assert not гейт.passed
    assert any("H3" in причина for причина in гейт.blocking_failures)


def test_недостача_названа_числом_и_примером() -> None:
    """Кадр с `plates>0` без вердикта — незакрытый протокол §14."""
    гейт = check_gate(
        _полный_прогон(120), [_строка(1)], _аудит(gap=("2.jpg", "7.jpg")), allow_sa=False
    )

    assert not гейт.passed
    причина = next(п for п in гейт.blocking_failures if "аудит не закрыт" in п)
    assert "2 обязательных кадров без решённого вердикта" in причина
    assert "2.jpg" in причина


def test_спорный_кадр_блокирует_и_назван_спорным() -> None:
    """Кадр с `conflict` вердикт имеет, но вопрос по нему открыт: гейт блокирует, а
    сообщение не смеет называть его «без вердикта»."""
    гейт = check_gate(
        _полный_прогон(120), [_строка(1)], _аудит(conflicts=("2.jpg",)), allow_sa=False
    )

    assert not гейт.passed
    причина = next(п for п in гейт.blocking_failures if "conflict" in п)
    assert "без решённого вердикта" not in причина
    assert "2.jpg" in причина


def test_битый_манифест_блокирует() -> None:
    строка = _строка(1)
    порченая = replace(строка, page_url="")
    гейт = check_gate(_полный_прогон(120), [порченая], _аудит(), allow_sa=False)
    assert not гейт.passed
    assert any("H4" in причина for причина in гейт.blocking_failures)


def test_низкое_покрытие_не_блокирует() -> None:
    """H1 — это измеренный дефицит, он открывает второй источник, а не чинит код."""
    гейт = check_gate(_полный_прогон(30), [_строка(1)], _аудит(), allow_sa=False)
    assert гейт.passed
    assert any("H1" in причина for причина in гейт.advisory_failures)


def test_недобор_ядрового_класса_не_блокирует_но_виден() -> None:
    итоги = [_итог(f"класс_{n}", 15) for n in range(146)]
    итоги += [_итог(H2_CORE_CLASSES[0], 3)]
    итоги += [_итог(class_id, 15) for class_id in H2_CORE_CLASSES[1:]]
    гейт = check_gate(итоги, [_строка(1)], _аудит(), allow_sa=False)
    assert гейт.passed
    assert any("H2" in причина for причина in гейт.advisory_failures)


def test_сборка_не_проверяет_h1_h2_и_говорит_об_этом() -> None:
    """Без чисел прогона H1 и H2 неизвестны, а молчание читалось бы как «дефицита нет»:
    сборка по накопительному манифесту насчитывала «классов, добравших 30: 134» там, где
    прогон не добрал ни одного."""
    гейт = check_gate(None, [_строка(1)], _аудит(resolved_total=1), allow_sa=False)

    assert not any("H1:" in п or "H2:" in п for п in гейт.advisory_failures)
    assert any("не проверены" in п for п in гейт.advisory_failures)


def test_сборка_не_мешает_блокирующим_порогам() -> None:
    """H3 и H4 считаются по строкам манифеста, а они в сборке настоящие."""
    гейт = check_gate(None, [_строка(1)], _аудит(resolved_total=1, gap=("2.jpg",)), allow_sa=False)

    assert not гейт.passed
    assert any("H3" in причина for причина in гейт.blocking_failures)


def test_отчёт_называет_дефицитные_классы() -> None:
    итоги = [_итог("lada__2107", 4), _итог("uaz__patriot", 15)]
    аудит = _аудит()
    отчёт = render_report(
        итоги,
        [_строка(1)],
        check_gate(итоги, [_строка(1)], аудит, allow_sa=False),
        git_sha="deadbee",
        ran_at="2026-08-28",
        audit=аудит,
    )
    assert "lada__2107" in отчёт
    assert "deadbee" in отчёт
    assert "дефицит" in отчёт
    assert "закрыто вердиктом 100" in отчёт, (
        "«проверено» было тем самым словом, которое засчитывало открытый спор за проверку"
    )


def test_отчёт_замыкает_арифметику_найдено_применено_осталось() -> None:
    """§12.1: отчёт показывает найдено, применено, осталось и называет применённое
    выполненной работой. Обнулять счётчик ради зелёного цвета запрещено."""
    итоги = [_итог("lada__2107", 4)]
    аудит = _аудит(dirty_found=("b.jpg", "c.jpg"), dirty_applied=("b.jpg", "c.jpg"))
    отчёт = render_report(
        итоги,
        [_строка(1)],
        check_gate(итоги, [_строка(1)], аудит, allow_sa=False),
        git_sha="deadbee",
        ran_at="2026-08-28",
        audit=аудит,
    )

    assert "найдено 2, применено 2, осталось 0" in отчёт
    assert "выполненная работа" in отчёт


def test_отчёт_прогона_называет_свои_числа_снимком() -> None:
    """После применения вердиктов манифест уезжает от чисел прогона, и файл обязан сказать,
    что он снимок, а не текущее состояние: иначе 4324 строки в отчёте против 4317 в манифесте
    читаются как ошибка одного из двух."""
    итоги = [_итог("lada__2107", 4)]
    аудит = _аудит()
    отчёт = render_report(
        итоги,
        [_строка(1)],
        check_gate(итоги, [_строка(1)], аудит, allow_sa=False),
        git_sha="deadbee",
        ran_at="2026-08-28",
        audit=аудит,
    )

    assert "снимок" in отчёт
    assert "bench/harvest-c1.md" in отчёт
    assert "Прогон от 2026-08-28, `git_sha` — `deadbee`." in отчёт


def test_отчёт_печатает_разбор_причин_отказа() -> None:
    """Причины без своего журнала видны только здесь: отказ необратим."""
    итоги = [
        replace(_итог("lada__2107", 4), rejects={"over_redacted": 7, "alias_mismatch": 2}),
        replace(_итог("uaz__patriot", 15), rejects={"over_redacted": 1}),
    ]
    аудит = _аудит()
    отчёт = render_report(
        итоги,
        [_строка(1)],
        check_gate(итоги, [_строка(1)], аудит, allow_sa=False),
        git_sha="deadbee",
        ran_at="2026-08-28",
        audit=аудит,
    )
    assert "## Причины отказа" in отчёт
    assert "| `over_redacted` | 8 |" in отчёт, "числа складываются по всем классам"
    assert "| `alias_mismatch` | 2 |" in отчёт
    assert "дельта этого прогона (`ClassOutcome`)" in отчёт, (
        "иначе расхождение с пожизненными числами bench/harvest-c1.md читается как ошибка"
    )
