# tests/test_harvest_gate.py
from dataclasses import replace

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


def test_прогон_проходит_гейт() -> None:
    гейт = check_gate(
        _полный_прогон(120), [_строка(1)], audited_total=100, audited_dirty=0, allow_sa=False
    )
    assert гейт.passed
    assert гейт.blocking_failures == ()


def test_читаемый_номер_в_аудите_блокирует() -> None:
    """H3 блокирующий: это нарушение правила 1 Р-11, а не низкое покрытие."""
    гейт = check_gate(
        _полный_прогон(120), [_строка(1)], audited_total=100, audited_dirty=1, allow_sa=False
    )
    assert not гейт.passed
    assert any("H3" in причина for причина in гейт.blocking_failures)


def test_непроверенный_протокол_блокирует() -> None:
    """Гейт обязан различать «аудит чист» и «аудита не было»: 0 из 0 — не пройден."""
    непроверенная = replace(_строка(1), audit="")
    гейт = check_gate(
        _полный_прогон(120), [непроверенная], audited_total=0, audited_dirty=0, allow_sa=False
    )
    assert not гейт.passed
    assert any("H3" in причина for причина in гейт.blocking_failures)


def test_частичный_аудит_блокирует() -> None:
    """Кадр с `plates>0` без вердикта — незакрытый протокол §14."""
    непроверенная = replace(_строка(2), audit="")
    гейт = check_gate(
        _полный_прогон(120),
        [_строка(1), непроверенная],
        audited_total=1,
        audited_dirty=0,
        allow_sa=False,
    )
    assert not гейт.passed
    assert any("H3" in причина for причина in гейт.blocking_failures)


def test_спорный_кадр_блокирует_и_назван_спорным() -> None:
    """Кадр с `conflict` вердикт имеет, но вопрос по нему открыт: гейт блокирует, а
    сообщение не смеет называть его «без вердикта»."""
    спорный = replace(_строка(2), audit="conflict")
    гейт = check_gate(
        _полный_прогон(120),
        [_строка(1), спорный],
        audited_total=1,
        audited_dirty=0,
        allow_sa=False,
    )

    assert not гейт.passed
    причина = next(п for п in гейт.blocking_failures if "аудит не закрыт" in п)
    assert "без решённого вердикта" in причина
    assert "1 спорных `conflict`" in причина


def test_битый_манифест_блокирует() -> None:
    строка = _строка(1)
    порченая = replace(строка, page_url="")
    гейт = check_gate(
        _полный_прогон(120), [порченая], audited_total=100, audited_dirty=0, allow_sa=False
    )
    assert not гейт.passed
    assert any("H4" in причина for причина in гейт.blocking_failures)


def test_низкое_покрытие_не_блокирует() -> None:
    """H1 — это измеренный дефицит, он открывает второй источник, а не чинит код."""
    гейт = check_gate(
        _полный_прогон(30), [_строка(1)], audited_total=100, audited_dirty=0, allow_sa=False
    )
    assert гейт.passed
    assert any("H1" in причина for причина in гейт.advisory_failures)


def test_недобор_ядрового_класса_не_блокирует_но_виден() -> None:
    итоги = [_итог(f"класс_{n}", 15) for n in range(146)]
    итоги += [_итог(H2_CORE_CLASSES[0], 3)]
    итоги += [_итог(class_id, 15) for class_id in H2_CORE_CLASSES[1:]]
    гейт = check_gate(итоги, [_строка(1)], audited_total=100, audited_dirty=0, allow_sa=False)
    assert гейт.passed
    assert any("H2" in причина for причина in гейт.advisory_failures)


def test_отчёт_называет_дефицитные_классы() -> None:
    итоги = [_итог("lada__2107", 4), _итог("uaz__patriot", 15)]
    отчёт = render_report(
        итоги,
        [_строка(1)],
        check_gate(итоги, [_строка(1)], 100, 0, allow_sa=False),
        git_sha="deadbee",
        ran_at="2026-08-28",
        audited_total=100,
        audited_dirty=0,
    )
    assert "lada__2107" in отчёт
    assert "deadbee" in отчёт
    assert "дефицит" in отчёт
    assert "аудит H3: закрыто вердиктом 100, с дефектами 0" in отчёт, (
        "«проверено» было тем самым словом, которое засчитывало открытый спор за проверку"
    )


def test_отчёт_печатает_разбор_причин_отказа() -> None:
    """Причины без своего журнала видны только здесь: отказ необратим."""
    итоги = [
        replace(_итог("lada__2107", 4), rejects={"over_redacted": 7, "alias_mismatch": 2}),
        replace(_итог("uaz__patriot", 15), rejects={"over_redacted": 1}),
    ]
    отчёт = render_report(
        итоги,
        [_строка(1)],
        check_gate(итоги, [_строка(1)], 100, 0, allow_sa=False),
        git_sha="deadbee",
        ran_at="2026-08-28",
        audited_total=100,
        audited_dirty=0,
    )
    assert "## Причины отказа" in отчёт
    assert "| `over_redacted` | 8 |" in отчёт, "числа складываются по всем классам"
    assert "| `alias_mismatch` | 2 |" in отчёт
    assert "дельта этого прогона (`ClassOutcome`)" in отчёт, (
        "иначе расхождение с пожизненными числами bench/harvest-c1.md читается как ошибка"
    )
