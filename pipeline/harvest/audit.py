"""Протокол ручного аудита H3 и заведение его чисел в гейт.

Прогон 2026-08-29 провалил H3 вакуумно: гейт получил `audited_dirty=0`, потому что
аудит покрыл 2 кадра из 28 с номерами, а не потому что выборка чиста. Здесь охват задан
явно и проверяем: все кадры с найденными номерами/лицами, все кадры ядровых классов,
плюс случайные 200 из тех, где детектор не нашёл ничего, — контроль ложноотрицательных
пропусков.

`drop:plate` и `drop:face` — нарушение приватности: такие кадры удаляются из манифеста
**и** засчитываются в `audited_dirty`, потому что хотя бы один такой кадр означает, что
автоматика протекает системно (гейт блокирует — правится редакция, а не список файлов).
`drop:<прочее>` (не та модель, мусор) приватности не касается: кадр удаляется, в dirty
не идёт.

Вердикты приходят из двух каналов, и они не равны. Человек смотрел на пиксели, автомат
(дедуп, §8 дизайна C1) сравнивал метаданные, поэтому **автомат не перебивает человека**:
автоматический `drop:*` строке с `audit == "ok"` не удаляет её, а понижается до `conflict`
(`demote_over_human`). Применённые автовердикты уезжают в git-отслеживаемый журнал
`provenance/dedup_verdicts.csv`: строка, выброшенная из манифеста, иначе исчезает из
провенанса, а причина остаётся в файле рядом с данными, который git не видит.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from pipeline.harvest.manifest import HarvestRow

AUDIT_DIRTY = frozenset({"drop:plate", "drop:face"})
CORE_CLASSES = ("lada__2107", "lada__granta", "lada__niva_2121", "uaz__patriot")
RANDOM_CLEAN_SAMPLE = 200
#: Отметка человека, которую автомат не имеет права перебить (§8 дизайна C1).
HUMAN_OK = "ok"
#: Куда сводится след применённых автовердиктов. Рядом с манифестом, а не в каталоге
#: данных: и то и другое — юридический след выборки, обязанный пережить чистку данных.
DEFAULT_APPLIED_PATH = (
    Path(__file__).resolve().parent.parent.parent / "provenance" / "dedup_verdicts.csv"
)


@dataclass(frozen=True, slots=True)
class AppliedVerdict:
    """Автовердикт, применённый к манифесту: по какой строке и когда.

    `class_id` и `basis` дублируют колонки манифеста намеренно: строку с `drop:*` из
    манифеста удаляют, и без копии по журналу нельзя узнать даже, какой класс потерял кадр.
    """

    file: str
    class_id: str
    basis: str
    verdict: str
    applied: str


APPLIED_COLUMNS = tuple(поле.name for поле in fields(AppliedVerdict))


class AuditError(ValueError):
    """CSV вердиктов не годится."""


def _годен(вердикт: str) -> bool:
    return вердикт in {"ok", "conflict"} or вердикт.startswith("drop:")


def load_audit(path: Path) -> dict[str, str]:
    """Прочитать CSV `file,verdict` и провалидировать вердикты."""
    вердикты: dict[str, str] = {}
    for строка in csv.DictReader(path.read_text(encoding="utf-8").splitlines()):
        файл = (строка.get("file") or "").strip()
        вердикт = (строка.get("verdict") or "").strip()
        if not файл:
            continue
        if not _годен(вердикт):
            raise AuditError(f"{path}: непонятный вердикт {вердикт!r} у {файл}")
        вердикты[файл] = вердикт
    return вердикты


def merge_verdicts(*verdicts: Mapping[str, str]) -> dict[str, str]:
    """Склеить вердикты нескольких проходов; поздний проход побеждает ранний.

    Порядок именно такой: второй проход смотрит кадр повторно только тогда, когда
    человек решил пересмотреть своё решение, и новый вердикт обязан вытеснить старый,
    а не потонуть в нём.
    """
    итог: dict[str, str] = {}
    for пачка in verdicts:
        итог.update(пачка)
    return итог


def apply_audit(rows: Sequence[HarvestRow], verdicts: Mapping[str, str]) -> list[HarvestRow]:
    """Проставить колонку `audit` и выкинуть строки с любым `drop:*`."""
    итог: list[HarvestRow] = []
    for строка in rows:
        вердикт = verdicts.get(строка.file, строка.audit)
        if вердикт.startswith("drop:"):
            continue
        итог.append(replace(строка, audit=вердикт))
    return итог


def demote_over_human(verdicts: Mapping[str, str], rows: Sequence[HarvestRow]) -> dict[str, str]:
    """Понизить автоматический `drop:*` до `conflict` там, где строку уже проверил человек.

    Автомат не перебивает человека. Дедуп решает спор двух классов по расстоянию хешей и
    по базису заявления, человек — по самим пикселям, поэтому заявление, подкреплённое
    отметкой `ok` в манифесте, сильнее заявления, подкреплённого категорией Commons. Первое
    же живое вытеснение это и доказало: строка `mitsubishi__pajero` с `audit=ok` была снесена
    по расстоянию ровно 6 (сам порог) против кадра `mazda__cx5` — двух разных машин одного
    фотографа в одинаковой постановке. Понижение оставляет спор в манифесте отметкой
    `conflict`, то есть отдаёт его человеку, ровно как §8 обещает для равно подкреплённых
    заявлений.

    Понижается только **автоматический** канал. Вердикты `--audit` — это и есть человек, и
    его `drop:plate` обязан сработать поверх любого прежнего `ok`: кадр с читаемым номером
    остаётся нарушением приватности, кто бы его раньше ни одобрил.
    """
    проверено = {строка.file for строка in rows if строка.audit == HUMAN_OK}
    return {
        файл: "conflict" if вердикт.startswith("drop:") and файл in проверено else вердикт
        for файл, вердикт in verdicts.items()
    }


def describe_applied(
    verdicts: Mapping[str, str], rows: Sequence[HarvestRow], applied: str
) -> list[AppliedVerdict]:
    """Записи журнала по вердиктам и манифесту **до** их применения.

    Строки, которых в манифесте нет, пропускаются: вердикт на отсутствующий файл ничего не
    изменил, и записывать его в журнал применённых значило бы врать о применении.
    """
    по_файлу = {строка.file: строка for строка in rows}
    return [
        AppliedVerdict(
            file=файл,
            class_id=по_файлу[файл].class_id,
            basis=по_файлу[файл].basis,
            verdict=вердикт,
            applied=applied,
        )
        for файл, вердикт in verdicts.items()
        if файл in по_файлу
    ]


def append_applied(path: Path, verdicts: Sequence[AppliedVerdict]) -> None:
    """Дописать применённые автовердикты в журнал, создав файл с шапкой при первом вызове."""
    if not verdicts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Файл нулевой длины — файл без шапки (идиома `rejects.append_reject_row`): журнал
    # укорачивают `> файл`, и без проверки размера первая дописанная строка стала бы шапкой.
    новый = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(APPLIED_COLUMNS))
        if новый:
            писатель.writeheader()
        for запись in verdicts:
            писатель.writerow(asdict(запись))


def load_applied(path: Path) -> list[AppliedVerdict]:
    """Прочитать журнал целиком; отсутствие файла — пустой журнал, а не ошибка."""
    if not path.exists():
        return []
    return [
        AppliedVerdict(
            file=строка["file"],
            class_id=строка["class_id"],
            basis=строка["basis"],
            verdict=строка["verdict"],
            applied=строка["applied"],
        )
        for строка in csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    ]


def audit_counts(rows: Sequence[HarvestRow]) -> tuple[int, int]:
    """`(audited_total, audited_dirty)` по строкам до удаления dirty."""
    total = sum(1 for с in rows if с.audit)
    dirty = sum(1 for с in rows if с.audit in AUDIT_DIRTY)
    return total, dirty


def required_audit_files(rows: Sequence[HarvestRow], seed: int = 0) -> set[str]:
    """Какие `file` обязаны быть в аудите по протоколу §14."""
    надо: set[str] = set()
    чистые: list[str] = []
    for строка in rows:
        if строка.plates > 0 or строка.faces > 0 or строка.class_id in CORE_CLASSES:
            надо.add(строка.file)
        else:
            чистые.append(строка.file)
    выборка = random.Random(seed).sample(чистые, min(RANDOM_CLEAN_SAMPLE, len(чистые)))
    надо.update(выборка)
    return надо


def audit_gap(rows: Sequence[HarvestRow], seed: int = 0) -> set[str]:
    """Обязательные по §14 кадры, у которых нет вердикта.

    Отличает «аудит прошёл и чист» от «аудита не было»: `audited_dirty == 0` верно в обоих
    случаях, и на этой неоднозначности прогон 2026-08-29 отрапортовал H3 вакуумно. Пустая
    недостача — единственное доказательство, что протокол закрыт, а не пропущен.
    """
    надо = required_audit_files(rows, seed=seed)
    проверено = {строка.file for строка in rows if строка.audit}
    return надо - проверено
