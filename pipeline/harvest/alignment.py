"""Триаж выборки по именам файлов: кто уходит автоматически, кто идёт к человеку.

Модуль работает с манифестом целиком и потому отделён от `aliases.py`, который отвечает
на вопрос об одном имени и одном классе и нужен шлюзу на приёме — там нет ни манифеста,
ни отчёта. Разъём между модулями один: `alias_verdict`.

Сети здесь нет ни в одном пути: триаж считается по колонкам манифеста, а значит целиком
воспроизводим и целиком тестируем.

Автомат допущен только к однозначному случаю — совпала ровно одна чужая игла и не
совпала своя. Человеку нечего добавить к слову `cressida` в имени файла. Два и более
чужих класса означают, что иглы спорят между собой, и вердикт за человека выдавать
нельзя; `unknown` означает, что имя молчит, и решает тоже человек.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from pipeline.harvest.aliases import (
    AliasesError,
    AliasKind,
    AliasTable,
    alias_verdict,
    broad_needles,
    needle_conflicts,
    normalize_filename,
)
from pipeline.harvest.audit import apply_audit
from pipeline.harvest.manifest import HarvestRow
from pipeline.harvest.sources import SEARCH_BASIS_PREFIX

__all__ = [
    "SEARCH_BASIS_PREFIX",
    "AliasGateResult",
    "RowVerdict",
    "auto_drops",
    "check_alias_gate",
    "render_alignment_report",
    "review_queue",
    "triage",
]


@dataclass(frozen=True, slots=True)
class RowVerdict:
    """Что имя файла говорит о строке манифеста."""

    file: str
    class_id: str
    kind: AliasKind
    foreign: tuple[str, ...]
    needle: str


def triage(rows: Sequence[HarvestRow], table: AliasTable) -> list[RowVerdict]:
    """Вердикт на каждую строку поискового базиса. Строки из категорий не трогаются.

    Имя берётся из `page_url`: колонки `title` в манифесте нет, а `page_url` несёт то же
    имя, только процент-кодированным — это снимает `normalize_filename`. Форму URL
    `normalize_filename` знает одну — commons (`/wiki/File:`), и сегодня в манифесте
    других источников нет. Второй источник (Flickr, Openverse по Р-11) отдаст URL, от
    которого в токены уйдёт весь путь, и слово из домена сможет совпасть с иглой —
    к тому времени имя придётся брать из колонки источника, а не из адреса.

    Класс вне таблицы игл — не `unknown`, а остановка: молчаливый пропуск означал бы
    вердикт, посчитанный не по тому классу. `KeyError` из `alias_verdict` перевыбрасывается
    с именем файла и класса, потому что на манифесте в три с половиной тысячи строк голый
    `KeyError('lada__granta')` не говорит ни какая это строка, ни что править надо
    `catalog/model_aliases.yaml`.
    """
    итог: list[RowVerdict] = []
    for строка in rows:
        if not строка.basis.startswith(SEARCH_BASIS_PREFIX):
            continue
        try:
            вердикт = alias_verdict(normalize_filename(строка.page_url), строка.class_id, table)
        except KeyError as ошибка:
            raise AliasesError(
                f"{строка.file}: класса {строка.class_id} нет в таблице игл "
                f"(catalog/model_aliases.yaml)"
            ) from ошибка
        итог.append(
            RowVerdict(строка.file, строка.class_id, вердикт.kind, вердикт.foreign, вердикт.needle)
        )
    return итог


def auto_drops(verdicts: Sequence[RowVerdict], table: AliasTable) -> list[RowVerdict]:
    """Строки, которые программа вправе выкинуть сама.

    Три условия: чужой класс ровно один, своя игла не совпала (это уже `kind`), и
    сработавшая чужая игла не шире собственных игл класса.

    Третье условие добавлено после ревью и стоило половины автомата. На заполненной
    таблице 67 автодропов из 131 держались на широкой игле соседа: `Lada_Niva_2.jpg` в
    классе `lada__niva_2131` ловится голой иглой `niva` соседнего `lada__niva_2121`, а
    своя игла (`niva 2131`, `2131`) совпасть с таким именем не может по устройству. §11
    дизайна оправдывает автомат тем, что «человеку нечего добавить к слову `cressida` в
    имени файла» — но к слову `niva` человеку есть что добавить: он посмотрит на кадр и
    посчитает двери. Такие строки уходят в очередь, а не под нож.
    """
    широкие = _широкие_пары(table)
    return [в for в in verdicts if _автомат_вправе(в, широкие)]


def review_queue(verdicts: Sequence[RowVerdict], table: AliasTable) -> list[RowVerdict]:
    """Строки на глаз человеку: всё, что не своё и не отдано автомату.

    Дополнение считается, а не перечисляется: `own` плюс автодропы плюс очередь обязаны
    давать полный список вердиктов, иначе охват `G_AL2` закрыть нечем.
    """
    широкие = _широкие_пары(table)
    return [в for в in verdicts if в.kind != "own" and not _автомат_вправе(в, широкие)]


def _широкие_пары(table: AliasTable) -> frozenset[tuple[str, str, str]]:
    """Тройки «сосед, его широкая игла, наш класс» — на них автомат не допускается."""
    return frozenset(broad_needles(table))


def _автомат_вправе(verdict: RowVerdict, широкие: frozenset[tuple[str, str, str]]) -> bool:
    """Один предикат на два списка: разойтись им нельзя."""
    if verdict.kind != "foreign" or len(verdict.foreign) != 1:
        return False
    return (verdict.foreign[0], verdict.needle, verdict.class_id) not in широкие


@dataclass(frozen=True, slots=True)
class AliasGateResult:
    """Вердикт задачи B: что блокирует и что просто измерено."""

    passed: bool
    blocking_failures: tuple[str, ...]
    measured: tuple[str, ...]


def check_alias_gate(
    rows: Sequence[HarvestRow],
    table: AliasTable,
    all_verdicts: Mapping[str, str],
    alias_verdicts: Mapping[str, str],
    catalog_class_ids: Collection[str],
) -> AliasGateResult:
    """Три блокирующих порога задачи B плюс измеряемые числа.

    `rows` — манифест ДО чистки, `all_verdicts` — слитая карта всех проходов,
    `alias_verdicts` — вердикты только алиасного прохода (без H3/privacy).
    Вердикты применяются здесь же (`apply_audit`), потому что охват иначе не по чему
    считать: выкинутая строка из манифеста исчезает вместе со своим вердиктом.

    Разделение «блокирует / измерено» повторяет `pipeline/harvest/gate.py` и по той же
    причине: чистота выборки под нашим контролем, рост метрики — под контролем CLIP и
    содержимого чужого хранилища.
    """
    блокирующие: list[str] = []

    без_игл = sorted(c for c in catalog_class_ids if c not in table)
    if без_игл:
        блокирующие.append(
            f"G_AL1: у {len(без_игл)} классов каталога нет игл, например {без_игл[0]}"
        )
        return AliasGateResult(False, tuple(блокирующие), ())

    конфликты = needle_conflicts(table)
    if конфликты:
        первый, второй, игла = конфликты[0]
        блокирующие.append(
            f"G_AL1: взаимно неразличимых игл — {len(конфликты)}, "
            f"например «{игла}» у {первый} и {второй}"
        )

    вердикты_строк = triage(rows, table)
    помеченные = [в for в in вердикты_строк if в.kind != "own"]
    недостача = sorted(в.file for в in помеченные if в.file not in alias_verdicts)
    if недостача:
        блокирующие.append(
            f"G_AL2: без вердикта {len(недостача)} помеченных строк "
            f"(например {недостача[0]}); ноль найденной грязи на непросмотренном ничего не значит"
        )

    остатки = [в for в in triage(apply_audit(rows, all_verdicts), table) if в.kind != "own"]
    if остатки:
        блокирующие.append(
            f"G_AL3: в манифесте осталось {len(остатки)} поисковых строк без своей иглы "
            f"(например {остатки[0].file} в {остатки[0].class_id}); "
            f"кадр либо выкидывается, либо объясняется новой иглой"
        )

    широкие_пары = broad_needles(table)
    уникальных_игл = len({игла for _, игла, _ in широкие_пары})
    автоматом = len(auto_drops(помеченные, table))
    измерено = [
        f"строк поискового базиса: {len(вердикты_строк)}",
        f"помечено: {len(помеченные)}, из них автоматом {автоматом}, "
        f"человеку {len(помеченные) - автоматом}",
        f"широких игл (не блокирует): {уникальных_игл} (пар: {len(широкие_пары)})",
    ]
    return AliasGateResult(not блокирующие, tuple(блокирующие), tuple(измерено))


def render_alignment_report(
    rows: Sequence[HarvestRow],
    table: AliasTable,
    verdicts: Mapping[str, str],
    gate: AliasGateResult,
    git_sha: str,
    ran_at: str,
    categories: Mapping[str, bool] | None = None,
) -> str:
    """Собрать `bench/model-alignment.md`: числа задачи в отслеживаемом git месте.

    Отчёт существует потому, что сама спека живёт вне git (`docs/` в `.gitignore`), и
    без него от задачи не осталось бы ни одного проверяемого следа.
    """
    try:
        вердикты_строк = triage(rows, table)
    except AliasesError:
        вердикты_строк = []

    помеченные = [в for в in вердикты_строк if в.kind != "own"]
    по_классам = Counter(в.class_id for в in помеченные)
    всего_по_классам = Counter(в.class_id for в in вердикты_строк)
    уцелевшие = apply_audit(rows, verdicts)
    осталось_по_классам = Counter(с.class_id for с in уцелевшие)

    широкие_пары = broad_needles(table)

    строки_отчёта = [
        "# Калибровка эталонной линейки: чужие модели в классах (задача B)",
        "",
        f"Прогон от {ran_at}, `git_sha` — `{git_sha}`.",
        "",
        f"- строк в манифесте до чистки: {len(rows)}",
        f"- строк поискового базиса: {len(вердикты_строк)}",
        f"- помечено правилом игл: {len(помеченные)}",
        f"- строк осталось после применения вердиктов: {len(уцелевшие)}",
        f"- вердикт гейта: {'пройден' if gate.passed else 'НЕ ПРОЙДЕН'}",
        "",
        "## Блокирующие провалы",
        "",
        *([f"- {причина}" for причина in gate.blocking_failures] or ["Нет."]),
        "",
        "## Измеренные числа (не блокируют)",
        "",
        *([f"- {число}" for число in gate.measured] or ["Нет."]),
    ]
    if широкие_пары:
        строки_отчёта += [
            "",
            "### Широкие иглы соседа (не блокируют)",
            "",
            *[
                f"- `{класс}`: «{игла}» накрывает класс `{сосед}`"
                for класс, игла, сосед in широкие_пары
            ],
        ]
    строки_отчёта += [
        "",
        "## Классы с помеченными строками",
        "",
        "| Класс | Поисковых | Помечено | Осталось всего |",
        "|---|---|---|---|",
        *[
            f"| `{класс}` | {всего_по_классам[класс]} | {сколько} | {осталось_по_классам[класс]} |"
            for класс, сколько in по_классам.most_common()
        ],
    ]
    if categories is not None:
        отсутствуют = sorted(класс for класс, есть in categories.items() if not есть)
        строки_отчёта += [
            "",
            "## Проверка категорий Commons (вход в задачу C)",
            "",
            f"Проверено классов: {len(categories)}, категория не найдена или пуста у "
            f"{len(отсутствуют)}.",
            "",
            *[f"- `{класс}`" for класс in отсутствуют],
        ]
    return "\n".join(строки_отчёта) + "\n"
