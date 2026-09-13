"""CLI харвеста: прогон по классам каталога с провенансом и отчётом.

Запуск (веса детекторов и CLIP — из группы `train`):

    uv run --group train python -m scripts.harvest --contact mail@example.com

Контакт обязателен: обезличенный User-Agent — это то, за что хранилища банят по IP,
и §3.2 требует его прямо.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.detectors.yolo import YoloDetector
from pipeline.harvest.admission import build_admit
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, AliasTable, load_alias_table
from pipeline.harvest.audit import (
    DEFAULT_APPLIED_PATH,
    append_applied,
    apply_audit,
    demote_over_human,
    describe_applied,
    load_applied,
    load_audit,
    merge_verdicts,
    summarize_audit,
    suppress_applied,
)
from pipeline.harvest.backends import (
    ClipImageEncoder,
    UrllibTransport,
    YuNetFaceDetector,
    build_text_matrix,
)
from pipeline.harvest.categories import ExpandResult, expand
from pipeline.harvest.commons import CommonsSource, Transport
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.detectors import (
    DEFAULT_DETECTORS_MANIFEST,
    load_detectors_manifest,
    verify_weights,
)
from pipeline.harvest.filters import NEGATIVE_PROMPTS
from pipeline.harvest.flickr import FlickrSource
from pipeline.harvest.gate import check_gate, render_report
from pipeline.harvest.manifest import (
    DEFAULT_MANIFEST_PATH,
    load_manifest,
    write_manifest,
)
from pipeline.harvest.notices import DEFAULT_NOTICES_PATH, render_notices
from pipeline.harvest.queries import (
    DEFAULT_LATIN_NAMES_PATH,
    DEFAULT_QUERIES_PATH,
    ClassQueries,
    claimed_categories,
    load_latin_names,
    load_queries,
)
from pipeline.harvest.redaction_profile import (
    CURRENT_REDACTION_PROFILE,
    RedactionProfile,
    build_plate_detectors,
)
from pipeline.harvest.run import CategoryExpander, ClassOutcome, HarvestDeps, harvest_class
from pipeline.harvest.sources import HarvestSource
from pipeline.harvest.state import load_state, save_state
from pipeline.paths import harvest_dir

REPORT_PATH = Path(__file__).resolve().parent.parent / "bench" / "harvest-m0.md"
# Имя файла завязано в двух местах: сборщик пишет решения дедупа сюда (`HarvestDeps`),
# аудит читает их отсюда же (§8 дизайна). Разные литералы разошлись бы молча.
DEDUP_VERDICTS_FILENAME = "verdicts.dedup.csv"
# Тот же повод, что у DEDUP_VERDICTS_FILENAME: сборщик пишет журнал обхода сюда
# (`HarvestDeps`), отчёт C1 читает его отсюда же (`scripts/report_c1.py`).
EXPAND_LOG_FILENAME = "expand_log.csv"


# N818 просит суффикс `Error`; по-русски вышло бы `ОтсутствуетКлючError`. Имя пришло из
# интерфейса задачи (план C2, задача 4) и живёт в тесте, поэтому глушится правило, а не имя.
class ОтсутствуетКлюч(RuntimeError):  # noqa: N818
    """Ключ Flickr не задан: прогон обязан встать до первого сетевого запроса."""


def build_source(
    name: str, transport: Transport, environ: Mapping[str, str], allow_sa: bool
) -> HarvestSource:
    """Источник по имени. Ключ Flickr — только из окружения (§11 дизайна C2).

    Окружение приходит параметром, а не читается из `os.environ` внутри: иначе тест на
    отсутствие ключа зависел бы от машины, на которой запущен.

    Ключ не берётся из аргумента командной строки сознательно: контакт мы обязаны
    раскрывать по §3.2 SPEC, ключ обязаны не раскрывать, и общий механизм смешал бы два
    противоположных требования. Аргумент осел бы в истории шелла и в `ps`.
    """
    if name == "commons":
        return CommonsSource(transport)
    ключ = environ.get("FLICKR_API_KEY", "").strip()
    if not ключ:
        raise ОтсутствуетКлюч("FLICKR_API_KEY не задан в окружении")
    return FlickrSource(transport, ключ, allow_sa=allow_sa)


def source_queries(table: dict[str, ClassQueries], source_name: str) -> dict[str, ClassQueries]:
    """Таблица запросов под выбранный источник: у Flickr свои строки и никаких категорий.

    Подмена здесь, а не в `run.py`: цепочка решений прогона имён источников не знает
    (§5 дизайна C2), а `_запросы` (`run.py:428`) читает ровно `categories` и `terms`. Без
    подмены `flickr_terms` не читал бы никто, и прогон Flickr спрашивал бы у Flickr
    коммонсовские `terms`.

    Категории обнуляются, потому что у Flickr их нет вовсе. Оставленные, они стоили бы
    обхода дерева Commons по сети (`_категории`, `run.py:362`) ради запросов, которые
    `FlickrSource._параметры` всё равно отвергает. `depth`, `include` и `exclude` вместе с
    ними становятся недостижимы — их читает только обход.

    Класс без `flickr_terms` получает пустой набор запросов и просмотренный ноль. Это
    заявленное значение поля («из Flickr этот класс не добирается», `queries.py:39-40`), а не
    потеря.

    Оговорка про состояние: ключ токена пагинации — `f"{class_id}|{basis_of(запрос)}"`
    (`run.py:311`), и источник в нём не назван. `"Chevrolet Lanos"` стоит и в `terms`, и в
    `flickr_terms` одного класса, поэтому токены обоих источников ложатся под один ключ.
    Ущерб несимметричен. Flickr → Commons — смещение: номер страницы прочтётся как
    `gsroffset` (`commons.py:196`), офсет законен, выдача просто сдвинется. Commons → Flickr
    хуже: офсет прочтётся как номер страницы (`flickr.py:192-193`), страница за последней
    даст пустой `photo` и токен `None`, а `run.py:349-351` на этом выбросит ключ — класс
    закроется молчаливым нулём по всему терму. Потеря ограничена одним прогоном: тем же шагом
    ключ и выбрасывается, поэтому следующий прогон начнёт терм со страницы 1. Ловушка на
    будущее уже лежит в состоянии (`datsun__on_do|search:Datsun on-DO` → `"500"`): классу с
    таким токеном нельзя добавлять `flickr_terms`, не сняв его. Правка самого ключа тронула бы
    запрещённую §5 цепочку решений и обнулила бы живые коммонсовские токены, поэтому здесь
    оговорка, а не лечение.
    """
    if source_name != "flickr":
        return table
    return {
        class_id: replace(запросы, categories=(), terms=запросы.flickr_terms)
        for class_id, запросы in table.items()
    }


def build_deps(
    source: HarvestSource,
    expand_categories: CategoryExpander,
    *,
    root: Path,
    class_order: tuple[str, ...],
    latin_names: Mapping[str, str],
    alias_table: AliasTable,
    manifest_path: Path,
    clip_model: str,
    clip_weights: str,
    allow_sa: bool,
    checked: str,
    profile: RedactionProfile = CURRENT_REDACTION_PROFILE,
) -> HarvestDeps:
    """Собрать зависимости прогона: модели, детекторы и пути. Единственное такое место.

    Извлечено из `main`, потому что теми же весами и той же текстовой матрицей обязан считать
    переприём (`scripts/readmit_rejects.py`). Копия этой сборки разошлась бы с боевой при
    первой правке модели CLIP — молча, и переприём принимал бы кадры не тем конвейером.

    `import open_clip` остаётся внутри тела: пакет живёт в группе `train`, в `dev` его нет, и
    импорт на уровне модуля уронил бы обычный `pytest`. `expand_categories` приходит
    параметром — у боевого прогона это замыкание на транспорт, у переприёма заглушка.

    `profile` — параметры детекторов номеров целиком (§3.1 дизайна рекалибровки). Умолчание
    оставлено ради переприёма, который обязан считать боевым контуром и знать о нём не больше,
    чем сейчас; калибровочные прогоны будут подавать свой профиль явно.
    """
    import open_clip  # type: ignore[import-untyped]

    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    verify_weights(пины, root / "models")

    модель, _, препроцесс = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_weights
    )
    токенизатор = open_clip.get_tokenizer(clip_model)
    матрица_классов = build_text_matrix(
        [latin_names[класс] for класс in class_order], модель, токенизатор
    )
    матрица_негативов = build_text_matrix(
        [текст.removeprefix("a photo of a ") for текст in NEGATIVE_PROMPTS],
        модель,
        токенизатор,
    )
    plate_detectors = build_plate_detectors(profile, models_dir=root / "models", pins=пины)
    return HarvestDeps(
        source=source,
        vehicle_detector=YoloDetector(),
        plate_detectors=plate_detectors,
        face_detector=YuNetFaceDetector(root / "models" / "face.onnx"),
        encoder=ClipImageEncoder(модель, препроцесс),
        class_matrix=матрица_классов,
        class_order=class_order,
        negative_matrix=матрица_негативов,
        allow_sa=allow_sa,
        root=root,
        manifest_path=manifest_path,
        checked=checked,
        alias_table=alias_table,
        alias_rejects_path=root / "alias_rejects.csv",
        reject_log_path=root / "reject_log.csv",
        expand_categories=expand_categories,
        dedup_verdicts_path=root / DEDUP_VERDICTS_FILENAME,
        expand_log_path=root / EXPAND_LOG_FILENAME,
    )


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов. Контакт обязателен, остальное имеет умолчания."""
    парсер = argparse.ArgumentParser(description="Харвест открытого провенанса (Р-11)")
    парсер.add_argument("--contact", required=True, help="контакт для User-Agent (§3.2)")
    парсер.add_argument("--classes", nargs="*", default=None, help="подмножество класcов")
    парсер.add_argument("--allow-sa", action="store_true", help="разрешить Tier 3 CC-BY-SA")
    парсер.add_argument(
        "--source",
        choices=("commons", "flickr"),
        default="commons",
        help="хранилище кандидатов; для flickr нужен FLICKR_API_KEY в окружении",
    )
    парсер.add_argument("--clip-model", default="ViT-B-32", help="архитектура open_clip")
    парсер.add_argument("--clip-weights", default="laion2b_s34b_b79k", help="веса open_clip")
    парсер.add_argument(
        "--audit",
        nargs="*",
        default=[],
        help="CSV вердиктов ручного аудита H3 (§14); можно несколько, поздний важнее",
    )
    парсер.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "не собирать; применить накопленные вердикты дедупа и --audit к манифесту и "
            "пересобрать NOTICES. Отчёт прогона не трогает: его числа — снимок прогона, а "
            "сборка их не знает (числа сборки печатаются в stdout)"
        ),
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Прогнать харвест и записать отчёт, манифест и атрибуцию."""
    аргументы = build_parser().parse_args(argv)
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    порядок = tuple(класс.class_id for класс in каталог.classes)
    корень = harvest_dir()

    # `None` — «чисел прогона нет» (сборка по манифесту), список — числа именно этого запуска.
    итоги: list[ClassOutcome] | None
    if аргументы.report_only:
        # Подставных `ClassOutcome` здесь больше нет: `seen=0` и `rejects={}` были выдуманы, а
        # `accepted` из накопительного манифеста — не дельта прогона. Сборка по манифесту
        # насчитывала «классов, добравших 30: 134» там, где прогон не добрал ни одного
        # (пункт 8 плана C1). `None` означает «чисел прогона нет», и гейт про это знает.
        итоги = None
    else:
        # Один транспорт на источник и на обход: контактный User-Agent (§3.2) и интервал
        # между запросами (≤ 1 запрос / 2 с на домен, там же) задаются в одном месте.
        # Пейсер живёт в транспорте и потому общий для поиска, скачивания и обхода: два
        # прежних счётчика на вызывающих не срабатывали — у `search` его не было вовсе, а
        # у `expand` он обнулялся на каждом вызове.
        # Транспорт и источник стоят первыми в ветке: отказ по ключу Flickr обязан прийти
        # до первого сетевого байта (§11 дизайна C2), а `create_model_and_transforms` внутри
        # `build_deps` ниже на холодном кеше тянет веса по сети.
        транспорт = UrllibTransport(аргументы.contact)
        try:
            источник = build_source(аргументы.source, транспорт, os.environ, аргументы.allow_sa)
        except ОтсутствуетКлюч as ошибка:
            print(f"{ошибка}: экспортируйте ключ и повторите", file=sys.stderr)
            return 2

        запросы = source_queries(load_queries(DEFAULT_QUERIES_PATH), аргументы.source)
        имена = load_latin_names(DEFAULT_LATIN_NAMES_PATH)
        иглы = load_alias_table(DEFAULT_ALIASES_PATH)
        без_игл = [c for c in (аргументы.classes or порядок) if c not in иглы]
        if без_игл:
            перечень = ", ".join(без_игл)
            print(f"нет игл у классов: {перечень}. Шлюз забракует их целиком. Останов.")
            return 2

        # Один раз на весь прогон: карта не меняется между классами, а строить её заново на
        # каждый вызов `развернуть` значило бы пересчитывать одно и то же 150 раз.
        заявлено = claimed_categories(запросы)

        def развернуть(
            class_id: str,
            категория: str,
            глубина: int,
            include: tuple[str, ...],
            exclude: tuple[str, ...],
        ) -> ExpandResult:
            допустить = build_admit(class_id, иглы, заявлено)
            return expand(транспорт, категория, глубина, include, exclude, admit=допустить)

        # `DEFAULT_MANIFEST_PATH` передаётся параметром, а не читается внутри `build_deps`:
        # `tests/test_harvest_cli.py` подменяет этот модульный путь `monkeypatch.setattr`, и
        # чтение его в момент вызова — то, на чём подмена работает.
        зависимости = build_deps(
            источник,
            развернуть,
            root=корень,
            class_order=порядок,
            latin_names=имена,
            alias_table=иглы,
            manifest_path=DEFAULT_MANIFEST_PATH,
            clip_model=аргументы.clip_model,
            clip_weights=аргументы.clip_weights,
            allow_sa=аргументы.allow_sa,
            checked=date.today().isoformat(),
            profile=CURRENT_REDACTION_PROFILE,
        )

        состояние = load_state(корень / "state.json")
        # Засев манифестом: без него второй прогон принимает во второй класс кадр, который
        # первый уже принял в первый (§8 дизайна).
        дедуп = (
            DedupIndex.from_manifest(load_manifest(DEFAULT_MANIFEST_PATH))
            if DEFAULT_MANIFEST_PATH.exists()
            else DedupIndex()
        )
        итоги = []
        for class_id in аргументы.classes or порядок:
            итоги.append(harvest_class(запросы[class_id], зависимости, состояние, дедуп))
            save_state(корень / "state.json", состояние)
            print(
                f"{class_id}: принято={итоги[-1].accepted} просмотрено={итоги[-1].seen}",
                flush=True,
            )

    строки = load_manifest(DEFAULT_MANIFEST_PATH)
    # Файл дедупа идёт первым, а `--audit` — следом: `merge_verdicts` документирован как
    # «поздний проход побеждает ранний, потому что человек пересмотрел своё решение». Если
    # бы автоматика шла последней, она перебивала бы человека, и кадр, снесённый дедупом
    # (`drop:superseded`), человек не смог бы вернуть своим `ok` — тот переписывался бы на
    # каждом прогоне. Одного порядка мало: `ok`, который человек поставил в **прошлом**
    # прогоне, лежит в колонке манифеста, а не в `--audit` этого запуска, и автомат сносил
    # им подкреплённую строку молча. Поэтому автовердикты ещё и понижаются по манифесту
    # (`demote_over_human`, §8 дизайна C1).
    файл_дедупа = корень / DEDUP_VERDICTS_FILENAME
    запрошено = load_audit(файл_дедупа) if файл_дедупа.exists() else {}
    # Вторая линия защиты сверх усечения очереди ниже: усечение бережёт штатный путь, а
    # подавление — путь, где очередь его пережила (прогон прервался между применением и
    # усечением; очередь восстановили из копии; вердикты применила версия кода без усечения —
    # ровно случай, найденный финальным ревью ветки C1). Подавление обязано идти ДО
    # понижения, а не после: журнал (в том числе две строки старой схемы без `requested`)
    # хранит по каждой строке именно ЗАПРОШЕННЫЙ дедупом вердикт — до появления
    # `demote_over_human` понижения не было вообще, поэтому колонка `verdict` тех строк и
    # есть запрос. Понижение же зависит от текущей отметки в манифесте и может дать разный
    # результат от прогона к прогону (кадр мог получить `audit=ok` уже после того, как этот
    # самый запрос был применён и залогирован) — сверка по понижённому вердикту искала бы
    # то, что понижение даёт сегодня, а не то, что журнал помнит с прошлого раза, и не
    # находила бы совпадения там, где оно есть.
    незалогированное = suppress_applied(запрошено, load_applied(DEFAULT_APPLIED_PATH))
    авто = demote_over_human(незалогированное, строки)
    ручные = [load_audit(Path(п)) for п in (аргументы.audit or [])]
    вердикты = merge_verdicts(авто, *ручные)
    # Снимок манифеста до применения: по нему сводка аудита считает найденное, а после
    # `apply_audit` строк с `drop:*` в манифесте уже нет, и найденное было бы всегда нулём.
    до_применения = строки
    if вердикты:
        # Журнал — провенанс **удалённых** строк, поэтому из ручного канала в него идут только
        # `drop:*`. `drop:plate` руки иначе не оставляет следа нигде: строка ушла из манифеста,
        # очередь `--audit` не усекается, но отчётом и не читается, и §12.1
        # «найдено/применено» на следующем же прогоне отчиталось бы нулём. Ручные `ok` и
        # `conflict` строку в манифесте оставляют, и свою историю она несёт колонкой `audit`;
        # записывать их значило бы выдать ручной спор за спор дедупа в счёте §9
        # (`scripts/report_c1.py`) и дать подавлению ключ, которого человек не запрашивал.
        # Запрос ручной записи — сам вердикт человека: понижать его некому.
        удаления_человека = {
            файл: вердикт
            for файл, вердикт in merge_verdicts(*ручные).items()
            if вердикт.startswith("drop:")
        }
        # След применения — до `apply_audit`: после него строки с `drop:*` в манифесте уже
        # нет, и записывать в журнал будет нечего, кроме имени файла.
        сегодня = date.today().isoformat()
        append_applied(
            DEFAULT_APPLIED_PATH,
            [
                *describe_applied(незалогированное, авто, строки, сегодня),
                *describe_applied(удаления_человека, удаления_человека, строки, сегодня),
            ],
        )
        строки = apply_audit(строки, вердикты)
        write_manifest(DEFAULT_MANIFEST_PATH, строки)
    # Усечение — по исходной очереди (`запрошено`), а не по `авто` после подавления: очередь
    # прочитана и полностью учтена в любом случае, применилась она сейчас или была уже
    # применена раньше (тогда `авто` после `suppress_applied` пуст). Усечение по `авто`
    # оставляло бы на диске заряженную, но уже неактуальную очередь навсегда, если
    # подавление съело её целиком, — тот же файл перечитывался бы каждым следующим прогоном.
    # Без усечения вообще: вердикт применяется на каждом следующем прогоне, а прополка
    # (`scripts/prune_state.py`) пересобирает `done_items` из манифеста, где снесённой строки
    # уже нет. Кадр снова стал бы кандидатом, снова был бы принят под тем же детерминированным
    # именем `{sha}.jpg` и снова снесён тем же старым вердиктом — замкнутый цикл потери. След
    # применения остаётся в журнале выше.
    if запрошено:
        файл_дедупа.write_text("", encoding="utf-8")
    # Журнал перечитывается уже после дописывания: он и есть память о найденном, и сводка
    # обязана видеть в нём сегодняшние записи наравне со вчерашними (§12.1). Сшивать список в
    # памяти значило бы держать вторую версию правды рядом с файлом, который только что
    # дедуплицировал записи сам.
    аудит = summarize_audit(до_применения, вердикты, строки, load_applied(DEFAULT_APPLIED_PATH))
    гейт = check_gate(итоги, строки, аудит, allow_sa=аргументы.allow_sa)
    if итоги is None:
        # Отчёт прогона сборка не пишет: перезапись затирала бы git-отслеживаемую запись
        # прогона, а роль «свежие числа манифеста с честной шапкой» уже занята
        # `bench/harvest-c1.md` (`scripts/report_c1.py`). Один файл — одна роль.
        вердикт = "пройден" if гейт.passed else "НЕ ПРОЙДЕН"
        print(
            f"сборка из манифеста: строк {len(строки)}, аудит закрыт вердиктом "
            f"{аудит.resolved_total}, дефектов приватности найдено {len(аудит.dirty_found)}, "
            f"применено {len(аудит.dirty_applied)}, осталось {len(аудит.dirty_remaining)}, "
            f"вердикт гейта {вердикт}"
        )
        for причина in (*гейт.blocking_failures, *гейт.advisory_failures):
            print(f"  {причина}")
        print(f"{REPORT_PATH} не тронут: его пишет только настоящий прогон")
    else:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        REPORT_PATH.write_text(
            render_report(
                итоги,
                строки,
                гейт,
                sha,
                date.today().isoformat(),
                audit=аудит,
            ),
            encoding="utf-8",
        )
    DEFAULT_NOTICES_PATH.write_text(render_notices(строки), encoding="utf-8")
    return 0 if гейт.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
