"""CLI харвеста: прогон по классам каталога с провенансом и отчётом.

Запуск (веса детекторов и CLIP — из группы `train`):

    uv run --group train python -m scripts.harvest --contact mail@example.com

Контакт обязателен: обезличенный User-Agent — это то, за что хранилища банят по IP,
и §3.2 требует его прямо.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.detectors.yolo import YoloDetector
from pipeline.harvest.admission import build_admit
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.harvest.audit import (
    DEFAULT_APPLIED_PATH,
    append_applied,
    apply_audit,
    counts_with_verdicts,
    demote_over_human,
    describe_applied,
    load_applied,
    load_audit,
    merge_verdicts,
    suppress_applied,
)
from pipeline.harvest.backends import (
    ClipImageEncoder,
    OnnxBoxDetector,
    UrllibTransport,
    YuNetFaceDetector,
    build_text_matrix,
)
from pipeline.harvest.categories import ExpandResult, expand
from pipeline.harvest.commons import CommonsSource
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.detectors import (
    DEFAULT_DETECTORS_MANIFEST,
    load_detectors_manifest,
    verify_weights,
)
from pipeline.harvest.filters import NEGATIVE_PROMPTS
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
    claimed_categories,
    load_latin_names,
    load_queries,
)
from pipeline.harvest.run import HarvestDeps, harvest_class
from pipeline.harvest.state import load_state, save_state
from pipeline.harvest.tiling import TiledDetector
from pipeline.paths import harvest_dir

REPORT_PATH = Path(__file__).resolve().parent.parent / "bench" / "harvest-m0.md"
# Имя файла завязано в двух местах: сборщик пишет решения дедупа сюда (`HarvestDeps`),
# аудит читает их отсюда же (§8 дизайна). Разные литералы разошлись бы молча.
DEDUP_VERDICTS_FILENAME = "verdicts.dedup.csv"
# Тот же повод, что у DEDUP_VERDICTS_FILENAME: сборщик пишет журнал обхода сюда
# (`HarvestDeps`), отчёт C1 читает его отсюда же (`scripts/report_c1.py`).
EXPAND_LOG_FILENAME = "expand_log.csv"


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов. Контакт обязателен, остальное имеет умолчания."""
    парсер = argparse.ArgumentParser(description="Харвест открытого провенанса (Р-11)")
    парсер.add_argument("--contact", required=True, help="контакт для User-Agent (§3.2)")
    парсер.add_argument("--classes", nargs="*", default=None, help="подмножество класcов")
    парсер.add_argument("--allow-sa", action="store_true", help="разрешить Tier 3 CC-BY-SA")
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
            "не собирать; пересобрать bench/NOTICES из манифеста, применив накопленные "
            "вердикты дедупа и --audit к манифесту"
        ),
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Прогнать харвест и записать отчёт, манифест и атрибуцию."""
    аргументы = build_parser().parse_args(argv)
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    порядок = tuple(класс.class_id for класс in каталог.classes)
    корень = harvest_dir()

    if аргументы.report_only:
        from collections import Counter

        from pipeline.harvest.run import ClassOutcome

        по_классам = Counter(с.class_id for с in load_manifest(DEFAULT_MANIFEST_PATH))
        итоги = [
            ClassOutcome(class_id=c, seen=0, accepted=по_классам.get(c, 0), rejects={})
            for c in порядок
        ]
    else:
        import open_clip  # type: ignore[import-untyped]

        пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
        verify_weights(пины, корень / "models")

        запросы = load_queries(DEFAULT_QUERIES_PATH)
        имена = load_latin_names(DEFAULT_LATIN_NAMES_PATH)
        иглы = load_alias_table(DEFAULT_ALIASES_PATH)
        без_игл = [c for c in (аргументы.classes or порядок) if c not in иглы]
        if без_игл:
            перечень = ", ".join(без_игл)
            print(f"нет игл у классов: {перечень}. Шлюз забракует их целиком. Останов.")
            return 2

        модель, _, препроцесс = open_clip.create_model_and_transforms(
            аргументы.clip_model, pretrained=аргументы.clip_weights
        )
        токенизатор = open_clip.get_tokenizer(аргументы.clip_model)
        матрица_классов = build_text_matrix([имена[c] for c in порядок], модель, токенизатор)
        матрица_негативов = build_text_matrix(
            [текст.removeprefix("a photo of a ") for текст in NEGATIVE_PROMPTS],
            модель,
            токенизатор,
        )

        plate_detectors = (
            TiledDetector(OnnxBoxDetector(корень / "models" / f"plate.{пины['plate'].format}")),
            TiledDetector(
                YoloDetector(
                    weights=str(корень / "models" / f"plate_hires.{пины['plate_hires'].format}"),
                    conf=0.25,
                )
            ),
        )
        # Один транспорт на источник и на обход: контактный User-Agent (§3.2) и интервал
        # между запросами (≤ 1 запрос / 2 с на домен, там же) задаются в одном месте.
        # Пейсер живёт в транспорте и потому общий для поиска, скачивания и обхода: два
        # прежних счётчика на вызывающих не срабатывали — у `search` его не было вовсе, а
        # у `expand` он обнулялся на каждом вызове.
        транспорт = UrllibTransport(аргументы.contact)
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

        зависимости = HarvestDeps(
            source=CommonsSource(транспорт),
            vehicle_detector=YoloDetector(),
            plate_detectors=plate_detectors,
            face_detector=YuNetFaceDetector(корень / "models" / "face.onnx"),
            encoder=ClipImageEncoder(модель, препроцесс),
            class_matrix=матрица_классов,
            class_order=порядок,
            negative_matrix=матрица_негативов,
            allow_sa=аргументы.allow_sa,
            root=корень,
            manifest_path=DEFAULT_MANIFEST_PATH,
            checked=date.today().isoformat(),
            alias_table=иглы,
            alias_rejects_path=корень / "alias_rejects.csv",
            reject_log_path=корень / "reject_log.csv",
            expand_categories=развернуть,
            dedup_verdicts_path=корень / DEDUP_VERDICTS_FILENAME,
            expand_log_path=корень / EXPAND_LOG_FILENAME,
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
    # Счёт — безусловно и до `apply_audit`: отметки `audit` живут в манифесте постоянно
    # (не только на прогонах с новыми вердиктами), а `apply_audit` ниже выкидывает строки
    # с `drop:*` из манифеста — счёт после него навсегда давал бы `audited_dirty == 0`.
    audited_total, audited_dirty = counts_with_verdicts(строки, вердикты)
    if вердикты:
        # След применения — до `apply_audit`: после него строки с `drop:*` в манифесте уже
        # нет, и записывать в журнал будет нечего, кроме имени файла.
        append_applied(
            DEFAULT_APPLIED_PATH,
            describe_applied(незалогированное, авто, строки, date.today().isoformat()),
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
    гейт = check_gate(итоги, строки, audited_total, audited_dirty, allow_sa=аргументы.allow_sa)
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
            audited_total=audited_total,
            audited_dirty=audited_dirty,
        ),
        encoding="utf-8",
    )
    DEFAULT_NOTICES_PATH.write_text(render_notices(строки), encoding="utf-8")
    return 0 if гейт.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
