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
from dataclasses import replace
from datetime import date
from pathlib import Path

from pipeline.catalog import DEFAULT_CATALOG_PATH, load_catalog
from pipeline.detectors.yolo import YoloDetector
from pipeline.harvest.aliases import DEFAULT_ALIASES_PATH, load_alias_table
from pipeline.harvest.audit import apply_audit, audit_counts, load_audit, merge_verdicts
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
    load_latin_names,
    load_queries,
)
from pipeline.harvest.run import HarvestDeps, harvest_class
from pipeline.harvest.state import load_state, save_state
from pipeline.harvest.tiling import TiledDetector
from pipeline.paths import harvest_dir

REPORT_PATH = Path(__file__).resolve().parent.parent / "bench" / "harvest-m0.md"


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
        help="не собирать, только пересобрать bench/NOTICES из манифеста",
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Прогнать харвест и записать отчёт, манифест и атрибуцию."""
    аргументы = build_parser().parse_args(argv)
    каталог = load_catalog(DEFAULT_CATALOG_PATH)
    порядок = tuple(класс.class_id for класс in каталог.classes)

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

        корень = harvest_dir()
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
        # Транспорт один на источник и на обход: вежливость к хранилищу считается по
        # процессу, а не по вызывающему коду.
        транспорт = UrllibTransport(аргументы.contact)

        def развернуть(
            категория: str, глубина: int, include: tuple[str, ...], exclude: tuple[str, ...]
        ) -> ExpandResult:
            return expand(транспорт, категория, глубина, include, exclude)

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
            expand_categories=развернуть,
        )

        состояние = load_state(корень / "state.json")
        дедуп = DedupIndex()
        итоги = []
        for class_id in аргументы.classes or порядок:
            итоги.append(harvest_class(запросы[class_id], зависимости, состояние, дедуп))
            save_state(корень / "state.json", состояние)
            print(
                f"{class_id}: принято={итоги[-1].accepted} просмотрено={итоги[-1].seen}",
                flush=True,
            )

    строки = load_manifest(DEFAULT_MANIFEST_PATH)
    audited_total, audited_dirty = 0, 0
    if аргументы.audit:
        вердикты = merge_verdicts(*(load_audit(Path(п)) for п in аргументы.audit))
        с_вердиктами = [replace(s, audit=вердикты.get(s.file, s.audit)) for s in строки]
        audited_total, audited_dirty = audit_counts(с_вердиктами)
        строки = apply_audit(строки, вердикты)
        write_manifest(DEFAULT_MANIFEST_PATH, строки)
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
