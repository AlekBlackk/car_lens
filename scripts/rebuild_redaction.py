"""Переderivация манифеста харвеста по кэшу оригиналов — без единого запроса к Commons.

Смена детектора или правил редакции обязана пересчитываться по `raw/`, а не
перекачиванием (§8 дизайна). Это полная переderivация от стадии кропа вниз:
редакция → pHash → дедуп → запись. Заливка номера меняет пиксели кропа, значит меняется
pHash, значит меняется результат дедупа — покрытие и список дефицитных классов
пересобираются заново, это не патч колонок.

Старая отметка `audit` переносится только при точном равенстве декодированных пикселей
старого и нового JPEG. Отсутствующий или битый старый кроп безопасно сбрасывает отметку.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from pipeline.clean import phash_of
from pipeline.crop import VEHICLE_CLASSES, Detector, crop_image, select_primary
from pipeline.harvest.cache import cache_path, decode_image
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.jpeg import encode_jpeg, observed_pixels_equal, write_jpeg_atomic
from pipeline.harvest.manifest import (
    DEFAULT_MANIFEST_PATH,
    HarvestRow,
    append_row,
    load_manifest,
)
from pipeline.harvest.redact import redact_frame
from pipeline.harvest.run import is_crop_over_redacted
from pipeline.paths import harvest_dir


class RebuildError(ValueError):
    """Манифест нельзя безопасно пересобрать без неоднозначной записи кропов."""


def crop_destination(root: Path, row: HarvestRow) -> Path:
    """Безопасный путь кропа, сохраняющий authoritative имя из манифеста."""
    имя = Path(row.file)
    if (
        имя.is_absolute()
        or имя.name != row.file
        or "/" in row.file
        or "\\" in row.file
        or row.file in {"", ".", ".."}
    ):
        raise RebuildError(f"небезопасное имя кропа: {row.file!r}")
    return root / "crops" / row.class_id / row.file


def validate_crop_destinations(rows: Sequence[HarvestRow], root: Path) -> None:
    """До пересборки отвергнуть небезопасные и повторяющиеся пути кропов."""
    пути: set[Path] = set()
    for row in rows:
        путь = crop_destination(root, row)
        if путь in пути:
            raise RebuildError(f"повтор пути кропа: {row.class_id}/{row.file}")
        пути.add(путь)


@dataclass(frozen=True, slots=True)
class RebuildDeps:
    """Всё внешнее для переderivации. Ничего не создаётся внутри."""

    vehicle_detector: Detector
    plate_detectors: tuple[Detector, ...]
    face_detector: Detector
    root: Path


def preflight(rows: Sequence[HarvestRow], root: Path) -> list[str]:
    """SHA256 строк, для которых нет оригинала в кэше. Пустой список — кэш цел."""
    пропало: list[str] = []
    for строка in rows:
        if not cache_path(root, строка.source, строка.sha256, "jpg").exists():
            пропало.append(строка.sha256)
    return пропало


def rebuild_row(
    row: HarvestRow, deps: RebuildDeps, dedup: DedupIndex, reset_audit: bool = False
) -> tuple[str, HarvestRow | None]:
    """Прогнать один оригинал через новую редакцию. Причина отказа или новая строка."""
    сырьё = cache_path(deps.root, row.source, row.sha256, "jpg").read_bytes()
    try:
        кадр = decode_image(сырьё)
    except Exception:
        return "decode_failed", None

    все_боксы = deps.vehicle_detector.detect(кадр)
    высота, ширина = кадр.shape[:2]
    решение = select_primary(все_боксы, ширина, высота)
    if решение.reject_reason is not None:
        return решение.reject_reason, None
    if решение.is_multi_car:
        return "multi_car", None
    assert решение.box is not None

    машины = [d for d in все_боксы if d.cls in VEHICLE_CLASSES and d.conf > 0.5]
    итог = redact_frame(кадр, машины, deps.plate_detectors, deps.face_detector)
    if not итог.ok:
        return "redaction_failed", None

    кроп = crop_image(итог.image, решение.box)
    if is_crop_over_redacted(кроп):
        return "over_redacted", None

    хеш = phash_of(кроп)

    вердикт = dedup.check(хеш, row.class_id, row.basis)
    # §8: `supersede` объявляет победителем **эту** строку — категорийное заявление бьёт
    # поисковое, — поэтому это приём, а не отказ (`run.py:222` трактует исход так же).
    # Отказы — только `duplicate` и `conflict`. Трактовка вытеснения как отказа
    # систематически выбрасывала бы из перестройки именно категорийные строки: манифест
    # перебирается в своём порядке, поисковая строка лежит раньше и успевает в индекс.
    # Вытесненная строка при этом остаётся в перестроенном манифесте: она уже дописана, а
    # канала вердиктов (`verdicts.dedup.csv` живого прогона) у перестройки нет. Так и
    # нужно: молча удалять строку провенанса автомат не должен, спор двух заявлений уходит
    # человеку — как §8 обещает для равно подкреплённых, — а терять победителя было хуже.
    if вердикт.status not in ("new", "supersede"):
        return вердикт.status, None

    путь = crop_destination(deps.root, row)
    путь.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = encode_jpeg(кроп)
    except RuntimeError:
        return "write_failed", None
    audit = row.audit if not reset_audit and observed_pixels_equal(путь, encoded) else ""
    try:
        write_jpeg_atomic(путь, encoded)
    except OSError:
        return "write_failed", None
    dedup.add(хеш, row.class_id, путь.name, row.basis)

    return "accepted", replace(
        row,
        phash=хеш,
        plates=итог.plates,
        faces=итог.faces,
        bbox_x=float(решение.box.x),
        bbox_y=float(решение.box.y),
        bbox_w=float(решение.box.w),
        bbox_h=float(решение.box.h),
        bbox_area_frac=float(решение.area_frac),
        det_conf=float(решение.box.conf),
        audit=audit,
        checked=date.today().isoformat(),
    )


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Переderivация манифеста харвеста по кэшу")
    парсер.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    парсер.add_argument(
        "--out",
        type=Path,
        default=None,
        help="целевой манифест (по умолчанию перезаписывает --manifest)",
    )
    парсер.add_argument(
        "--models",
        type=Path,
        default=harvest_dir() / "models",
        help="каталог весов детекторов",
    )
    парсер.add_argument(
        "--data-root",
        type=Path,
        default=harvest_dir(),
        help="~/carlens-data/harvest",
    )
    парсер.add_argument(
        "--reset-audit",
        action="store_true",
        help="сбросить все вердикты аудита, включая неизменившиеся пиксели",
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Префлайт кэша, затем полная переderivация в новый манифест."""
    from pipeline.detectors.yolo import YoloDetector
    from pipeline.harvest.backends import YuNetFaceDetector
    from pipeline.harvest.detectors import (
        DEFAULT_DETECTORS_MANIFEST,
        load_detectors_manifest,
        verify_weights,
    )
    from pipeline.harvest.redaction_profile import (
        CURRENT_REDACTION_PROFILE,
        build_plate_detectors,
    )

    аргументы = build_parser().parse_args(argv)
    строки = load_manifest(аргументы.manifest)
    validate_crop_destinations(строки, аргументы.data_root)

    пропало = preflight(строки, аргументы.data_root)
    if пропало:
        print(f"ПРЕФЛАЙТ: нет оригинала для {len(пропало)} строк — пример {пропало[0]}")
        print("Эти строки не переderivировать без сети. Останов.")
        return 2

    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    verify_weights(пины, аргументы.models)
    plate_detectors = build_plate_detectors(
        CURRENT_REDACTION_PROFILE, models_dir=аргументы.models, pins=пины
    )
    deps = RebuildDeps(
        vehicle_detector=YoloDetector(),
        plate_detectors=plate_detectors,
        face_detector=YuNetFaceDetector(аргументы.models / "face.onnx"),
        root=аргументы.data_root,
    )

    целевой_манифест = аргументы.out if аргументы.out is not None else аргументы.manifest
    временный = целевой_манифест.with_suffix(".rebuild.csv")
    временный.unlink(missing_ok=True)
    dedup = DedupIndex()
    принято = 0
    причины: dict[str, int] = {}
    for строка in строки:
        причина, новая = rebuild_row(строка, deps, dedup, reset_audit=аргументы.reset_audit)
        if новая is None:
            причины[причина] = причины.get(причина, 0) + 1
            continue
        append_row(временный, новая)
        принято += 1
    временный.replace(целевой_манифест)
    print(f"переderivировано: {принято} из {len(строки)}; отсев: {причины}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
