"""Переderivация манифеста харвеста по кэшу оригиналов — без единого запроса к Commons.

Смена детектора или правил редакции обязана пересчитываться по `raw/`, а не
перекачиванием (§8 дизайна). Это полная переderivация от стадии кропа вниз:
редакция → pHash → дедуп → запись. Заливка номера меняет пиксели кропа, значит меняется
pHash, значит меняется результат дедупа — покрытие и список дефицитных классов
пересобираются заново, это не патч колонок.

Аудит после переderivации начинается с чистого листа: старые отметки `audit`
сбрасываются, потому что относились к другим пикселям.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import cv2

from pipeline.clean import phash_of
from pipeline.crop import VEHICLE_CLASSES, Detector, crop_image, select_primary
from pipeline.harvest.cache import cache_path, decode_image
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.manifest import (
    DEFAULT_MANIFEST_PATH,
    HarvestRow,
    append_row,
    load_manifest,
)
from pipeline.harvest.redact import redact_frame
from pipeline.harvest.run import is_crop_over_redacted
from pipeline.paths import harvest_dir


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
    row: HarvestRow, deps: RebuildDeps, dedup: DedupIndex, keep_audit: bool = False
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

    вердикт = dedup.check(хеш, row.class_id)
    if вердикт.status != "new":
        return вердикт.status, None

    путь = deps.root / "crops" / row.class_id / f"{row.sha256}.jpg"
    путь.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(путь), кроп)
    dedup.add(хеш, row.class_id, путь.name)

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
        audit=row.audit if keep_audit else "",
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
        "--keep-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="сохранять вердикты аудита (Р-11: контур редакции заморожен)",
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Префлайт кэша, затем полная переderivация в новый манифест."""
    from pipeline.detectors.yolo import YoloDetector
    from pipeline.harvest.backends import OnnxBoxDetector, YuNetFaceDetector
    from pipeline.harvest.detectors import (
        DEFAULT_DETECTORS_MANIFEST,
        load_detectors_manifest,
        verify_weights,
    )
    from pipeline.harvest.tiling import TiledDetector

    аргументы = build_parser().parse_args(argv)
    строки = load_manifest(аргументы.manifest)

    пропало = preflight(строки, аргументы.data_root)
    if пропало:
        print(f"ПРЕФЛАЙТ: нет оригинала для {len(пропало)} строк — пример {пропало[0]}")
        print("Эти строки не переderivировать без сети. Останов.")
        return 2

    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    verify_weights(пины, аргументы.models)
    plate_detectors = (
        TiledDetector(OnnxBoxDetector(аргументы.models / f"plate.{пины['plate'].format}")),
        TiledDetector(
            YoloDetector(
                weights=str(аргументы.models / f"plate_hires.{пины['plate_hires'].format}"),
                conf=0.25,
            )
        ),
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
        причина, новая = rebuild_row(строка, deps, dedup, keep_audit=аргументы.keep_audit)
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
