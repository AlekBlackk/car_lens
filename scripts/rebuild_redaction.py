"""Переderivация манифеста харвеста по кэшу оригиналов — без единого запроса к Commons.

Смена детектора или правил редакции обязана пересчитываться по `raw/`, а не
перекачиванием (§8 дизайна). Это полная переderivация от стадии кропа вниз:
редакция → pHash → дедуп → запись. Заливка номера меняет пиксели кропа, значит меняется
pHash, значит меняется результат дедупа — покрытие и список дефицитных классов
пересобираются заново, это не патч колонок.

Вход только читается. Результат целиком ложится в новый отсутствующий `--out-root`:
манифест, `crops/`, `rejects.csv`, паспорт профиля и отчёт. Пересборка «поверх» боевых
данных невозможна по устройству — калибровка перебирает профили, и прогон, изменивший
вход, лишил бы следующий профиль того же основания для сравнения.

Старая отметка `audit` переносится только при точном равенстве декодированных пикселей
старого и нового JPEG. Отсутствующий или битый старый кроп безопасно сбрасывает отметку.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import cv2
import numpy as np

from pipeline.clean import phash_of
from pipeline.crop import VEHICLE_CLASSES, Detector, crop_image, select_primary
from pipeline.harvest.cache import cache_path, decode_image
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.detectors import DetectorPin
from pipeline.harvest.jpeg import encode_jpeg, observed_pixels_equal, write_jpeg_atomic
from pipeline.harvest.manifest import (
    DEFAULT_MANIFEST_PATH,
    HarvestRow,
    load_manifest,
    write_manifest,
)
from pipeline.harvest.redact import redact_frame
from pipeline.harvest.redaction_profile import RedactionProfile, profile_id
from pipeline.harvest.run import is_crop_over_redacted
from pipeline.paths import harvest_dir

# Имена, которые workspace резервирует за собой. `--out` не может занять ни одно из них:
# манифест, перезаписанный отчётом, выглядел бы как пустая пересборка.
RESERVED_WORKSPACE_NAMES = frozenset(
    {"crops", "rejects.csv", "observations.csv", "report.json", "profile.json"}
)


class RebuildError(ValueError):
    """Манифест нельзя безопасно пересобрать без неоднозначной записи кропов."""


def safe_component(value: str, kind: str) -> str:
    """Проверить, что строка манифеста — ровно один сегмент пути, а не маршрут наружу.

    Проверка идёт по правилам Windows даже в WSL: `C:crop.jpg` и `\\\\server\\share` —
    пути только там, и манифест, собранный на одной из двух половин рабочего места,
    обязан отвергаться на обеих одинаково.
    """
    имя = PureWindowsPath(value)
    if (
        имя.is_absolute()
        or имя.drive
        or имя.name != value
        or "/" in value
        or "\\" in value
        or ":" in value
        or value in {"", ".", ".."}
    ):
        raise RebuildError(f"небезопасное {kind}: {value!r}")
    return value


def crop_destination(root: Path, row: HarvestRow) -> Path:
    """Безопасный путь кропа, сохраняющий authoritative имена класса и файла из манифеста.

    Класс проверяется наравне с именем файла. `load_manifest` не ограничивает `class_id`
    ничем, а `validate` требует от него только непустоты, — значит один сегмент `..` в
    колонке класса уводил бы запись кропа за пределы workspace, мимо всей изоляции.
    """
    class_id = safe_component(row.class_id, "имя класса")
    file = safe_component(row.file, "имя кропа")
    return root / "crops" / class_id / file


def validate_crop_destinations(rows: Sequence[HarvestRow], root: Path) -> None:
    """До пересборки отвергнуть небезопасные и повторяющиеся пути кропов."""
    пути: set[tuple[str, str]] = set()
    for row in rows:
        crop_destination(root, row)
        путь = (row.class_id.casefold(), row.file.casefold())
        if путь in пути:
            raise RebuildError(f"повтор пути кропа: {row.class_id}/{row.file}")
        пути.add(путь)


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Зарезервированные пути одного изолированного результата пересборки."""

    root: Path
    manifest: Path
    crops: Path
    rejects: Path
    observations: Path
    report: Path
    profile: Path


@dataclass(frozen=True, slots=True)
class RebuildDeps:
    """Внешние зависимости и раздельные корни чтения/записи переderivации."""

    vehicle_detector: Detector
    plate_detectors: tuple[Detector, ...]
    face_detector: Detector
    input_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class RowOutcome:
    """Что переderivация сделала с одной строкой входного манифеста."""

    source: HarvestRow
    reason: str
    rebuilt: HarvestRow | None


@dataclass(frozen=True, slots=True)
class AuditCounts:
    """Судьба отметок H3 входного манифеста, разложенная на непересекающиеся корзины."""

    in_input: int
    preserved: int
    reset: int
    missing: int
    corrupt: int
    dropped_with_row: int

    def __post_init__(self) -> None:
        разложено = (
            self.preserved + self.reset + self.missing + self.corrupt + self.dropped_with_row
        )
        if разложено != self.in_input:
            raise RebuildError(f"отметки аудита не сходятся: {разложено} против {self.in_input}")


def validate_workspace_roots(out_root: Path, data_root: Path) -> tuple[Path, Path]:
    """Разрешить корни и отвергнуть повторный, вложенный или несуществующий workspace."""
    try:
        input_root = data_root.resolve(strict=True)
    except OSError as ошибка:
        raise RebuildError(f"каталог данных не существует: {data_root}") from ошибка
    output_root = out_root.resolve(strict=False)
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
        raise RebuildError("корни ввода и workspace не могут совпадать или быть вложенными")
    if output_root.exists():
        raise RebuildError(f"workspace уже существует: {output_root}")
    if not output_root.parent.is_dir():
        raise RebuildError(f"родитель workspace не существует: {output_root.parent}")
    return input_root, output_root


def resolve_manifest_target(output_root: Path, out: Path | None) -> Path:
    """Целевой манифест внутри workspace; по умолчанию `<out-root>/manifest.csv`.

    Относительный `--out` считается от workspace, а не от текущего каталога: иначе одна и
    та же команда, запущенная из двух разных мест, писала бы манифест в разные файлы.
    Наружу цель не выходит ни через `..`, ни абсолютным путём, ни симлинком — `resolve`
    снимает и то, и другое, и остаётся проверить вложенность уже настоящего пути.
    """
    if out is None:
        return output_root / "manifest.csv"
    цель = (out if out.is_absolute() else output_root / out).resolve(strict=False)
    if цель == output_root or not цель.is_relative_to(output_root):
        raise RebuildError(f"--out ведёт за пределы workspace {output_root}: {out}")
    if цель.relative_to(output_root).parts[0] in RESERVED_WORKSPACE_NAMES:
        raise RebuildError(f"--out занимает служебное имя workspace: {out}")
    return цель


def prepare_workspace(out_root: Path, data_root: Path) -> WorkspacePaths:
    """Проверить изоляцию и создать новый workspace без продолжения старого."""
    _, output_root = validate_workspace_roots(out_root, data_root)
    output_root.mkdir()
    crops = output_root / "crops"
    crops.mkdir()
    return WorkspacePaths(
        root=output_root,
        manifest=output_root / "manifest.csv",
        crops=crops,
        rejects=output_root / "rejects.csv",
        observations=output_root / "observations.csv",
        report=output_root / "report.json",
        profile=output_root / "profile.json",
    )


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
    сырьё = cache_path(deps.input_root, row.source, row.sha256, "jpg").read_bytes()
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

    старый_путь = crop_destination(deps.input_root, row)
    новый_путь = crop_destination(deps.output_root, row)
    новый_путь.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = encode_jpeg(кроп)
    except RuntimeError:
        return "write_failed", None
    audit = row.audit if not reset_audit and observed_pixels_equal(старый_путь, encoded) else ""
    try:
        write_jpeg_atomic(новый_путь, encoded)
    except OSError:
        return "write_failed", None
    dedup.add(хеш, row.class_id, новый_путь.name, row.basis)

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
        "--out-root",
        type=Path,
        required=True,
        help="новый отсутствующий каталог изолированного workspace",
    )
    парсер.add_argument(
        "--out",
        type=Path,
        default=None,
        help="имя манифеста внутри --out-root (по умолчанию manifest.csv)",
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


def current_git_sha() -> str:
    """Полный Git SHA, которым помечается профильный workspace."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    ).stdout.strip()


def current_timestamp() -> str:
    """UTC-время паспорта с явным суффиксом Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_rejects(path: Path, rejects: Sequence[tuple[HarvestRow, str]]) -> None:
    fields = ("file", "class_id", "source", "source_item_id", "sha256", "reason")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row, reason in rejects:
            writer.writerow(
                {
                    "file": row.file,
                    "class_id": row.class_id,
                    "source": row.source,
                    "source_item_id": row.source_item_id,
                    "sha256": row.sha256,
                    "reason": reason,
                }
            )


def _old_crop_state(row: HarvestRow, input_root: Path) -> str:
    path = crop_destination(input_root, row)
    if not path.exists():
        return "missing"
    try:
        data = path.read_bytes()
        if not data:
            return "corrupt"
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return "corrupt"
    return "present" if decoded is not None else "corrupt"


def _audit_counts(outcomes: Sequence[RowOutcome], *, input_root: Path) -> AuditCounts:
    """Разложить отметки H3 по судьбе, не переигрывая решение `rebuild_row` заново.

    Перенос читается с самой построенной строки: `rebuild_row` либо копирует отметку
    входа, либо ставит пустую, третьего у него нет, — поэтому непустая отметка на выходе и
    есть «перенесена». Считать здесь тот же предикат во второй раз нельзя: офлайн-копия
    правила расходится с кодом молча и печатает правдоподобное неверное число.

    Старый кроп открывается только там, где отметка потеряна, и только чтобы назвать
    причину. Это диагноз, а не решение: решение уже принято и уже записано.
    """
    in_input = preserved = reset = missing = corrupt = dropped = 0
    for исход in outcomes:
        if not исход.source.audit:
            continue
        in_input += 1
        if исход.rebuilt is None:
            dropped += 1
        elif исход.rebuilt.audit:
            preserved += 1
        else:
            состояние = _old_crop_state(исход.source, input_root)
            if состояние == "missing":
                missing += 1
            elif состояние == "corrupt":
                corrupt += 1
            else:
                reset += 1
    return AuditCounts(in_input, preserved, reset, missing, corrupt, dropped)


def _profile_document(
    *,
    profile: RedactionProfile,
    pins: Mapping[str, DetectorPin],
    git_sha: str,
    stage_sha256: str,
    input_manifest: Path,
    input_root: Path,
) -> dict[str, Any]:
    return {
        "profile": asdict(profile),
        "profile_id": profile_id(
            profile,
            pins=pins,
            git_sha=git_sha,
            stage_sha256=stage_sha256,
        ),
        "git_sha": git_sha,
        "stage_sha256": stage_sha256,
        "detector_pins": {name: asdict(pin) for name, pin in sorted(pins.items())},
        "input_manifest_sha256": _sha256(input_manifest),
        "input_root": str(input_root),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "timestamp": current_timestamp(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Префлайт входа целиком, затем переderivация в новый изолированный workspace."""
    # Р-15: тяжёлое — веса и `ultralytics` — импортируется только на реальном прогоне.
    # `RedactionProfile` и `profile_id` чисты и взяты сверху; остальные три имени остаются
    # здесь, чтобы тесты подменяли их на модуле определения, а не на копии в этом файле.
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
        detector_stage_sha256,
    )

    аргументы = build_parser().parse_args(argv)

    # Всё, что способно упасть, срабатывает до создания workspace. Порядок не косметика:
    # workspace обязан быть новым и пустым, поэтому прогон, оборвавшийся уже после
    # `mkdir`, оставил бы каталог, который блокирует собственный повтор.
    input_root, output_root = validate_workspace_roots(аргументы.out_root, аргументы.data_root)
    целевой_манифест = resolve_manifest_target(output_root, аргументы.out)
    манифест = аргументы.manifest.resolve(strict=True)
    строки = load_manifest(манифест)
    validate_crop_destinations(строки, input_root)

    пропало = preflight(строки, input_root)
    if пропало:
        print(f"ПРЕФЛАЙТ: нет оригинала для {len(пропало)} строк — пример {пропало[0]}")
        print("Эти строки не переderivировать без сети. Останов.")
        return 2

    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    verify_weights(пины, аргументы.models)
    паспорт = _profile_document(
        profile=CURRENT_REDACTION_PROFILE,
        pins=пины,
        git_sha=current_git_sha(),
        stage_sha256=detector_stage_sha256(),
        input_manifest=манифест,
        input_root=input_root,
    )
    plate_detectors = build_plate_detectors(
        CURRENT_REDACTION_PROFILE, models_dir=аргументы.models, pins=пины
    )
    vehicle_detector = YoloDetector()
    face_detector = YuNetFaceDetector(аргументы.models / "face.onnx")

    workspace = replace(
        prepare_workspace(аргументы.out_root, input_root), manifest=целевой_манифест
    )
    deps = RebuildDeps(
        vehicle_detector=vehicle_detector,
        plate_detectors=plate_detectors,
        face_detector=face_detector,
        input_root=input_root,
        output_root=workspace.root,
    )

    dedup = DedupIndex()
    исходы: list[RowOutcome] = []
    for строка in строки:
        причина, новая = rebuild_row(строка, deps, dedup, reset_audit=аргументы.reset_audit)
        исходы.append(RowOutcome(строка, причина, новая))

    accepted = [исход.rebuilt for исход in исходы if исход.rebuilt is not None]
    rejects = [(исход.source, исход.reason) for исход in исходы if исход.rebuilt is None]
    reasons = Counter(исход.reason for исход in исходы if исход.rebuilt is None)

    write_manifest(workspace.manifest, accepted)
    _write_rejects(workspace.rejects, rejects)
    _write_json(workspace.profile, паспорт)
    # Отчёт называет числа прогона и ничего не утверждает про успех калибровки: решение
    # «профиль годится» принимается не здесь и не по этим полям.
    report = {
        "counts": {
            "input": len(строки),
            "accepted": len(accepted),
            "rejected": len(rejects),
        },
        "reject_reasons": dict(sorted(reasons.items())),
        "audit": asdict(_audit_counts(исходы, input_root=input_root)),
        "output_hashes": {
            workspace.manifest.relative_to(workspace.root).as_posix(): _sha256(workspace.manifest),
            "profile.json": _sha256(workspace.profile),
            "rejects.csv": _sha256(workspace.rejects),
        },
    }
    _write_json(workspace.report, report)

    print(f"переderivировано: {len(accepted)} из {len(строки)}; отсев: {dict(reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
