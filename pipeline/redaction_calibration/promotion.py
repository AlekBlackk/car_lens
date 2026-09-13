"""Хешированный снимок данных и журналируемая установка его поверх боевых (§12 спеки).

Продвижение — отдельное внешнее действие. Здесь оно распадается на три шага, между
которыми боевые данные не остаются в промежуточном виде:

1. `build_snapshot` копирует входы в новый каталог снимка и хеширует каждый файл;
2. `validate_snapshot` сверяет хеши и чистоту назначения, ничего не записывая;
3. `promote`/`rollback` ставят снимок на место по журналируемому протоколу.

`promote` перед первым же изменением снимает вытесняемые боевые данные вторым снимком
(`PreviousSnapshotRequest`) — тем же `build_snapshot`. Без этого целью отката оставался бы
только снимок, который кто-то не забыл собрать заранее, а §12 требует отката целым
согласованным снимком.

Назначения живут на двух файловых системах: манифест — файл репозитория на диске Windows,
кропы, сплиты и индекс — каталоги данных в WSL. Одного общего атомарного переименования
между ними не существует, и переносить данные через границу переименованием запрещено.
Поэтому переключение — последовательность атомарных переименований, по одному на каждое
назначение, каждое внутри своей файловой системы, с журналом и восстановлением уже
переключённых назначений при отказе любого следующего. Наблюдаемое состояние назначения
после отработки протокола (включая его собственное восстановление) — целиком старое либо
целиком новое.

Полнота снимка — не булево, а уровень. `base` несёт манифест и кропы, `complete` — ещё и
сплиты с артефактами индекса. Продвинуть можно любой: до пересборки производных боевые
данные состоят из манифеста и кропов. Откатиться на `base` можно только пока производных
на назначении нет; как только появились сплиты или индекс, частичный откат запрещён
спекой §12 — смешение старых входов с новыми производными и есть та ошибка, ради которой
уровень существует.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

from pipeline.redaction_calibration.state import (
    CalibrationStateError,
    PromotionRecord,
    PromotionState,
)

Operation: TypeAlias = Literal["promote", "rollback"]

#: Порядок установки. Манифест первым: он — авторитет строк, и назначение, где кропы уже
#: новые, а манифест ещё старый, читается как «манифест ссылается на пропавшие кропы».
SNAPSHOT_GROUPS: Final = ("manifest", "crops", "splits", "index")
BASE_GROUPS: Final = ("manifest", "crops")
DERIVED_GROUPS: Final = ("splits", "index")

MANIFEST_NAME: Final = "manifest.csv"
PROVENANCE_NAME: Final = "provenance.json"
SPLIT_FILES: Final = ("holdout_ru.parquet", "holdout_ru_report.json", "pool.parquet")
INDEX_FILES: Final = (
    "catalog_index.bin",
    "catalog_meta.json",
    "fewshot_metrics.json",
    "report.json",
)

_SNAPSHOT_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-fA-F]{64}")
_GIT_SHA_PATTERN: Final = re.compile(r"[0-9a-fA-F]{40}")
_TOKEN_LENGTH: Final = 16
_SCHEMA_VERSION: Final = 1


class PromotionError(RuntimeError):
    """Снимок неполон, назначение не чисто или установка не может продолжаться."""


class SnapshotTier(StrEnum):
    """Уровень полноты снимка: только входы или входы вместе с производными."""

    BASE = "base"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Опись снимка: что в нём лежит, с какими хешами и по каким свидетелям собрано."""

    snapshot_id: str
    tier: SnapshotTier
    created_at: str
    root: Path
    artifacts: tuple[str, ...]
    hashes: dict[str, str]
    input_witnesses: dict[str, str]

    @property
    def result_sha256(self) -> str:
        """SHA256 описи — тот самый `snapshot_result_sha256` автомата продвижения.

        Считается по содержимому, а не по расположению: перенос каталога снимка не должен
        превращать его в другой снимок.
        """
        return _document_sha256(snapshot_document(self))

    @property
    def groups(self) -> tuple[str, ...]:
        """Группы назначений, которые этот снимок способен установить."""
        return BASE_GROUPS if self.tier is SnapshotTier.BASE else SNAPSHOT_GROUPS


@dataclass(frozen=True, slots=True)
class PromotionDestinations:
    """Боевые адреса установки. Все — параметры: калибровка не знает боевых путей сама."""

    manifest: Path
    crops: Path
    journal: Path
    splits: Path | None = None
    index: Path | None = None

    def path(self, group: str) -> Path | None:
        """Путь назначения группы; `None` — группа на этом назначении не адресована."""
        if group == "manifest":
            return self.manifest
        if group == "crops":
            return self.crops
        if group == "splits":
            return self.splits
        if group == "index":
            return self.index
        raise PromotionError(f"неизвестная группа назначения: {group}")


@dataclass(frozen=True, slots=True)
class PreviousSnapshotRequest:
    """Куда продвижение обязано сложить вытесняемые боевые данные, прежде чем их вытеснить.

    Спека §12 требует, чтобы после начала производных шагов откат выполнялся целым
    согласованным снимком. Значит снимок вытесняемого состояния обязано делать само
    продвижение: иначе единственной целью отката остаётся снимок, который кто-то не забыл
    собрать заранее. Свидетели описывают *продвигаемый прочь* профиль — тот, что сейчас
    стоит на боевых данных, — и их знает вызывающий, а не этот модуль.
    """

    root: Path
    snapshot_id: str
    witnesses: Mapping[str, str]
    created_at: str


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Что именно было установлено; его SHA256 — свидетель для автомата продвижения."""

    operation: Operation
    snapshot_id: str
    tier: SnapshotTier
    switched: tuple[str, ...]
    snapshot_result_sha256: str
    previous_snapshot_id: str | None = None
    previous_snapshot_result_sha256: str | None = None

    @property
    def result_sha256(self) -> str:
        return _document_sha256(promotion_result_document(self))


def snapshot_document(snapshot: SnapshotManifest) -> dict[str, Any]:
    """Каноническое содержимое описи — ровно то, что лежит в `provenance.json`."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "tier": snapshot.tier.value,
        "created_at": snapshot.created_at,
        "artifacts": list(snapshot.artifacts),
        "hashes": dict(sorted(snapshot.hashes.items())),
        "input_witnesses": dict(sorted(snapshot.input_witnesses.items())),
    }


def promotion_result_document(result: PromotionResult) -> dict[str, Any]:
    """Каноническое содержимое результата установки."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "operation": result.operation,
        "snapshot_id": result.snapshot_id,
        "tier": result.tier.value,
        "switched": list(result.switched),
        "snapshot_result_sha256": result.snapshot_result_sha256,
        "previous_snapshot_id": result.previous_snapshot_id,
        "previous_snapshot_result_sha256": result.previous_snapshot_result_sha256,
    }


def confirmation_token(snapshot: SnapshotManifest, *, operation: Operation) -> str:
    """Токен подтверждения, привязанный к снимку и к операции.

    Привязка к операции не косметика: подтверждение отката не должно случайно
    сработать как подтверждение продвижения того же снимка.
    """
    digest = hashlib.sha256(f"{operation}:{snapshot.result_sha256}".encode()).hexdigest()
    return digest[:_TOKEN_LENGTH]


def _document_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{payload}\n".encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _tree_files(root: Path, *, label: str) -> dict[str, Path]:
    """Обычные файлы дерева по относительным posix-путям; ссылки и устройства — отказ.

    Снимок обязан нести байты, а не указатели на них: символьная ссылка, скопированная
    как ссылка, после установки указывала бы на данные другой версии.
    """
    collected: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise PromotionError(f"{label}: не обычный файл {path.relative_to(root).as_posix()}")
        collected[path.relative_to(root).as_posix()] = path
    if not collected:
        raise PromotionError(f"{label}: пусто в {root}")
    return collected


def _require_files(files: Mapping[str, Path], required: Sequence[str], *, label: str) -> None:
    missing = sorted(set(required) - set(files))
    if missing:
        raise PromotionError(f"{label}: в снимке нет обязательных файлов {missing}")


def _check_index_provenance(index_files: Mapping[str, Path], pool_sha256: str) -> None:
    """Сверить связи производного индекса: со своими файлами и с pool, из которого собран.

    `report.json` сборки индекса называет хеши `catalog_index.bin` и `catalog_meta.json`,
    но не называет сплит-источник — эту связь снимок обязан принести свидетелем снаружи.
    """
    report = _read_json(index_files["report.json"])
    links = (("catalog_index.bin", "index_sha256"), ("catalog_meta.json", "meta_sha256"))
    for name, field in links:
        declared = str(report.get(field, ""))
        actual = _sha256(index_files[name])
        if declared != actual:
            raise PromotionError(
                f"индекс: report.json {field} = {declared or 'нет'}, а {name} даёт {actual}"
            )
    declared_pool = report.get("pool_sha256")
    if declared_pool is not None and str(declared_pool) != pool_sha256:
        raise PromotionError(
            f"индекс: report.json pool_sha256 = {declared_pool}, а в снимке pool {pool_sha256}"
        )


def _validate_witnesses(witnesses: Mapping[str, str], *, complete: bool) -> dict[str, str]:
    """Потребовать полный набор свидетелей: профиль, геометрия и версия кода (§13 спеки).

    `git_sha` стоит наравне с хешем профиля намеренно: производный результат без версии
    кода спека запрещает, а тот же снимок, собранный другим коммитом, — другой результат.
    Формат тот же, что у `baseline.create_baseline_passport`: полные 40 шестнадцатеричных.
    """
    required = [
        "canonical_sha256",
        "geometry_witness_sha256",
        "git_sha",
        "profile_id",
        "profile_sha256",
    ]
    if complete:
        required.append("index_pool_sha256")
    missing = sorted(set(required) - set(witnesses))
    if missing:
        raise PromotionError(f"нет свидетелей входа: {missing}")
    for name in ("profile_sha256", "geometry_witness_sha256", "canonical_sha256"):
        if _SHA256_PATTERN.fullmatch(witnesses[name]) is None:
            raise PromotionError(f"{name}: нужен полный шестнадцатеричный SHA256")
    if _GIT_SHA_PATTERN.fullmatch(witnesses["git_sha"]) is None:
        raise PromotionError("git_sha: нужен полный Git SHA из 40 шестнадцатеричных символов")
    return dict(witnesses)


def build_snapshot(
    snapshot_root: Path,
    *,
    snapshot_id: str,
    manifest_path: Path,
    crops_root: Path,
    splits_root: Path | None = None,
    index_root: Path | None = None,
    witnesses: Mapping[str, str],
    created_at: str,
) -> SnapshotManifest:
    """Скопировать входы в новый каталог снимка, захешировать всё и записать опись.

    Корни передаются явно и по отдельности, а не выводятся из workspace: тем же кодом
    снимается и калибровочный workspace, и боевые данные — снимок боевых данных и есть
    то, на что откатывается неудачное продвижение.
    """
    if _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None:
        raise PromotionError(f"небезопасный snapshot_id: {snapshot_id!r}")
    if snapshot_root.exists():
        raise PromotionError(f"каталог снимка уже существует: {snapshot_root}")
    if (splits_root is None) != (index_root is None):
        raise PromotionError("сплиты и индекс входят в снимок только вместе")
    complete = splits_root is not None
    checked_witnesses = _validate_witnesses(witnesses, complete=complete)

    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{snapshot_root.name}.staging-", dir=snapshot_root.parent)
    )
    try:
        artifacts = _copy_inputs(
            staging,
            manifest_path=manifest_path,
            crops_root=crops_root,
            splits_root=splits_root,
            index_root=index_root,
            witnesses=checked_witnesses,
        )
        snapshot = SnapshotManifest(
            snapshot_id=snapshot_id,
            tier=SnapshotTier.COMPLETE if complete else SnapshotTier.BASE,
            created_at=created_at,
            root=snapshot_root,
            artifacts=tuple(sorted(artifacts)),
            hashes={name: _sha256(staging / name) for name in sorted(artifacts)},
            input_witnesses=checked_witnesses,
        )
        _write_json(staging / PROVENANCE_NAME, snapshot_document(snapshot))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Каталог снимка появляется целиком одним переименованием внутри своей файловой
    # системы: половина снимка не должна выглядеть как снимок.
    os.replace(staging, snapshot_root)
    return snapshot


def _copy_inputs(
    staging: Path,
    *,
    manifest_path: Path,
    crops_root: Path,
    splits_root: Path | None,
    index_root: Path | None,
    witnesses: Mapping[str, str],
) -> list[str]:
    """Разложить входы по раскладке снимка §13 и проверить обязательный состав групп."""
    if not manifest_path.is_file():
        raise PromotionError(f"нет манифеста: {manifest_path}")
    shutil.copy2(manifest_path, staging / MANIFEST_NAME)
    artifacts = [MANIFEST_NAME]

    for group, source, required in (
        ("crops", crops_root, ()),
        ("splits", splits_root, SPLIT_FILES),
        ("index", index_root, INDEX_FILES),
    ):
        if source is None:
            continue
        if not source.is_dir():
            raise PromotionError(f"{group}: нет каталога {source}")
        files = _tree_files(source, label=group)
        _require_files(files, required, label=group)
        for relative, path in files.items():
            target = staging / group / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            artifacts.append(f"{group}/{relative}")

    if index_root is not None and splits_root is not None:
        pool_sha256 = _sha256(staging / "splits" / "pool.parquet")
        if witnesses["index_pool_sha256"] != pool_sha256:
            raise PromotionError(
                f"index_pool_sha256 = {witnesses['index_pool_sha256']}, "
                f"а pool.parquet снимка даёт {pool_sha256}"
            )
        _check_index_provenance(
            {name: staging / "index" / name for name in INDEX_FILES}, pool_sha256
        )
    return artifacts


def load_snapshot(snapshot_root: Path) -> SnapshotManifest:
    """Прочитать опись ранее собранного снимка без проверки его файлов."""
    provenance = snapshot_root / PROVENANCE_NAME
    if not provenance.is_file():
        raise PromotionError(f"нет описи снимка: {provenance}")
    document = _read_json(provenance)
    if document.get("schema_version") != _SCHEMA_VERSION:
        raise PromotionError(f"неизвестная версия описи снимка: {document.get('schema_version')}")
    return SnapshotManifest(
        snapshot_id=str(document["snapshot_id"]),
        tier=SnapshotTier(document["tier"]),
        created_at=str(document["created_at"]),
        root=snapshot_root,
        artifacts=tuple(str(name) for name in document["artifacts"]),
        hashes={str(name): str(value) for name, value in document["hashes"].items()},
        input_witnesses={
            str(name): str(value) for name, value in document["input_witnesses"].items()
        },
    )


def _group_of(artifact: str) -> str:
    return "manifest" if artifact == MANIFEST_NAME else artifact.split("/", 1)[0]


def _artifacts_of(snapshot: SnapshotManifest, group: str) -> tuple[str, ...]:
    return tuple(name for name in snapshot.artifacts if _group_of(name) == group)


def _sidecar(destination: Path, *, snapshot_id: str, operation: str, kind: str) -> Path:
    """Служебный путь рядом с назначением — на той же файловой системе, что и оно."""
    return destination.parent / f".{destination.name}.{operation}-{snapshot_id}.{kind}"


def validate_snapshot(snapshot: SnapshotManifest, destinations: PromotionDestinations) -> None:
    """Сверить хеши снимка и чистоту назначения, не записав в назначение ни байта."""
    for name in snapshot.artifacts:
        path = snapshot.root / name
        if not path.is_file():
            raise PromotionError(f"в снимке нет файла {name}")
        actual = _sha256(path)
        if actual != snapshot.hashes[name]:
            raise PromotionError(f"{name}: хеш {actual} не совпал с описью {snapshot.hashes[name]}")

    if destinations.journal.exists():
        raise PromotionError(
            f"журнал предыдущей установки не убран: {destinations.journal}; "
            "разберите оборванный прогон вручную"
        )
    for group in snapshot.groups:
        destination = destinations.path(group)
        if destination is None:
            raise PromotionError(f"снимок несёт группу {group}, но её назначение не задано")
        for operation in ("promote", "rollback"):
            for kind in ("staging", "previous"):
                leftover = _sidecar(
                    destination, snapshot_id=snapshot.snapshot_id, operation=operation, kind=kind
                )
                if leftover.exists():
                    raise PromotionError(f"назначение занято остатком {kind}: {leftover}")
        if not destination.parent.is_dir():
            raise PromotionError(f"нет каталога назначения: {destination.parent}")


def _has_derived_artifacts(destinations: PromotionDestinations) -> bool:
    """Есть ли на назначении хоть один производный артефакт сплитов или индекса."""
    for group, required in (("splits", SPLIT_FILES), ("index", INDEX_FILES)):
        root = destinations.path(group)
        if root is not None and root.is_dir() and any((root / name).exists() for name in required):
            return True
    return False


def _require_token(
    snapshot: SnapshotManifest, *, operation: Operation, confirmation_token_value: str
) -> None:
    if confirmation_token_value != confirmation_token(snapshot, operation=operation):
        raise PromotionError(
            f"токен подтверждения не подходит для {operation} снимка {snapshot.snapshot_id}"
        )


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def promote(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    *,
    record: PromotionRecord,
    confirmation_token: str,
    previous: PreviousSnapshotRequest,
    checkpoint: Callable[[str], None] | None = None,
) -> PromotionResult:
    """Снять вытесняемые данные снимком, затем установить новый снимок поверх боевых.

    Автомат продвижения здесь только проверяется, но не двигается: перевод записи в
    `promoted` остаётся за вызывающим, который свидетельствует его `result_sha256`.
    Так один и тот же результат нельзя записать в автомат дважды по-разному.
    """
    if record.state is not PromotionState.PROMOTABLE:
        raise CalibrationStateError(
            f"продвижение разрешено только из promotable, а запись в {record.state.value}"
        )
    if record.snapshot_result_sha256 != snapshot.result_sha256:
        raise CalibrationStateError(
            "snapshot_result_sha256 записи не совпадает с описью продвигаемого снимка"
        )
    _require_same_profile(record, snapshot)
    _require_token(snapshot, operation="promote", confirmation_token_value=confirmation_token)
    validate_snapshot(snapshot, destinations)

    # Снимок вытесняемых данных снимается до единого изменения назначения: если его нельзя
    # собрать, продвигать нечего — откатываться будет некуда.
    retained = capture_destinations(destinations, previous)
    return _install(
        snapshot,
        destinations,
        operation="promote",
        confirmation_token_value=confirmation_token,
        checkpoint=checkpoint,
        previous_snapshot_id=retained.snapshot_id,
        previous_snapshot_result_sha256=retained.result_sha256,
    )


def _require_same_profile(record: PromotionRecord, snapshot: SnapshotManifest) -> None:
    """Не дать записи, годной для профиля A, продвинуть снимок, названный профилем B."""
    witnesses = snapshot.input_witnesses
    for field, witness in (
        ("profile_id", "profile_id"),
        ("profile_sha256", "profile_sha256"),
        ("geometry_witness_sha256", "geometry_witness_sha256"),
    ):
        expected = witnesses.get(witness)
        actual = getattr(record, field)
        if actual != expected:
            raise CalibrationStateError(
                f"{field} записи ({actual}) не совпадает со свидетелем снимка ({expected})"
            )


def capture_destinations(
    destinations: PromotionDestinations, previous: PreviousSnapshotRequest
) -> SnapshotManifest:
    """Снять текущее содержимое назначения снимком — целью будущего отката.

    Уровень снимка выводится из назначения: есть производные — `complete`, нет — `base`.
    Ровно это и требуется, чтобы откат после производных шагов был целым (§12), а откат
    до них оставался разрешённым.

    `index_pool_sha256` для снимка *уже установленных* данных считается по их же
    `pool.parquet`, а не приносится снаружи: пара «индекс и pool» здесь не заявляется, а
    фиксируется как есть. Внутренние связи индекса (`index_sha256`, `meta_sha256`) при
    этом сверяются полностью, как и для любого другого снимка.
    """
    derived = _has_derived_artifacts(destinations)
    splits_root = destinations.splits if derived else None
    index_root = destinations.index if derived else None
    witnesses = dict(previous.witnesses)
    if splits_root is not None:
        witnesses["index_pool_sha256"] = _sha256(splits_root / "pool.parquet")
    return build_snapshot(
        previous.root,
        snapshot_id=previous.snapshot_id,
        manifest_path=destinations.manifest,
        crops_root=destinations.crops,
        splits_root=splits_root,
        index_root=index_root,
        witnesses=witnesses,
        created_at=previous.created_at,
    )


def rollback(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    *,
    confirmation_token: str,
    checkpoint: Callable[[str], None] | None = None,
) -> PromotionResult:
    """Вернуть назначение к сохранённому снимку целиком.

    Запись автомата здесь не требуется намеренно: обратного ребра у автомата §11.3 нет,
    восстановление оформляется новой кампанией (§12), а единственным человеческим
    шлагбаумом остаётся токен подтверждения.
    """
    if snapshot.tier is SnapshotTier.BASE and _has_derived_artifacts(destinations):
        raise PromotionError(
            "частичный откат запрещён: на назначении уже есть сплиты или индекс, "
            "а снимок несёт только манифест и кропы (§12 спеки)"
        )
    return _install(
        snapshot,
        destinations,
        operation="rollback",
        confirmation_token_value=confirmation_token,
        checkpoint=checkpoint,
    )


def _install(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    *,
    operation: Operation,
    confirmation_token_value: str,
    checkpoint: Callable[[str], None] | None,
    previous_snapshot_id: str | None = None,
    previous_snapshot_result_sha256: str | None = None,
) -> PromotionResult:
    """Журналируемая установка: подготовка, журнал, переключения, уборка.

    Токен и снимок проверяются повторно и здесь: `rollback` заходит сюда напрямую, а
    `promote` — после снятия вытесняемых данных, и обе двери обязаны быть заперты.
    """
    _require_token(snapshot, operation=operation, confirmation_token_value=confirmation_token_value)
    validate_snapshot(snapshot, destinations)
    step = checkpoint if checkpoint is not None else _no_checkpoint
    groups = snapshot.groups

    try:
        for group in groups:
            _stage_group(snapshot, destinations, group, operation=operation)
            step(f"staged:{group}")
    except BaseException:
        _discard_staging(snapshot, destinations, groups, operation=operation)
        raise

    _write_json(
        destinations.journal,
        {
            "schema_version": _SCHEMA_VERSION,
            "operation": operation,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_result_sha256": snapshot.result_sha256,
            "tier": snapshot.tier.value,
            "groups": list(groups),
        },
    )
    try:
        step("journal_written")
    except BaseException:
        _discard_staging(snapshot, destinations, groups, operation=operation)
        destinations.journal.unlink()
        raise

    switched: list[str] = []
    try:
        for group in groups:
            step(f"before_switch:{group}")
            _switch_group(snapshot, destinations, group, operation=operation, step=step)
            switched.append(group)
            step(f"after_switch:{group}")
    except BaseException:
        _restore(snapshot, destinations, switched, operation=operation)
        _discard_staging(snapshot, destinations, groups, operation=operation)
        # Журнал остаётся намеренно: он единственное свидетельство оборванной установки и
        # он же блокирует слепой повтор через `validate_snapshot`.
        raise

    for group in groups:
        destination = _required_destination(destinations, group)
        _remove(
            _sidecar(
                destination,
                snapshot_id=snapshot.snapshot_id,
                operation=operation,
                kind="previous",
            )
        )
    destinations.journal.unlink()
    return PromotionResult(
        operation=operation,
        snapshot_id=snapshot.snapshot_id,
        tier=snapshot.tier,
        switched=tuple(switched),
        snapshot_result_sha256=snapshot.result_sha256,
        previous_snapshot_id=previous_snapshot_id,
        previous_snapshot_result_sha256=previous_snapshot_result_sha256,
    )


def _no_checkpoint(step: str) -> None:
    """Пустая точка наблюдения протокола; тесты подменяют её инъекцией отказа."""


def _required_destination(destinations: PromotionDestinations, group: str) -> Path:
    destination = destinations.path(group)
    if destination is None:
        raise PromotionError(f"назначение группы {group} не задано")
    return destination


def _stage_group(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    group: str,
    *,
    operation: Operation,
) -> None:
    """Разложить группу рядом с её назначением и пересверить хеши уже скопированного."""
    destination = _required_destination(destinations, group)
    staging = _sidecar(
        destination, snapshot_id=snapshot.snapshot_id, operation=operation, kind="staging"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in _artifacts_of(snapshot, group):
        source = snapshot.root / name
        target = staging if group == "manifest" else staging / name.split("/", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        actual = _sha256(target)
        if actual != snapshot.hashes[name]:
            raise PromotionError(f"{name}: подготовленная копия дала хеш {actual}")


def _switch_group(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    group: str,
    *,
    operation: Operation,
    step: Callable[[str], None],
) -> None:
    """Переключить одно назначение внутри его собственной файловой системы.

    Переименований здесь два, и между ними назначение не существует: старое уже уехало в
    `.previous`, новое ещё не встало. Это единственное окно во всём протоколе, где отказ
    операционной системы способен уничтожить назначение, поэтому окно закрывается прямо
    здесь — своим же обратным переименованием, — а не общим `_restore` снаружи: снаружи
    группа ещё не числится переключённой, и восстанавливать её было бы некому.
    """
    destination = _required_destination(destinations, group)
    staging = _sidecar(
        destination, snapshot_id=snapshot.snapshot_id, operation=operation, kind="staging"
    )
    previous = _sidecar(
        destination, snapshot_id=snapshot.snapshot_id, operation=operation, kind="previous"
    )
    moved_aside = False
    if destination.exists():
        os.replace(destination, previous)
        moved_aside = True
    try:
        step(f"switching:{group}")
        os.replace(staging, destination)
    except BaseException:
        if moved_aside:
            os.replace(previous, destination)
        raise


def _restore(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    switched: Sequence[str],
    *,
    operation: Operation,
) -> None:
    """Вернуть уже переключённые назначения к прежнему содержимому, в обратном порядке."""
    for group in reversed(list(switched)):
        destination = _required_destination(destinations, group)
        previous = _sidecar(
            destination, snapshot_id=snapshot.snapshot_id, operation=operation, kind="previous"
        )
        _remove(destination)
        if previous.exists():
            os.replace(previous, destination)


def _discard_staging(
    snapshot: SnapshotManifest,
    destinations: PromotionDestinations,
    groups: Sequence[str],
    *,
    operation: Operation,
) -> None:
    for group in groups:
        destination = destinations.path(group)
        if destination is None:
            continue
        _remove(
            _sidecar(
                destination, snapshot_id=snapshot.snapshot_id, operation=operation, kind="staging"
            )
        )
