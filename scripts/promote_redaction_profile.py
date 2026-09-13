"""Снять калибровочный workspace снимком и поставить его поверх боевых данных (§12).

Три подкоманды и один принцип: без явного токена подтверждения ни одна из них боевые
данные не меняет.

    uv run python -m scripts.promote_redaction_profile snapshot --snapshot <каталог> ...
    uv run python -m scripts.promote_redaction_profile promote --snapshot <каталог>
    uv run python -m scripts.promote_redaction_profile rollback --snapshot <каталог>

`promote` и `rollback` без `--confirm` — сухой прогон: они проверяют снимок и назначение и
печатают токен, который надо вернуть в `--confirm`. Токен считается по содержимому снимка и
по имени операции, поэтому вслепую скопированный из соседнего прогона не подойдёт.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pipeline.harvest.manifest import DEFAULT_MANIFEST_PATH
from pipeline.paths import crops_dir, data_root, splits_dir
from pipeline.redaction_calibration.promotion import (
    Operation,
    PreviousSnapshotRequest,
    PromotionDestinations,
    PromotionError,
    PromotionResult,
    SnapshotManifest,
    build_snapshot,
    confirmation_token,
    load_snapshot,
    promote,
    promotion_result_document,
    rollback,
    snapshot_document,
    validate_snapshot,
)
from pipeline.redaction_calibration.state import (
    CalibrationStateError,
    PromotionRecord,
    PromotionState,
    transition_promotion,
)
from scripts.build_fewshot_index import index_dir

WITNESS_OPTIONS = (
    "profile-id",
    "profile-sha256",
    "geometry-witness-sha256",
    "canonical-sha256",
    "git-sha",
    "index-pool-sha256",
)

# Откат меняет данные, но не автомат: обратного ребра у §11.3 нет. Значит запись,
# оставшаяся в `promoted`, продолжает свидетельствовать `promotion_result_sha256`
# данных, которых на диске больше нет, и человек обязан это увидеть, а не угадать.
ROLLBACK_NOTICE = (
    "ВНИМАНИЕ: запись автомата продвижения теперь лжёт.\n"
    "  Данные откачены, но обратного перехода у автомата §11.3 нет: запись\n"
    "  продолжает свидетельствовать promotion_result_sha256 данных, которых на\n"
    "  диске больше нет. По §12 спеки восстановление оформляется НОВОЙ КАМПАНИЕЙ,\n"
    "  а не правкой этой записи."
)


def current_git_sha() -> str:
    """Полный SHA текущего HEAD — свидетель версии кода для §13."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    ).stdout.strip()


def default_journal() -> Path:
    """Журнал лежит в каталоге данных, а не в репозитории: он след прогона, не артефакт."""
    return data_root() / ".promotion-journal.json"


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Снимок продвижения и его установка")
    подкоманды = парсер.add_subparsers(dest="command", required=True)

    снимок = подкоманды.add_parser("snapshot", help="собрать хешированный снимок")
    снимок.add_argument("--snapshot", type=Path, required=True, help="новый каталог снимка")
    снимок.add_argument("--snapshot-id", default=None, help="по умолчанию имя каталога")
    снимок.add_argument("--manifest", type=Path, required=True)
    снимок.add_argument("--crops", type=Path, required=True)
    снимок.add_argument("--splits", type=Path, default=None)
    снимок.add_argument("--index", type=Path, default=None)
    for option in WITNESS_OPTIONS:
        снимок.add_argument(f"--{option}", default=None)

    for name, help_text in (
        ("promote", "установить снимок поверх боевых данных"),
        ("rollback", "вернуть боевые данные к снимку"),
    ):
        команда = подкоманды.add_parser(name, help=help_text)
        команда.add_argument("--snapshot", type=Path, required=True)
        команда.add_argument("--manifest-dest", type=Path, default=DEFAULT_MANIFEST_PATH)
        команда.add_argument("--crops-dest", type=Path, default=None)
        команда.add_argument("--splits-dest", type=Path, default=None)
        команда.add_argument("--index-dest", type=Path, default=None)
        команда.add_argument("--journal", type=Path, default=None)
        команда.add_argument("--confirm", default=None, help="токен из сухого прогона")
        команда.add_argument(
            "--record", type=Path, default=None, help="JSON записи автомата продвижения"
        )
        if name == "promote":
            команда.add_argument(
                "--previous-root",
                type=Path,
                default=None,
                help="новый каталог для снимка вытесняемых боевых данных",
            )
            команда.add_argument(
                "--previous-witnesses",
                type=Path,
                default=None,
                help="JSON со свидетелями профиля, стоящего сейчас на боевых данных",
            )
    return парсер


def destinations_from_arguments(arguments: argparse.Namespace) -> PromotionDestinations:
    """Собрать назначения, оставляя боевые умолчания там, где ничего не задано явно."""
    return PromotionDestinations(
        manifest=arguments.manifest_dest,
        crops=arguments.crops_dest or crops_dir(),
        splits=arguments.splits_dest or splits_dir(),
        index=arguments.index_dest or index_dir(),
        journal=arguments.journal or default_journal(),
    )


def _load_json_document(path: Path, *, label: str) -> dict[str, Any]:
    """Прочитать JSON, переведя порчу файла в отказ команды, а не в traceback."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ошибка:
        raise PromotionError(f"{label} {path}: {ошибка}") from ошибка
    if not isinstance(document, dict):
        raise PromotionError(f"{label} {path}: ожидался объект JSON")
    return cast(dict[str, Any], document)


def load_record(path: Path) -> PromotionRecord:
    document = _load_json_document(path, label="запись автомата")
    try:
        state = PromotionState(document["state"])
    except (KeyError, ValueError) as ошибка:
        raise PromotionError(f"запись автомата {path}: нечитаемое state ({ошибка})") from ошибка
    return PromotionRecord(
        state=state,
        profile_id=document.get("profile_id"),
        profile_sha256=document.get("profile_sha256"),
        geometry_witness_sha256=document.get("geometry_witness_sha256"),
        snapshot_result_sha256=document.get("snapshot_result_sha256"),
        promotion_result_sha256=document.get("promotion_result_sha256"),
        h3_result_sha256=document.get("h3_result_sha256"),
        end_to_end_result_sha256=document.get("end_to_end_result_sha256"),
        full_pool_result_sha256=document.get("full_pool_result_sha256"),
    )


def write_record(path: Path, record: PromotionRecord) -> None:
    document = {
        "state": record.state.value,
        "profile_id": record.profile_id,
        "profile_sha256": record.profile_sha256,
        "geometry_witness_sha256": record.geometry_witness_sha256,
        "snapshot_result_sha256": record.snapshot_result_sha256,
        "promotion_result_sha256": record.promotion_result_sha256,
        "h3_result_sha256": record.h3_result_sha256,
        "end_to_end_result_sha256": record.end_to_end_result_sha256,
        "full_pool_result_sha256": record.full_pool_result_sha256,
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _witnesses_from_arguments(arguments: argparse.Namespace) -> dict[str, str]:
    """Свидетели из опций; версия кода по умолчанию — HEAD текущего дерева."""
    witnesses = {
        option.replace("-", "_"): value
        for option in WITNESS_OPTIONS
        if (value := getattr(arguments, option.replace("-", "_"))) is not None
    }
    witnesses.setdefault("git_sha", current_git_sha())
    return witnesses


def _run_snapshot(arguments: argparse.Namespace) -> int:
    witnesses = _witnesses_from_arguments(arguments)
    snapshot = build_snapshot(
        arguments.snapshot,
        snapshot_id=arguments.snapshot_id or arguments.snapshot.name,
        manifest_path=arguments.manifest,
        crops_root=arguments.crops,
        splits_root=arguments.splits,
        index_root=arguments.index,
        witnesses=witnesses,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    print(json.dumps(snapshot_document(snapshot), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"снимок {snapshot.snapshot_id} ({snapshot.tier.value}): {snapshot.result_sha256}")
    return 0


def _dry_run(
    snapshot: SnapshotManifest, destinations: PromotionDestinations, operation: Operation
) -> int:
    validate_snapshot(snapshot, destinations)
    print(f"снимок {snapshot.snapshot_id} ({snapshot.tier.value}) проверен, назначение чисто")
    for group in snapshot.groups:
        print(f"  {group} → {destinations.path(group)}")
    token = confirmation_token(snapshot, operation=operation)
    print(f"сухой прогон. Для установки повторите с --confirm {token}")
    return 0


def _report(result: PromotionResult) -> None:
    document = promotion_result_document(result)
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"{result.operation}: {result.snapshot_id} → {', '.join(result.switched)}")


def _run_promote(arguments: argparse.Namespace) -> int:
    snapshot = load_snapshot(arguments.snapshot)
    destinations = destinations_from_arguments(arguments)
    if arguments.confirm is None:
        return _dry_run(snapshot, destinations, "promote")
    if arguments.confirm != confirmation_token(snapshot, operation="promote"):
        print("токен подтверждения не подходит этому снимку: боевые данные не тронуты")
        return 2
    if arguments.record is None:
        print("--record обязателен: продвижение проверяется записью автомата")
        return 2

    if arguments.previous_root is None or arguments.previous_witnesses is None:
        print("--previous-root и --previous-witnesses обязательны: без снимка")
        print("вытесняемых данных откатываться будет некуда (§12 спеки)")
        return 2

    record = load_record(arguments.record)
    previous = PreviousSnapshotRequest(
        root=arguments.previous_root,
        snapshot_id=arguments.previous_root.name,
        witnesses={
            str(key): str(value)
            for key, value in _load_json_document(
                arguments.previous_witnesses, label="свидетели вытесняемого профиля"
            ).items()
        },
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    result = promote(
        snapshot,
        destinations,
        record=record,
        confirmation_token=arguments.confirm,
        previous=previous,
    )
    write_record(
        arguments.record,
        transition_promotion(
            record, PromotionState.PROMOTED, promotion_result_sha256=result.result_sha256
        ),
    )
    _report(result)
    return 0


def _run_rollback(arguments: argparse.Namespace) -> int:
    snapshot = load_snapshot(arguments.snapshot)
    destinations = destinations_from_arguments(arguments)
    if arguments.confirm is None:
        return _dry_run(snapshot, destinations, "rollback")
    if arguments.confirm != confirmation_token(snapshot, operation="rollback"):
        print("токен подтверждения не подходит этому снимку: боевые данные не тронуты")
        return 2
    _report(rollback(snapshot, destinations, confirmation_token=arguments.confirm))
    print(ROLLBACK_NOTICE)
    if arguments.record is None:
        print("  Устаревшая запись: та, что вела это продвижение (--record её назовёт).")
    else:
        record = load_record(arguments.record)
        print(f"  Устаревшая запись: {arguments.record}, состояние {record.state.value},")
        print(f"  promotion_result_sha256 {record.promotion_result_sha256}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Разобрать подкоманду и выполнить её, переводя отказ снимка в код возврата."""
    arguments = build_parser().parse_args(argv)
    runners = {
        "snapshot": _run_snapshot,
        "promote": _run_promote,
        "rollback": _run_rollback,
    }
    try:
        return runners[arguments.command](arguments)
    except (PromotionError, CalibrationStateError) as ошибка:
        # `CalibrationStateError` — самая вероятная ошибка оператора (запись не в
        # `promotable`), и она обязана выглядеть отказом команды, а не падением.
        print(f"ОТКАЗ: {ошибка}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
