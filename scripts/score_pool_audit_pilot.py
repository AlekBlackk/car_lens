"""Счёт пилота аудита пула против эталона 300-выборки.

Гейт выполнимости формата из спеки `2026-09-18-pool-privacy-audit-design.md`: человек
обязан переоткрыть все известные номера (`вердикт` в `bench/plate-leak-sample.csv`) и все
известные лица (`KNOWN_FACE_LEAK_NUMBERS`). Код 0 — формат годен, 1 — формат провален и
основной проход идёт в `--layout single`, 2 — пилот не закончен.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pipeline.pool_audit import pilot_truth, read_registry, score_pilot

DEFAULT_SAMPLE = Path("bench") / "plate-leak-sample.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Счёт пилота аудита пула")
    parser.add_argument("--export", type=Path, required=True, help="выгрузка страницы пилота")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sample = pd.read_csv(arguments.sample, dtype={"sha256": str})
    numbers = dict(zip(sample["sha256"], sample["номер"], strict=True))
    score = score_pilot(read_registry(arguments.export), pilot_truth(sample))

    plate_found = score.plate_expected - len(score.plate_misses)
    face_found = score.face_expected - len(score.face_misses)
    print(f"номера: {plate_found}/{score.plate_expected}")
    print(f"лица: {face_found}/{score.face_expected}")
    misses = [("пропущен номер", i) for i in score.plate_misses]
    misses += [("пропущено лицо", i) for i in score.face_misses]
    for label, photo_id in misses:
        print(f"  {label}: #{int(numbers[photo_id]):03d} {photo_id}")
    if score.pending_ids:
        print(f"пилот не закончен: {len(score.pending_ids)} кадров без вердикта")
        return 2
    if not score.passed:
        print("ФОРМАТ ПРОВАЛЕН: основной проход — --layout single")
        return 1
    print("пилот пройден: формат годен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
