"""Замер few-shot по `holdout_ru` (Р-11, §13.2).

Скрипт читает `catalog_index.bin`, сверяет его с `catalog_meta.json` (проверка
`verify_meta`, §13.2), прогоняет энкодер по кадрам `holdout_ru`, считает ранжирование по
центроидам, вычисляет метрики, проверяет гейт G_FS1–G_FS3 и пишет отчёт в
`bench/fewshot-m2.md`, а рядом с ним — копию `report.json` сборки индекса, которым замер
получен.

Имена классов берутся из `catalog_meta.json`, а не загружаются из каталога YAML заново:
замер обязан оценивать классы ровно в тех слотах и под теми именами, под которыми они
собраны в бинарный индекс.

`open_clip` и `torch` импортируются внутри `main` (Р-15): чистые функции модуля
(`idx_to_class_id_of`, `missing_holdout_classes`, `evaluate`) тестируются в базовом
окружении без весов.

Запуск:

    uv run --group train python -m scripts.eval_fewshot
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.fewshot.gate import check_gate, render_report
from pipeline.fewshot.metrics import FEWSHOT_TOP_K, FewShotMetrics, aggregate_fewshot
from pipeline.index_format import read_index, verify_meta
from pipeline.knn import ranked_by_centroids
from pipeline.paths import harvest_dir, splits_dir
from pipeline.schema import read_split
from scripts.build_fewshot_index import (
    ENCODER_ARCH,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    crops_by_class,
    index_dir,
)

REPORT_PATH = Path(__file__).resolve().parent.parent / "bench" / "fewshot-m2.md"
DEFAULT_SPLIT = "holdout_ru"


def idx_to_class_id_of(meta: Mapping[str, Any]) -> dict[int, str]:
    """`class_id` по индексу класса: позиция записи в `classes` — и есть индекс (§13.2)."""
    return {номер: класс["class_id"] for номер, класс in enumerate(meta["classes"])}


def missing_holdout_classes(
    idx_to_class_id: Mapping[int, str], metrics: FewShotMetrics
) -> tuple[str, ...]:
    """Классы индекса, которых нет в эталоне, в порядке индекса.

    Они не провал и не успех: порог по ним не измерен, и отчёт обязан назвать их отдельно
    от классов, которые измерены и не дотянули.
    """
    return tuple(
        class_id
        for _, class_id in sorted(idx_to_class_id.items())
        if class_id not in metrics.per_class_support
    )


def read_split_report(report_path: Path) -> tuple[str, ...]:
    """Прочитать список дроблёных классов из отчёта сплита, если файл существует."""
    if not report_path.exists():
        return ()
    данные = json.loads(report_path.read_text(encoding="utf-8"))
    return tuple(данные.get("refined_classes", ()))


def сохранить_отчёт_сборки(report_path: Path, build_report: str) -> Path:
    """Положить `report.json` сборки индекса рядом с отчётом замера, вернуть путь копии.

    Числа замера без индекса, которым они получены, неразбираемы: перезамер C1 свёлся к
    пересборке вслепую именно потому, что индекс не сохранили, и пять чисел из семи не
    воспроизвелись (`bench/fewshot-m2.md`, «Воспроизводимость перезамера»). Каталог данных
    живёт на одной машине и переписывается следующей сборкой, отчёт — под git, поэтому
    провенанс кладётся рядом с числами, а не остаётся ссылкой на путь в WSL.

    Текст пишется как прочитан, а не пересобирается из разобранного json: копия обязана
    говорить то же, что отчёт сборки, вплоть до порядка ключей.
    """
    путь = report_path.with_name(f"{report_path.stem}-index-report.json")
    путь.write_text(build_report, encoding="utf-8")
    return путь


def evaluate(
    embeddings: np.ndarray,
    targets: Sequence[int],
    centroids: np.ndarray,
    idx_to_class_id: Mapping[int, str],
) -> FewShotMetrics:
    """Ранжирование по центроидам и метрики Р-11 одним шагом.

    Отдельная функция, а не три строки в `main`: связка «ранжирование → метрика» — то место,
    где ошибка даёт не падение, а неверный вердикт гейта, поэтому она проверяется тестом.
    """
    return aggregate_fewshot(
        ranked_by_centroids(embeddings, centroids, FEWSHOT_TOP_K),
        targets,
        idx_to_class_id=idx_to_class_id,
    )


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов замера."""
    парсер = argparse.ArgumentParser(description="Замер few-shot по holdout_ru (Р-11)")
    парсер.add_argument("--index", type=Path, default=None, help="каталог с catalog_index.bin")
    парсер.add_argument("--split", default=DEFAULT_SPLIT)
    парсер.add_argument("--report", type=Path, default=REPORT_PATH)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Замерить эталон, проверить гейт, записать отчёт. `0` — гейт пройден."""
    import open_clip  # type: ignore[import-untyped]

    from pipeline.fewshot.embed import embed_crops
    from pipeline.harvest.backends import ClipImageEncoder

    аргументы = build_parser().parse_args(argv)
    каталог_индекса = аргументы.index or index_dir()
    заголовок, центроиды = read_index(каталог_индекса / "catalog_index.bin")
    байты_меты = (каталог_индекса / "catalog_meta.json").read_bytes()
    verify_meta(заголовок, байты_меты)
    мета = json.loads(байты_меты)
    имена = idx_to_class_id_of(мета)

    сборка_текст = (каталог_индекса / "report.json").read_text(encoding="utf-8")
    сборка = json.loads(сборка_текст)

    кадр = read_split(splits_dir() / f"{аргументы.split}.parquet")
    группы = crops_by_class(кадр, harvest_dir() / "crops")

    # Те же веса, что в сборке индекса. Индекс и запрос, посчитанные разными энкодерами,
    # дают косинус, который ничего не значит, поэтому пин один и импортируется, а не
    # повторяется строкой.
    модель, _, препроцесс = open_clip.create_model_and_transforms(
        ENCODER_ARCH, pretrained=ENCODER_WEIGHTS
    )
    энкодер = ClipImageEncoder(модель.eval(), препроцесс)

    куски: list[np.ndarray] = []
    метки: list[int] = []
    for номер, пути in группы:
        свои = embed_crops(пути, энкодер)
        куски.append(свои)
        метки += [номер] * len(свои)
        print(f"{имена[номер]}: {len(свои)} кадров эталона", flush=True)
    эмбеддинги = np.concatenate(куски)

    метрики = evaluate(эмбеддинги, метки, центроиды, имена)
    гейт = check_gate(метрики, catalog_classes=заголовок.num_classes)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    отчёт_сплита_путь = splits_dir() / f"{аргументы.split}_report.json"
    дроблёные = read_split_report(отчёт_сплита_путь)

    аргументы.report.parent.mkdir(parents=True, exist_ok=True)
    аргументы.report.write_text(
        render_report(
            метрики,
            гейт,
            git_sha=sha,
            run_date=date.today().isoformat(),
            encoder=ENCODER_NAME,
            index_sha256=str(сборка["index_sha256"]),
            holdout_frames=len(метки),
            pool_frames=int(сборка["embeddings_total"]),
            catalog_classes=заголовок.num_classes,
            missing_holdout=missing_holdout_classes(имена, метрики),
            refined_classes=дроблёные,
        ),
        encoding="utf-8",
    )
    путь_сборки = сохранить_отчёт_сборки(аргументы.report, сборка_текст)
    (каталог_индекса / "fewshot_metrics.json").write_text(
        json.dumps(
            {
                "top1": метрики.top1,
                "top3": метрики.top3,
                "per_class_top3": метрики.per_class_top3,
                "per_class_support": метрики.per_class_support,
                "gate_passed": гейт.passed,
                "gate_failures": list(гейт.failures),
                "deficient_classes": list(гейт.deficient_classes),
                "refined_classes": list(дроблёные),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"top-1 {метрики.top1:.4f}, top-3 {метрики.top3:.4f}")
    print(f"отчёт сборки индекса: {путь_сборки}")
    for провал in гейт.failures:
        print(f"  ✗ {провал}")
    return 0 if гейт.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
