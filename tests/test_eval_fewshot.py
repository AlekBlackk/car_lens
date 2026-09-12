"""Чистая часть замера few-shot: имена классов, недостача эталона, связка метрик."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.fewshot.metrics import FewShotMetrics
from scripts.build_fewshot_index import crops_by_class
from scripts.eval_fewshot import (
    build_parser,
    evaluate,
    idx_to_class_id_of,
    missing_holdout_classes,
    ordered_photo_ids,
    predictions_from_ranking,
    read_split_report,
    resolve_paths,
    write_predictions,
    сохранить_отчёт_сборки,
)

МЕТА = {
    "classes": [
        {"class_id": "lada__2107", "members": ["lada__2107"]},
        {"class_id": "kia__rio", "members": ["kia__rio"]},
        {"class_id": "uaz__patriot", "members": ["uaz__patriot"]},
    ]
}


def test_карты_имён_собираются_по_позиции_в_мете() -> None:
    assert idx_to_class_id_of(МЕТА) == {0: "lada__2107", 1: "kia__rio", 2: "uaz__patriot"}


def test_недостающие_классы_названы_в_порядке_индекса() -> None:
    имена = {0: "lada__2107", 1: "kia__rio", 2: "uaz__patriot"}
    метрики = FewShotMetrics(
        top1=1.0,
        top3=1.0,
        per_class_top3={"lada__2107": 1.0, "uaz__patriot": 1.0},
        per_class_support={"lada__2107": 5, "uaz__patriot": 3},
    )
    assert missing_holdout_classes(имена, метрики) == ("kia__rio",)


def test_evaluate_связывает_ранжирование_и_агрегацию() -> None:
    """Точный косинус 1.0 на правильном центроиде — попадание top-1 и top-3."""
    генератор = np.random.default_rng(3)
    сырые = генератор.normal(size=(3, 2, 4))
    центроиды = (сырые / np.linalg.norm(сырые, axis=2, keepdims=True)).astype(np.float32)
    эмбеддинги = np.stack([центроиды[0, 0], центроиды[1, 1], центроиды[2, 0]])
    метки = [0, 1, 2]
    имена = {0: "a", 1: "b", 2: "c"}
    итог = evaluate(эмбеддинги, метки, центроиды, имена)
    assert итог.top1 == 1.0
    assert итог.top3 == 1.0
    assert set(итог.per_class_top3) == {"a", "b", "c"}


def test_evaluate_ловит_ошибки_ранжирования() -> None:
    """Размерности эмбеддингов и центроидов обязаны совпадать."""
    import pytest

    центроиды = np.zeros((3, 2, 8), dtype=np.float32)
    эмбеддинги = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        evaluate(эмбеддинги, [0, 1], центроиды, {0: "a", 1: "b", 2: "c"})


def test_чтение_отчёта_сплита_извлекает_дроблёные_классы(tmp_path: Path) -> None:
    import json

    путь = tmp_path / "holdout_ru_report.json"
    путь.write_text(json.dumps({"refined_classes": ["bmw__x5", "lada__granta"]}), encoding="utf-8")
    assert read_split_report(путь) == ("bmw__x5", "lada__granta")


def test_чтение_отчёта_сплита_когда_нет_файла_возвращает_пустой_кортеж(tmp_path: Path) -> None:
    путь = tmp_path / "несуществующий_report.json"
    assert read_split_report(путь) == ()


def test_отчёт_сборки_ложится_рядом_с_отчётом_замера(tmp_path: Path) -> None:
    """Имя сайдкара выводится из имени отчёта: файлы обязаны находиться друг по другу."""
    отчёт = tmp_path / "fewshot-m2.md"
    текст = '{\n  "index_sha256": "9c1f",\n  "embeddings_total": 7\n}'

    путь = сохранить_отчёт_сборки(отчёт, текст)

    assert путь == tmp_path / "fewshot-m2-index-report.json"
    assert путь.read_text(encoding="utf-8") == текст


def test_отчёт_сборки_переписывается_повторным_замером(tmp_path: Path) -> None:
    """Перезамер обновляет сайдкар, а не дописывает: рядом с числами лежит их индекс."""
    отчёт = tmp_path / "fewshot-m2.md"
    сохранить_отчёт_сборки(отчёт, '{"index_sha256": "старый"}')

    путь = сохранить_отчёт_сборки(отчёт, '{"index_sha256": "новый"}')

    assert путь.read_text(encoding="utf-8") == '{"index_sha256": "новый"}'


def test_photo_ids_exactly_match_crops_by_class_path_order(tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "photo_id": ["two-b", "zero-a", "two-a", "one-a"],
            "class_idx": [2, 0, 2, 1],
            "crop_path": [
                "class-two/two-b.jpg",
                "class-zero/zero-a.jpg",
                "class-two/two-a.jpg",
                "class-one/one-a.jpg",
            ],
        }
    )
    path_order = [path.stem for _, paths in crops_by_class(frame, tmp_path) for path in paths]

    assert ordered_photo_ids(frame) == path_order == ["zero-a", "one-a", "two-b", "two-a"]


def test_prediction_export_preserves_every_frame_in_embedding_order(tmp_path: Path) -> None:
    predictions = predictions_from_ranking(
        ["photo-b", "photo-a"],
        [2, 1],
        np.asarray([[2, 0, 1], [0, 1, 2]], dtype=np.int64),
    )
    path = tmp_path / "predictions.csv"

    write_predictions(path, predictions)

    assert path.read_text(encoding="utf-8") == (
        "photo_id,target,top_indices\nphoto-b,2,2 0 1\nphoto-a,1,0 1 2\n"
    )


def test_prediction_export_rejects_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError, match="несовпадение длин"):
        predictions_from_ranking(["only"], [1, 2], np.asarray([[1, 0, 2]], dtype=np.int64))


def test_explicit_eval_paths_are_parsed_and_preserved() -> None:
    arguments = build_parser().parse_args(
        [
            "--split-path",
            "/fixture/holdout.parquet",
            "--crops-root",
            "/fixture/crops",
            "--predictions-out",
            "/fixture/predictions.csv",
        ]
    )

    paths = resolve_paths(arguments)

    assert paths.split_path == Path("/fixture/holdout.parquet")
    assert paths.crops_root == Path("/fixture/crops")
    assert paths.predictions_out == Path("/fixture/predictions.csv")


def test_eval_path_defaults_remain_current_split_and_crops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_root = Path("/legacy/splits")
    harvest_root = Path("/legacy/harvest")
    monkeypatch.setattr("scripts.eval_fewshot.splits_dir", lambda: split_root)
    monkeypatch.setattr("scripts.eval_fewshot.harvest_dir", lambda: harvest_root)

    arguments = build_parser().parse_args([])
    paths = resolve_paths(arguments)

    assert arguments.split_path is None
    assert arguments.crops_root is None
    assert arguments.predictions_out is None
    assert paths.split_path == split_root / "holdout_ru.parquet"
    assert paths.crops_root == harvest_root / "crops"
    assert paths.predictions_out is None
