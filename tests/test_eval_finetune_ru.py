"""Тесты замера дообученной модели на эталоне (Р-11, задача 6).

Проверяются:
1. Главная трасса замера (ранжирование через центроиды индекса пула).
2. Справочная трасса замера (ранжирование через логиты головы ArcFace).
3. Извлечение порядка имён классов из `catalog_meta.json` индекса.
4. Создание отчёта сравнения по пути из флага `--out`.
5. Разбор флагов командной строки со значениями по умолчанию.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.fewshot.metrics import FEWSHOT_TOP_K, FewShotMetrics
from pipeline.knn import ranked_by_centroids
from pipeline.paths import data_root, harvest_dir, splits_dir
from scripts.eval_finetune_ru import (
    build_parser,
    evaluate_head_track,
    evaluate_index_track,
    idx_to_class_id_of,
    load_index_classes,
    main,
)


def test_главная_трасса_идёт_через_центроиды() -> None:
    """Главная трасса: ранжирование по центроидам совпадает с ranked_by_centroids."""
    генератор = np.random.default_rng(42)
    сырые_центроиды = генератор.normal(size=(4, 2, 8))
    центроиды = (сырые_центроиды / np.linalg.norm(сырые_центроиды, axis=2, keepdims=True)).astype(
        np.float32
    )

    # 4 кадра-эмбеддинга: 0 и 1 точно совпадают с центроидами 0 и 1 (попадание top-1)
    эмбеддинги = np.stack(
        [
            центроиды[0, 0],
            центроиды[1, 0],
            центроиды[2, 1],
            (центроиды[3, 0] + центроиды[3, 1]) / 2.0,
        ]
    ).astype(np.float32)
    эмбеддинги = (эмбеддинги / np.linalg.norm(эмбеддинги, axis=1, keepdims=True)).astype(np.float32)
    метки = [0, 1, 2, 3]
    имена = {0: "lada__2107", 1: "kia__rio", 2: "uaz__patriot", 3: "gaz__gazelle"}

    ранжирование, метрики = evaluate_index_track(эмбеддинги, метки, центроиды, имена)

    ожидаемое_ранжирование = ranked_by_centroids(эмбеддинги, центроиды, k=FEWSHOT_TOP_K)
    assert ранжирование == ожидаемое_ранжирование
    assert isinstance(метрики, FewShotMetrics)
    assert [len(строка) for строка in ранжирование] == [FEWSHOT_TOP_K] * len(метки)
    assert метрики.top1 == 1.0
    assert метрики.top3 == 1.0


def test_справочная_трасса_идёт_через_голову() -> None:
    """Справочная трасса: ранжирование по логитам совпадает с ranked_predictions."""
    pytest.importorskip("torch")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from training.evaluate import ranked_predictions
    from training.model import ArcFaceHead, CarClassifier

    число_классов = 4
    устройство = torch.device("cpu")
    # Три кадра с заранее известными логитами
    логиты = torch.tensor(
        [
            [0.1, 0.9, 0.4, 0.2],  # argmax = 1 (top-1), top-3 = [1, 2, 3]
            [0.3, 0.1, 0.2, 0.8],  # argmax = 3 (top-1), top-3 = [3, 0, 2]
            [0.7, 0.2, 0.5, 0.1],  # argmax = 0 (top-1), top-3 = [0, 2, 1]
        ]
    )
    метки = torch.tensor([1, 3, 2])
    загрузчик = DataLoader(TensorDataset(логиты, метки), batch_size=2)

    class ТождественныйБэкбон(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.num_features = число_классов

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    модель = CarClassifier(
        ТождественныйБэкбон(), число_классов, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    модель.embedding = nn.Identity()
    голова = ArcFaceHead(число_классов, число_классов, s=32.0, m=0.5)
    with torch.no_grad():
        голова.weight.copy_(torch.eye(число_классов))

    имена = {0: "lada__2107", 1: "kia__rio", 2: "uaz__patriot", 3: "gaz__gazelle"}

    ранжирование, цели, метрики = evaluate_head_track(
        модель, голова, загрузчик, device=устройство, idx_to_class_id=имена
    )

    ожидаемые_ранги, ожидаемые_цели = ranked_predictions(
        модель, голова, загрузчик, device=устройство, k=FEWSHOT_TOP_K
    )
    assert ранжирование == ожидаемые_ранги
    assert цели == ожидаемые_цели
    assert [len(строка) for строка in ранжирование] == [FEWSHOT_TOP_K] * len(метки)
    assert цели == [1, 3, 2]
    # Кадр 0: top-1 верный (1). Кадр 1: top-1 верный (3).
    # Кадр 2: top-1 неверный (0 != 2), но top-3 верный (2 in [0, 2, 1]).
    assert метрики.top1 == pytest.approx(2 / 3)
    assert метрики.top3 == pytest.approx(1.0)


def test_имена_классов_из_меты_индекса(tmp_path: Path) -> None:
    """При расхождении порядка в YAML и в мете берётся порядок из меты индекса."""
    мета = {
        "classes": [
            {"class_id": "uaz__patriot", "members": ["uaz__patriot"]},
            {"class_id": "lada__2107", "members": ["lada__2107"]},
            {"class_id": "kia__rio", "members": ["kia__rio"]},
        ]
    }
    путь_меты = tmp_path / "catalog_meta.json"
    путь_меты.write_text(json.dumps(мета), encoding="utf-8")

    имена = load_index_classes(tmp_path)

    assert имена == {0: "uaz__patriot", 1: "lada__2107", 2: "kia__rio"}
    assert idx_to_class_id_of(мета) == {0: "uaz__patriot", 1: "lada__2107", 2: "kia__rio"}


def test_отчёт_записан_по_пути_из_флага(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Отчёт пишется точно по пути из `--out` и содержит таблицу лицом к лицу."""
    целевой_отчёт = tmp_path / "custom_dir" / "custom_finetune_report.md"
    from pipeline.fewshot.gate import G_FS2_CORE_CLASSES
    from pipeline.index_format import DTYPE_FLOAT32, IndexHeader

    базис_поклассово = {класс: 0.50 for класс in G_FS2_CORE_CLASSES}
    базис_опора = {класс: 10 for класс in G_FS2_CORE_CLASSES}
    базис_json = tmp_path / "fewshot_metrics.json"
    базис_json.write_text(
        json.dumps(
            {
                "top1": 0.4412,
                "top3": 0.6290,
                "per_class_top3": базис_поклассово,
                "per_class_support": базис_опора,
            }
        ),
        encoding="utf-8",
    )

    каталог_индекса = tmp_path / "mock_index"
    каталог_индекса.mkdir()
    (каталог_индекса / "catalog_meta.json").write_text(
        json.dumps(
            {"classes": [{"class_id": класс, "members": [класс]} for класс in G_FS2_CORE_CLASSES]}
        ),
        encoding="utf-8",
    )
    (каталог_индекса / "catalog_index.bin").write_bytes(b"dummy_index_bin")

    чекпоинт_файл = tmp_path / "last.pt"
    чекпоинт_файл.write_bytes(b"dummy_pt")

    фиктивные_метрики_индекс = FewShotMetrics(
        top1=0.48,
        top3=0.65,
        per_class_top3={класс: 0.60 for класс in G_FS2_CORE_CLASSES},
        per_class_support={класс: 10 for класс in G_FS2_CORE_CLASSES},
    )
    фиктивные_метрики_голова = FewShotMetrics(
        top1=0.42,
        top3=0.61,
        per_class_top3={класс: 0.55 for класс in G_FS2_CORE_CLASSES},
        per_class_support={класс: 10 for класс in G_FS2_CORE_CLASSES},
    )

    фиктивный_заголовок = IndexHeader(
        embedding_dim=256,
        num_classes=len(G_FS2_CORE_CLASSES),
        centroids_per_class=4,
        dtype=DTYPE_FLOAT32,
        normalized=True,
        meta_sha256=b"\x00" * 32,
    )

    monkeypatch.setattr(
        "scripts.eval_finetune_ru.read_index",
        lambda путь: (фиктивный_заголовок, np.zeros((2, 4, 256), dtype=np.float32)),
    )
    monkeypatch.setattr(
        "scripts.eval_finetune_ru.verify_meta",
        lambda заголовок, байты: None,
    )
    monkeypatch.setattr(
        "scripts.eval_finetune_ru._run_model_evaluation",
        lambda **kwargs: (
            фиктивные_метрики_индекс,
            фиктивные_метрики_голова,
            {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "epochs": 60, "seed": 1337},
            22,
            "abc1234",
        ),
    )

    код_возврата = main(
        [
            "--checkpoint",
            str(чекпоинт_файл),
            "--index",
            str(каталог_индекса),
            "--baseline",
            str(базис_json),
            "--out",
            str(целевой_отчёт),
        ]
    )

    assert код_возврата == 0
    assert целевой_отчёт.is_file()
    текст = целевой_отчёт.read_text(encoding="utf-8")
    assert "## Лицом к лицу: одна линейка, один код метрики" in текст
    assert "| величина | дообучение | few-shot базис M2 | дельта |" in текст
    assert "top-1" in текст
    assert "top-3" in текст


def test_разбор_флагов_даёт_значения_по_умолчанию() -> None:
    """Три пути по умолчанию совпадают с splits_dir(), harvest_dir() и каталогом few-shot."""
    парсер = build_parser()
    аргументы = парсер.parse_args(["--checkpoint", "last.pt", "--index", "some_index"])

    assert аргументы.holdout == splits_dir() / "holdout_ru.parquet"
    assert аргументы.crops_root == harvest_dir() / "crops"
    assert аргументы.baseline == data_root() / "index/fewshot_openclip_b32/fewshot_metrics.json"
    assert аргументы.weights == "raw"
    assert аргументы.device == "cuda"
    assert аргументы.out == Path("bench/finetune-ru.md")
    assert аргументы.preregistration == Path("bench/finetune-ru-preregistration.md")
    assert аргументы.train_minutes == 0.0


def test_config_sha256_дамп_при_отсутствии_файлов(tmp_path: Path) -> None:
    """При отсутствии config.yaml рядом sha256 считается детерминированно из словаря."""
    import hashlib

    from scripts.eval_finetune_ru import _config_sha256

    конфиг = {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "epochs": 60}
    чекпоинт = tmp_path / "last.pt"
    хеш = _config_sha256(чекпоинт, конфиг)
    ожидаемый = hashlib.sha256(json.dumps(конфиг, sort_keys=True).encode("utf-8")).hexdigest()
    assert хеш == ожидаемый


def test_git_sha_возвращает_запасной_при_ошибке_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если git недоступен или завершается с ошибкой, возвращается запасной sha."""
    import subprocess

    from scripts.eval_finetune_ru import _git_sha

    def mock_check_output(*args: object, **kwargs: object) -> bytes:
        raise subprocess.SubprocessError("git not found")

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    assert _git_sha("fallback_sha123") == "fallback_sha123"
    assert _git_sha(None) == "unknown"


def test_отсутствие_ema_state_роняет_замер_с_понятной_ошибкой(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """При заказе weights='ema' отсутствие ключа ema_state вызывает понятный ValueError."""
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    чекпоинт_файл = tmp_path / "last.pt"
    # Чекпоинт без ema_state
    фиктивный_чекпоинт = {
        "config": {
            "backbone": "mobilenetv4_conv_medium.e500_r256_in1k",
            "embedding_dim": 256,
            "img_size": 320,
            "batch_size": 32,
            "num_workers": 0,
        },
        "class_index": {"lada__2107": 0},
        "model_state": {},
        "head_state": {},
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    torch.save(фиктивный_чекпоинт, чекпоинт_файл)

    monkeypatch.setattr(
        "training.model.create_model",
        lambda *args, **kwargs: (torch.nn.Identity(), torch.nn.Identity()),
    )

    with pytest.raises(ValueError, match=r"ключ 'ema_state' отсутствует в чекпоинте"):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((1, 4, 256), dtype=np.float32),
            idx_to_class_id={0: "lada__2107"},
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="ema",
            device="cpu",
        )
