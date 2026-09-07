"""Тесты замера дообученной модели на эталоне (Р-11, задача 6).

Проверяются:
1. Главная трасса замера (ранжирование через центроиды индекса пула).
2. Справочная трасса замера (ранжирование через логиты головы ArcFace).
3. Извлечение порядка имён классов из `catalog_meta.json` индекса.
4. Создание отчёта сравнения по пути из флага `--out`.
5. Разбор флагов командной строки со значениями по умолчанию.
"""

from __future__ import annotations

import hashlib
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
    yaml_контент = (
        "classes:\n"
        "  - id: lada__2107\n"
        "    brand: Lada\n"
        "    models: [lada__2107]\n"
        "  - id: uaz__patriot\n"
        "    brand: UAZ\n"
        "    models: [uaz__patriot]\n"
        "  - id: kia__rio\n"
        "    brand: Kia\n"
        "    models: [kia__rio]\n"
    )
    (tmp_path / "catalog.yaml").write_text(yaml_контент, encoding="utf-8")

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

    # Порядок строго из catalog_meta.json (0: uaz, 1: lada, 2: kia), не из YAML (lada=0)
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
    хеш_чекпоинта = hashlib.sha256(чекпоинт_файл.read_bytes()).hexdigest()
    (каталог_индекса / "catalog_meta.json").write_text(
        json.dumps(
            {
                "model_weights_sha256": хеш_чекпоинта,
                "classes": [
                    {"class_id": класс, "members": [класс]} for класс in G_FS2_CORE_CLASSES
                ],
            }
        ),
        encoding="utf-8",
    )
    # Эталон свой, а не умолчание: иначе тест молча опирался бы на настоящий
    # holdout_ru.parquet в каталоге данных и падал бы там, где данных нет.
    эталон_файл = tmp_path / "holdout_ru.parquet"
    эталон_файл.write_bytes(b"dummy_parquet")
    пул_файл = tmp_path / "training_pool.parquet"
    пул_файл.write_bytes(b"training_pool")

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
    monkeypatch.setattr("pipeline.schema.read_split", lambda путь: [object(), object(), object()])

    код_возврата = main(
        [
            "--checkpoint",
            str(чекпоинт_файл),
            "--index",
            str(каталог_индекса),
            "--baseline",
            str(базис_json),
            "--pool",
            str(пул_файл),
            "--holdout",
            str(эталон_файл),
            "--out",
            str(целевой_отчёт),
        ]
    )

    assert код_возврата == 0
    assert целевой_отчёт.is_file()
    текст = целевой_отчёт.read_text(encoding="utf-8")
    assert "## Лицом к лицу: одна линейка, один код метрики" in текст
    assert "| величина | дообучение | few-shot базис M2 | дельта |" in текст
    assert "## Две трассы одного прогона" in текст
    assert "## Пороги G_FS1–G_FS3: справочная линия" in текст
    assert "## Поклассовые дельты: десять вверх и десять вниз" in текст
    assert "## Ожидание против факта" in текст
    assert "## Четыре искажения, названные до прогона" in текст
    assert hashlib.sha256(пул_файл.read_bytes()).hexdigest() in текст
    assert "кадров 3" in текст
    assert "top-1" in текст
    assert "top-3" in текст


def test_разбор_флагов_даёт_значения_по_умолчанию() -> None:
    """Пути по умолчанию совпадают со splits_dir(), harvest_dir() и каталогом few-shot."""
    парсер = build_parser()
    аргументы = парсер.parse_args(["--checkpoint", "last.pt", "--index", "some_index"])

    assert аргументы.pool == splits_dir() / "pool.parquet"
    assert аргументы.holdout == splits_dir() / "holdout_ru.parquet"
    assert аргументы.crops_root == harvest_dir() / "crops"
    assert аргументы.baseline == data_root() / "index/fewshot_openclip_b32/fewshot_metrics.json"
    assert аргументы.weights == "raw"
    assert аргументы.device == "cuda"
    assert аргументы.out == Path("bench/finetune-ru.md")
    assert аргументы.preregistration == Path("bench/finetune-ru-preregistration.md")
    assert аргументы.train_minutes == 0.0


def test_config_sha256_из_канонического_конфига_чекпоинта(tmp_path: Path) -> None:
    """Mutable config.yaml рядом не подменяет конфиг, фактически загруженный из checkpoint."""
    import hashlib

    from scripts.eval_finetune_ru import _config_sha256

    конфиг = {"epochs": 60, "backbone": "mobilenetv4_conv_medium.e500_r256_in1k"}
    чекпоинт = tmp_path / "last.pt"
    (tmp_path / "config.yaml").write_text("epochs: 999\n", encoding="utf-8")

    хеш = _config_sha256(чекпоинт, конфиг)

    канон = b'{"backbone":"mobilenetv4_conv_medium.e500_r256_in1k","epochs":60}'
    assert хеш == hashlib.sha256(канон).hexdigest()


def test_git_sha_предпочитает_чекпоинт_текущему_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Провенанс модели — commit обучения из checkpoint, а не текущий commit замера."""
    import subprocess

    from scripts.eval_finetune_ru import _git_sha

    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"current_checkout\n")
    assert _git_sha("training_checkpoint") == "training_checkpoint"


def test_git_sha_берёт_checkout_без_sha_чекпоинта(monkeypatch: pytest.MonkeyPatch) -> None:
    """Старый checkpoint без git_sha использует текущий checkout как запасной источник."""
    import subprocess

    from scripts.eval_finetune_ru import _git_sha

    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"current_checkout\n")
    assert _git_sha(None) == "current_checkout"


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
            index_embedding_dim=256,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="ema",
            device="cpu",
        )


@pytest.mark.parametrize("пропущенный_ключ", ["config", "class_index", "head_state", "model_state"])
def test_отсутствие_обязательного_ключа_чекпоинта_роняет_замер(
    tmp_path: Path, пропущенный_ключ: str
) -> None:
    """Отсутствие любого обязательного ключа чекпоинта роняет замер с понятным ValueError."""
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    чекпоинт = {
        "config": {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "embedding_dim": 256},
        "class_index": {"lada__2107": 0},
        "model_state": {},
        "head_state": {},
    }
    del чекпоинт[пропущенный_ключ]

    чекпоинт_файл = tmp_path / f"broken_{пропущенный_ключ}.pt"
    torch.save(чекпоинт, чекпоинт_файл)

    with pytest.raises(ValueError, match=rf"ключ '{пропущенный_ключ}' отсутствует в чекпоинте"):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((1, 4, 256), dtype=np.float32),
            idx_to_class_id={0: "lada__2107"},
            index_embedding_dim=256,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="raw",
            device="cpu",
        )


def test_рассогласование_классов_чекпоинта_и_индекса_роняет_замер(tmp_path: Path) -> None:
    """Если классы чекпоинта разошлись с метой индекса (Р-21), выполнение завершается с ошибкой."""
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    чекпоинт = {
        "config": {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "embedding_dim": 256},
        "class_index": {"lada__2107": 0, "kia__rio": 1},
        "model_state": {},
        "head_state": {},
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    чекпоинт_файл = tmp_path / "last.pt"
    torch.save(чекпоинт, чекпоинт_файл)

    # Мета с другими классами (uaz вместо kia)
    idx_to_class_id = {0: "lada__2107", 1: "uaz__patriot"}

    with pytest.raises(
        ValueError, match=r"нумерация классов чекпоинта разошлась с каталогом индекса"
    ):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((2, 4, 256), dtype=np.float32),
            idx_to_class_id=idx_to_class_id,
            index_embedding_dim=256,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="raw",
            device="cpu",
        )


def test_несовпадение_размерности_центроидов_и_меты_роняет_evaluate_index_track() -> None:
    """Размерность первого измерения центроидов должна совпадать с числом классов в мете."""
    центроиды = np.zeros((3, 4, 16), dtype=np.float32)
    эмбеддинги = np.zeros((2, 16), dtype=np.float32)
    имена = {0: "a", 1: "b"}  # 2 класса против 3 центроидов

    with pytest.raises(
        ValueError, match=r"размерность центроидов \(3\) не совпадает с числом классов в мете \(2\)"
    ):
        evaluate_index_track(эмбеддинги, [0, 1], центроиды, имена)


def test_несуществующие_файлы_роняют_main(tmp_path: Path) -> None:
    """Отсутствие чекпоинта, индекса или базиса роняет main с FileNotFoundError."""
    существующий_чекпоинт = tmp_path / "last.pt"
    существующий_чекпоинт.write_bytes(b"dummy")

    каталог_индекса = tmp_path / "index"
    каталог_индекса.mkdir()
    (каталог_индекса / "catalog_index.bin").write_bytes(b"dummy")
    (каталог_индекса / "catalog_meta.json").write_bytes(b"{}")

    существующий_базис = tmp_path / "baseline.json"
    существующий_базис.write_bytes(b"{}")

    # 1. Нет чекпоинта
    with pytest.raises(FileNotFoundError, match=r"чекпоинт не найден"):
        main(["--checkpoint", str(tmp_path / "missing.pt"), "--index", str(каталог_индекса)])

    # 2. Нет бинарного индекса
    (каталог_индекса / "catalog_index.bin").unlink()
    with pytest.raises(FileNotFoundError, match=r"файл индекса не найден"):
        main(["--checkpoint", str(существующий_чекпоинт), "--index", str(каталог_индекса)])
    (каталог_индекса / "catalog_index.bin").write_bytes(b"dummy")

    # 3. Нет catalog_meta.json
    (каталог_индекса / "catalog_meta.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"файл меты индекса не найден"):
        main(["--checkpoint", str(существующий_чекпоинт), "--index", str(каталог_индекса)])
    (каталог_индекса / "catalog_meta.json").write_bytes(b"{}")

    # 4. Нет базиса
    with pytest.raises(FileNotFoundError, match=r"файл базиса не найден"):
        main(
            [
                "--checkpoint",
                str(существующий_чекпоинт),
                "--index",
                str(каталог_индекса),
                "--baseline",
                str(tmp_path / "missing_baseline.json"),
            ]
        )


def test_отрицательное_время_обучения_роняет_main(tmp_path: Path) -> None:
    """Отрицательное значение --train-minutes недопустимо."""
    чекпоинт = tmp_path / "last.pt"
    чекпоинт.write_bytes(b"dummy")

    индекс = tmp_path / "index"
    индекс.mkdir()
    (индекс / "catalog_index.bin").write_bytes(b"dummy")
    (индекс / "catalog_meta.json").write_bytes(b"{}")

    базис = tmp_path / "baseline.json"
    базис.write_bytes(b"{}")

    with pytest.raises(ValueError, match=r"train_minutes не может быть отрицательным"):
        main(
            [
                "--checkpoint",
                str(чекпоинт),
                "--index",
                str(индекс),
                "--baseline",
                str(базис),
                "--train-minutes",
                "-10.5",
            ]
        )


@pytest.mark.parametrize("пропущенный_ключ", ["img_size", "mean", "std"])
def test_отсутствие_параметра_предобработки_роняет_замер(
    tmp_path: Path, пропущенный_ключ: str
) -> None:
    """Параметры предобработки требуются, а не подставляются умолчанием (Р-21).

    Чужие `mean`, `std` или `img_size` меняют эмбеддинг, а значит и замеряемое число,
    молча: отчёт получился бы правдоподобным и неверным. `train.py` пишет все три в
    каждый чекпоинт, поэтому их отсутствие — признак битого файла, а не старого формата.
    """
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    чекпоинт = {
        "config": {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "embedding_dim": 256},
        "class_index": {"lada__2107": 0},
        "model_state": {},
        "head_state": {},
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    del чекпоинт[пропущенный_ключ]

    чекпоинт_файл = tmp_path / f"no_{пропущенный_ключ}.pt"
    torch.save(чекпоинт, чекпоинт_файл)

    with pytest.raises(ValueError, match=rf"ключ '{пропущенный_ключ}' отсутствует в чекпоинте"):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((1, 4, 256), dtype=np.float32),
            idx_to_class_id={0: "lada__2107"},
            index_embedding_dim=256,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="raw",
            device="cpu",
        )


@pytest.mark.parametrize("пропущенное_поле", ["backbone", "embedding_dim"])
def test_отсутствие_поля_конфига_роняет_замер(tmp_path: Path, пропущенное_поле: str) -> None:
    """Архитектура и размерность берутся из конфига чекпоинта, а не из зашитой константы.

    Зашитое умолчание собрало бы другую сеть под теми же весами: `load_state_dict` упал бы
    стеной несовпавших ключей, из которой причину не видно, а совпади размерности — замер
    пошёл бы по чужой архитектуре.
    """
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    конфиг = {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "embedding_dim": 256}
    del конфиг[пропущенное_поле]
    чекпоинт = {
        "config": конфиг,
        "class_index": {"lada__2107": 0},
        "model_state": {},
        "head_state": {},
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    чекпоинт_файл = tmp_path / f"no_{пропущенное_поле}.pt"
    torch.save(чекпоинт, чекпоинт_файл)

    with pytest.raises(
        ValueError, match=rf"поле '{пропущенное_поле}' отсутствует в config чекпоинта"
    ):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((1, 4, 256), dtype=np.float32),
            idx_to_class_id={0: "lada__2107"},
            index_embedding_dim=256,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="raw",
            device="cpu",
        )


def test_рассогласование_классов_называет_разошедшийся_класс(tmp_path: Path) -> None:
    """Сообщение о рассогласовании называет слот и обе стороны, а не только счётчики.

    При равном числе классов «(150 против 150 классов)» не говорит, что чинить: разошёлся
    порядок, а какой именно класс уехал — из текста не видно.
    """
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    чекпоинт = {
        "config": {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "embedding_dim": 256},
        "class_index": {"lada__2107": 0, "kia__rio": 1},
        "model_state": {},
        "head_state": {},
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    чекпоинт_файл = tmp_path / "last.pt"
    torch.save(чекпоинт, чекпоинт_файл)

    with pytest.raises(ValueError, match=r"слот 1.*kia__rio.*uaz__patriot"):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((2, 4, 256), dtype=np.float32),
            idx_to_class_id={0: "lada__2107", 1: "uaz__patriot"},
            index_embedding_dim=256,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="raw",
            device="cpu",
        )


def test_размерность_индекса_против_чекпоинта_роняет_замер(tmp_path: Path) -> None:
    """Индекс, собранный другим энкодером, отказывается до первого форварда.

    Косинус между эмбеддингом 256 и центроидом 512 не определён вовсе: без этой проверки
    отказ пришёл бы из numpy на matmul, без имён и после полного прохода по эталону.
    """
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    чекпоинт = {
        "config": {"backbone": "mobilenetv4_conv_medium.e500_r256_in1k", "embedding_dim": 256},
        "class_index": {"lada__2107": 0},
        "model_state": {},
        "head_state": {},
        "img_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    чекпоинт_файл = tmp_path / "last.pt"
    torch.save(чекпоинт, чекпоинт_файл)

    with pytest.raises(ValueError, match=r"embedding_dim чекпоинта \(256\) не совпадает.*\(512\)"):
        _run_model_evaluation(
            checkpoint_path=чекпоинт_файл,
            centroids=np.zeros((1, 4, 512), dtype=np.float32),
            idx_to_class_id={0: "lada__2107"},
            index_embedding_dim=512,
            holdout_path=tmp_path / "holdout.parquet",
            crops_root=tmp_path / "crops",
            weights="raw",
            device="cpu",
        )


def test_две_трассы_требуют_одинаковые_цели(tmp_path: Path) -> None:
    """Второй проход по loader не может молча измерять другой порядок или набор кадров."""
    pytest.importorskip("torch")
    import torch

    from scripts.eval_finetune_ru import _run_model_evaluation

    class ФиктивныйМодуль:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        def load_state_dict(self, state: object) -> None:
            pass

        def to(self, device: object) -> None:
            pass

    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "config": {"backbone": "stub", "embedding_dim": 2},
            "class_index": {"lada__2107": 0},
            "model_state": {},
            "head_state": {},
            "img_size": 2,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        checkpoint,
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "training.model.create_model",
        lambda *args, **kwargs: (ФиктивныйМодуль(), ФиктивныйМодуль()),
    )
    monkeypatch.setattr("training.samples.read_samples", lambda *args, **kwargs: [object()])
    monkeypatch.setattr("training.dataset.build_transforms", lambda *args, **kwargs: object())
    monkeypatch.setattr("training.dataset.make_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "training.build_index.collect_embeddings",
        lambda *args, **kwargs: (np.array([[1.0, 0.0]], dtype=np.float32), [0]),
    )
    метрики = FewShotMetrics(1.0, 1.0, {"lada__2107": 1.0}, {"lada__2107": 1})
    monkeypatch.setattr(
        "scripts.eval_finetune_ru.evaluate_index_track",
        lambda *args, **kwargs: ([[0]], метрики),
    )
    monkeypatch.setattr(
        "scripts.eval_finetune_ru.evaluate_head_track",
        lambda *args, **kwargs: ([[0]], [1], метрики),
    )

    try:
        with pytest.raises(ValueError, match="цели главной и справочной трасс не совпадают"):
            _run_model_evaluation(
                checkpoint_path=checkpoint,
                centroids=np.zeros((1, 4, 2), dtype=np.float32),
                idx_to_class_id={0: "lada__2107"},
                index_embedding_dim=2,
                holdout_path=tmp_path / "holdout.parquet",
                crops_root=tmp_path / "crops",
                weights="raw",
                device="cpu",
            )
    finally:
        monkeypatch.undo()


def test_несовпадение_хеша_чекпоинта_и_меты_индекса_роняет_main(tmp_path: Path) -> None:
    """Индекс обязан быть построен ровно тем checkpoint, который сейчас измеряется."""
    индекс = tmp_path / "index"
    индекс.mkdir()
    (индекс / "catalog_index.bin").write_bytes(b"dummy")
    (индекс / "catalog_meta.json").write_text(
        json.dumps({"model_weights_sha256": "0" * 64, "classes": []}),
        encoding="utf-8",
    )
    чекпоинт = tmp_path / "last.pt"
    чекпоинт.write_bytes(b"other checkpoint")
    пул = tmp_path / "pool.parquet"
    пул.write_bytes(b"pool")
    эталон = tmp_path / "holdout.parquet"
    эталон.write_bytes(b"holdout")
    базис = tmp_path / "baseline.json"
    базис.write_text("{}", encoding="utf-8")

    from pipeline.index_format import DTYPE_FLOAT32, IndexHeader

    заголовок = IndexHeader(1, 1, 1, DTYPE_FLOAT32, True, b"\x00" * 32)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "scripts.eval_finetune_ru.read_index",
        lambda путь: (заголовок, np.zeros((1, 1, 1), dtype=np.float32)),
    )
    monkeypatch.setattr("scripts.eval_finetune_ru.verify_meta", lambda *args: None)
    try:
        with pytest.raises(ValueError, match=r"model_weights_sha256 меты.*sha256 чекпоинта"):
            main(
                [
                    "--checkpoint",
                    str(чекпоинт),
                    "--index",
                    str(индекс),
                    "--pool",
                    str(пул),
                    "--holdout",
                    str(эталон),
                    "--baseline",
                    str(базис),
                ]
            )
    finally:
        monkeypatch.undo()


def test_отсутствующий_пул_роняет_main(tmp_path: Path) -> None:
    """Провенанс не подменяет отсутствующий обучающий пул значениями unknown и 0."""
    индекс = tmp_path / "index"
    индекс.mkdir()
    (индекс / "catalog_index.bin").write_bytes(b"dummy")
    (индекс / "catalog_meta.json").write_text("{}", encoding="utf-8")
    чекпоинт = tmp_path / "last.pt"
    чекпоинт.write_bytes(b"dummy")
    эталон = tmp_path / "holdout.parquet"
    эталон.write_bytes(b"holdout")
    базис = tmp_path / "baseline.json"
    базис.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"обучающий пул не найден"):
        main(
            [
                "--checkpoint",
                str(чекпоинт),
                "--index",
                str(индекс),
                "--pool",
                str(tmp_path / "missing-pool.parquet"),
                "--holdout",
                str(эталон),
                "--baseline",
                str(базис),
            ]
        )


def test_отсутствующий_эталон_роняет_main(tmp_path: Path) -> None:
    """Эталон проверяется на входе, наравне с чекпоинтом, индексом и базисом.

    Без этой проверки отсутствие файла всплывало бы только внутри `read_samples`, после
    загрузки torch и весов, а провенанс отчёта носил бы `holdout_sha256 = "unknown"`.
    """
    индекс = tmp_path / "index"
    индекс.mkdir()
    (индекс / "catalog_index.bin").write_bytes(b"dummy")
    (индекс / "catalog_meta.json").write_text("{}", encoding="utf-8")
    чекпоинт = tmp_path / "last.pt"
    чекпоинт.write_bytes(b"dummy")
    базис = tmp_path / "fewshot_metrics.json"
    базис.write_text("{}", encoding="utf-8")
    пул = tmp_path / "pool.parquet"
    пул.write_bytes(b"pool")

    with pytest.raises(FileNotFoundError, match=r"эталон не найден"):
        main(
            [
                "--checkpoint",
                str(чекпоинт),
                "--index",
                str(индекс),
                "--baseline",
                str(базис),
                "--pool",
                str(пул),
                "--holdout",
                str(tmp_path / "нет-такого.parquet"),
            ]
        )
