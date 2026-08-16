"""Тесты модели и головы ArcFace (M-1.2, §4.2 дизайна). Требуют torch."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from training.model import ArcFaceHead, create_model


def test_фабрика_даёт_эмбеддинг_нужной_размерности() -> None:
    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=171, pretrained=False
    )
    модель.eval()
    эмбеддинг = модель(torch.randn(2, 3, 320, 320))
    assert эмбеддинг.shape == (2, 256)
    assert голова(эмбеддинг).shape == (2, 171)


def test_фабрика_отдаёт_mean_и_std_из_бэкбона() -> None:
    модель, _ = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=3, pretrained=False
    )
    assert len(модель.mean) == 3
    assert len(модель.std) == 3
    assert all(0.0 < значение < 1.0 for значение in модель.mean + модель.std)


def test_эмбеддинг_ненормированный() -> None:
    модель, _ = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=3, pretrained=False
    )
    модель.eval()
    нормы = модель(torch.randn(4, 3, 320, 320)).norm(dim=1)
    assert not torch.allclose(нормы, torch.ones_like(нормы), atol=1e-3)


def test_голова_без_маржи_даёт_косинус_на_s() -> None:
    голова = ArcFaceHead(8, 4, s=32.0, m=0.0)
    эмбеддинг = torch.randn(5, 8)
    цели = torch.tensor([0, 1, 2, 3, 0])
    с_целью = голова(эмбеддинг, цели)
    без_цели = голова(эмбеддинг)
    assert torch.allclose(с_целью, без_цели, atol=1e-5)


def test_маржа_снижает_логит_истинного_класса() -> None:
    голова = ArcFaceHead(8, 4, s=32.0, m=0.5)
    эмбеддинг = torch.randn(5, 8)
    цели = torch.tensor([0, 1, 2, 3, 0])
    без_маржи = голова(эмбеддинг)
    с_маржой = голова(эмбеддинг, цели)
    истинные_без = без_маржи.gather(1, цели[:, None])
    истинные_с = с_маржой.gather(1, цели[:, None])
    assert bool((истинные_с < истинные_без).all())


def test_target_none_не_зависит_от_маржи() -> None:
    голова = ArcFaceHead(8, 4, s=32.0, m=0.2)
    эмбеддинг = torch.randn(3, 8)
    было = голова(эмбеддинг)
    голова.set_margin(0.5)
    стало = голова(эмбеддинг)
    assert torch.allclose(было, стало, atol=1e-6)


def test_оценка_на_синтетическом_загрузчике(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    from training.evaluate import evaluate

    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=3, pretrained=False
    )
    картинки = torch.randn(4, 3, 64, 64)
    метки = torch.tensor([0, 1, 2, 0])
    загрузчик = DataLoader(TensorDataset(картинки, метки), batch_size=2)
    метрики = evaluate(
        модель,
        голова,
        загрузчик,
        idx_to_model_id={0: "acura__rl", 1: "acura__tl", 2: "bmw__x5"},
        device=torch.device("cpu"),
    )
    assert 0.0 <= метрики.top1_model <= 1.0
    assert метрики.top5_model >= метрики.top1_model
    assert set(метрики.per_class_recall) <= {"acura__rl", "acura__tl", "bmw__x5"}
    assert np.isfinite(метрики.top1_brand)


def test_top1_model_совпадает_с_ручным_подсчётом() -> None:
    """Численная проверка: `top1_model` — не тавтология `<= 1.0`, а argmax-доля.

    `num_classes=8 > TOP_K` — иначе `top5` покрывал бы все классы и
    `top5_model` был бы равен 1.0 по построению, независимо от поведения кода.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from training.evaluate import evaluate

    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=8, pretrained=False
    )
    модель.eval()
    картинки = torch.randn(9, 3, 64, 64)
    метки = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0])
    загрузчик = DataLoader(TensorDataset(картинки, метки), batch_size=4)
    idx_to_model_id = {индекс: f"марка{индекс}__модель{индекс}" for индекс in range(8)}

    метрики = evaluate(
        модель, голова, загрузчик, idx_to_model_id=idx_to_model_id, device=torch.device("cpu")
    )

    with torch.no_grad():
        эталонные_логиты = голова(модель(картинки))
    эталонные_предсказания = эталонные_логиты.argmax(dim=1)
    эталонный_top1 = (эталонные_предсказания == метки).float().mean().item()

    assert метрики.top1_model == pytest.approx(эталонный_top1)
    assert метрики.top5_model < 1.0


def test_режимы_модели_и_головы_восстанавливаются_по_отдельности() -> None:
    """Порча состояния через общий флаг — регрессия из ревью Task 7."""
    from torch.utils.data import DataLoader, TensorDataset

    from training.evaluate import evaluate

    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=3, pretrained=False
    )
    модель.train()
    голова.eval()
    картинки = torch.randn(4, 3, 64, 64)
    метки = torch.tensor([0, 1, 2, 0])
    загрузчик = DataLoader(TensorDataset(картинки, метки), batch_size=2)

    evaluate(
        модель,
        голова,
        загрузчик,
        idx_to_model_id={0: "acura__rl", 1: "acura__tl", 2: "bmw__x5"},
        device=torch.device("cpu"),
    )

    assert модель.training is True
    assert голова.training is False


def test_режимы_восстанавливаются_при_исключении_внутри_цикла() -> None:
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset

    from training.evaluate import evaluate

    class ПадающийDataset(Dataset[tuple[Tensor, int]]):  # type: ignore[misc]
        """Датасет, ломающийся на чтении батча — для проверки `finally`."""

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[Tensor, int]:
            raise RuntimeError("тест: ошибка чтения батча")

    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=3, pretrained=False
    )
    модель.train()
    голова.eval()
    загрузчик = DataLoader(ПадающийDataset(), batch_size=2)

    with pytest.raises(RuntimeError):
        evaluate(
            модель,
            голова,
            загрузчик,
            idx_to_model_id={0: "acura__rl", 1: "acura__tl", 2: "bmw__x5"},
            device=torch.device("cpu"),
        )

    assert модель.training is True
    assert голова.training is False
