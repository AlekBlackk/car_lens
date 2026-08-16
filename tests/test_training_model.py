"""Тесты модели и головы ArcFace (M-1.2, §4.2 дизайна). Требуют torch."""

import pytest

pytest.importorskip("torch")

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
