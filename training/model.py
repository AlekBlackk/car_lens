"""Бэкбон, голова эмбеддинга и ArcFace (§4.2).

Наружу `CarClassifier` отдаёт **ненормированный** эмбеддинг (Р-18):
L2-нормализация делается внутри `ArcFaceHead` при обучении и при сборке
индекса эталонов на M-1.3. Нормализация в двух местах — верный способ
нормировать дважды и получить константу вместо признака.

`mean`/`std` берутся из `pretrained_cfg` бэкбона, а не хардкодятся: при
переходе на фолбэк из §4.2 хардкод разъехался бы с весами молча.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class ArcFaceHead(nn.Module):  # type: ignore[misc]
    """Косинусный классификатор с угловой маржой."""

    def __init__(self, embedding_dim: int, num_classes: int, *, s: float, m: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def set_margin(self, m: float) -> None:
        """Обновить маржу по расписанию §4.2. Вызывается раз в эпоху."""
        self.m = m

    def forward(self, embedding: Tensor, target: Tensor | None = None) -> Tensor:
        """Логиты. Без `target` — `cosine * s`, ими считаются метрики."""
        косинус = F.linear(F.normalize(embedding), F.normalize(self.weight))
        косинус = косинус.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if target is None:
            return косинус * self.s
        с_маржой = torch.cos(torch.acos(косинус) + self.m)
        маска = F.one_hot(target, num_classes=self.weight.shape[0]).bool()
        return torch.where(маска, с_маржой, косинус) * self.s


class CarClassifier(nn.Module):  # type: ignore[misc]
    """Бэкбон + голова эмбеддинга: `BN → Linear(bias=False) → BN` (§4.2)."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        *,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # `num_features` — каналы после `forward_features()`, ДО головы бэкбона.
        # У моделей с расширяющей conv-головой (например, mobilenetv4_conv_medium:
        # 960 -> 1280) именно `head_hidden_size` совпадает с реальной шириной
        # тензора, который отдаёт вызов backbone(x) при num_classes=0. Для
        # бэкбонов без отдельной головы оба атрибута равны, так что откат к
        # `num_features` там, где `head_hidden_size` не определён, безопасен.
        число_признаков = getattr(backbone, "head_hidden_size", backbone.num_features)
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(число_признаков),
            nn.Linear(число_признаков, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.mean = mean
        self.std = std

    def forward(self, image: Tensor) -> Tensor:
        """Эмбеддинг размерности `embedding_dim`, ненормированный."""
        признаки: Tensor = self.backbone(image)
        эмбеддинг: Tensor = self.embedding(признаки)
        return эмбеддинг


def create_model(
    backbone: str,
    *,
    num_classes: int,
    embedding_dim: int = 256,
    pretrained: bool = True,
    s: float = 32.0,
    m: float = 0.5,
) -> tuple[CarClassifier, ArcFaceHead]:
    """Собрать модель и голову.

    `pretrained=False` существует ради тестов: они не должны ходить в сеть
    за весами.
    """
    сеть = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
    конфиг = сеть.pretrained_cfg
    среднее = (float(конфиг["mean"][0]), float(конфиг["mean"][1]), float(конфиг["mean"][2]))
    отклонение = (float(конфиг["std"][0]), float(конфиг["std"][1]), float(конфиг["std"][2]))
    модель = CarClassifier(сеть, embedding_dim, mean=среднее, std=отклонение)
    голова = ArcFaceHead(embedding_dim, num_classes, s=s, m=m)
    return модель, голова
