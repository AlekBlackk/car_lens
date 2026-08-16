"""Оценка на сплите: прогон загрузчика через модель и сведение метрик.

Разложение логитов в индексы top-5 делается здесь, а арифметика метрик — в
`training.metrics`, который torch не импортирует и потому проверяется в CI.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from training.metrics import Metrics, aggregate
from training.model import ArcFaceHead, CarClassifier

TOP_K = 5


@torch.no_grad()  # type: ignore[untyped-decorator]  # torch.no_grad не типизирован (follow_imports=skip)
def evaluate(
    model: CarClassifier,
    head: ArcFaceHead,
    loader: DataLoader[tuple[Tensor, int]],
    *,
    idx_to_model_id: Mapping[int, str],
    device: torch.device,
) -> Metrics:
    """Метрики закрытого множества. Модель и голова переводятся в `eval`."""
    режим_модели, режим_головы = model.training, head.training
    model.eval()
    head.eval()
    предсказания: list[list[int]] = []
    цели: list[int] = []
    try:
        for картинки, метки in loader:
            логиты = head(model(картинки.to(device, non_blocking=True)))
            сколько = min(TOP_K, логиты.shape[1])
            лучшие = логиты.topk(сколько, dim=1).indices
            предсказания.extend(лучшие.cpu().tolist())
            цели.extend(int(метка) for метка in метки)
    finally:
        model.train(режим_модели)
        head.train(режим_головы)
    return aggregate(предсказания, цели, idx_to_model_id=idx_to_model_id)
