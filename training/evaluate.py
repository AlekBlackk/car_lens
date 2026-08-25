"""Оценка на сплите: прогон загрузчика через модель и сведение метрик.

Разложение логитов в индексы top-5 делается здесь, а арифметика метрик — в
`training.metrics`, который torch не импортирует и потому проверяется в CI.

Autocast здесь нет намеренно, это не забытая оптимизация. `evaluate` считает
`head(model(...))` одним выражением, и под autocast голова попала бы внутрь
целиком — а `ArcFaceHead` численно корректен только в fp32 (`acos` рядом с `±1`,
мина Task 8). Покупать этим нечего: валидация — порядка 13% форвардов прогона и
уже идёт под `no_grad`, то есть речь о единицах процентов общего времени.

Если валидация однажды станет узким местом, правильная форма правки — как в
`train_one_epoch`: autocast только на бэкбоне, а голова вызывается вне него, на
`head(эмбеддинг.float())`.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from training.metrics import TOP_K, Metrics, aggregate
from training.model import ArcFaceHead, CarClassifier


# `torch.no_grad` не типизирован (follow_imports = "skip" на torch), отсюда ignore.
@torch.no_grad()  # type: ignore[untyped-decorator]
def evaluate(
    model: CarClassifier,
    head: ArcFaceHead,
    loader: DataLoader[tuple[Tensor, int]],
    *,
    idx_to_class_id: Mapping[int, str],
    device: torch.device,
) -> Metrics:
    """Метрики закрытого множества. Модель и голова переводятся в `eval`.

    Голова уже ArcFace: её выходов должно быть не меньше `TOP_K`, иначе top-5
    посчитать нечем — см. проверку ниже.
    """
    # Проверка здесь, а не в `aggregate`: там узкий список приходит уже обезличенным
    # и падает сообщением про строку выборки («строка 0: 3 предсказаний»), из которого
    # причину не видно. Ширина логитов равна числу выходов головы, так что узость
    # известна до первого форварда — падаем сразу и по делу.
    классов = int(head.weight.shape[0])
    if классов < TOP_K:
        raise ValueError(
            f"классов меньше {TOP_K} ({классов}), top-5 не определён: "
            f"метрики требуют минимум {TOP_K} выходов головы"
        )
    режим_модели, режим_головы = model.training, head.training
    model.eval()
    head.eval()
    предсказания: list[list[int]] = []
    цели: list[int] = []
    try:
        for картинки, метки in loader:
            логиты = head(model(картинки.to(device, non_blocking=True)))
            лучшие = логиты.topk(TOP_K, dim=1).indices
            предсказания.extend(лучшие.cpu().tolist())
            цели.extend(int(метка) for метка in метки)
    finally:
        model.train(режим_модели)
        head.train(режим_головы)
    return aggregate(предсказания, цели, idx_to_class_id=idx_to_class_id)
