"""Тесты модели и головы ArcFace (M-1.2, §4.2 дизайна). Требуют torch."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

import numpy as np
import timm
import torch

from training.model import ArcFaceHead, create_model

# Классов шесть, а не три: `evaluate` сверяет число выходов головы с `TOP_K` до
# форварда и зовёт `topk(TOP_K)` без клампинга — на трёх классах он отказался бы
# считать («классов меньше 5»). Шесть классов гоняют полный путь `topk(5)`.
ИНДЕКС_КЛАССОВ = {
    0: "acura__rl",
    1: "acura__tl",
    2: "bmw__x5",
    3: "chevrolet__corvette",
    4: "chevrolet__express",
    5: "ford__f150",
}

# Боевой бэкбон и второй, у которого `mean`/`std` другие: ImageNet против
# inception-овых `(0.5, 0.5, 0.5)`. На одном бэкбоне проверка провенанса
# невозможна — совпадение с ImageNet одинаково объясняется и чтением конфига,
# и хардкодом ровно тех же чисел.
БЭКБОНЫ = ("mobilenetv4_conv_medium.e500_r256_in1k", "tf_mobilenetv3_small_100.in1k")


def test_фабрика_даёт_эмбеддинг_нужной_размерности() -> None:
    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=171, pretrained=False
    )
    модель.eval()
    эмбеддинг = модель(torch.randn(2, 3, 320, 320))
    assert эмбеддинг.shape == (2, 256)
    assert голова(эмбеддинг).shape == (2, 171)


def test_фабрика_отдаёт_mean_и_std_из_бэкбона() -> None:
    """`mean`/`std` совпадают с `pretrained_cfg`, а не с захардкоженным ImageNet.

    Конфиг берётся через `timm.get_pretrained_cfg`, то есть в обход `create_model`:
    сравнение модели с её же полями хардкод не поймало бы. Проверяются оба бэкбона
    из `БЭКБОНЫ` и отдельно — что константы у них правда разные. Это цена §11 п.6:
    на фолбэке бэкбона хардкод разъехался бы с весами молча.
    """
    конфиги = [timm.get_pretrained_cfg(имя) for имя in БЭКБОНЫ]
    первый, второй = конфиги
    assert (первый.mean, первый.std) != (второй.mean, второй.std), (
        "бэкбоны в БЭКБОНЫ перестали различаться по mean/std — пара ничего не проверяет"
    )
    for имя, конфиг in zip(БЭКБОНЫ, конфиги, strict=True):
        модель, _ = create_model(имя, num_classes=len(ИНДЕКС_КЛАССОВ), pretrained=False)
        assert модель.mean == tuple(конфиг.mean), имя
        assert модель.std == tuple(конфиг.std), имя


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


def test_голова_под_autocast_отдаёт_fp32_и_клампит() -> None:
    """Контракт fp32 у головы: под bf16-autocast она всё равно считает в fp32.

    Правка ради будущего вызывающего (`build_index.py` на M-1.3), поэтому проверять
    надо именно под autocast: на fp32-пути `.float()` — no-op, и его отсутствие там
    неотличимо. Вес и эмбеддинг выставлены так, что косинус на диагонали ровно `1.0`
    — тот единственный вход, где разница видна: в fp32 `clamp` вернёт `1 - 1e-7`, а в
    bf16 граница `1 - 1e-7` сама округляется до `1.0` (мантисса 8 бит), клэмп
    становится тождеством и `acos` уезжает в `NaN` по градиенту.

    CUDA не нужна: bf16-autocast на cpu в torch 2.12 работает.
    """
    голова = ArcFaceHead(4, 3, s=32.0, m=0.5)
    with torch.no_grad():
        голова.weight.copy_(torch.eye(3, 4))
    эмбеддинг = torch.eye(3, 4).requires_grad_()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        логиты = голова(эмбеддинг)
    assert логиты.dtype is torch.float32, "голова отдала не fp32 — приведение косинуса потерялось"
    # В bf16 здесь вышло бы ровно `s`: клэмп выродился бы в тождество.
    assert float(логиты.max().detach()) < голова.s, "clamp не сработал: косинус дошёл до ±1"

    with torch.autocast("cpu", dtype=torch.bfloat16):
        голова(эмбеддинг, torch.tensor([0, 1, 2])).sum().backward()
    градиент = эмбеддинг.grad
    assert градиент is not None
    assert bool(torch.isfinite(градиент).all()), "NaN в градиенте: `acos` получил ровно ±1"


def test_оценка_на_синтетическом_загрузчике(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    from training.evaluate import evaluate

    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k",
        num_classes=len(ИНДЕКС_КЛАССОВ),
        pretrained=False,
    )
    картинки = torch.randn(4, 3, 64, 64)
    метки = torch.tensor([0, 1, 2, 0])
    загрузчик = DataLoader(TensorDataset(картинки, метки), batch_size=2)
    метрики = evaluate(
        модель,
        голова,
        загрузчик,
        idx_to_class_id=ИНДЕКС_КЛАССОВ,
        device=torch.device("cpu"),
    )
    assert 0.0 <= метрики.top1_model <= 1.0
    assert метрики.top5_model >= метрики.top1_model
    assert set(метрики.per_class_recall) <= set(ИНДЕКС_КЛАССОВ.values())
    assert np.isfinite(метрики.top1_brand)


def test_меньше_пяти_классов_отвергается_понятным_сообщением() -> None:
    """Узкая голова падает в `evaluate`, а не внутри `aggregate`.

    До явной проверки `topk(min(TOP_K, ширина))` молча отдавал список шириной три,
    и падал уже `aggregate` сообщением «строка 0: 3 предсказаний, а top5 требует 5»
    — по сути верным, но про причину не говорящим. Тест закрепляет, что причина
    называется там, где она видна: у головы меньше пяти выходов.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from training.evaluate import evaluate

    модель, голова = create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k", num_classes=4, pretrained=False
    )
    выборка = TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
    узкий_индекс = {0: "acura__rl", 1: "acura__tl", 2: "bmw__x5", 3: "ford__f150"}
    with pytest.raises(ValueError, match=r"классов меньше 5 \(4\), top-5 не определён"):
        evaluate(
            модель,
            голова,
            DataLoader(выборка, batch_size=2),
            idx_to_class_id=узкий_индекс,
            device=torch.device("cpu"),
        )


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
    idx_to_class_id = {индекс: f"марка{индекс}__модель{индекс}" for индекс in range(8)}

    метрики = evaluate(
        модель, голова, загрузчик, idx_to_class_id=idx_to_class_id, device=torch.device("cpu")
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
        "mobilenetv4_conv_medium.e500_r256_in1k",
        num_classes=len(ИНДЕКС_КЛАССОВ),
        pretrained=False,
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
        idx_to_class_id=ИНДЕКС_КЛАССОВ,
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
        "mobilenetv4_conv_medium.e500_r256_in1k",
        num_classes=len(ИНДЕКС_КЛАССОВ),
        pretrained=False,
    )
    модель.train()
    голова.eval()
    загрузчик = DataLoader(ПадающийDataset(), batch_size=2)

    with pytest.raises(RuntimeError):
        evaluate(
            модель,
            голова,
            загрузчик,
            idx_to_class_id=ИНДЕКС_КЛАССОВ,
            device=torch.device("cpu"),
        )

    assert модель.training is True
    assert голова.training is False
