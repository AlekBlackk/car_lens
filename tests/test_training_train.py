"""Смоук-прогон обучения (M-1.2, §9 п.12 дизайна). Требует torch."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

import cv2
import torch

from pipeline.schema import COLUMNS, write_split

# Классов шесть, а не три: `evaluate` сверяет число выходов головы с `TOP_K` до
# форварда и зовёт `topk(TOP_K)` без клампинга — на трёх классах он отказался бы
# считать («классов меньше 5»). Шесть классов гоняют полный путь `topk(5)`.
КЛАССЫ = {
    "acura__rl": 0,
    "acura__tl": 1,
    "bmw__x5": 2,
    "chevrolet__corvette": 3,
    "chevrolet__express": 4,
    "ford__f150": 5,
}
# Базовый lr конфига ниже: тест расписания сверяется с ним, а не с литералом,
# чтобы не стать пустым, если конфиг теста однажды поправят.
БАЗОВЫЙ_LR = 1.0e-3


def каталог(tmp_path: Path) -> Path:
    файл = tmp_path / "catalog.yaml"
    файл.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "brands": {
                    "Acura": {
                        "models": {"rl": {"display": "Acura RL"}, "tl": {"display": "Acura TL"}}
                    },
                    "BMW": {"models": {"x5": {"display": "BMW X5"}}},
                    "Chevrolet": {
                        "models": {
                            "corvette": {"display": "Chevrolet Corvette"},
                            "express": {"display": "Chevrolet Express"},
                        }
                    },
                    "Ford": {"models": {"f150": {"display": "Ford F-150"}}},
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return файл


def конфиг(tmp_path: Path) -> Path:
    файл = tmp_path / "config.yaml"
    файл.write_text(
        yaml.safe_dump(
            {
                "backbone": "mobilenetv4_conv_medium.e500_r256_in1k",
                "embedding_dim": 32,
                "img_size": 64,
                "batch_size": 2,
                "epochs": 2,
                "optimizer": "AdamW",
                "lr": БАЗОВЫЙ_LR,
                "weight_decay": 0.05,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "arcface": {
                    "s": 32.0,
                    "margin_start": 0.2,
                    "margin_end": 0.5,
                    "margin_warmup_epochs": 1,
                },
                "amp": "bf16",
                "ema_decay": 0.9998,
                "grad_clip": 1.0,
                "backbone_lr_mult": 0.1,
                "backbone_lr_mult_epochs": 1,
                "num_workers": 0,
                "seed": 1337,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return файл


def данные(корень: Path) -> None:
    """20 синтетических кропов, разложенных по train/val/test."""
    кропы = корень / "crops" / "stanford"
    кропы.mkdir(parents=True)
    сплиты = корень / "splits"
    сплиты.mkdir(parents=True)
    генератор = np.random.default_rng(0)
    номер = 0
    for имя, сколько in (("train", 12), ("val", 4), ("test", 4)):
        строки = []
        for _ in range(сколько):
            номер += 1
            файл = f"stanford-{номер:05d}.jpg"
            cv2.imwrite(
                str(кропы / файл),
                генератор.integers(0, 255, (80, 80, 3), dtype=np.uint8),
            )
            модель_id, индекс = list(КЛАССЫ.items())[номер % len(КЛАССЫ)]
            строки.append(
                {
                    "photo_id": f"0f8e4f70-aadc-4f70-9f8e-{номер:012d}",
                    "listing_id": f"stanford_{номер:06d}",
                    "source": "stanford",
                    "crop_path": файл,
                    "brand": модель_id.split("__")[0],
                    "model_id": модель_id,
                    "class_idx": индекс,
                    "body_style": "sedan",
                    "year": 2012,
                    "bbox": [1.0, 2.0, 30.0, 20.0],
                    "bbox_area_frac": 0.4,
                    "det_conf": 0.9,
                    "phash": 1234567890123456789 + номер,
                    "raw_model_string": модель_id,
                    "is_multi_car": False,
                }
            )
        write_split(pd.DataFrame(строки, columns=list(COLUMNS)), сплиты / f"{имя}.parquet")


def test_смоук_прогон_пишет_чекпоинт_и_карточку(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.train import main

    данные(tmp_path)
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))
    путь_конфига = конфиг(tmp_path)

    код = main(
        [
            "--config",
            str(путь_конфига),
            "--catalog",
            str(каталог(tmp_path)),
            "--device",
            "cpu",
            "--no-pretrained",
        ]
    )
    assert код == 0

    прогоны = sorted((tmp_path / "runs").glob("m1_2_train_*"))
    assert len(прогоны) == 1
    прогон = прогоны[0]
    for имя in ("config.yaml", "manifest.json", "metrics.json", "report.md", "best.pt", "last.pt"):
        assert (прогон / имя).is_file(), f"нет артефакта {имя}"

    # Копия конфига в каталоге прогона обязана быть побайтной: пересборка YAML
    # из `TrainConfig` дала бы файл с другим порядком ключей и форматом чисел,
    # и сравнить два прогона глазами стало бы труднее.
    assert (прогон / "config.yaml").read_bytes() == путь_конфига.read_bytes()

    метрики = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))
    assert len(метрики["epochs"]) == 2
    assert метрики["epochs"][0]["loss"] > 0
    # Проверка формата, а не EMA: наличие ключей говорит, что оценка EMA доехала до
    # файла, но не что усреднение вообще работало — это проверяет
    # `test_ema_обновляется_на_каждом_шаге`.
    for ключ in ("val", "test", "val_ema", "test_ema"):
        assert "top1_model" in метрики[ключ], f"в metrics.json нет {ключ}.top1_model"

    чекпоинт = torch.load(прогон / "best.pt", map_location="cpu", weights_only=False)
    for ключ in (
        "model_state",
        "ema_state",
        "head_state",
        "class_index",
        "img_size",
        "mean",
        "std",
        "config",
        "epoch",
        "metrics",
        "git_sha",
    ):
        assert ключ in чекпоинт, f"в чекпоинте нет поля {ключ}"
    assert чекпоинт["class_index"] == КЛАССЫ
    # `best.pt` пишется по улучшению, а не всегда: в нём должна лежать эпоха с
    # максимальным `val top1_model` (при равенстве — первая из них, отбор идёт
    # по строгому `>`), а не последняя.
    журнал = метрики["epochs"]
    лучшая = max(range(len(журнал)), key=lambda номер: журнал[номер]["val_top1_model"])
    assert чекпоинт["epoch"] == лучшая

    # Верхний уровень `metrics.json`: по нему сверяют, какие веса поедут в M-1.3, не
    # пересчитывая `max` по журналу и не открывая `best.pt` через torch.
    assert метрики["best_epoch"] == лучшая, "best_epoch в metrics.json не совпал с best.pt"
    assert метрики["best_val_top1"] == pytest.approx(журнал[лучшая]["val_top1_model"])
    # Прогон на cpu: torch не учитывает VRAM, ноль здесь — правильное значение.
    assert метрики["peak_vram_mb"] == pytest.approx(0.0)

    манифест = json.loads((прогон / "manifest.json").read_text(encoding="utf-8"))
    # `config.yaml` рядом — копия файла до override'ов, поэтому применённые флаги
    # должны быть в манифесте: иначе прогон необъясним по артефактам.
    assert манифест["cli_overrides"] == {"no_pretrained": True}

    # Карточка — единственный человекочитаемый артефакт вехи, и проверка «файл
    # существует» проходит на пустом файле. Проверяются четыре вещи, которые в ней
    # обязаны быть: строка таблицы метрик, лучшая эпоха, пик VRAM и дата.
    карточка = (прогон / "report.md").read_text(encoding="utf-8")
    assert "| val | " in карточка, "в карточке нет строки таблицы про val"
    assert f"Лучшая эпоха: {лучшая}," in карточка, "в карточке нет лучшей эпохи"
    assert "- Пик VRAM: 0 МБ" in карточка, "в карточке нет строки про пик VRAM"
    # Дата в карточке обязана совпадать с датой в имени каталога прогона. Проверка
    # сквозная и на подмену метки старта на «сейчас» не ловится: смоук стартует и
    # рендерит в один день. За это отвечает `test_дата_карточки_из_метки_старта`.
    дата_каталога = прогон.name.removeprefix("m1_2_train_")[:8]
    ожидаемая = f"{дата_каталога[:4]}-{дата_каталога[4:6]}-{дата_каталога[6:8]}"
    assert f"Дата: {ожидаемая}" in карточка, "дата карточки не совпала с именем каталога"


def test_дата_карточки_из_метки_старта(tmp_path: Path) -> None:
    """`render_report` печатает дату старта, а не дату рендеринга.

    Метка старта здесь заведомо из другого дня, поэтому возврат к `datetime.now(UTC)`
    внутри `render_report` виден сразу. Сквозная проверка в смоуке этого не может:
    там старт и рендеринг попадают в одни сутки, и обе даты совпадают при любой
    реализации.
    """
    from datetime import UTC, datetime

    from training.config import load_config
    from training.metrics import Metrics
    from training.train import render_report

    метрики = Metrics(
        top1_model=0.5, top5_model=0.75, top1_brand=0.5, per_class_recall={"acura__rl": 0.5}
    )
    журнал = [{"epoch": 0, "loss": 1.0, "lr": БАЗОВЫЙ_LR, "val_top1_model": 0.5, "seconds": 1.0}]

    карточка = render_report(
        load_config(конфиг(tmp_path)),
        журнал,
        метрики,
        метрики,
        val_ema=метрики,
        test_ema=метрики,
        best_epoch=0,
        best_val_top1=0.5,
        device="cpu",
        seconds=61.0,
        peak_vram_mb=0.0,
        started_at=datetime(2020, 1, 1, 23, 59, tzinfo=UTC),
    )

    assert "Дата: 2020-01-01" in карточка
    сегодня = datetime.now(UTC).date().isoformat()
    assert сегодня not in карточка, f"в карточке дата рендеринга ({сегодня}), а не старта"


def test_ema_обновляется_на_каждом_шаге(tmp_path: Path) -> None:
    """Ловит невызванный `ema.update_parameters` внутри `train_one_epoch`.

    Сравнивать веса EMA с сырыми бессмысленно: при `ema_decay: 0.9998` за три
    шага они разойдутся на величину, неотличимую от шума, а при выключенном
    обновлении разойдутся тоже — сырые-то едут. Считается число усреднений
    (`n_averaged` внутри `AveragedModel`): без вызова оно останется нулём.
    """
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    from torch.utils.data import DataLoader, TensorDataset

    from training.config import load_config
    from training.model import create_model
    from training.train import make_optimizer, train_one_epoch

    настройки = load_config(конфиг(tmp_path))
    модель, голова = create_model(
        настройки.backbone,
        num_classes=len(КЛАССЫ),
        embedding_dim=настройки.embedding_dim,
        pretrained=False,
        s=настройки.arcface.s,
        m=настройки.arcface.margin_start,
    )
    картинки = torch.randn(6, 3, настройки.img_size, настройки.img_size)
    метки = torch.tensor([0, 1, 2, 0, 1, 2])
    загрузчик = DataLoader(TensorDataset(картинки, метки), batch_size=настройки.batch_size)
    шагов = 3  # 6 картинок при batch_size 2
    оптимизатор = make_optimizer(
        модель, голова, lr=настройки.lr, weight_decay=настройки.weight_decay
    )
    ema = AveragedModel(модель, multi_avg_fn=get_ema_multi_avg_fn(настройки.ema_decay))
    assert int(ema.n_averaged) == 0

    train_one_epoch(
        модель,
        голова,
        загрузчик,
        оптимизатор,
        ema,
        epoch=0,
        config=настройки,
        device=torch.device("cpu"),
        steps_per_epoch=шагов,
        limit_batches=None,
    )

    assert int(ema.n_averaged) == шагов, "ema.update_parameters вызвана не на каждом шаге"


def test_override_epochs_валидируется_после_применения(tmp_path: Path) -> None:
    """`--epochs 1` при `warmup_epochs: 1` обязан падать, а не молча разогреваться.

    Override едет после `load_config`, поэтому без перепроверки бюджетов прогон
    целиком остался бы в разогреве — и никто бы не узнал.
    """
    from training.config import ConfigError
    from training.train import main

    with pytest.raises(ConfigError, match="--epochs: warmup_epochs=1 не меньше epochs=1"):
        main(
            [
                "--config",
                str(конфиг(tmp_path)),
                "--catalog",
                str(каталог(tmp_path)),
                "--device",
                "cpu",
                "--no-pretrained",
                "--epochs",
                "1",
            ]
        )


def test_лимит_батчей_учитывается_в_знаменателе_расписания(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--limit-batches` должен резать и число шагов, и знаменатель расписания.

    В конфиге теста 12 кропов и batch 2, то есть 6 шагов в полной эпохе. Если
    знаменатель расписания остаётся равным 6, а фактически делается 2 шага, то
    прогон целиком сидит в разогреве: `warmup_epochs=1` растянут на 6 шагов, а
    случится их всего 2 — lr нулевой эпохи не доходит до `base_lr`, и косинус
    не начинается.
    """
    from training.train import main

    данные(tmp_path)
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    код = main(
        [
            "--config",
            str(конфиг(tmp_path)),
            "--catalog",
            str(каталог(tmp_path)),
            "--device",
            "cpu",
            "--no-pretrained",
            "--limit-batches",
            "2",
        ]
    )
    assert код == 0

    прогон = sorted((tmp_path / "runs").glob("m1_2_train_*"))[0]
    журнал = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))["epochs"]
    assert len(журнал) == 2
    # `warmup_epochs=1`: последний шаг нулевой эпохи — последний шаг разогрева,
    # а разогрев по `lr_at` заканчивается ровно на `base_lr`.
    assert журнал[0]["lr"] == pytest.approx(БАЗОВЫЙ_LR)
    # Дальше идёт косинус вниз, поэтому эпоха 1 обязана уйти ниже `base_lr`.
    assert журнал[1]["lr"] < БАЗОВЫЙ_LR

    # Оба применённых флага обязаны попасть в манифест, а не только последний.
    манифест = json.loads((прогон / "manifest.json").read_text(encoding="utf-8"))
    assert манифест["cli_overrides"] == {"limit_batches": 2, "no_pretrained": True}
