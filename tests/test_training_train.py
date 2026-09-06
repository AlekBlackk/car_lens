"""Смоук-прогон обучения (M-1.2, §9 п.12 дизайна). Требует torch."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

import cv2
import torch
from torch import Tensor

from pipeline.schema import COLUMNS, write_split
from training.metrics import Metrics
from training.model import ArcFaceHead

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


def каталог_с_группой(tmp_path: Path) -> Path:
    """Тот же каталог плюс седьмая модель, объединённая с `acura__rl` в группу.

    Семь моделей и шесть классов: числа заведомо разные, поэтому голова,
    посчитанная по моделям, отличается от посчитанной по классам одним
    выходом. Группа встаёт на место `acura__rl` (минимальный `model_id`
    участников), так что остальные пять индексов совпадают с `КЛАССЫ` —
    синтетические данные годятся без переразметки.
    """
    сырой = yaml.safe_load(каталог(tmp_path).read_text(encoding="utf-8"))
    сырой["brands"]["Honda"] = {"models": {"legend": {"display": "Honda Legend"}}}
    сырой["groups"] = {
        "grp__legend": {
            "members": ["acura__rl", "honda__legend"],
            "display": "Acura RL / Honda Legend",
        }
    }
    файл = tmp_path / "catalog_с_группой.yaml"
    файл.write_text(yaml.safe_dump(сырой, sort_keys=False, allow_unicode=True), encoding="utf-8")
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


def данные(
    корень: Path,
    *,
    кропы: Path | None = None,
    состав: Sequence[tuple[str, int]] = (("train", 12), ("val", 4), ("test", 4)),
) -> Path:
    """20 синтетических кропов, разложенных по train/val/test. Отдаёт корень кропов.

    `кропы` — куда лечь файлам; по умолчанию `crops/stanford`, то есть туда, где их
    ищет контур M-1.2 без флагов. `состав` — какие сплиты и по сколько кадров.
    Русскому пути нужен один `pool` под своим корнем, и ни `crops/stanford`, ни
    `train.parquet` при этом не создаётся вовсе: прогон, взявший умолчание вместо
    флага, упадёт на отсутствии файла, а не подменит данные молча.
    """
    каталог_кропов = кропы if кропы is not None else корень / "crops" / "stanford"
    каталог_кропов.mkdir(parents=True)
    сплиты = корень / "splits"
    сплиты.mkdir(parents=True)
    генератор = np.random.default_rng(0)
    номер = 0
    for имя, сколько in состав:
        строки = []
        for _ in range(сколько):
            номер += 1
            файл = f"stanford-{номер:05d}.jpg"
            cv2.imwrite(
                str(каталог_кропов / файл),
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
    return каталог_кропов


def данные_пула(корень: Path) -> Path:
    """Данные русского пути: 12 кропов под `harvest/crops` и один сплит `pool`."""
    return данные(корень, кропы=корень / "harvest" / "crops", состав=(("pool", 12),))


def прогон_пула(корень: Path, *, дополнительно: Sequence[str] = ()) -> Path:
    """Прогон русского пути на фикстуре; отдаёт каталог его артефактов.

    Общие флаги: кропы из `harvest/crops`, обучающий сплит `pool`, пустой
    `--eval-splits`. `дополнительно` дописывается перед ним — свой префикс каталога
    или свою лицензию тесты передают сами. `--eval-splits` идёт последним нарочно:
    при `nargs="*"` он забирает всё до следующего флага, и пустым списком остаётся
    только в конце.
    """
    from training.train import main

    кропы = данные_пула(корень)
    код = main(
        [
            "--config",
            str(конфиг(корень)),
            "--catalog",
            str(каталог(корень)),
            "--device",
            "cpu",
            "--no-pretrained",
            "--limit-batches",
            "2",
            "--crops-root",
            str(кропы),
            "--train-split",
            "pool",
            *дополнительно,
            "--eval-splits",
        ]
    )
    assert код == 0
    прогоны = sorted(путь for путь in (корень / "runs").iterdir() if путь.is_dir())
    assert len(прогоны) == 1, f"каталогов прогона {len(прогоны)}, ожидался один"
    return прогоны[0]


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


def test_голова_считается_по_классам_а_не_по_моделям(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Выходов головы ArcFace столько же, сколько классов, а не моделей (§13.2).

    Каталог здесь — семь моделей и шесть классов: платформенная группа
    занимает один слот `catalog_index.bin` и один выход головы. Голова,
    посчитанная по `len(каталог)`, дала бы седьмой выход, который никогда
    не активируется, — и сборка индекса в M-1.3 упала бы на сверке
    нумерации, потому что `class_index` каталога знает шесть ключей.

    Заодно проверяется, чем размечен `class_index` чекпоинта: ключи —
    `class_id`, и `model_id` участников группы в них не встречается.
    Обратное отображение из него раскрашивает `per_class_recall`, и подписи
    по `model_id` показывали бы одного участника вместо класса.
    """
    from training.train import main

    данные(tmp_path)
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    код = main(
        [
            "--config",
            str(конфиг(tmp_path)),
            "--catalog",
            str(каталог_с_группой(tmp_path)),
            "--device",
            "cpu",
            "--no-pretrained",
            "--limit-batches",
            "2",
        ]
    )
    assert код == 0

    прогон = sorted((tmp_path / "runs").glob("m1_2_train_*"))[0]
    чекпоинт = torch.load(прогон / "best.pt", map_location="cpu", weights_only=False)
    assert чекпоинт["class_index"] == {
        "grp__legend": 0,
        "acura__tl": 1,
        "bmw__x5": 2,
        "chevrolet__corvette": 3,
        "chevrolet__express": 4,
        "ford__f150": 5,
    }
    # Семь моделей в YAML — шесть выходов головы. Литерал 6, а не
    # `len(чекпоинт["class_index"])`: сверять размер головы с той же картой,
    # из которой он посчитан, — тавтология, тест прошёл бы и на семи выходах.
    assert чекпоинт["head_state"]["weight"].shape[0] == 6


def test_дата_карточки_из_метки_старта(tmp_path: Path) -> None:
    """`render_report` печатает дату старта, а не дату рендеринга.

    Метка старта здесь заведомо из другого дня, поэтому возврат к `datetime.now(UTC)`
    внутри `render_report` виден сразу. Сквозная проверка в смоуке этого не может:
    там старт и рендеринг попадают в одни сутки, и обе даты совпадают при любой
    реализации.
    """
    from datetime import UTC, datetime

    from training.config import load_config
    from training.train import render_report

    метрики = Metrics(
        top1_model=0.5, top5_model=0.75, top1_brand=0.5, per_class_recall={"acura__rl": 0.5}
    )
    журнал = [{"epoch": 0, "loss": 1.0, "lr": БАЗОВЫЙ_LR, "val_top1_model": 0.5, "seconds": 1.0}]

    карточка = render_report(
        load_config(конфиг(tmp_path)),
        журнал,
        {"val": метрики, "test": метрики},
        card_title="M-1.2",
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


def test_пустые_eval_splits_обходятся_без_проверочных_данных(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогон без проверочного сплита доходит до конца: в фикстуре только обучающий.

    Русский путь §3 дизайна: валидационного среза нет по числам, а эталон в обучении
    не смотрится вовсе. Ни `val.parquet`, ни `test.parquet` в фикстуре не существует,
    поэтому контур, который всё ещё собирает проверочные загрузчики или зовёт
    `evaluate`, падает на отсутствии файла, а не проходит молча.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(tmp_path)

    assert not (tmp_path / "splits" / "val.parquet").exists(), "фикстура не та: val на месте"
    assert (прогон / "last.pt").is_file()
    метрики = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))
    assert len(метрики["epochs"]) == 2, "эпох в конфиге две"
    assert метрики["epochs"][0]["loss"] > 0
    assert "val_top1_model" not in метрики["epochs"][0], "поэпохальный замер всё-таки был"
    # Финального замера тоже не было, и ключей сплитов в файле нет вовсе: пустые
    # словари читались бы как «померили и не получили ничего».
    лишние = {"val", "test", "pool"} & set(метрики)
    assert not лишние, f"в metrics.json попал финальный замер: {sorted(лишние)}"


def test_без_валидации_best_pt_не_пишется(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без проверочного сплита отбор лучших весов выключен целиком.

    Нацелить отбор на эталон значило бы выбрать веса по линейке (§3 дизайна), а без
    замера выбирать нечем вовсе. Поэтому `best.pt` не появляется, а не дублирует
    `last.pt`, и `best_epoch` остаётся -1 — по нему видно, что отбора не было.
    Ключ `metrics` чекпоинта при этом `None`: ни один читатель чекпоинта его не
    смотрит — ни `training/build_index.py`, ни `scripts/eval_index.py`, ни `export/`.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(tmp_path)

    assert (прогон / "best.pt").exists() is False
    метрики = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))
    assert метрики["best_epoch"] == -1
    чекпоинт = torch.load(прогон / "last.pt", map_location="cpu", weights_only=False)
    assert чекпоинт["metrics"] is None


def test_корень_кропов_и_имя_обучающего_сплита_из_флагов(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Кропы читаются из переданного корня, обучающий parquet — по переданному имени.

    Умолчаний в фикстуре нет физически: ни каталога `crops/stanford`, ни
    `train.parquet`. Прогон, прошедший обе эпохи с ненулевым loss, прочитал именно
    переданное: `read_samples` падает на первом же отсутствующем файле кропа.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(tmp_path)

    assert not (tmp_path / "crops").exists(), "умолчание кропов на месте: тест ничего не ловит"
    assert not (tmp_path / "splits" / "train.parquet").exists(), "умолчание сплита на месте"
    журнал = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))["epochs"]
    assert [запись["epoch"] for запись in журнал] == [0, 1]
    assert all(запись["loss"] > 0 for запись in журнал)


def test_префикс_каталога_прогона_из_флага(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Каталог прогона называется по переданному префиксу, а не по `m1_2_train`.

    Прогон по русскому пулу — не M-1.2, и складывать его артефакты в каталог с чужим
    именем значит терять их среди прогонов вехи при первом же `ls`.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(tmp_path, дополнительно=["--run-prefix", "pool_ru_train"])

    assert прогон.name.startswith("pool_ru_train_"), f"каталог прогона {прогон.name}"


def test_манифест_несёт_датасет_и_лицензию_из_флагов(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`manifest.json` пишет переданные датасет и лицензию, а не литералы Stanford.

    Манифест — единственное место, где записано, на чём обучались и на каких правах.
    Зашитые `stanford_cars` и `research-only` в прогоне по русскому пулу были бы
    прямым враньём об источнике кадров и об их лицензии.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(
        tmp_path, дополнительно=["--dataset", "ru_pool", "--licence", "mixed-cc-by"]
    )

    манифест = json.loads((прогон / "manifest.json").read_text(encoding="utf-8"))
    assert манифест["datasets"] == ["ru_pool"]
    assert манифест["licence"] == "mixed-cc-by"


def test_манифест_несёт_подмену_данных_в_cli_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Флаги данных и раскладки едут в `cli_overrides`: иначе прогон необъясним.

    `config.yaml` рядом с манифестом — копия файла до override'ов, датасет и лицензия
    лежат отдельными полями, заголовок карточки виден в самой карточке. А вот из какого
    корня читались кропы, как звался обучающий сплит и был ли замер — кроме манифеста,
    узнать негде. Пишется только отличие от умолчания, поэтому у прогона M-1.2 объект
    остаётся прежним: это сверяет `test_лимит_батчей_учитывается_в_знаменателе_расписания`.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(tmp_path, дополнительно=["--run-prefix", "pool_ru_train"])

    переопределения = json.loads((прогон / "manifest.json").read_text(encoding="utf-8"))[
        "cli_overrides"
    ]
    assert Path(переопределения["crops_root"]) == tmp_path / "harvest" / "crops"
    assert переопределения["train_split"] == "pool"
    assert переопределения["eval_splits"] == []
    assert переопределения["run_prefix"] == "pool_ru_train"


def test_заголовок_карточки_из_флага(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Первая строка `report.md` — переданный заголовок, а не веха M-1.2.

    Эту строку копируют в решения, и прогон по русскому пулу, представляющийся вехой
    M-1.2, врал бы в самом заметном месте карточки.
    """
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))

    прогон = прогон_пула(tmp_path, дополнительно=["--card-title", "дообучения по русскому пулу"])

    карточка = (прогон / "report.md").read_text(encoding="utf-8")
    assert карточка.startswith("# Карточка прогона дообучения по русскому пулу\n")
    assert "M-1.2" not in карточка.splitlines()[0]


def test_карточка_без_проверочных_сплитов_говорит_об_этом(tmp_path: Path) -> None:
    """Пустые итоги: карточка объясняет, почему метрик нет, и таблицу не печатает.

    Пустая таблица или строка нулей читалась бы как «померили и получили нуль».
    Вместо неё — одно объяснение: замер эталона идёт отдельным шагом. Лучшей эпохи
    в карточке тоже нет: отбора не было, и печатать `-1` значит печатать мусор.
    """
    from datetime import UTC, datetime

    from training.config import load_config
    from training.train import render_report

    журнал = [
        {
            "epoch": 0,
            "loss": 2.0,
            "lr": БАЗОВЫЙ_LR,
            "margin": 0.2,
            "train_top1": 0.25,
            "seconds": 5.0,
        }
    ]

    карточка = render_report(
        load_config(конфиг(tmp_path)),
        журнал,
        {},
        card_title="дообучения по русскому пулу",
        best_epoch=-1,
        best_val_top1=-1.0,
        device="cpu",
        seconds=61.0,
        peak_vram_mb=0.0,
        started_at=datetime(2020, 1, 1, 23, 59, tzinfo=UTC),
    )

    assert "Проверочных сплитов" in карточка
    assert "отдельным шагом" in карточка
    assert "| Сплит |" not in карточка, "карточка печатает таблицу метрик, которых нет"
    assert "Десять худших" not in карточка
    assert "Лучшая эпоха" not in карточка, "карточка печатает лучшую эпоху, которой не было"
    # Столбец точности в таблице эпох остаётся, но обучающий: подпись «учится или
    # сломано» — единственное поэпохальное число такого прогона.
    assert "| Эпоха | loss | lr | train top1 | сек |" in карточка
    assert "| 0 | 2.0000 | 1.00e-03 | 0.2500 | 5 |" in карточка


ОЖИДАЕМАЯ_КАРТОЧКА_STANFORD = """\
# Карточка прогона M-1.2

Дата: 2020-01-01

- Бэкбон: `mobilenetv4_conv_medium.e500_r256_in1k`
- Классов: 3 из встреченных в val
- Эпох: 2, batch 2, img_size 64
- Устройство: cpu, время прогона: 25.0 мин
- Пик VRAM: 2048 МБ

## Метрики

| Сплит | top1_model | top5_model | top1_brand |
|---|---|---|---|
| val | 0.5100 | 0.7100 | 0.6100 |
| test | 0.5200 | 0.7200 | 0.6200 |
| val (EMA) | 0.5300 | 0.7300 | 0.6300 |
| test (EMA) | 0.5400 | 0.7400 | 0.6400 |

Лучшая эпоха: 1, val top1_model = 0.5100. Именно её веса
лежат в `best.pt`, и в M-1.3 поедут они, а не финальные из таблицы выше —
порог ниже сверяйте с этим числом.

Строки `(EMA)` считаются один раз в конце прогона, и читать их как ответ на
вопрос «сырые веса или усреднённые» нельзя. В `AveragedModel` завёрнут только
бэкбон: голова ArcFace остаётся сырой, а BN-буферы torch при `use_buffers=False`
копирует из живой модели на каждом `update_parameters`. Измерен, значит, гибрид —
усреднённый бэкбон с сырой головой и сырыми буферами, а не усреднённая модель.
Числа посчитаны и воспроизводимы, но аргументом за или против усреднения не
являются: веса для M-1.3 берутся сырые, из `best.pt`. Подробности — долг 7 в §11
дизайна; честная оценка требует усреднять голову вместе с бэкбоном и стоит ещё
одного прогона, поэтому на M-1.2 не делалась.

Отдельно: числа `(EMA)` посчитаны на финальном усреднении, то есть на
`ema_state` из `last.pt`. В `best.pt` лежит `ema_state` лучшей по сырым весам
эпохи — другие тензоры, и к числам выше они отношения не имеют.

Пороги §5.2 к этому прогону не применяются (Р-13): они про 150 классов на
уличных данных. Единственный порог здесь — `val top1_model >= 0.50`, пол
здравомыслия, отделяющий работающий пайплайн от бага в разметке или
нормализации.

## Десять худших классов по recall на val

- `acura__tl`: 0.100
- `bmw__x5`: 0.200
- `acura__rl`: 0.300

## По эпохам

| Эпоха | loss | lr | val top1 | сек |
|---|---|---|---|---|
| 0 | 2.5000 | 1.00e-03 | 0.4000 | 12 |
| 1 | 1.2500 | 5.00e-04 | 0.5100 | 11 |
"""


def test_карточка_stanford_не_изменилась(tmp_path: Path) -> None:
    """Текст карточки для ключей `val`/`test`/`val_ema`/`test_ema` — прежний, до байта.

    Главная страховка задачи: `render_report` перешёл с четырёх именованных метрик на
    словарь итогов, и таблица теперь строится обходом ключей. Эталон выше снят с
    реализации до правки. Значения различаются в каждом разряде, а непустой
    поклассовый recall только у `val`, поэтому перепутанные строки, столбцы или взятый
    не первым не-EMA ключ видны, а не совпадают случайно. Лишний ключ `train_top1` в
    журнале — тоже проверка: в таблице эпох вехи M-1.2 он появиться не должен.
    """
    from datetime import UTC, datetime

    from training.config import load_config
    from training.train import render_report

    поклассово = {"acura__rl": 0.30, "acura__tl": 0.10, "bmw__x5": 0.20}
    итоги = {
        "val": Metrics(0.51, 0.71, 0.61, поклассово),
        "test": Metrics(0.52, 0.72, 0.62, {"ford__f150": 0.01}),
        "val_ema": Metrics(0.53, 0.73, 0.63, {"ford__f150": 0.02}),
        "test_ema": Metrics(0.54, 0.74, 0.64, {"ford__f150": 0.03}),
    }
    журнал = [
        {
            "epoch": 0,
            "loss": 2.5,
            "lr": 1.0e-3,
            "margin": 0.2,
            "train_top1": 0.125,
            "val_top1_model": 0.4,
            "val_top5_model": 0.6,
            "val_top1_brand": 0.5,
            "seconds": 12.0,
        },
        {
            "epoch": 1,
            "loss": 1.25,
            "lr": 5.0e-4,
            "margin": 0.5,
            "train_top1": 0.5,
            "val_top1_model": 0.51,
            "val_top5_model": 0.71,
            "val_top1_brand": 0.61,
            "seconds": 11.0,
        },
    ]

    карточка = render_report(
        load_config(конфиг(tmp_path)),
        журнал,
        итоги,
        card_title="M-1.2",
        best_epoch=1,
        best_val_top1=0.51,
        device="cpu",
        seconds=1500.0,
        peak_vram_mb=2048.0,
        started_at=datetime(2020, 1, 1, 23, 59, tzinfo=UTC),
    )

    assert карточка == ОЖИДАЕМАЯ_КАРТОЧКА_STANFORD


def логиты_с_argmax(предсказания: Sequence[int]) -> Tensor:
    """Логиты батча, у которых argmax равен переданным индексам классов."""
    тензор = torch.zeros(len(предсказания), len(КЛАССЫ))
    for строка, класс in enumerate(предсказания):
        тензор[строка, класс] = 9.0
    return тензор


class ГоловаСЗаданнымиЛогитами(ArcFaceHead):
    """Голова, отдающая заготовленные логиты: попадания top-1 известны наперёд.

    Настоящая голова считает логиты по своим весам, и известного ответа у теста не
    было бы. Заготовка складывается с `0.0 * embedding.sum()`: значений это не меняет,
    но связывает логиты с графом — иначе `backward()` внутри `train_one_epoch` падает
    на тензоре, у которого нет градиента.

    `штраф_маржи` подражает настоящей голове: при переданном `target` колонка верного
    класса понижается на эту величину, без `target` заготовка отдаётся как есть. Так
    видно, по каким логитам считается обучающий top-1 — с маржой или без неё. Батч
    определяется вызовом с целью: `train_one_epoch` зовёт голову с целью для функции
    потерь, а затем на том же эмбеддинге без цели.
    """

    def __init__(
        self,
        логиты: Sequence[Tensor],
        *,
        embedding_dim: int,
        num_classes: int,
        штраф_маржи: float = 0.0,
    ) -> None:
        super().__init__(embedding_dim, num_classes, s=1.0, m=0.0)
        self.заготовки = list(логиты)
        self.штраф_маржи = штраф_маржи
        self.батчей = 0
        self.текущая = логиты[0]

    def forward(self, embedding: Tensor, target: Tensor | None = None) -> Tensor:
        """Заготовка текущего батча; с `target` колонка верного класса штрафуется."""
        if target is None:
            return self.текущая + 0.0 * embedding.sum()
        self.текущая = self.заготовки[self.батчей]
        self.батчей += 1
        штраф = torch.zeros_like(self.текущая)
        штраф[torch.arange(len(target)), target] = self.штраф_маржи
        return self.текущая - штраф + 0.0 * embedding.sum()


def прогнать_эпоху_на_заготовках(tmp_path: Path, *, штраф_маржи: float) -> tuple[float, int]:
    """Одна эпоха `train_one_epoch` на заготовленных логитах.

    Батчи нарочно разного размера, 4 и 2, заготовленный argmax даёт четыре попадания из
    шести кадров: доля по кадрам — 4/6, среднее батчевых точностей — 0.5, и одна
    реализация от другой отличима.

    Возвращает обучающий top-1 и число проходов бэкбона, посчитанное хуком: проходов
    обязано быть по одному на батч, потому что логиты без маржи считаются повторным
    вызовом головы на уже полученном эмбеддинге, а не вторым проходом сети.
    """
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    from torch.utils.data import DataLoader, TensorDataset

    from training.config import load_config
    from training.model import create_model
    from training.train import make_optimizer, train_one_epoch

    настройки = load_config(конфиг(tmp_path))
    модель, _ = create_model(
        настройки.backbone,
        num_classes=len(КЛАССЫ),
        embedding_dim=настройки.embedding_dim,
        pretrained=False,
        s=настройки.arcface.s,
        m=настройки.arcface.margin_start,
    )
    голова = ГоловаСЗаданнымиЛогитами(
        [логиты_с_argmax([0, 1, 2, 3]), логиты_с_argmax([0, 0])],
        embedding_dim=настройки.embedding_dim,
        num_classes=len(КЛАССЫ),
        штраф_маржи=штраф_маржи,
    )
    проходов = 0

    def считать_проход(*_: object) -> None:
        nonlocal проходов
        проходов += 1

    модель.register_forward_hook(считать_проход)
    картинки = torch.randn(6, 3, настройки.img_size, настройки.img_size)
    метки = torch.arange(6)
    загрузчик = DataLoader(TensorDataset(картинки, метки), batch_size=4)
    оптимизатор = make_optimizer(
        модель, голова, lr=настройки.lr, weight_decay=настройки.weight_decay
    )
    ema = AveragedModel(модель, multi_avg_fn=get_ema_multi_avg_fn(настройки.ema_decay))

    _, _, обучающий_top1 = train_one_epoch(
        модель,
        голова,
        загрузчик,
        оптимизатор,
        ema,
        epoch=0,
        config=настройки,
        device=torch.device("cpu"),
        steps_per_epoch=2,
        limit_batches=None,
    )
    return обучающий_top1, проходов


def test_обучающий_top1_считается_по_батчам(tmp_path: Path) -> None:
    """Обучающий top-1 — доля попаданий по кадрам, а не среднее батчевых точностей.

    Заготовленный argmax даёт четыре попадания из шести кадров при батчах 4 и 2:
    правильный ответ 4/6, среднее батчевых точностей — 0.5. Заодно сверяется, что
    проходов бэкбона два, по одному на батч: логиты без маржи считаются повторным
    вызовом головы на уже полученном эмбеддинге, а второй проход сети стоил бы
    столько же, сколько первый.
    """
    обучающий_top1, проходов = прогнать_эпоху_на_заготовках(tmp_path, штраф_маржи=0.0)

    # Метки батчей — 0,1,2,3 и 4,5; заготовленный argmax — 0,1,2,3 и 0,0.
    assert обучающий_top1 == pytest.approx(4 / 6)
    assert проходов == 2, "проходов бэкбона больше, чем батчей: считается второй форвард"


def test_обучающий_top1_не_зависит_от_маржи(tmp_path: Path) -> None:
    """Число считается по логитам без маржи: заведомо большой штраф его не меняет.

    `ArcFaceHead` с `target` понижает колонку верного класса
    (`training/model.py:47-49`), а `m` едет 0.2 → 0.5 за первые десять эпох
    (`training/schedule.py:32-34`). Считай подпись «учится или сломано» по логитам с
    маржой — и линейка двигалась бы вместе с числом ровно на тех эпохах, где её
    читают: исправно обучающийся контур показал бы плоский или падающий `train_top1`.
    Штраф здесь больше разброса заготовок, поэтому реализация, взявшая логиты с
    маржой, покажет 0.0 вместо 4/6.
    """
    обучающий_top1, _ = прогнать_эпоху_на_заготовках(tmp_path, штраф_маржи=50.0)

    assert обучающий_top1 == pytest.approx(4 / 6)


def test_журнал_несёт_train_top1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Обучающий top-1 попадает в журнал каждой эпохи — и без валидации, и с ней.

    Подпись «учится или сломано» (§5 дизайна) читается только из артефакта: в
    `metrics.json` прогона M-1.2 её нет вовсе, поэтому и «у M-1.2 она уходила к
    единице» ничем не подтверждено. Русскому пути она нужна тем более: там это
    единственное поэпохальное число.
    """
    from training.train import main

    пул = tmp_path / "пул"
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(пул))

    прогон = прогон_пула(пул)

    журнал = json.loads((прогон / "metrics.json").read_text(encoding="utf-8"))["epochs"]
    assert журнал, "журнал пуст: проверять нечего"
    assert all("train_top1" in запись for запись in журнал)
    assert all(0.0 <= запись["train_top1"] <= 1.0 for запись in журнал)

    # Второй прогон — контур Stanford со своими val/test: ключ обязан быть и там, и
    # рядом с поэпохальным замером, а не вместо него.
    stanford = tmp_path / "stanford"
    stanford.mkdir()
    данные(stanford)
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(stanford))
    код = main(
        [
            "--config",
            str(конфиг(stanford)),
            "--catalog",
            str(каталог(stanford)),
            "--device",
            "cpu",
            "--no-pretrained",
            "--limit-batches",
            "2",
        ]
    )
    assert код == 0

    прогон_вехи = sorted((stanford / "runs").glob("m1_2_train_*"))[0]
    журнал_вехи = json.loads((прогон_вехи / "metrics.json").read_text(encoding="utf-8"))["epochs"]
    assert all("train_top1" in запись for запись in журнал_вехи)
    assert all("val_top1_model" in запись for запись in журнал_вехи)
