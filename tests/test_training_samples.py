"""Тесты чтения сплита в образцы для обучения (M-1.2, §4.1 дизайна)."""

from pathlib import Path

import pandas as pd
import pytest

from pipeline.schema import COLUMNS, write_split
from training.samples import DatasetError, Sample, read_samples


def строка(**переопределения: object) -> dict[str, object]:
    основа: dict[str, object] = {
        "photo_id": "0f8e4f70-aadc-4f70-9f8e-000000000001",
        "listing_id": "stanford_000001",
        "source": "stanford",
        "crop_path": "stanford-00001.jpg",
        "brand": "Acura",
        "model_id": "acura__rl",
        "class_idx": 0,
        "body_style": "sedan",
        "year": 2012,
        "bbox": [10.0, 20.0, 300.0, 200.0],
        "bbox_area_frac": 0.42,
        "det_conf": 0.91,
        "phash": 1234567890123456789,
        "raw_model_string": "Acura RL Sedan 2012",
        "is_multi_car": False,
    }
    основа.update(переопределения)
    return основа


def подготовить(tmp_path: Path, *строки: dict[str, object]) -> tuple[Path, Path]:
    """Записать parquet и создать файлы кропов, упомянутые в нём."""
    кропы = tmp_path / "crops" / "stanford"
    кропы.mkdir(parents=True)
    for запись in строки:
        (кропы / str(запись["crop_path"])).write_bytes(b"jpeg")
    файл = tmp_path / "train.parquet"
    write_split(pd.DataFrame(list(строки), columns=list(COLUMNS)), файл)
    return файл, кропы


def test_образцы_читаются_и_путь_склеен(tmp_path: Path) -> None:
    файл, кропы = подготовить(tmp_path, строка())
    образцы = read_samples(файл, кропы, num_classes=171, skip_multi_car=True)
    assert образцы == [Sample(path=кропы / "stanford-00001.jpg", class_idx=0, is_multi_car=False)]


def test_null_class_idx_отвергается(tmp_path: Path) -> None:
    файл, кропы = подготовить(tmp_path, строка(class_idx=None, model_id=None, brand=None))
    with pytest.raises(DatasetError, match="class_idx"):
        read_samples(файл, кропы, num_classes=171, skip_multi_car=True)


def test_class_idx_вне_диапазона_отвергается(tmp_path: Path) -> None:
    файл, кропы = подготовить(tmp_path, строка(class_idx=171))
    with pytest.raises(DatasetError, match="вне диапазона"):
        read_samples(файл, кропы, num_classes=171, skip_multi_car=True)


def test_отсутствующий_кроп_отвергается(tmp_path: Path) -> None:
    файл, кропы = подготовить(tmp_path, строка())
    (кропы / "stanford-00001.jpg").unlink()
    with pytest.raises(DatasetError, match="нет файла кропа"):
        read_samples(файл, кропы, num_classes=171, skip_multi_car=True)


def test_многомашинные_выбрасываются_по_флагу(tmp_path: Path) -> None:
    файл, кропы = подготовить(
        tmp_path,
        строка(),
        строка(
            photo_id="0f8e4f70-aadc-4f70-9f8e-000000000002",
            listing_id="stanford_000002",
            crop_path="stanford-00002.jpg",
            is_multi_car=True,
        ),
    )
    без_многомашинных = read_samples(файл, кропы, num_classes=171, skip_multi_car=True)
    со_всеми = read_samples(файл, кропы, num_classes=171, skip_multi_car=False)
    assert len(без_многомашинных) == 1
    assert len(со_всеми) == 2
    assert со_всеми[1].is_multi_car is True


def test_пустой_сплит_отвергается(tmp_path: Path) -> None:
    """Сплит без строк — дыра в данных, а не пустой список образцов."""
    файл, кропы = подготовить(tmp_path)
    with pytest.raises(DatasetError, match="ни одного образца"):
        read_samples(файл, кропы, num_classes=171, skip_multi_car=True)


def test_сплит_из_одних_многомашинных_отвергается(tmp_path: Path) -> None:
    """Фильтр §3.5, съевший весь сплит, — тот же пустой датасет.

    Сообщение обязано показать число отброшенных: иначе непонятно, сплит пуст
    или его выел `skip_multi_car`.
    """
    файл, кропы = подготовить(tmp_path, строка(is_multi_car=True))
    with pytest.raises(DatasetError, match="отброшено многомашинных: 1"):
        read_samples(файл, кропы, num_classes=171, skip_multi_car=True)
    # Без фильтра та же строка читается: пустоту делает именно фильтр.
    assert len(read_samples(файл, кропы, num_classes=171, skip_multi_car=False)) == 1
