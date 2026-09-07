"""Тесты CLI замера частоты утечки номера (правило 1 Р-11, §7)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.plate_leak import BoundReport, LeakBound, LeakFrame, StratumBound
from pipeline.schema import COLUMNS, write_split
from scripts.measure_plate_leak import build_parser, main, report_to_dict, write_manifest_verdicts


def сделать_pool_parquet(путь: Path, кадров: int) -> None:
    строки = [
        {
            "photo_id": f"sha{n}",
            "listing_id": f"l-{n}",
            "source": "commons",
            "crop_path": f"class{n % 2}/sha{n}.jpg",
            "brand": "Lada",
            "model_id": "lada__2107",
            "class_idx": 0,
            "body_style": None,
            "year": None,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "bbox_area_frac": 0.5,
            "det_conf": 0.9,
            "phash": n,
            "raw_model_string": None,
            "is_multi_car": False,
        }
        for n in range(кадров)
    ]
    write_split(pd.DataFrame(строки, columns=list(COLUMNS)), путь)


def сделать_манифест(путь: Path, строк: int) -> None:
    шапка = (
        "file,class_id,source,source_item_id,page_url,file_url,author,license,license_tier,"
        "checked,basis,sha256,phash,clip_rank,clip_margin,plates,faces,bbox_x,bbox_y,bbox_w,"
        "bbox_h,bbox_area_frac,det_conf,audit"
    )
    строки = [шапка]
    for n in range(строк):
        строки.append(
            f"sha{n}.jpg,lada__2107,commons,item{n},http://x/{n},http://x/{n}.jpg,a,cc-by,2,"
            f"1,clip,sha{n},{n},1,0.5,0,0,0,0,10,10,0.5,0.9,ok"
        )
    путь.write_text("\n".join(строки) + "\n", encoding="utf-8")


def сделать_выборку_с_вердиктами(путь: Path) -> None:
    выборка = pd.DataFrame(
        {
            "номер": [1, 2, 3, 4],
            "sha256": ["sha0", "sha1", "sha2", "sha3"],
            "pageid": ["item0", "item1", "item2", "item3"],
            "класс": ["lada__2107"] * 4,
            "plates": [1, 1, 0, 0],
            "faces": [0, 0, 0, 0],
            "штамп": ["ok"] * 4,
            "вердикт": ["leak", "clean_filled", "clean_no_plate", "leak_hist"],
            "заметка": ["", "", "", "старое фото"],
        }
    )
    выборка.to_csv(путь, index=False)


def test_report_to_dict_даёт_проценты_и_кадры() -> None:
    отчёт = BoundReport(
        sample_size=10,
        verdict_counts={"leak": 2, "clean_filled": 8},
        bounds=(
            LeakBound(
                definition="строго",
                leaks=2,
                sample_size=10,
                share=0.2,
                ci=(0.05, 0.45),
                pool_estimate=20,
                pool_interval=(5, 45),
                accepted_interval=(10, 90),
            ),
        ),
        strata=(StratumBound(stratum="plates_0", seen=4, leaks=1, share=0.25, ci=(0.01, 0.5)),),
        leak_frames=(
            LeakFrame(
                number=1,
                verdict="leak",
                pageid="item0",
                class_id="lada__2107",
                plates=1,
                faces=0,
                sha256="sha0",
                note="",
            ),
        ),
    )

    словарь = report_to_dict(отчёт)

    assert словарь["выборка"] == 10
    assert словарь["строго"]["утечек"] == 2
    assert словарь["строго"]["доля_%"] == 20.0
    assert словарь["страты"]["plates_0"]["просмотрено"] == 4
    assert словарь["кадры_утечек"] == [
        {
            "number": 1,
            "verdict": "leak",
            "pageid": "item0",
            "class_id": "lada__2107",
            "plates": 1,
            "faces": 0,
            "sha256": "sha0",
            "note": "",
        }
    ]


def test_write_manifest_verdicts_пишет_только_строгие(tmp_path: Path) -> None:
    merged = pd.DataFrame(
        {
            "sha256": ["b", "a", "c"],
            "вердикт": ["leak", "leak", "leak_hist"],
        }
    )
    выход = tmp_path / "verdicts.csv"

    записано = write_manifest_verdicts(merged, выход)

    assert записано == 2
    строки = list(csv.DictReader(выход.read_text(encoding="utf-8").splitlines()))
    assert [r["file"] for r in строки] == ["a.jpg", "b.jpg"]  # отсортировано по sha256
    assert {r["verdict"] for r in строки} == {"drop:plate"}


def test_main_отсутствие_выборки_роняет_с_понятной_ошибкой(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="выборка не найдена"):
        main(["--sample", str(tmp_path / "нет.csv")])


def test_main_считает_и_пишет_отчёт_и_вердикты(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))
    сделать_pool_parquet(tmp_path / "splits" / "pool.parquet", кадров=100)
    манифест = tmp_path / "manifest.csv"
    сделать_манифест(манифест, строк=200)
    выборка = tmp_path / "sample.csv"
    сделать_выборку_с_вердиктами(выборка)
    отчёт_путь = tmp_path / "report.json"
    вердикты_путь = tmp_path / "verdicts.plate-leak.csv"

    код = main(
        [
            "--sample",
            str(выборка),
            "--manifest",
            str(манифест),
            "--out-report",
            str(отчёт_путь),
            "--out-verdicts",
            str(вердикты_путь),
        ]
    )

    assert код == 0
    отчёт = json.loads(отчёт_путь.read_text(encoding="utf-8"))
    # 1 leak из 4 в выборке -> перенос на популяцию 100: round(100 * 1/4) = 25.
    assert отчёт["выборка"] == 4
    assert отчёт["строго"]["утечек"] == 1
    assert отчёт["строго"]["кадров_в_пуле_оценка"] == 25
    assert отчёт["с_историческими"]["утечек"] == 2

    строки_вердиктов = list(csv.DictReader(вердикты_путь.read_text(encoding="utf-8").splitlines()))
    assert [r["file"] for r in строки_вердиктов] == ["sha0.jpg"]


def test_main_сводит_черновик_с_вердиктами_человека(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--verdicts` переключает `--sample` из готового отчёта в черновик до свода."""
    monkeypatch.setenv("CARLENS_DATA_ROOT", str(tmp_path))
    сделать_pool_parquet(tmp_path / "splits" / "pool.parquet", кадров=100)
    манифест = tmp_path / "manifest.csv"
    сделать_манифест(манифест, строк=200)

    черновик = tmp_path / "draft.csv"
    pd.DataFrame(
        {
            "номер": [1, 2],
            "sha256": ["sha0", "sha1"],
            "pageid": ["item0", "item1"],
            "класс": ["lada__2107", "lada__2107"],
            "plates": [1, 0],
            "faces": [0, 0],
            "штамп": ["ok", "ok"],
            "crop_path": ["lada__2107/sha0.jpg", "lada__2107/sha1.jpg"],
        }
    ).to_csv(черновик, index=False)
    вердикты = tmp_path / "verdicts_review.csv"
    pd.DataFrame(
        {"номер": [1, 2], "вердикт": ["leak", "clean_no_plate"], "заметка": ["", ""]}
    ).to_csv(вердикты, index=False)
    итог = tmp_path / "final.csv"

    код = main(
        [
            "--sample",
            str(черновик),
            "--verdicts",
            str(вердикты),
            "--out-sample",
            str(итог),
            "--manifest",
            str(манифест),
            "--out-report",
            str(tmp_path / "report.json"),
            "--out-verdicts",
            str(tmp_path / "verdicts.plate-leak.csv"),
        ]
    )

    assert код == 0
    смерженная = pd.read_csv(итог)
    assert list(смерженная.columns) == [
        "номер",
        "sha256",
        "pageid",
        "класс",
        "plates",
        "faces",
        "штамп",
        "вердикт",
        "заметка",
    ]
    assert "crop_path" not in смерженная.columns
    отчёт = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert отчёт["строго"]["утечек"] == 1


def test_разбор_флагов_даёт_значения_по_умолчанию() -> None:
    аргументы = build_parser().parse_args([])
    assert аргументы.sample == Path("bench") / "plate-leak-sample.csv"
    assert аргументы.split == "pool"
    assert аргументы.out_report == Path("bench") / "plate-leak-bound-report.json"
    assert аргументы.out_verdicts == Path("verdicts.plate-leak.csv")
