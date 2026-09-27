"""Листы просмотра негативов по видам, с номерами, впаянными в картинку.

Рядом с листом — `<вид>_NN.txt`: номер → `photo_id`, `raw_model_string`. Сомнительный кроп
уходит в `catalog/external/negatives_exclude.csv` строкой `photo,<photo_id>,<причина>`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pipeline.negatives import NEG_KINDS, neg_kind_of
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import read_split
from scripts.external_sheets import PER_SHEET, render_sheet


def main(argv: Sequence[str] | None = None) -> int:
    парсер = argparse.ArgumentParser(description="Листы просмотра негативов")
    парсер.add_argument("--out", type=Path, default=data_root() / "negatives-review")
    парсер.add_argument("--split", default="negatives", help="имя parquet в splits/")
    аргументы = парсер.parse_args(argv)

    кадр = read_split(splits_dir() / f"{аргументы.split}.parquet")
    кадр = кадр.assign(вид=[neg_kind_of(str(л)) for л in кадр["listing_id"]])
    аргументы.out.mkdir(parents=True, exist_ok=True)
    корень = harvest_dir() / "crops"
    for вид in NEG_KINDS:
        строки = (
            кадр[кадр["вид"] == вид].sort_values(["listing_id", "photo_id"]).reset_index(drop=True)
        )
        for лист, начало in enumerate(range(0, len(строки), PER_SHEET)):
            кусок = строки.iloc[начало : начало + PER_SHEET]
            имя = f"{вид}_{лист:02d}"
            render_sheet([корень / п for п in кусок["crop_path"]], начало).save(
                аргументы.out / f"{имя}.png"
            )
            (аргументы.out / f"{имя}.txt").write_text(
                "".join(
                    f"{начало + н}\t{pid}\t{сырое}\n"
                    for н, (pid, сырое) in enumerate(
                        zip(кусок["photo_id"], кусок["raw_model_string"], strict=True)
                    )
                ),
                encoding="utf-8",
            )
    print(f"листы: {аргументы.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
