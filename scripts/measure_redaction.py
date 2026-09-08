"""Замер остатков повторной проверки редакции по кэшу оригиналов, без сети.

`redaction_failed` означает «детектор номеров сработал во второй раз», а не «остался
читаемый номер»: диагноз `bench/redaction-failed-c2.md` показал на шести кадрах, что
второй проход ловит собственную заливку и боксы невозможной для номера формы. Замер
раскладывает все такие остатки по трём корзинам и даёт числа, на которых выбираются
пороги правила.

Кадры гоняются через `redact_plates` — ту самую функцию, которой потом будет решать
`redact_frame`. Своей копии логики здесь нет намеренно: копия разошлась бы молча.

Детекторы и их веса импортируются внутри `main` (Р-15): чистые функции модуля —
`корзина`, `измерить_кадр`, `ячейка`, `листы` — тестируются в базовом окружении, где
ни весов, ни `ultralytics` нет.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from pipeline.crop import VEHICLE_CLASSES, Detection, Detector
from pipeline.harvest.cache import cache_path, decode_image
from pipeline.harvest.redact import fill_overlap, plate_aspect, redact_plates
from pipeline.paths import harvest_dir

ЯЧЕЙКА_W = 160
ЯЧЕЙКА_H = 120
СЕТКА = 10
НА_СТРАНИЦЕ = СЕТКА * СЕТКА
ПОЛЕ = 0.4


@dataclass(frozen=True, slots=True)
class Остаток:
    """Один бокс, найденный повторным проходом поверх заливки."""

    source_item_id: str
    class_id: str
    sha256: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    aspect: float
    overlap: float
    filled_count: int


MEASURE_COLUMNS = tuple(поле.name for поле in fields(Остаток))


def корзина(остаток: Остаток, aspect_min: float, overlap_min: float) -> str:
    """`A` — гасится заливкой, `B` — гасится формой, `V` — не гасится ничем.

    Перекрытие имеет приоритет: остаток, подошедший под оба предиката, — это заливка,
    увиденная детектором, а её форма следствие, а не причина. Корзины непересекающиеся,
    иначе один и тот же бокс попал бы на два контактных листа и был бы посчитан дважды.

    Ровные значения границ гасятся не одинаково: `overlap == overlap_min` — уже `A`, а
    `aspect == aspect_min` — ещё `V`. Пороги выбираются ровно на потолках Global
    Constraints, поэтому задача 6 обязана взять эти же операторы: иначе выбранное
    человеком число будет означать в продакшене чуть другое.
    """
    if остаток.overlap >= overlap_min:
        return "A"
    if остаток.aspect < aspect_min:
        return "B"
    return "V"


def измерить_кадр(
    image: np.ndarray,
    vehicle_boxes: Sequence[Detection],
    plate_detectors: Sequence[Detector],
    source_item_id: str,
    class_id: str,
    sha256: str,
) -> tuple[np.ndarray, list[Остаток]]:
    """Прогнать кадр через стадию номеров и померить каждый остаток.

    Возвращает отредактированный кадр вместе с остатками: вырезка для контактного листа
    берётся именно из него — глазами проверяется, читается ли номер **после** заливки.
    """
    рабочая = image.copy()
    стадия = redact_plates(рабочая, vehicle_boxes, plate_detectors)
    остатки = [
        Остаток(
            source_item_id=source_item_id,
            class_id=class_id,
            sha256=sha256,
            x=int(бокс.x),
            y=int(бокс.y),
            w=int(бокс.w),
            h=int(бокс.h),
            conf=float(бокс.conf),
            aspect=float(plate_aspect(бокс)),
            overlap=fill_overlap(бокс, стадия.filled),
            filled_count=len(стадия.filled),
        )
        for бокс in стадия.residuals
    ]
    return рабочая, остатки


def ячейка(image: np.ndarray, остаток: Остаток) -> np.ndarray:
    """Вырезка остатка с полем контекста, ужатая под сетку, с подписью латиницей."""
    высота, ширина = image.shape[:2]
    dx = int(остаток.w * ПОЛЕ)
    dy = int(остаток.h * ПОЛЕ)
    x0 = max(0, остаток.x - dx)
    y0 = max(0, остаток.y - dy)
    x1 = min(ширина, остаток.x + остаток.w + dx)
    y1 = min(высота, остаток.y + остаток.h + dy)
    вырезка = image[y0:y1, x0:x1]
    if вырезка.size == 0:
        вырезка = np.zeros((ЯЧЕЙКА_H, ЯЧЕЙКА_W, 3), dtype=np.uint8)
    плитка = cv2.resize(вырезка, (ЯЧЕЙКА_W, ЯЧЕЙКА_H), interpolation=cv2.INTER_AREA)
    подпись = f"{остаток.source_item_id} {остаток.w}x{остаток.h}"
    числа = f"a={остаток.aspect:.2f} ov={остаток.overlap:.2f}"
    for текст, база in ((подпись, 14), (числа, 114)):
        # Дважды, чёрным толще и зелёным поверх: на светлых ячейках — шильдик на белой
        # панели, линия крыши на фоне неба — одна зелёная подпись почти теряется, а
        # человек по ней связывает ячейку со строкой CSV.
        cv2.putText(плитка, текст, (3, база), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 3)
        cv2.putText(плитка, текст, (3, база), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
    return плитка


def листы(ячейки: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Склеить ячейки в страницы 10×10.

    Пагинация не косметика: в корзину A уйдут сотни боксов, и один склеенный лист на
    четыреста ячеек — простыня в пятнадцать тысяч пикселей по стороне, которую не
    откроет просмотрщик.
    """
    страницы: list[np.ndarray] = []
    for начало in range(0, len(ячейки), НА_СТРАНИЦЕ):
        кусок = list(ячейки[начало : начало + НА_СТРАНИЦЕ])
        пустая = np.zeros((ЯЧЕЙКА_H, ЯЧЕЙКА_W, 3), dtype=np.uint8)
        кусок += [пустая] * (НА_СТРАНИЦЕ - len(кусок))
        ряды = [np.hstack(кусок[ряд * СЕТКА : (ряд + 1) * СЕТКА]) for ряд in range(СЕТКА)]
        страницы.append(np.vstack(ряды))
    return страницы


def записать_замер(path: Path, остатки: Sequence[Остаток]) -> None:
    """Перезаписать таблицу замера целиком: она производная от кэша и карты."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(MEASURE_COLUMNS))
        писатель.writeheader()
        for остаток in остатки:
            писатель.writerow(asdict(остаток))


def записать_лист(path: Path, лист: np.ndarray) -> None:
    """Записать страницу, не поверив `cv2.imwrite` на слово.

    `imwrite` отдаёт `False` на путях с не-ASCII символами — имя уходит в системную
    кодировку, а под Windows это не UTF-8; `pipeline/harvest/run.py:213` держит тот же
    фолбэк по той же причине. Без проверки человек после получасового прогона получил бы
    CSV без единого листа и ни одного сообщения о том, почему.
    """
    if cv2.imwrite(str(path), лист):
        return
    удалось, буфер = cv2.imencode(".png", лист)
    if not удалось:
        raise RuntimeError(f"лист не кодируется в PNG: {path}")
    path.write_bytes(буфер.tobytes())


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="замер остатков повторной проверки редакции")
    парсер.add_argument("--data-root", type=Path, default=None, help="каталог данных харвеста")
    парсер.add_argument("--models", type=Path, default=None, help="каталог весов детекторов")
    парсер.add_argument("--out", type=Path, default=None, help="каталог листов и CSV")
    парсер.add_argument("--aspect-min", type=float, default=1.0, help="граница w/h: ниже — B")
    парсер.add_argument("--overlap-min", type=float, default=0.5, help="доля заливки: выше — A")
    парсер.add_argument("--limit", type=int, default=0, help="0 — все строки карты")
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Прогнать карту по кэшу, записать CSV и контактные листы по корзинам."""
    from pipeline.detectors.yolo import YoloDetector
    from pipeline.harvest.detectors import (
        DEFAULT_DETECTORS_MANIFEST,
        load_detectors_manifest,
        verify_weights,
    )
    from pipeline.harvest.redaction_profile import (
        CURRENT_REDACTION_PROFILE,
        build_plate_detectors,
    )

    аргументы = build_parser().parse_args(argv)
    корень = аргументы.data_root if аргументы.data_root is not None else harvest_dir()
    модели = аргументы.models if аргументы.models is not None else корень / "models"
    выход = аргументы.out if аргументы.out is not None else корень / "measure"
    выход.mkdir(parents=True, exist_ok=True)

    пины = load_detectors_manifest(DEFAULT_DETECTORS_MANIFEST)
    verify_weights(пины, модели)
    plate_detectors = build_plate_detectors(CURRENT_REDACTION_PROFILE, models_dir=модели, pins=пины)
    vehicle_detector = YoloDetector()

    строки = list(csv.DictReader((корень / "reject_index.csv").read_text("utf-8").splitlines()))
    if аргументы.limit:
        строки = строки[: аргументы.limit]

    все_остатки: list[Остаток] = []
    ячейки: dict[str, list[np.ndarray]] = {"A": [], "B": [], "V": []}
    пропущено = 0
    битые: list[str] = []
    без_остатков = 0
    for номер, строка in enumerate(строки, 1):
        if not строка["sha256"]:
            пропущено += 1
            continue
        путь = cache_path(корень, "commons", строка["sha256"], "jpg")
        if not путь.exists():
            пропущено += 1
            continue
        try:
            кадр = decode_image(путь.read_bytes())
        except Exception:  # причин у битого файла кэша много, ответ один
            # Не `raise`: CSV пишется после цикла, и один обрезанный файл унёс бы
            # с собой все посчитанные строки — на 554 кадрах это десятки минут.
            # Поимённо, а не счётчиком: строки в CSV у битого кадра нет, и без
            # имени непонятно, какой файл кэша чистить и какую страницу качать.
            битые.append(f"{строка['source_item_id']}:{строка['sha256']}")
            continue
        машины = [
            d for d in vehicle_detector.detect(кадр) if d.cls in VEHICLE_CLASSES and d.conf > 0.5
        ]
        отредактированный, остатки = измерить_кадр(
            кадр,
            машины,
            plate_detectors,
            строка["source_item_id"],
            строка["class_id"],
            строка["sha256"],
        )
        if not остатки:
            # Не аномалия: `redaction_failed` ставит и стадия лиц
            # (`pipeline/harvest/redact.py:198-207`), а карта задачи 3 фильтруется
            # одной причиной и потому держит кадры обеих категорий вперемешку.
            # Правило задачи 6 таким кадрам не поможет: это часть популяции.
            без_остатков += 1
        for остаток in остатки:
            имя = корзина(остаток, аргументы.aspect_min, аргументы.overlap_min)
            ячейки[имя].append(ячейка(отредактированный, остаток))
        все_остатки += остатки
        if номер % 25 == 0:
            print(f"{номер} / {len(строки)}")

    записать_замер(выход / "reject_measure.csv", все_остатки)
    for имя, плитки in ячейки.items():
        for страница, лист in enumerate(листы(плитки), 1):
            записать_лист(выход / f"sheet_{имя}_{страница:02d}.png", лист)
        print(f"корзина {имя}: {len(плитки)} остатков")
    print(f"кадров без оригинала: {пропущено}; битых файлов кэша: {len(битые)}")
    if битые:
        print(f"битые (source_item_id:sha256): {' '.join(битые)}")
    print(f"кадров без остатков номеров при замере: {без_остатков} (верхняя оценка стадии лиц)")
    print(f"таблица: {выход / 'reject_measure.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
