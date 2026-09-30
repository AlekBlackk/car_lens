"""Паритет fp16-детектора с fp32 на настоящих кадрах (M4, бюджет размера §5.2).

Сравниваются не тензоры, а то, что детектор отдаёт дальше: боксы после порога, NMS и
разворота в координаты кадра. От боксов зависит только кроп, а при совпадении боксов кроп
и всё, что после него (эмбеддинг, пороги, индекс), идентичны — поэтому пороги
распознавателя fp16 детектора не трогает.

Обе модели гоняет питоновский двойник конвейера устройства `scripts.gen_device_golden.
детектировать`, тот же, что даёт эталон для `ParityTest.kt`. Чистая часть модуля
(сопоставление, счётчики, вердикт) не тянет экспорт и проверяется в обычной venv; тяжёлые
импорты живут в `main`.

Пороги гейта записаны в `bench/fp16-detector-preregistration.md` до замера и здесь
продублированы константами: менять их после замера нельзя.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.crop import Detection, iou

# Ниже этого IoU два бокса одного класса — уже не «тот же бокс», а два разных.
PAIR_MIN_IOU = 0.5
# Допуск устройства в `ParityTest.kt`: бокс из другой модели считается тем же при IoU ≥ 0.9.
PRIMARY_MIN_IOU = 0.9

GATE_EQUAL_COUNT = 0.995
GATE_UNPAIRED = 0.005
GATE_IOU_MIN = 0.9
GATE_IOU_MEDIAN = 0.999
GATE_PRIMARY = 0.005

WORST_LISTED = 5


def match(эталон: Sequence[Detection], проверяемый: Sequence[Detection]) -> list[tuple[int, int]]:
    """Пары индексов боксов одного класса: жадно по убыванию уверенности эталона.

    Каждому боксу эталона достаётся свободный бокс того же класса с наибольшим IoU, если он
    не ниже `PAIR_MIN_IOU`. Не вошедшие в пары боксы обеих сторон — расхождения.
    """
    занятые: set[int] = set()
    пары: list[tuple[int, int]] = []
    порядок = sorted(range(len(эталон)), key=lambda i: эталон[i].conf, reverse=True)
    for i in порядок:
        лучший, кандидат = 0.0, None
        for j, бокс in enumerate(проверяемый):
            if j in занятые or бокс.cls != эталон[i].cls:
                continue
            пересечение = iou(эталон[i], бокс)
            if пересечение > лучший:
                лучший, кандидат = пересечение, j
        if кандидат is not None and лучший >= PAIR_MIN_IOU:
            занятые.add(кандидат)
            пары.append((i, кандидат))
    return пары


def primary_agrees(эталон: Sequence[Detection], проверяемый: Sequence[Detection]) -> bool:
    """Главный бокс (с наибольшей уверенностью) тот же: класс и IoU ≥ `PRIMARY_MIN_IOU`."""
    if not эталон or not проверяемый:
        return not эталон and not проверяемый
    первый = max(эталон, key=lambda d: d.conf)
    второй = max(проверяемый, key=lambda d: d.conf)
    return первый.cls == второй.cls and iou(первый, второй) >= PRIMARY_MIN_IOU


@dataclass
class SetTotals:
    """Счётчики одного набора кадров."""

    name: str
    frames: int = 0
    unreadable: int = 0
    frames_with_boxes: int = 0
    equal_count: int = 0
    boxes_fp32: int = 0
    boxes_fp16: int = 0
    unpaired: int = 0
    primary_disagree: int = 0
    ious: list[float] = field(default_factory=list)
    conf_deltas: list[float] = field(default_factory=list)
    # (IoU, кадр): чтобы худшую пару можно было открыть глазами, а не верить медиане.
    worst: list[tuple[float, str]] = field(default_factory=list)
    disagreeing_frames: list[str] = field(default_factory=list)

    def add(self, кадр: str, эталон: Sequence[Detection], проверяемый: Sequence[Detection]) -> None:
        пары = match(эталон, проверяемый)
        self.frames += 1
        self.frames_with_boxes += int(bool(эталон))
        self.equal_count += int(len(эталон) == len(проверяемый))
        self.boxes_fp32 += len(эталон)
        self.boxes_fp16 += len(проверяемый)
        self.unpaired += len(эталон) + len(проверяемый) - 2 * len(пары)
        согласны = primary_agrees(эталон, проверяемый)
        self.primary_disagree += int(not согласны)
        for i, j in пары:
            пересечение = iou(эталон[i], проверяемый[j])
            self.ious.append(пересечение)
            self.conf_deltas.append(abs(эталон[i].conf - проверяемый[j].conf))
            self.worst = sorted([*self.worst, (пересечение, кадр)])[:WORST_LISTED]
        if len(эталон) != len(проверяемый) or not согласны:
            self.disagreeing_frames.append(кадр)


def _median(значения: Sequence[float]) -> float:
    return float(np.median(значения)) if значения else 0.0


def verdict(наборы: Sequence[SetTotals]) -> dict[str, Any]:
    """Гейт предрегистрации: каждое условие со значением, пределом и признаком прохождения."""
    ious = [значение for набор in наборы for значение in набор.ious]
    боксов = sum(набор.boxes_fp32 for набор in наборы)
    кадров = sum(набор.frames for набор in наборы)
    условия: dict[str, dict[str, Any]] = {}
    for набор in наборы:
        доля = набор.equal_count / набор.frames if набор.frames else 0.0
        условия[f"a_equal_count[{набор.name}]"] = {
            "value": доля,
            "limit": GATE_EQUAL_COUNT,
            "ok": доля >= GATE_EQUAL_COUNT,
        }
    без_пары = sum(набор.unpaired for набор in наборы) / боксов if боксов else 1.0
    условия["b_unpaired"] = {
        "value": без_пары,
        "limit": GATE_UNPAIRED,
        "ok": без_пары <= GATE_UNPAIRED,
    }
    минимум = min(ious, default=0.0)
    условия["c_iou_min"] = {"value": минимум, "limit": GATE_IOU_MIN, "ok": минимум >= GATE_IOU_MIN}
    медиана = _median(ious)
    условия["c_iou_median"] = {
        "value": медиана,
        "limit": GATE_IOU_MEDIAN,
        "ok": медиана >= GATE_IOU_MEDIAN,
    }
    расходятся = sum(набор.primary_disagree for набор in наборы) / кадров if кадров else 1.0
    условия["d_primary"] = {
        "value": расходятся,
        "limit": GATE_PRIMARY,
        "ok": расходятся <= GATE_PRIMARY,
    }
    return {"conditions": условия, "passed": all(у["ok"] for у in условия.values())}


def render_report(
    наборы: Sequence[SetTotals], итог: dict[str, Any], детекторы: dict[str, str]
) -> str:
    строки = [
        "# Детектор fp16 против fp32 — паритет боксов",
        "",
        "Предрегистрация: `bench/fp16-detector-preregistration.md`. Замер: "
        "`python -m export.detector_parity`.",
        "",
        f"fp32 `{детекторы['fp32']}`, fp16 `{детекторы['fp16']}`.",
        "",
        "## Наборы",
        "",
        "| набор | кадров | нечитаемых | с боксами | равное число | боксов fp32 | боксов fp16 "
        "| без пары | главный расходится |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for н in наборы:
        строки.append(
            f"| {н.name} | {н.frames} | {н.unreadable} | {н.frames_with_boxes} | {н.equal_count} "
            f"| {н.boxes_fp32} | {н.boxes_fp16} | {н.unpaired} | {н.primary_disagree} |"
        )
    ious = [значение for н in наборы for значение in н.ious]
    дельты = [значение for н in наборы for значение in н.conf_deltas]
    строки += [
        "",
        f"Пар боксов {len(ious)}: IoU min {min(ious, default=0.0):.4f}, "
        f"медиана {_median(ious):.5f}; |Δconf| max {max(дельты, default=0.0):.4f}, "
        f"медиана {_median(дельты):.5f}.",
        "",
        "## Гейт",
        "",
        "| условие | значение | предел | |",
        "|---|---|---|---|",
    ]
    for имя, у in итог["conditions"].items():
        знак = "≥" if имя.startswith(("a_", "c_")) else "≤"
        строки.append(
            f"| {имя} | {у['value']:.5f} | {знак} {у['limit']} | {'да' if у['ok'] else 'НЕТ'} |"
        )
    строки += ["", f"**Вердикт: {'гейт пройден' if итог['passed'] else 'гейт НЕ пройден'}.**", ""]
    строки += ["## Худшие пары IoU", ""]
    худшие = sorted(запись for н in наборы for запись in н.worst)[:WORST_LISTED]
    строки += [f"- {значение:.4f} — `{кадр}`" for значение, кадр in худшие] or ["- пар нет"]
    строки += ["", "## Кадры с расхождением (число боксов или главный бокс)", ""]
    расходящиеся = [кадр for н in наборы for кадр in н.disagreeing_frames]
    строки += [f"- `{кадр}`" for кадр in расходящиеся[:20]] or ["- нет"]
    if len(расходящиеся) > 20:
        строки.append(f"- … и ещё {len(расходящиеся) - 20}")
    return "\n".join(строки) + "\n"


# Наборы и объём фиксированы предрегистрацией: 0 — все кадры набора.
FRAME_SETS = (("mad_cars", 500), ("coco", 500), ("manual", 0))


def _pick_frames(корень: Path, n: int) -> list[Path]:
    """Равный шаг по отсортированному списку: тот же выбор при каждом запуске."""
    файлы = sorted(корень.rglob("*.jpg"))
    if n <= 0 or len(файлы) <= n:
        return файлы
    шаг = len(файлы) // n
    return файлы[::шаг][:n]


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Паритет боксов fp16-детектора с fp32")
    парсер.add_argument("--detector", type=Path, required=True, help="detector_fp32.tflite")
    парсер.add_argument(
        "--frames-root", type=Path, required=True, help="каталог с mad_cars/, coco/, manual/"
    )
    парсер.add_argument("--out", type=Path, required=True, help="куда писать отчёт (markdown)")
    return парсер.parse_args()


def main() -> None:
    # Тяжёлые импорты здесь: чистая часть модуля должна подниматься в venv без экспорта.
    import json

    import cv2

    from scripts.gen_device_golden import ПРЕДЕЛ_ДЛИННОЙ, детектировать

    аргументы = _parse_args()
    fp32 = аргументы.detector
    fp16 = fp32.with_name("detector_fp16.tflite")
    контракт = json.loads(fp32.with_name("detector_report.json").read_text("utf-8"))["contract"]

    наборы: list[SetTotals] = []
    for имя, число in FRAME_SETS:
        итог_набора = SetTotals(имя)
        кадры = _pick_frames(аргументы.frames_root / имя, число)
        for номер, путь in enumerate(кадры, 1):
            картинка = cv2.imread(str(путь))
            if картинка is None:
                итог_набора.unreadable += 1
                continue
            # Ужатие как в эталоне устройства: детектор всё равно вписывает кадр в 416.
            высота, ширина = картинка.shape[:2]
            масштаб = min(1.0, ПРЕДЕЛ_ДЛИННОЙ / max(высота, ширина))
            if масштаб < 1.0:
                картинка = cv2.resize(
                    картинка,
                    (round(ширина * масштаб), round(высота * масштаб)),
                    interpolation=cv2.INTER_AREA,
                )
            итог_набора.add(
                f"{имя}/{путь.name}",
                детектировать(fp32, контракт, картинка),
                детектировать(fp16, контракт, картинка),
            )
            if номер % 100 == 0:
                print(f"{имя}: {номер}/{len(кадры)}", flush=True)
        наборы.append(итог_набора)

    итог = verdict(наборы)
    аргументы.out.write_text(
        render_report(наборы, итог, {"fp32": fp32.name, "fp16": fp16.name}), encoding="utf-8"
    )
    print(f"{'гейт пройден' if итог['passed'] else 'гейт НЕ пройден'}: {аргументы.out}")


if __name__ == "__main__":
    main()
