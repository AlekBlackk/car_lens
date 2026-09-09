"""Редакция номеров и лиц — правило 1 решения Р-11.

Кроп машины не удаляет её собственный номер, поэтому стадия обязательная и обхода у неё
нет: кадр, который не удалось замыть, отбраковывается целиком.

Редакция полнокадровая и идёт до кропа (§6): номер фоновой машины, попавшей краем в
сохраняемый кроп, обязан уйти так же, как собственный номер главной. Кроп вырезается из
уже очищенного кадра — всё, что в него попадёт, по построению отредактировано.

Номера заливаются сплошным цветом сразу: полумеры с размытием оставляли символы
читаемыми. Лица сохраняют лестницу размыв→заливка — размытого лица достаточно для «не
идентифицируется», а чёрный прямоугольник вредит данным.

Вход и выход — BGR: детекторы OpenCV работают в нём же, конверсий здесь нет.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import cv2
import numpy as np

from pipeline.crop import Detection, Detector
from pipeline.harvest.tiling import TileObservation, nms_merge

MIN_PLATE_W_PX = 16
MIN_FACE_SIDE_PX = 20
PLATE_FILL_PAD = 0.15
VEHICLE_SEARCH_PAD = 0.15
MIN_KERNEL = 15

# Пороги выбраны замером 554 отбитых кандидатов, `bench/redaction-thresholds.md`.
# Правило 1 Р-11 требует убрать читаемый номер, а не отбить кадр по любому повторному
# срабатыванию: детектор ловит собственную чёрную заливку и боксы, номером быть не
# могущие. Оба порога действуют только на повторную проверку — первый проход заливает
# всё, что нашёл.
MIN_PLATE_ASPECT = 0.66
FILL_OVERLAP_LIMIT = 0.7

NamedDetector: TypeAlias = tuple[str, Detector]
PlateDetectorInput: TypeAlias = Detector | NamedDetector
_DEFAULT_DETECTOR_NAMES = ("plate", "plate_hires")


class _TiledDetector(Protocol):
    """Необязательная трассируемая поверхность нарезанного детектора."""

    def detect_with_tiles(
        self, image: np.ndarray
    ) -> tuple[list[Detection], tuple[TileObservation, ...]]: ...


@dataclass(frozen=True, slots=True)
class RawPlateObservation:
    """Одно срабатывание до общего NMS в координатах полного кадра."""

    detector: str
    zone: tuple[int, int, int, int]
    tile: tuple[int, int, int, int]
    box: Detection


@dataclass(frozen=True, slots=True)
class PlateDetectionTrace:
    """Фактический результат детекции и необязательные наблюдения до NMS."""

    raw: tuple[RawPlateObservation, ...] = ()
    merged: tuple[Detection, ...] = ()


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Итог редакции одного кадра с необязательной трассой обоих проходов номеров."""

    image: np.ndarray
    plates: int
    faces: int
    ok: bool
    first_pass: PlateDetectionTrace = PlateDetectionTrace()
    second_pass: PlateDetectionTrace = PlateDetectionTrace()


@dataclass(frozen=True, slots=True)
class PlateStage:
    """Итог стадии номеров: что залито и что нашёл поверх этого повторный проход.

    `found` — сколько боксов нашёл первый проход, `filled` — сколько срезов он успел
    нарисовать. Числа расходятся на вырожденных боксах, и в манифест идёт `found`:
    колонку `plates` читает аудит, и её смысл — «сколько номеров было в кадре».
    """

    found: int
    filled: tuple[tuple[int, int, int, int], ...]
    residuals: tuple[Detection, ...]
    first_pass: PlateDetectionTrace = PlateDetectionTrace()
    second_pass: PlateDetectionTrace = PlateDetectionTrace()


def blur_kernel(bbox_w: int) -> int:
    """Нечётный радиус размытия, пропорциональный ширине бокса."""
    return max(MIN_KERNEL, (bbox_w // 3) | 1)


def vehicle_search_box(
    box: Detection, frame_w: int, frame_h: int, pad: float = VEHICLE_SEARCH_PAD
) -> tuple[int, int, int, int]:
    """Бокс ТС, расширенный на `pad`: номер часто ниже бампера, вне тесного бокса."""
    dx = box.w * pad
    dy = box.h * pad
    x0 = max(0, round(box.x - dx))
    y0 = max(0, round(box.y - dy))
    x1 = min(frame_w, round(box.x + box.w + dx))
    y1 = min(frame_h, round(box.y + box.h + dy))
    return x0, y0, x1, y1


def _крупные(боксы: Sequence[Detection], min_w: int, min_h: int) -> list[Detection]:
    """Отсечь детекции ниже порога различимости."""
    return [б for б in боксы if б.w >= min_w and б.h >= min_h]


def _именованные_детекторы(
    detectors: Sequence[PlateDetectorInput],
) -> tuple[NamedDetector, ...]:
    """Прикрепить стабильные имена на границе вызова, не меняя фабрику Task 1."""
    именованные: list[NamedDetector] = []
    for номер, значение in enumerate(detectors):
        if isinstance(значение, tuple):
            именованные.append(значение)
            continue
        имя = (
            _DEFAULT_DETECTOR_NAMES[номер]
            if номер < len(_DEFAULT_DETECTOR_NAMES)
            else f"plate_{номер + 1}"
        )
        именованные.append((имя, значение))
    return tuple(именованные)


def detect_plates(
    image: np.ndarray,
    vehicle_boxes: Sequence[Detection],
    named_detectors: Sequence[PlateDetectorInput],
    *,
    trace: bool = False,
) -> PlateDetectionTrace:
    """Найти номера по кадру и зонам ТС, дополнительно сохранив сырые наблюдения."""
    высота, ширина = image.shape[:2]
    зоны: list[tuple[int, int, int, int]] = [(0, 0, ширина, высота)]
    зоны += [vehicle_search_box(б, ширина, высота) for б in vehicle_boxes]

    найдено: list[Detection] = []
    наблюдения: list[RawPlateObservation] | None = [] if trace else None
    for имя, детектор in _именованные_детекторы(named_detectors):
        for зона in зоны:
            x0, y0, x1, y1 = зона
            вырез = image[y0:y1, x0:x1]
            if вырез.size == 0:
                continue
            if trace and hasattr(детектор, "detect_with_tiles"):
                локальные, тайлы = cast(_TiledDetector, детектор).detect_with_tiles(вырез)
                if наблюдения is not None:
                    for наблюдение in тайлы:
                        tx0, ty0, tx1, ty1 = наблюдение.tile
                        наблюдения.append(
                            RawPlateObservation(
                                имя,
                                зона,
                                (tx0 + x0, ty0 + y0, tx1 + x0, ty1 + y0),
                                Detection(
                                    наблюдение.box.x + x0,
                                    наблюдение.box.y + y0,
                                    наблюдение.box.w,
                                    наблюдение.box.h,
                                    наблюдение.box.conf,
                                    наблюдение.box.cls,
                                ),
                            )
                        )
            else:
                локальные = детектор.detect(вырез)
                if наблюдения is not None:
                    for бокс in локальные:
                        наблюдения.append(
                            RawPlateObservation(
                                имя,
                                зона,
                                зона,
                                Detection(
                                    бокс.x + x0,
                                    бокс.y + y0,
                                    бокс.w,
                                    бокс.h,
                                    бокс.conf,
                                    бокс.cls,
                                ),
                            )
                        )
            for бокс in локальные:
                найдено.append(
                    Detection(
                        бокс.x + x0,
                        бокс.y + y0,
                        бокс.w,
                        бокс.h,
                        бокс.conf,
                        бокс.cls,
                    )
                )
    слитые = nms_merge(_крупные(найдено, MIN_PLATE_W_PX, 1))
    return PlateDetectionTrace(tuple(наблюдения or ()), tuple(слитые))


def _залить_номер(image: np.ndarray, box: Detection) -> tuple[int, int, int, int] | None:
    """Сплошная заливка нулём, расширенная на PLATE_FILL_PAD.

    Возвращает нарисованный срез, чтобы повторная проверка могла отличить собственную
    заливку от пропущенного номера. `None` — бокс выродился после обрезки границами
    кадра и заливки не было.
    """
    высота, ширина = image.shape[:2]
    dx = box.w * PLATE_FILL_PAD
    dy = box.h * PLATE_FILL_PAD
    x0 = max(0, int(box.x - dx))
    y0 = max(0, int(box.y - dy))
    x1 = min(ширина, int(box.x + box.w + dx))
    y1 = min(высота, int(box.y + box.h + dy))
    if x1 <= x0 or y1 <= y0:
        return None
    image[y0:y1, x0:x1] = 0
    return x0, y0, x1, y1


def _замыть_лицо(image: np.ndarray, box: Detection, заливкой: bool) -> None:
    """Одна ступень замыва лица на месте."""
    высота, ширина = image.shape[:2]
    x0 = max(0, int(box.x))
    y0 = max(0, int(box.y))
    x1 = min(ширина, int(box.x + box.w))
    y1 = min(высота, int(box.y + box.h))
    if x1 <= x0 or y1 <= y0:
        return
    if заливкой:
        image[y0:y1, x0:x1] = 0
        return
    ядро = blur_kernel(x1 - x0)
    image[y0:y1, x0:x1] = cv2.GaussianBlur(image[y0:y1, x0:x1], (ядро, ядро), 0)


def redact_plates(
    image: np.ndarray,
    vehicle_boxes: Sequence[Detection],
    detectors: Sequence[PlateDetectorInput],
    *,
    trace: bool = False,
) -> PlateStage:
    """Залить номера на месте и вернуть срезы заливки с остатками второго прохода.

    Стадия вынесена из `redact_frame` публичной функцией не ради красоты: порог
    значимости остатка обязан считаться тем же кодом, который потом по нему решает.
    Офлайн-замер, повторяющий эту логику своими руками, дал бы правдоподобное неверное
    число — проект уже платил за такую копию.

    Остатки возвращаются без фильтрации: решает `significant_residuals`, а замеру нужны
    все, вместе с теми, которые правило погасит.
    """
    первый = detect_plates(image, vehicle_boxes, detectors, trace=trace)
    залито: list[tuple[int, int, int, int]] = []
    for бокс in первый.merged:
        срез = _залить_номер(image, бокс)
        if срез is not None:
            залито.append(срез)
    второй = detect_plates(image, vehicle_boxes, detectors, trace=trace)
    first_pass = первый if trace else PlateDetectionTrace()
    second_pass = второй if trace else PlateDetectionTrace()
    return PlateStage(
        len(первый.merged),
        tuple(залито),
        второй.merged,
        first_pass,
        second_pass,
    )


def fill_overlap(box: Detection, filled: Sequence[tuple[int, int, int, int]]) -> float:
    """Доля площади бокса, накрытая объединением залитых срезов.

    Объединение считается растеризацией в булеву маску размером с сам бокс: площадь
    остатка мала, стоимость незаметна, а формула включений-исключений на произвольном
    числе пересекающихся прямоугольников — источник тихих ошибок ровно в ту сторону,
    в которую ошибаться нельзя (перекрытие больше настоящего гасит живой остаток).
    """
    x0, y0 = int(box.x), int(box.y)
    x1, y1 = int(box.x + box.w), int(box.y + box.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    маска = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for зал_x0, зал_y0, зал_x1, зал_y1 in filled:
        общий_x0 = max(x0, зал_x0)
        общий_y0 = max(y0, зал_y0)
        общий_x1 = min(x1, зал_x1)
        общий_y1 = min(y1, зал_y1)
        if общий_x1 > общий_x0 and общий_y1 > общий_y0:
            маска[общий_y0 - y0 : общий_y1 - y0, общий_x0 - x0 : общий_x1 - x0] = True
    return float(маска.mean())


def plate_aspect(box: Detection) -> float:
    """Отношение сторон бокса, по которому судит порог `MIN_PLATE_ASPECT`.

    Одна на предикат и на офлайн-замер (`scripts/measure_redaction.py`) по той же причине,
    что и `fill_overlap`: порог обязан считать тот код, который потом по нему решает.
    Замер, повторивший формулу своими руками, дал бы правдоподобное неверное число —
    проект уже платил за такую копию.
    """
    return box.w / max(box.h, 1.0)


def significant_residuals(stage: PlateStage) -> list[Detection]:
    """Остатки повторного прохода, которые правда означают «номер остался».

    Два предиката стоят на разных основаниях, и разницу важно держать в голове.

    Перекрытие — структурное: если больше чем `FILL_OVERLAP_LIMIT` площади остатка лежит
    под собственной заливкой, детектор увидел чёрный прямоугольник, который редакция сама
    и нарисовала. Читаемому номеру там взяться неоткуда.

    Отношение сторон — эмпирическое. Замер 554 отбитых кадров
    (`bench/redaction-thresholds.md`) опроверг ходовую посылку «номера такой формы не
    бывает ни в одной стране»: детектор ловит фрагменты настоящих номеров — блок региона,
    половину двухрядного, — а у фрагмента отношение сторон любое. Значение держится не на
    форме номеров, а на замере: `0.66` — наименьшее круглое число в окне
    `(0.6512793, 0.7364900]`, где диагноз C2 (кадр `69548496`) лечится и не гасится ни один
    остаток, признанный читаемым **человеком**. Нижняя граница окна — это требование
    вылечить диагноз, а не отсутствие протечек: протечек нет и при 0.50. Остаток #74 (блок
    региона `77 RUS`, символов самого номера не видно) при `0.66` гаснет — нечитаемым его
    признал человек на гейте §4 отчёта, а не замер. Менять число можно только вместе с
    новым замером и новым решением по этой развилке.

    Всё остальное отбивает кадр, как отбивало.
    """
    значимые: list[Detection] = []
    for бокс in stage.residuals:
        if fill_overlap(бокс, stage.filled) >= FILL_OVERLAP_LIMIT:
            continue
        if plate_aspect(бокс) < MIN_PLATE_ASPECT:
            continue
        значимые.append(бокс)
    return значимые


def redact_frame(
    image: np.ndarray,
    vehicle_boxes: Sequence[Detection],
    plate_detectors: Sequence[PlateDetectorInput],
    face_detector: Detector,
    *,
    trace: bool = False,
) -> RedactionResult:
    """Очистить весь кадр: номера — заливкой, лица — лестницей, с повторной проверкой.

    Кадр отбивается не по любому остатку повторной проверки номеров, а только по значимому:
    что значимо, решает `significant_residuals`.
    """
    рабочая = image.copy()

    номера = redact_plates(рабочая, vehicle_boxes, plate_detectors, trace=trace)
    if significant_residuals(номера):
        return RedactionResult(
            рабочая,
            номера.found,
            0,
            False,
            номера.first_pass,
            номера.second_pass,
        )

    лица = _крупные(face_detector.detect(рабочая), MIN_FACE_SIDE_PX, MIN_FACE_SIDE_PX)
    лица_число = len(лица)
    for заливкой in (False, True):
        if not лица:
            break
        for бокс in лица:
            _замыть_лицо(рабочая, бокс, заливкой)
        лица = _крупные(face_detector.detect(рабочая), MIN_FACE_SIDE_PX, MIN_FACE_SIDE_PX)
    ok = not лица
    return RedactionResult(
        рабочая,
        номера.found,
        лица_число,
        ok,
        номера.first_pass,
        номера.second_pass,
    )
