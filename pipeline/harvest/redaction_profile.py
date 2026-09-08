"""Профиль редакции номеров, его провенанс и единственная фабрика plate-детекторов.

Три инструмента — боевой харвест, переderivация и замер — строили пару детекторов
номеров тремя почти одинаковыми блоками. Копии молчаливо расходятся при первой же
правке порога, и тогда «пересобрано тем же контуром» держится на совпадении, которое
никто не проверяет. Здесь эта пара собирается один раз, из профиля.

Профиль называет параметры явно, включая те, что раньше приходили умолчаниями классов
(`TiledDetector.tile`, `OnnxBoxDetector.conf`). Умолчание — это параметр, о котором
прогон не знает, что он у него есть: его правка меняет пиксели кропов, не оставляя следа
ни в одном артефакте. `CURRENT_REDACTION_PROFILE` повторяет прежние эффективные значения
буква в букву, поэтому переход на фабрику ничего не меняет в результатах.

`profile_id` отвечает на вопрос «этим ли контуром получены кропы»: он смешивает
канонический JSON параметров, SHA256 весов, полный Git SHA и хеш байтов кода стадии
детекции. Кода в списке свидетелей достаточно, чтобы правка `redact.py` без правки
параметров не выдавала себя за прежний профиль.

Тяжёлые классы детекторов импортируются внутри фабрики (Р-15): модуль обязан
импортироваться в базовом окружении, где нет ни весов, ни `ultralytics`, — иначе профиль
и его провенанс нельзя было бы протестировать.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.crop import Detector
from pipeline.harvest.detectors import DetectorPin

_ПАКЕТ = Path(__file__).resolve().parent

# Файлы, чьи байты способны изменить найденные боксы при неизменных параметрах: сама
# редакция, нарезка, обёртки детекторов и эта фабрика. Пути не хешируются — только имена
# и содержимое, иначе идентификатор зависел бы от каталога чекаута.
DETECTOR_STAGE_SOURCES = (
    _ПАКЕТ / "redact.py",
    _ПАКЕТ / "tiling.py",
    _ПАКЕТ / "backends.py",
    _ПАКЕТ.parent / "detectors" / "yolo.py",
    Path(__file__).resolve(),
)


@dataclass(frozen=True, slots=True)
class DetectorProfile:
    """Параметры одного детектора номеров: порог и нарезка кадра."""

    confidence: float
    tile: int
    overlap: float
    iou_merge: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence вне [0, 1]: {self.confidence}")
        if self.tile < 1:
            raise ValueError(f"tile меньше 1: {self.tile}")
        # Верхняя граница перекрытия открыта: при 1.0 шаг тайла обнуляется и нарезка
        # перестала бы двигаться по кадру.
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap вне [0, 1): {self.overlap}")
        # Нулевой порог NMS сливал бы любые два бокса, поэтому нижняя граница открыта.
        if not 0.0 < self.iou_merge <= 1.0:
            raise ValueError(f"iou_merge вне (0, 1]: {self.iou_merge}")


@dataclass(frozen=True, slots=True)
class RedactionProfile:
    """Пара детекторов номеров: SSD `plate` и высокоразрешающий YOLO `plate_hires`."""

    plate: DetectorProfile
    plate_hires: DetectorProfile


# Значения повторяют эффективные параметры прогонов до шага 0: `OnnxBoxDetector.conf`,
# `TiledDetector.tile`/`overlap`/`iou_merge` брались умолчаниями классов, а `conf=0.25`
# у YOLO задавался на месте вызова.
CURRENT_REDACTION_PROFILE = RedactionProfile(
    plate=DetectorProfile(confidence=0.5, tile=1280, overlap=0.25),
    plate_hires=DetectorProfile(confidence=0.25, tile=1280, overlap=0.25),
)


def canonical_profile_json(profile: RedactionProfile) -> bytes:
    """Каноническая сериализация профиля: сортировка ключей, компактные разделители.

    Порядок полей в памяти зависит от того, как записан класс, а идентификатор прогона
    зависеть от этого не должен — иначе перестановка двух строк в исходнике объявила бы
    старые кропы полученными другим профилем.
    """
    текст = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{текст}\n".encode()


def detector_stage_sha256() -> str:
    """Хеш байтов кода стадии детекции — свидетель «код тот же» без Git-состояния."""
    хеш = hashlib.sha256()
    for путь in DETECTOR_STAGE_SOURCES:
        # Имя файла входит в хеш, чтобы перенос строки из одного файла в другой не давал
        # прежнего значения на склеенных байтах.
        хеш.update(путь.name.encode("utf-8"))
        хеш.update(hashlib.sha256(путь.read_bytes()).digest())
    return хеш.hexdigest()


def profile_id(
    profile: RedactionProfile,
    *,
    pins: Mapping[str, DetectorPin],
    git_sha: str,
    stage_sha256: str,
) -> str:
    """Идентификатор прогона: параметры, веса, ревизия дерева и код стадии вместе.

    Свидетели перечислены явно и порознь: расхождение любого из них означает другие
    пиксели, а один общий «хеш всего» не дал бы сказать, чем именно прогоны разошлись.
    """
    хеш = hashlib.sha256()
    хеш.update(canonical_profile_json(profile))
    for имя in sorted(pins):
        хеш.update(f"weights:{имя}={pins[имя].sha256}\n".encode())
    хеш.update(f"git={git_sha}\n".encode())
    хеш.update(f"stage={stage_sha256}\n".encode())
    return хеш.hexdigest()


def build_plate_detectors(
    profile: RedactionProfile,
    *,
    models_dir: Path,
    pins: Mapping[str, DetectorPin],
) -> tuple[Detector, Detector]:
    """Пара детекторов номеров по профилю. Единственное место их сборки в проекте.

    Формат весов берётся из пинов, а не из литерала расширения: манифест разрешает и
    `onnx`, и `pt`, и рассинхрон имени файла с манифестом означал бы прогон на весах,
    SHA256 которых никто не сверял.
    """
    from pipeline.detectors.yolo import YoloDetector
    from pipeline.harvest.backends import OnnxBoxDetector
    from pipeline.harvest.tiling import TiledDetector

    ssd = TiledDetector(
        OnnxBoxDetector(
            models_dir / f"plate.{pins['plate'].format}",
            conf=profile.plate.confidence,
        ),
        tile=profile.plate.tile,
        overlap=profile.plate.overlap,
        iou_merge=profile.plate.iou_merge,
    )
    hires = TiledDetector(
        YoloDetector(
            weights=str(models_dir / f"plate_hires.{pins['plate_hires'].format}"),
            conf=profile.plate_hires.confidence,
        ),
        tile=profile.plate_hires.tile,
        overlap=profile.plate_hires.overlap,
        iou_merge=profile.plate_hires.iou_merge,
    )
    return ssd, hires
