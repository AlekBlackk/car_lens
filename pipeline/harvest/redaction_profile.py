"""Профиль редакции номеров, его провенанс и единственная фабрика plate-детекторов.

Три инструмента — боевой харвест, переderivация и замер — собирают набор детекторов
номеров из единого профиля. Копии молчаливо расходятся при первой же правке порога,
и тогда «пересобрано тем же контуром» держится на совпадении, которое никто не проверяет.
Здесь детекторы собираются один раз, из профиля.

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
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, overload

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

_INSTANCE_NAME_RE = re.compile(r"^[a-z0-9_]+$")


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
class DetectorEntry:
    """Запись одного детектора в профиле редакции.

    - instance: уникальное имя экземпляра (^[a-z0-9_]+$), попадает в observations.csv;
    - weights: ключ пина весов из detectors_manifest.csv (например, plate или plate_hires);
    - params: параметры нарезки и порога детектора.
    """

    instance: str
    weights: str
    params: DetectorProfile

    def __post_init__(self) -> None:
        if not isinstance(self.instance, str) or not _INSTANCE_NAME_RE.match(self.instance):
            raise ValueError(
                f"недопустимое имя экземпляра {self.instance!r}: "
                f"ожидался шаблон {_INSTANCE_NAME_RE.pattern}"
            )
        if not isinstance(self.weights, str) or not _INSTANCE_NAME_RE.match(self.weights):
            raise ValueError(
                f"недопустимый ключ весов {self.weights!r}: "
                f"ожидался шаблон {_INSTANCE_NAME_RE.pattern}"
            )
        if not isinstance(self.params, DetectorProfile):
            raise TypeError(
                "параметры детектора должны быть DetectorProfile, "
                f"получено {type(self.params).__name__}"
            )


@dataclass(frozen=True, slots=True)
class RedactionProfile:
    """Упорядоченный набор именованных детекторов номеров.

    Хранится отсортированным по instance. Повторяющиеся имена и пустой набор запрещены.
    """

    detectors: tuple[DetectorEntry, ...]

    def __init__(
        self,
        detectors: (
            Sequence[DetectorEntry | tuple[str, DetectorProfile] | tuple[str, str, DetectorProfile]]
            | None
        ) = None,
        *,
        plate: DetectorProfile | None = None,
        plate_hires: DetectorProfile | None = None,
    ) -> None:
        if detectors is None:
            if plate is not None and plate_hires is not None:
                detectors = (
                    DetectorEntry(instance="plate", weights="plate", params=plate),
                    DetectorEntry(
                        instance="plate_hires", weights="plate_hires", params=plate_hires
                    ),
                )
            elif plate is not None or plate_hires is not None:
                raise ValueError(
                    "необходимо указать оба детектора 'plate' и 'plate_hires' "
                    "или передать 'detectors'"
                )
            else:
                raise ValueError("профиль редакции не может быть пустым")
        elif plate is not None or plate_hires is not None:
            raise ValueError(
                "нельзя передавать одновременно 'detectors' "
                "и именованные аргументы 'plate'/'plate_hires'"
            )
        if not detectors:
            raise ValueError("профиль редакции не может быть пустым")

        нормализованные: list[DetectorEntry] = []
        for пункт in detectors:
            if isinstance(пункт, DetectorEntry):
                нормализованные.append(пункт)
            elif isinstance(пункт, (tuple, list)):
                if len(пункт) == 2:
                    имя, params = пункт
                    if not isinstance(params, DetectorProfile):
                        raise TypeError(
                            "параметры детектора должны быть DetectorProfile, "
                            f"получено {type(params).__name__}"
                        )
                    нормализованные.append(DetectorEntry(instance=имя, weights=имя, params=params))
                elif len(пункт) == 3:
                    instance, weights, params = пункт
                    if not isinstance(params, DetectorProfile):
                        raise TypeError(
                            "параметры детектора должны быть DetectorProfile, "
                            f"получено {type(params).__name__}"
                        )
                    нормализованные.append(
                        DetectorEntry(instance=instance, weights=weights, params=params)
                    )
                else:
                    raise ValueError(f"недопустимый кортеж детектора: {пункт}")
            else:
                raise TypeError(f"недопустимый элемент профиля: {type(пункт).__name__}")

        имена = [д.instance for д in нормализованные]
        if len(имена) != len(set(имена)):
            raise ValueError(f"повторяющиеся имена детекторов в профиле: {имена}")

        отсортированные = tuple(sorted(нормализованные, key=lambda д: д.instance))
        object.__setattr__(self, "detectors", отсортированные)

    def __iter__(self) -> Iterator[DetectorEntry]:
        return iter(self.detectors)

    def __len__(self) -> int:
        return len(self.detectors)

    @overload
    def __getitem__(self, key: int) -> DetectorEntry: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[DetectorEntry, ...]: ...

    @overload
    def __getitem__(self, key: str) -> DetectorEntry: ...

    def __getitem__(self, key: int | slice | str) -> DetectorEntry | tuple[DetectorEntry, ...]:
        if isinstance(key, str):
            entry = self.get(key)
            if entry is None:
                raise KeyError(f"детектор {key!r} отсутствует в профиле")
            return entry
        return self.detectors[key]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(d.instance == item for d in self.detectors)
        return item in self.detectors

    def get(self, instance: str) -> DetectorEntry | None:
        for d in self.detectors:
            if d.instance == instance:
                return d
        return None

    @property
    def plate(self) -> DetectorProfile:
        entry = self.get("plate")
        if entry is None:
            raise AttributeError("в профиле нет детектора 'plate'")
        return entry.params

    @property
    def plate_hires(self) -> DetectorProfile:
        entry = self.get("plate_hires")
        if entry is None:
            raise AttributeError("в профиле нет детектора 'plate_hires'")
        return entry.params


CURRENT_REDACTION_PROFILE = RedactionProfile(
    (
        DetectorEntry(
            instance="plate",
            weights="plate",
            params=DetectorProfile(confidence=0.5, tile=1280, overlap=0.25),
        ),
        DetectorEntry(
            instance="plate_hires",
            weights="plate_hires",
            params=DetectorProfile(confidence=0.25, tile=1280, overlap=0.25),
        ),
    )
)


def canonical_profile_json(profile: RedactionProfile) -> bytes:
    """Каноническая сериализация профиля: сортировка ключей, компактные разделители.

    Порядок записей гарантирован сортировкой по instance при конструировании, а ключи
    каждого словаря упорядочиваются sort_keys=True.
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
    Веса подмешиваются по каждому различному ключу weights (§3.4 дизайна).
    """
    хеш = hashlib.sha256()
    хеш.update(canonical_profile_json(profile))
    for weights_key in sorted({entry.weights for entry in profile.detectors}):
        if weights_key not in pins:
            raise ValueError(f"нет пина для весов: {weights_key}")
        хеш.update(f"weights:{weights_key}={pins[weights_key].sha256}\n".encode())
    хеш.update(f"git={git_sha}\n".encode())
    хеш.update(f"stage={stage_sha256}\n".encode())
    return хеш.hexdigest()


NamedDetector: TypeAlias = tuple[str, Detector]


def build_plate_detectors(
    profile: RedactionProfile,
    *,
    models_dir: Path,
    pins: Mapping[str, DetectorPin],
) -> tuple[NamedDetector, ...]:
    """Набор именованных детекторов номеров по профилю. Единственное место их сборки.

    Формат весов берётся из пинов, а не из литерала расширения: манифест разрешает и
    `onnx`, и `pt`, и рассинхрон имени файла с манифестом означал бы прогон на весах,
    SHA256 которых никто не сверял. Фабрика требует пин на каждый ключ весов и падает
    при его отсутствии.
    """
    from pipeline.detectors.yolo import YoloDetector
    from pipeline.harvest.backends import OnnxBoxDetector
    from pipeline.harvest.tiling import TiledDetector

    результат: list[NamedDetector] = []
    for entry in profile.detectors:
        if entry.weights not in pins:
            raise ValueError(f"нет пина для весов {entry.weights!r}")
        pin = pins[entry.weights]
        model_path = models_dir / f"{entry.weights}.{pin.format}"
        if pin.format == "onnx":
            inner: Detector = OnnxBoxDetector(
                model_path,
                conf=entry.params.confidence,
            )
        elif pin.format == "pt":
            inner = YoloDetector(
                weights=str(model_path),
                conf=entry.params.confidence,
            )
        else:
            raise ValueError(f"неподдерживаемый формат весов {pin.format!r} для {entry.weights!r}")

        tiled = TiledDetector(
            inner,
            tile=entry.params.tile,
            overlap=entry.params.overlap,
            iou_merge=entry.params.iou_merge,
        )
        результат.append((entry.instance, tiled))
    return tuple(результат)
