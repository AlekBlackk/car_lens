"""Эталон паритета для устройства: двенадцать кадров в двух уровнях (§9 дизайна M-1.5).

Уровень A (`tensor`) — готовый кроп `320×320` до нормализации: устройство обязано
получить тот же эмбеддинг, что и питон, при побитово одинаковом входе. Уровень B
(`frame`) — целый кадр и бокс: сюда добавляются кроп, паддинг и ресайз. Разделение
нужно для диагностики: падает A — виноват рантайм или нормализация, A зелёный и B
красный — виноват кроп или ресайз.

PNG, а не JPEG: `cv2.imdecode` и `BitmapFactory` реализуют IDCT по-разному, и
расхождение декодеров добавилось бы к тому, что тест измеряет.

Детектор здесь — экспортированный `detector_fp32.tflite`, а не веса ultralytics.
Причин две. Во-первых, `ultralytics` нет в группе `export`, а ставить его в это
окружение нельзя: он доустанавливает конвертер прямо туда, где запущен, и
пинованная venv экспорта перестаёт соответствовать `uv.lock`. Во-вторых, так тест
паритета меряет то, ради чего написан — препроцессинг и постобработку на Kotlin
против питоновских, — а не качество экспорта детектора, которое проверяется
отчётом контракта на M-1.5 и паритетом на M-1.4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from export.to_tflite import _run_tflite
from pipeline.crop import Detection, crop_image, iou, select_primary
from pipeline.index_format import read_index
from pipeline.knn import top5_by_centroids
from pipeline.paths import raw_dir, splits_dir
from pipeline.schema import read_split
from training.dataset import build_transforms

КАДРОВ = 12
ПРЕДЕЛ_ДЛИННОЙ = 720
# PNG жмётся до упора: фикстура коммитится, а разница между уровнями сжатия — только
# время записи, пиксели одинаковы.
СЖАТИЕ = (cv2.IMWRITE_PNG_COMPRESSION, 9)
МИНИМАЛЬНАЯ_СТОРОНА = 320
ИСХОДНИКИ = "stanford_cars"
ПРЕФИКС_ФОТО = "stanford-"


class GoldenError(RuntimeError):
    """Эталон собрать нельзя."""


def sha256_of_file(path: Path) -> str:
    """Пятая копия в дереве и по той же причине: модуль не тянет за собой экспорт."""
    хеш = hashlib.sha256()
    with path.open("rb") as файл:
        for кусок in iter(lambda: файл.read(1 << 20), b""):
            хеш.update(кусок)
    return хеш.hexdigest()


def подготовить_кадр(путь: Path) -> np.ndarray:
    """Прочитать фото и ужать до короткой стороны `МИНИМАЛЬНАЯ_СТОРОНА`, не увеличивая.

    Мера — короткая сторона, а не длинная: фикстура коммитится, и лишние пиксели
    здесь ничего не проверяют. Ужимание по длинной стороне отбрасывало бы вытянутые
    кадры (короткая уезжает под 320, а тест требует минимум 320) и подставляло бы
    вместо них кадры покрупнее — то есть увеличивало бы фикстуру, а не уменьшало.
    """
    изображение = cv2.imread(str(путь))
    if изображение is None:
        raise GoldenError(f"не читается {путь}")
    высота, ширина = изображение.shape[:2]
    масштаб = min(
        1.0,
        МИНИМАЛЬНАЯ_СТОРОНА / min(высота, ширина),
        ПРЕДЕЛ_ДЛИННОЙ / max(высота, ширина),
    )
    if масштаб == 1.0:
        return изображение
    return cv2.resize(
        изображение,
        (round(ширина * масштаб), round(высота * масштаб)),
        interpolation=cv2.INTER_AREA,
    )


def letterbox(
    изображение: np.ndarray, размер: int, поле: int
) -> tuple[np.ndarray, float, float, float]:
    """Вписать кадр в квадрат с серыми полями. Тот же расчёт, что в `Letterbox.forFrame`."""
    высота, ширина = изображение.shape[:2]
    масштаб = min(размер / ширина, размер / высота)
    новая_ширина = round(ширина * масштаб)
    новая_высота = round(высота * масштаб)
    сжатое = cv2.resize(изображение, (новая_ширина, новая_высота), interpolation=cv2.INTER_LINEAR)
    отступ_x = (размер - ширина * масштаб) / 2
    отступ_y = (размер - высота * масштаб) / 2
    слева = round(отступ_x - 0.1)
    сверху = round(отступ_y - 0.1)
    холст = np.full((размер, размер, 3), поле, dtype=np.uint8)
    холст[сверху : сверху + новая_высота, слева : слева + новая_ширина] = сжатое
    return холст, масштаб, отступ_x, отступ_y


def _nms(боксы: list[Detection], порог: float, максимум: int) -> list[Detection]:
    """Жадный NMS — тот же алгоритм и тот же порядок, что в `Boxes.nms` на Kotlin."""
    оставленные: list[Detection] = []
    for кандидат in sorted(боксы, key=lambda d: d.conf, reverse=True):
        if len(оставленные) >= максимум:
            break
        if all(iou(оставленный, кандидат) <= порог for оставленный in оставленные):
            оставленные.append(кандидат)
    return оставленные


def детектировать(
    модель: Path, контракт: dict[str, Any], изображение: np.ndarray
) -> list[Detection]:
    """Питоновский двойник `Detector` на Kotlin: тот же граф, тот же разбор выхода."""
    размер = int(контракт["imgsz"])
    вписанное, масштаб, отступ_x, отступ_y = letterbox(
        изображение, размер, int(контракт["letterbox_pad_value"])
    )
    rgb = cv2.cvtColor(вписанное, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    тензор = np.transpose(rgb, (2, 0, 1))[None, ...]
    сырой = _run_tflite(модель, тензор)
    выход = np.asarray(сырой, dtype=np.float32)
    if выход.ndim == 3:
        выход = выход[0]
    if контракт["output_layout"] == "CN":
        каналы = выход
    elif контракт["output_layout"] == "NC":
        каналы = выход.T
    else:
        raise GoldenError(f"неизвестный output_layout {контракт['output_layout']!r}")

    единицы = float(размер) if контракт["coord_units"] == "normalized" else 1.0
    классы = [int(значение) for значение in контракт["vehicle_classes"]]
    скоры = каналы[4:, :][классы, :]
    лучшие = скоры.max(axis=0)
    лучший_класс = np.asarray(классы)[скоры.argmax(axis=0)]
    кандидаты: list[Detection] = []
    for якорь in np.flatnonzero(лучшие >= 0.35):
        cx, cy, ширина_бокса, высота_бокса = каналы[:4, якорь] * единицы
        кандидаты.append(
            Detection(
                x=float(cx - ширина_бокса / 2),
                y=float(cy - высота_бокса / 2),
                w=float(ширина_бокса),
                h=float(высота_бокса),
                conf=float(лучшие[якорь]),
                cls=int(лучший_класс[якорь]),
            )
        )
    выжившие = _nms(кандидаты, 0.5, int(контракт["max_det"]))
    # Разворот в координаты кадра идёт после NMS: боксы сравнимы между собой в
    # координатах тензора, и порядок «сначала развернуть» дал бы другой набор.
    return [
        Detection(
            x=(бокс.x - отступ_x) / масштаб,
            y=(бокс.y - отступ_y) / масштаб,
            w=бокс.w / масштаб,
            h=бокс.h / масштаб,
            conf=бокс.conf,
            cls=бокс.cls,
        )
        for бокс in выжившие
    ]


def эмбеддинг(модель: Path, кроп_rgb: np.ndarray, преобразование: Any) -> np.ndarray:
    """Кроп RGB `320×320` → нормализованный эмбеддинг, как на устройстве."""
    тензор = преобразование(image=кроп_rgb)["image"].numpy()[None, ...]
    сырой = _run_tflite(модель, тензор.astype(np.float32))
    вектор = np.asarray(сырой, dtype=np.float32)[0]
    норма = float(np.linalg.norm(вектор))
    if not np.isfinite(норма) or норма <= 1e-12:
        raise GoldenError(f"норма эмбеддинга {норма}: нормализовать нечего")
    нормированный: np.ndarray = (вектор / норма).astype(np.float32)
    return нормированный


def _тройка(значения: Any, имя: str) -> tuple[float, float, float]:
    """Три числа на канал, а не сколько записано: чужая длина роняет здесь, а не в графе."""
    числа = [float(значение) for значение in значения]
    if len(числа) != 3:
        raise GoldenError(f"{имя}: {len(числа)} значений, нужно ровно три")
    return числа[0], числа[1], числа[2]


def _исходник(photo_id: str) -> Path:
    имя = photo_id.removeprefix(ПРЕФИКС_ФОТО)
    return raw_dir() / ИСХОДНИКИ / "images" / f"{имя}.jpg"


def _кандидаты(сплит: Path) -> list[str]:
    """Двенадцать кадров детерминированно: сортировка и равный шаг, как в калибровке.

    Возвращается весь упорядоченный список, а не двенадцать штук: кадр может
    отсеяться (нет исходника, детектор ничего не нашёл), и тогда берётся следующий
    за ним — иначе эталон зависел бы от того, какие кадры отсеялись.
    """
    кадр = read_split(сплит)
    упорядоченные = sorted(str(значение) for значение in кадр["photo_id"])
    if len(упорядоченные) < КАДРОВ:
        raise GoldenError(f"{сплит}: {len(упорядоченные)} строк, нужно хотя бы {КАДРОВ}")
    шаг = len(упорядоченные) // КАДРОВ
    порядок = [упорядоченные[позиция * шаг] for позиция in range(КАДРОВ)]
    остальные = [photo_id for photo_id in упорядоченные if photo_id not in set(порядок)]
    return порядок + остальные


def _parse_args() -> argparse.Namespace:
    парсер = argparse.ArgumentParser(description="Эталон паритета устройства")
    парсер.add_argument("--run", type=Path, required=True, help="каталог прогона обучения")
    парсер.add_argument("--detector", type=Path, required=True, help="detector_fp32.tflite")
    парсер.add_argument("--out", type=Path, default=Path("tests/fixtures/device"))
    парсер.add_argument("--split", type=Path, default=None, help="parquet сплита, умолчание val")
    парсер.add_argument(
        "--index", type=Path, default=None, help="каталог индекса, умолчание run/index"
    )
    return парсер.parse_args()


def main() -> None:
    аргументы = _parse_args()
    экспорт = аргументы.run / "export"
    отчёт = json.loads((экспорт / "export_report.json").read_text(encoding="utf-8"))
    манифест = отчёт["manifest"]
    классификатор = экспорт / "classifier_int8.tflite"
    if not классификатор.is_file():
        raise GoldenError(f"нет {классификатор}: экспорт шёл с --skip-int8?")
    контракт = json.loads(
        аргументы.detector.with_name("detector_report.json").read_text(encoding="utf-8")
    )["contract"]

    индекс = аргументы.index or (аргументы.run / "index")
    _, центроиды = read_index(индекс / "catalog_index.bin")
    преобразование = build_transforms(
        int(манифест["img_size"]),
        train=False,
        mean=_тройка(манифест["mean"], "mean"),
        std=_тройка(манифест["std"], "std"),
    )

    каталог_кадров = аргументы.out / "frame"
    каталог_тензоров = аргументы.out / "tensor"
    for каталог in (каталог_кадров, каталог_тензоров):
        каталог.mkdir(parents=True, exist_ok=True)

    записи_кадров: list[dict[str, Any]] = []
    записи_тензоров: list[dict[str, Any]] = []
    отсеяно: dict[str, int] = {}
    for photo_id in _кандидаты(аргументы.split or (splits_dir() / "val.parquet")):
        if len(записи_кадров) == КАДРОВ:
            break
        исходник = _исходник(photo_id)
        if not исходник.is_file():
            отсеяно["нет исходника"] = отсеяно.get("нет исходника", 0) + 1
            continue
        кадр = подготовить_кадр(исходник)
        высота, ширина = кадр.shape[:2]
        if min(высота, ширина) < МИНИМАЛЬНАЯ_СТОРОНА:
            отсеяно["мелкий кадр"] = отсеяно.get("мелкий кадр", 0) + 1
            continue

        номер = len(записи_кадров)
        имя_кадра = f"frame/{номер:03d}.png"
        # Кадр сохраняется до детекции, и дальше всё считается по перечитанному
        # файлу: устройство получит именно эти пиксели, а не те, что были в памяти.
        if not cv2.imwrite(str(аргументы.out / имя_кадра), кадр, СЖАТИЕ):
            raise GoldenError(f"не удалось записать {имя_кадра}")
        перечитанный_кадр = cv2.imread(str(аргументы.out / имя_кадра))
        if перечитанный_кадр is None:
            raise GoldenError(f"{имя_кадра} записан, но не читается обратно")
        кадр = перечитанный_кадр

        боксы = детектировать(аргументы.detector, контракт, кадр)
        решение = select_primary(боксы, ширина, высота)
        if решение.box is None:
            (аргументы.out / имя_кадра).unlink()
            отсеяно[решение.reject_reason or "нет бокса"] = (
                отсеяно.get(решение.reject_reason or "нет бокса", 0) + 1
            )
            continue
        бокс = решение.box

        кроп = crop_image(кадр, бокс, pad=0.08)
        кроп_rgb = cv2.cvtColor(кроп, cv2.COLOR_BGR2RGB)
        готовый = cv2.resize(
            кроп_rgb,
            (int(манифест["img_size"]), int(манифест["img_size"])),
            interpolation=cv2.INTER_LINEAR,
        )
        имя_тензора = f"tensor/{номер:03d}.png"
        if not cv2.imwrite(
            str(аргументы.out / имя_тензора), cv2.cvtColor(готовый, cv2.COLOR_RGB2BGR), СЖАТИЕ
        ):
            raise GoldenError(f"не удалось записать {имя_тензора}")
        # Эмбеддинг считается по перечитанному PNG, а не по массиву в памяти: иначе
        # эталон описывал бы вход, которого на устройстве не будет.
        перечитанный = cv2.imread(str(аргументы.out / имя_тензора))
        if перечитанный is None:
            raise GoldenError(f"{имя_тензора} записан, но не читается обратно")
        вектор = эмбеддинг(
            классификатор, cv2.cvtColor(перечитанный, cv2.COLOR_BGR2RGB), преобразование
        )
        top5 = top5_by_centroids(вектор[None, :], центроиды)[0]

        описание_бокса = {
            "x": float(бокс.x),
            "y": float(бокс.y),
            "w": float(бокс.w),
            "h": float(бокс.h),
        }
        записи_тензоров.append(
            {"png": имя_тензора, "embedding": [float(з) for з in вектор], "top5": top5}
        )
        записи_кадров.append(
            {
                "png": имя_кадра,
                # Связь кадра с его кропом явная: инструментальный тест уровня B
                # сравнивает свой кроп с этим PNG, и соглашение «одинаковый индекс»
                # держать в голове нельзя.
                "tensor_png": имя_тензора,
                "photo_id": photo_id,
                "box": описание_бокса,
                "detector_box": описание_бокса,
                "detector_conf": float(бокс.conf),
                "embedding": [float(з) for з in вектор],
                "top5": top5,
            }
        )

    if len(записи_кадров) != КАДРОВ:
        raise GoldenError(f"собрано {len(записи_кадров)} кадров из {КАДРОВ}; отсев: {отсеяно}")

    эталон = {
        "checkpoint_sha256": манифест["checkpoint_sha256"],
        "classifier_sha256": sha256_of_file(классификатор),
        "detector_sha256": sha256_of_file(аргументы.detector),
        "embedding_dim": int(манифест["embedding_dim"]),
        "img_size": int(манифест["img_size"]),
        "tensor": записи_тензоров,
        "frame": записи_кадров,
    }
    путь = аргументы.out / "golden.json"
    путь.write_text(json.dumps(эталон, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"эталон записан в {путь}; отсев: {отсеяно or 'нет'}")


if __name__ == "__main__":
    main()
