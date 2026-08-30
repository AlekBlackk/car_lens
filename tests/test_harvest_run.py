# tests/test_harvest_run.py
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.crop import Detection
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.manifest import load_manifest
from pipeline.harvest.queries import ClassQueries
from pipeline.harvest.run import HarvestDeps, harvest_class, process_candidate
from pipeline.harvest.sources import Candidate, SearchPage, SourceQuery
from pipeline.harvest.state import ACCEPTED_TARGET, SEEN_BUDGET, HarvestState


def _jpeg(ширина: int = 1200, высота: int = 900, seed: int = 0) -> bytes:
    буфер = io.BytesIO()
    if seed == 0:
        Image.new("RGB", (ширина, высота), (30, 60, 90)).save(буфер, format="JPEG")
    else:
        rng = np.random.default_rng(seed)
        массив = rng.integers(0, 255, size=(высота, ширина, 3), dtype=np.uint8)
        Image.fromarray(массив).save(буфер, format="JPEG")
    return буфер.getvalue()


class ПодставнойИсточник:
    name = "commons"

    def __init__(self, страницы: list[SearchPage]) -> None:
        self.страницы = list(страницы)
        self.скачано = 0

    def search(self, query: SourceQuery) -> SearchPage:
        return self.страницы.pop(0) if self.страницы else SearchPage((), None)

    def fetch(self, candidate: Candidate) -> bytes:
        self.скачано += 1
        return _jpeg()


class СписокДетектор:
    """Отдаёт заготовленные боксы: по одному списку на каждый вызов detect."""

    def __init__(self, сценарий: list[list[Detection]]) -> None:
        self.сценарий = list(сценарий)
        self.вызовы = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.вызовы += 1
        return self.сценарий.pop(0) if self.сценарий else []


class ФиксированныйДетектор:
    def __init__(self, боксы: list[Detection]) -> None:
        self.боксы = боксы

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.боксы


class ПостоянныйЭнкодер:
    def __init__(self, вектор: np.ndarray) -> None:
        self.вектор = вектор

    def encode_rgb(self, image: np.ndarray) -> np.ndarray:
        return self.вектор


def _кандидат(class_id: str = "lada__granta", **правки: object) -> Candidate:
    поля: dict[str, object] = {
        "source": "commons",
        "source_item_id": "12345",
        "class_id": class_id,
        "page_url": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "file_url": "https://upload.wikimedia.org/X.jpg",
        "license_raw": "CC BY 4.0",
        "author": '<a href="/wiki/User:X">Иван Петров</a>',
        "title": "File:X.jpg",
        "width": 1200,
        "height": 900,
        "basis": "category:Lada Granta",
    }
    поля.update(правки)
    return Candidate(**поля)  # type: ignore[arg-type]


def _зависимости(
    tmp_path: Path,
    источник: ПодставнойИсточник | None = None,
    **правки: object,
) -> HarvestDeps:
    машина = Detection(x=100.0, y=100.0, w=700.0, h=500.0, conf=0.9, cls=2)
    поля: dict[str, object] = {
        "source": источник if источник is not None else ПодставнойИсточник([]),
        "vehicle_detector": ФиксированныйДетектор([машина]),
        "plate_detectors": (),
        "face_detector": ФиксированныйДетектор([]),
        "encoder": ПостоянныйЭнкодер(np.array([1.0, 0.0], dtype=np.float32)),
        "class_matrix": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "class_order": ("lada__granta", "uaz__patriot"),
        "negative_matrix": np.array([[0.0, 1.0]], dtype=np.float32),
        "allow_sa": False,
        "root": tmp_path,
        "manifest_path": tmp_path / "harvest_manifest.csv",
        "checked": "2026-08-28",
    }
    поля.update(правки)
    return HarvestDeps(**поля)  # type: ignore[arg-type]


_deps = _зависимости


def test_годный_кандидат_принимается(tmp_path: Path) -> None:
    источник = ПодставнойИсточник([])
    причина, строка = process_candidate(_кандидат(), _зависимости(tmp_path, источник), DedupIndex())
    assert причина == "accepted"
    assert строка is not None
    assert строка.author == "Иван Петров", "HTML из extmetadata обязан сняться"
    assert строка.license_tier == 2
    assert строка.class_id == "lada__granta"
    assert (tmp_path / "crops" / "lada__granta" / f"{строка.sha256}.jpg").exists()


def test_мелкий_файл_отсекается_до_скачивания(tmp_path: Path) -> None:
    источник = ПодставнойИсточник([])
    причина, строка = process_candidate(
        _кандидат(width=400, height=300), _зависимости(tmp_path, источник), DedupIndex()
    )
    assert причина == "too_small_file"
    assert строка is None
    assert источник.скачано == 0, "качать заведомо негодное — лишний запрос к хранилищу"


def test_share_alike_без_флага_отсекается(tmp_path: Path) -> None:
    причина, _ = process_candidate(
        _кандидат(license_raw="CC BY-SA 4.0"),
        _зависимости(tmp_path, ПодставнойИсточник([])),
        DedupIndex(),
    )
    assert причина == "license_tier"


def test_кадр_без_машины_отбраковывается(tmp_path: Path) -> None:
    зависимости = _зависимости(
        tmp_path, ПодставнойИсточник([]), vehicle_detector=ФиксированныйДетектор([])
    )
    причина, _ = process_candidate(_кандидат(), зависимости, DedupIndex())
    assert причина == "no_vehicle"


def test_непробиваемый_номер_отбраковывает_кадр(tmp_path: Path) -> None:
    номер = Detection(x=200.0, y=400.0, w=120.0, h=40.0, conf=0.9, cls=0)
    зависимости = _зависимости(
        tmp_path, ПодставнойИсточник([]), plate_detectors=(ФиксированныйДетектор([номер]),)
    )
    причина, _ = process_candidate(_кандидат(), зависимости, DedupIndex())
    assert причина == "redaction_failed"


def test_фоновая_машина_в_кропе_очищается(tmp_path: Path) -> None:
    """Кроп вырезается из уже очищенного кадра — номер второй машины не проскакивает."""
    главная = Detection(40.0, 40.0, 520.0, 400.0, 0.96, 2)
    фоновая = Detection(580.0, 60.0, 200.0, 160.0, 0.7, 2)
    номер_фоновой = Detection(620.0, 200.0, 70.0, 20.0, 0.9, 0)

    deps = _deps(
        tmp_path,
        vehicle_detector=ФиксированныйДетектор([главная, фоновая]),
        plate_detectors=(СписокДетектор([[номер_фоновой], [], [], [], [], []]),),
        face_detector=СписокДетектор([[]]),
    )
    причина, строка = process_candidate(_кандидат("lada__granta"), deps, DedupIndex())
    assert причина == "accepted"
    assert строка is not None and строка.plates == 1


def test_мусорный_кадр_режется_по_негативам(tmp_path: Path) -> None:
    зависимости = _зависимости(
        tmp_path,
        ПодставнойИсточник([]),
        encoder=ПостоянныйЭнкодер(np.array([0.0, 1.0], dtype=np.float32)),
        negative_matrix=np.array([[0.0, 1.0]], dtype=np.float32),
        class_matrix=np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )
    причина, _ = process_candidate(_кандидат(), зависимости, DedupIndex())
    assert причина == "not_a_car_photo"


def test_повтор_отсекается_дедупом(tmp_path: Path) -> None:
    зависимости = _зависимости(tmp_path, ПодставнойИсточник([]))
    индекс = DedupIndex()
    process_candidate(_кандидат(), зависимости, индекс)
    причина, _ = process_candidate(_кандидат(source_item_id="99"), зависимости, индекс)
    assert причина == "duplicate"


def test_прогон_класса_останавливается_на_бюджете(tmp_path: Path) -> None:
    """Стоп по бюджету просмотренных: источник бесконечен, прогон — нет."""
    страницы = [
        SearchPage(
            tuple(_кандидат(source_item_id=f"{p}_{n}", width=400) for n in range(50)),
            "t",
        )
        for p in range(10)
    ]
    источник = ПодставнойИсточник(страницы)
    состояние = HarvestState({}, {}, {}, set())
    итог = harvest_class(
        ClassQueries("lada__granta", ("Lada Granta",), ()),
        _зависимости(tmp_path, источник),
        состояние,
        DedupIndex(),
    )
    assert итог.seen == SEEN_BUDGET
    assert итог.accepted == 0
    assert итог.rejects["too_small_file"] == SEEN_BUDGET


def test_прогон_класса_останавливается_на_цели(tmp_path: Path) -> None:
    class РазныеИсточник(ПодставнойИсточник):
        def fetch(self, candidate: Candidate) -> bytes:
            self.скачано += 1
            return _jpeg(seed=int(candidate.source_item_id) + 1)

    кандидаты = tuple(_кандидат(source_item_id=str(n)) for n in range(50))
    источник = РазныеИсточник([SearchPage(кандидаты, None)])
    состояние = HarvestState({}, {}, {}, set())
    зависимости = _зависимости(tmp_path, источник)
    итог = harvest_class(
        ClassQueries("lada__granta", ("Lada Granta",), ()), зависимости, состояние, DedupIndex()
    )
    assert итог.accepted == ACCEPTED_TARGET
    assert состояние.accepted["lada__granta"] == ACCEPTED_TARGET
    assert len(load_manifest(зависимости.manifest_path)) == ACCEPTED_TARGET


def test_обработанный_элемент_не_берётся_дважды(tmp_path: Path) -> None:
    источник = ПодставнойИсточник([SearchPage((_кандидат(),), None)])
    состояние = HarvestState({}, {}, {}, {"commons:12345"})
    итог = harvest_class(
        ClassQueries("lada__granta", ("Lada Granta",), ()),
        _зависимости(tmp_path, источник),
        состояние,
        DedupIndex(),
    )
    assert итог.seen == 0
    assert источник.скачано == 0


def test_класс_вне_каталога_это_ошибка(tmp_path: Path) -> None:
    """Порядок классов задаёт строки матрицы CLIP: чужой class_id — не «пропустить»."""
    with pytest.raises(ValueError, match="нет_такого"):
        process_candidate(
            _кандидат(class_id="нет_такого"),
            _зависимости(tmp_path, ПодставнойИсточник([])),
            DedupIndex(),
        )


def test_принятая_строка_несёт_бокс_и_уверенность(tmp_path: Path) -> None:
    источник = ПодставнойИсточник([])
    причина, строка = process_candidate(
        _кандидат(), _зависимости(tmp_path, источник), DedupIndex()
    )
    assert причина == "accepted"
    assert строка is not None
    assert строка.det_conf == pytest.approx(0.9)
    assert (строка.bbox_x, строка.bbox_y) == (100.0, 100.0)
    assert (строка.bbox_w, строка.bbox_h) == (700.0, 500.0)
    assert 0.0 < строка.bbox_area_frac <= 1.0

