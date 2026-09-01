# tests/test_harvest_run.py
import csv
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pipeline.crop import Detection
from pipeline.harvest.aliases import REJECT_COLUMNS
from pipeline.harvest.audit import load_audit
from pipeline.harvest.categories import ExpandResult
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.expand_log import load_expand_rows
from pipeline.harvest.manifest import load_manifest
from pipeline.harvest.queries import ClassQueries
from pipeline.harvest.rejects import load_rejects
from pipeline.harvest.run import (
    MAX_BLACK_PIXEL_FRAC,
    HarvestDeps,
    _запросы,
    harvest_class,
    is_crop_over_redacted,
    process_candidate,
)
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


def _обход_без_сети(
    class_id: str, категория: str, глубина: int, include: tuple[str, ...], exclude: tuple[str, ...]
) -> ExpandResult:
    """Заглушка обхода по умолчанию: категория сама себе дерево, ни одного запроса."""
    return ExpandResult((категория,), 0, 0, 0, False, 0)


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
        "alias_table": {"lada__granta": ("granta",), "uaz__patriot": ("patriot",)},
        "alias_rejects_path": tmp_path / "alias_rejects.csv",
        "reject_log_path": tmp_path / "reject_log.csv",
        "expand_categories": _обход_без_сети,
        "dedup_verdicts_path": tmp_path / "verdicts.dedup.csv",
        "expand_log_path": tmp_path / "expand_log.csv",
    }
    поля.update(правки)
    return HarvestDeps(**поля)  # type: ignore[arg-type]


_deps = _зависимости


def _хеш_кадра(tmp_path: Path) -> int:
    """Хеш кропа, который даст цепочка решений: берётся из прогона, а не из копии кода."""
    _, строка = process_candidate(_кандидат(), _зависимости(tmp_path), DedupIndex())
    assert строка is not None
    return строка.phash


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


def test_вытеснение_пишет_вердикт_и_принимает_кадр(tmp_path: Path) -> None:
    вердикты = tmp_path / "verdicts.dedup.csv"
    deps = _зависимости(tmp_path, dedup_verdicts_path=вердикты)
    дедуп = DedupIndex()
    дедуп.add(_хеш_кадра(tmp_path), "daewoo__lanos", "a.jpg", "search:lanos")

    причина, _ = process_candidate(_кандидат(basis="category:Chevrolet Lanos"), deps, дедуп)

    assert причина == "accepted"
    assert load_audit(вердикты) == {"a.jpg": "drop:superseded"}


def test_спор_двух_категорий_помечает_старую_строку(tmp_path: Path) -> None:
    вердикты = tmp_path / "verdicts.dedup.csv"
    deps = _зависимости(tmp_path, dedup_verdicts_path=вердикты)
    дедуп = DedupIndex()
    дедуп.add(_хеш_кадра(tmp_path), "daewoo__lanos", "a.jpg", "category:Daewoo Lanos")

    причина, строка = process_candidate(_кандидат(basis="category:Chevrolet Lanos"), deps, дедуп)

    assert (причина, строка) == ("conflict", None)
    assert load_audit(вердикты) == {"a.jpg": "conflict"}


def test_провал_записи_после_вытеснения_не_теряет_обе_строки(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если кроп не записался, вердикт `drop:superseded` не должен ставиться заранее:
    иначе `write_failed` терял бы и старую строку (снесена вердиктом), и новую
    (не появилась) вместо одной.
    """
    вердикты = tmp_path / "verdicts.dedup.csv"
    хеш = _хеш_кадра(tmp_path)
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, None))
    deps = _зависимости(tmp_path, dedup_verdicts_path=вердикты)
    дедуп = DedupIndex()
    дедуп.add(хеш, "daewoo__lanos", "a.jpg", "search:lanos")

    причина, строка = process_candidate(_кандидат(basis="category:Chevrolet Lanos"), deps, дедуп)

    assert (причина, строка) == ("write_failed", None)
    assert not вердикты.exists(), "старую строку нельзя приговаривать без новой взамен"


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
    причина, строка = process_candidate(_кандидат(), _зависимости(tmp_path, источник), DedupIndex())
    assert причина == "accepted"
    assert строка is not None
    assert строка.det_conf == pytest.approx(0.9)
    assert (строка.bbox_x, строка.bbox_y) == (100.0, 100.0)
    assert (строка.bbox_w, строка.bbox_h) == (700.0, 500.0)
    assert 0.0 < строка.bbox_area_frac <= 1.0


def test_is_crop_over_redacted_порог() -> None:
    черный = np.zeros((100, 100, 3), dtype=np.uint8)
    светлый = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert is_crop_over_redacted(черный)
    assert not is_crop_over_redacted(светлый)

    # 51% черного при пороге 50%
    смешанный_брак = np.full((100, 100, 3), 128, dtype=np.uint8)
    смешанный_брак[:51, :] = 0
    assert is_crop_over_redacted(смешанный_брак, max_black_frac=MAX_BLACK_PIXEL_FRAC)

    # 49% черного при пороге 50%
    смешанный_годный = np.full((100, 100, 3), 128, dtype=np.uint8)
    смешанный_годный[:49, :] = 0
    assert not is_crop_over_redacted(смешанный_годный, max_black_frac=MAX_BLACK_PIXEL_FRAC)


def test_залитый_чёрным_кроп_отбраковывается_на_приёме(tmp_path: Path) -> None:
    """Огромный бокс номера заливает всю машину черным -> отказ over_redacted."""
    гигантский_номер = Detection(x=100.0, y=100.0, w=700.0, h=500.0, conf=0.9, cls=0)
    зависимости = _зависимости(
        tmp_path,
        ПодставнойИсточник([]),
        plate_detectors=(СписокДетектор([[гигантский_номер], []]),),
    )
    причина, строка = process_candidate(_кандидат(), зависимости, DedupIndex())
    assert причина == "over_redacted"
    assert строка is None


def test_поисковый_кандидат_с_чужим_именем_бракуется_до_скачивания(tmp_path: Path) -> None:
    """21 строка `lexus__lx`, названная Land Cruiser, — ровно этот случай."""
    источник = ПодставнойИсточник([])
    кандидат = _кандидат(basis="search:Lada Granta", title="File:UAZ_Patriot_2019_front.jpg")
    причина, строка = process_candidate(кандидат, _зависимости(tmp_path, источник), DedupIndex())
    assert причина == "alias_mismatch"
    assert строка is None
    assert источник.скачано == 0, "проверка идёт по метаданным, трафик нулевой"


def test_имя_без_единого_совпадения_тоже_бракуется(tmp_path: Path) -> None:
    """`unknown` на поисковом пути так же жёсток, как `foreign` (§9 дизайна)."""
    кандидат = _кандидат(basis="search:Lada Granta", title="File:IMG_0412.jpg")
    причина, _ = process_candidate(
        кандидат, _зависимости(tmp_path, ПодставнойИсточник([])), DedupIndex()
    )
    assert причина == "alias_mismatch"


def test_кандидат_из_категории_с_мусорным_именем_проходит(tmp_path: Path) -> None:
    """Категория Commons размечена людьми — это принятое решение проекта."""
    кандидат = _кандидат(basis="category:Lada Granta", title="File:IMG_0412.jpg")
    причина, строка = process_candidate(
        кандидат, _зависимости(tmp_path, ПодставнойИсточник([])), DedupIndex()
    )
    assert причина == "accepted"
    assert строка is not None


def test_журнал_получает_строку_об_отказе(tmp_path: Path) -> None:
    """Без журнала ложный отказ невосстановим: `done_items` помнит кандидата вечно."""
    кандидат = _кандидат(basis="search:Lada Granta", title="File:UAZ_Patriot_2019_front.jpg")
    журнал = tmp_path / "alias_rejects.csv"
    process_candidate(
        кандидат,
        _зависимости(tmp_path, ПодставнойИсточник([]), alias_rejects_path=журнал),
        DedupIndex(),
    )
    записи = list(csv.DictReader(журнал.read_text(encoding="utf-8").splitlines()))
    assert len(записи) == 1
    assert tuple(записи[0]) == REJECT_COLUMNS
    assert записи[0]["verdict"] == "foreign"
    assert записи[0]["foreign"] == "uaz__patriot"
    assert записи[0]["normalized"] == "uaz patriot 2019 front"


def test_свой_поисковый_кандидат_проходит_и_журнал_остаётся_пустым(tmp_path: Path) -> None:
    """Ложный отказ дороже пропуска: `done_items` жжёт выдачу класса безвозвратно."""
    журнал = tmp_path / "alias_rejects.csv"
    кандидат = _кандидат(basis="search:Lada Granta", title="File:Lada_Granta_2019_front.jpg")
    причина, строка = process_candidate(
        кандидат,
        _зависимости(tmp_path, ПодставнойИсточник([]), alias_rejects_path=журнал),
        DedupIndex(),
    )
    assert причина == "accepted"
    assert строка is not None
    assert not журнал.exists(), "своё имя не оставляет следа в журнале отказов"


def test_неудача_кодирования_кропа_не_оставляет_битого_файла(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Иначе на диск лёг бы кроп в нуль байт, а строка манифеста ушла бы как `accepted`."""
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, None))
    причина, строка = process_candidate(
        _кандидат(), _зависимости(tmp_path, ПодставнойИсточник([])), DedupIndex()
    )
    assert причина == "write_failed"
    assert строка is None
    assert not list((tmp_path / "crops").rglob("*.jpg")), "битого кропа на диске нет"


def test_обход_подставляет_подкатегории_перед_терминами(tmp_path: Path) -> None:
    вызовы: list[tuple[str, str, int, tuple[str, ...], tuple[str, ...]]] = []

    def обход(
        class_id: str,
        категория: str,
        глубина: int,
        include: tuple[str, ...],
        exclude: tuple[str, ...],
    ) -> ExpandResult:
        вызовы.append((class_id, категория, глубина, include, exclude))
        return ExpandResult((категория, f"{категория} (DC)"), 1, 0, 0, False, 0)

    # Глубина, include и exclude заданы различимыми: обе последние — `tuple[str, ...]`, и
    # перестановка их местами прошла бы проверку типов молча.
    queries = ClassQueries(
        "kia__rio",
        ("Kia Rio",),
        ("Kia Rio",),
        depth=2,
        include=("Kia Rio (UB)",),
        exclude=("Kia Rio (JB)",),
    )
    состояние = HarvestState({}, {}, {}, set())
    deps = _зависимости(tmp_path, expand_categories=обход)

    список = _запросы(queries, deps, состояние)

    assert [(з.category, з.text) for з in список] == [
        ("Kia Rio", None),
        ("Kia Rio (DC)", None),
        (None, "Kia Rio"),
    ]
    assert вызовы == [("kia__rio", "Kia Rio", 2, ("Kia Rio (UB)",), ("Kia Rio (JB)",))]
    assert состояние.categories["kia__rio|Kia Rio"] == ("Kia Rio", "Kia Rio (DC)")
    журнал = load_expand_rows(tmp_path / "expand_log.csv")
    assert [(с.class_id, с.category, с.categories) for с in журнал] == [("kia__rio", "Kia Rio", 2)]


def test_дерево_из_состояния_не_перезапрашивается(tmp_path: Path) -> None:
    def обход(
        class_id: str,
        категория: str,
        глубина: int,
        include: tuple[str, ...],
        exclude: tuple[str, ...],
    ) -> ExpandResult:
        raise AssertionError("обход не должен вызываться при кэше в состоянии")

    queries = ClassQueries("kia__rio", ("Kia Rio",), ())
    состояние = HarvestState({}, {}, {}, set())
    состояние.categories["kia__rio|Kia Rio"] = ("Kia Rio", "Kia Rio (QB)")
    deps = _зависимости(tmp_path, expand_categories=обход)

    список = _запросы(queries, deps, состояние)

    assert [з.category for з in список] == ["Kia Rio", "Kia Rio (QB)"]
    assert load_expand_rows(tmp_path / "expand_log.csv") == []


def test_отказ_попадает_в_журнал(tmp_path: Path) -> None:
    журнал = tmp_path / "reject_log.csv"
    источник = ПодставнойИсточник([SearchPage((_кандидат(width=400),), None)])
    deps = _зависимости(tmp_path, источник, reject_log_path=журнал)
    состояние = HarvestState({}, {}, {}, set())

    harvest_class(ClassQueries("lada__granta", (), ("Lada Granta",)), deps, состояние, DedupIndex())

    assert [с.reason for с in load_rejects(журнал)] == ["too_small_file"]
    # Basis в строке — кандидата, а не запроса: запрос здесь текстовый (`search:Lada Granta`),
    # и подмена одного другим прошла бы проверку одной причины молча.
    отказ = load_rejects(журнал)[0]
    assert (отказ.class_id, отказ.basis, отказ.checked) == (
        "lada__granta",
        "category:Lada Granta",
        "2026-08-28",
    )
