# pipeline/harvest/run.py
"""Оркестратор: цепочка решений и стоп-критерий.

Порядок шагов не произволен. Дешёвые проверки идут раньше сетевых, сетевые — раньше
модельных: лицензия и размеры известны из метаданных выдачи, и кандидат, негодный по
ним, не стоит ни одного запроса к хранилищу.

Каждый отказ возвращается причиной, а не молча: журнал причин — это материал отчёта, по
которому потом видно, что именно съело выборку.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.clean import phash_of
from pipeline.crop import VEHICLE_CLASSES, Detection, Detector, crop_image, select_primary
from pipeline.harvest.aliases import (
    AliasReject,
    AliasTable,
    alias_verdict,
    append_reject,
    normalize_filename,
)
from pipeline.harvest.cache import crop_path, decode_image, digest, store
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.filters import ImageEncoder, clip_verdict
from pipeline.harvest.licenses import canonical_name, is_allowed, tier_of
from pipeline.harvest.manifest import HarvestRow, append_row, strip_html
from pipeline.harvest.queries import ClassQueries
from pipeline.harvest.redact import redact_frame
from pipeline.harvest.sources import (
    SEARCH_BASIS_PREFIX,
    Candidate,
    HarvestSource,
    SourceQuery,
    basis_of,
)
from pipeline.harvest.state import (
    HarvestState,
    is_class_done,
)

MIN_SIDE = 640
PAGE_LIMIT = 50
MAX_BLACK_PIXEL_FRAC = 0.50
BLACK_PIXEL_THRESHOLD = 8


def is_crop_over_redacted(
    crop: np.ndarray,
    max_black_frac: float = MAX_BLACK_PIXEL_FRAC,
    black_threshold: int = BLACK_PIXEL_THRESHOLD,
) -> bool:
    """Проверить, не съела ли заливка редакции машину целиком.

    Если доля почти чёрных пикселей (яркость < black_threshold) превышает max_black_frac,
    кадр бракуется с причиной over_redacted.
    """
    if crop.size == 0:
        return True
    серый = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    доля = float((серый < black_threshold).mean())
    return доля > max_black_frac


@dataclass(frozen=True, slots=True)
class HarvestDeps:
    """Всё внешнее, что нужно прогону. Ничего из этого модуль не создаёт сам."""

    source: HarvestSource
    vehicle_detector: Detector
    plate_detectors: tuple[Detector, ...]
    face_detector: Detector
    encoder: ImageEncoder
    class_matrix: np.ndarray
    class_order: tuple[str, ...]
    negative_matrix: np.ndarray
    allow_sa: bool
    root: Path
    manifest_path: Path
    checked: str
    alias_table: AliasTable
    alias_rejects_path: Path


@dataclass(frozen=True, slots=True)
class ClassOutcome:
    """Итог одного класса: сколько просмотрено, сколько принято и что отсеяло."""

    class_id: str
    seen: int
    accepted: int
    rejects: dict[str, int]


def process_candidate(
    candidate: Candidate, deps: HarvestDeps, dedup: DedupIndex
) -> tuple[str, HarvestRow | None]:
    """Провести кандидата по цепочке решений. Причина `accepted` означает приём."""
    # `class_order` повторяет порядок `catalog.classes` (тот же, что даёт
    # `catalog.class_index`), поэтому позиция в кортеже — это номер строки матрицы
    # текстовых векторов. Класс не из каталога роняет прогон `ValueError`, а не
    # пропускается: молчаливый пропуск означал бы ранг, посчитанный не по тому классу.
    if candidate.class_id not in deps.class_order:
        raise ValueError(f"Класс {candidate.class_id} отсутствует в catalog/deps.class_order")
    индекс_класса = deps.class_order.index(candidate.class_id)

    if not is_allowed(candidate.license_raw, allow_sa=deps.allow_sa):
        return "license_tier", None
    if min(candidate.width, candidate.height) < MIN_SIDE:
        return "too_small_file", None

    # Шлюз принадлежности классу. Только поисковый путь: категория Commons размечена
    # людьми и проходит даже с именем IMG_0412.jpg. Место в цепочке не случайно —
    # проверка идёт по метаданным выдачи, до единого байта трафика, как и лицензия выше.
    if candidate.basis.startswith(SEARCH_BASIS_PREFIX):
        имя = normalize_filename(candidate.title or candidate.page_url)
        вердикт_иглы = alias_verdict(имя, candidate.class_id, deps.alias_table)
        if вердикт_иглы.kind != "own":
            append_reject(
                deps.alias_rejects_path,
                AliasReject(
                    class_id=candidate.class_id,
                    basis=candidate.basis,
                    source=candidate.source,
                    source_item_id=candidate.source_item_id,
                    title=candidate.title,
                    normalized=имя,
                    verdict=вердикт_иглы.kind,
                    foreign=";".join(вердикт_иглы.foreign),
                ),
            )
            return "alias_mismatch", None

    try:
        данные = deps.source.fetch(candidate)
    except Exception:
        return "http", None
    if not данные:
        return "http", None
    try:
        кадр = decode_image(данные)
    except Exception:
        return "http", None
    store(deps.root, candidate.source, данные, "jpg")

    все_боксы = deps.vehicle_detector.detect(кадр)
    высота, ширина = кадр.shape[:2]
    решение = select_primary(все_боксы, ширина, высота)
    if решение.reject_reason is not None:
        return решение.reject_reason, None
    if решение.is_multi_car:
        return "multi_car", None
    assert решение.box is not None

    вердикт = clip_verdict(
        кадр, deps.encoder, deps.class_matrix, deps.negative_matrix, индекс_класса
    )
    if not вердикт.is_car_photo:
        return "not_a_car_photo", None

    машины_в_кадре = [d for d in все_боксы if d.cls in VEHICLE_CLASSES and d.conf > 0.5]
    кроп, номера, лица, замыто = redact_and_crop(кадр, решение.box, машины_в_кадре, deps)
    if not замыто:
        return "redaction_failed", None
    if is_crop_over_redacted(кроп):
        return "over_redacted", None

    хеш = phash_of(кроп)

    решение_дедуп = dedup.check(хеш, candidate.class_id)
    if решение_дедуп.status != "new":
        return решение_дедуп.status, None

    sha = digest(данные)
    путь = crop_path(deps.root, candidate.class_id, sha)
    путь.parent.mkdir(parents=True, exist_ok=True)
    # `cv2.imwrite` отдаёт False на путях с не-ASCII символами: имя уходит в системную
    # кодировку, а под Windows это не UTF-8. Фолбэк кодирует в память и пишет байты сам.
    # Флаг `imencode` проверяется: без проверки на диск лёг бы файл в нуль байт, а строка
    # манифеста ушла бы как `accepted` — провенанс указывал бы на битый кроп.
    if not cv2.imwrite(str(путь), кроп):
        удалось, буфер = cv2.imencode(".jpg", кроп)
        if not удалось:
            return "write_failed", None
        путь.write_bytes(буфер.tobytes())
    dedup.add(хеш, candidate.class_id, путь.name)

    тир = tier_of(candidate.license_raw)
    assert тир is not None  # проверено `is_allowed` выше
    return "accepted", HarvestRow(
        file=путь.name,
        class_id=candidate.class_id,
        source=candidate.source,
        source_item_id=candidate.source_item_id,
        page_url=candidate.page_url,
        file_url=candidate.file_url,
        author=strip_html(candidate.author),
        license=canonical_name(candidate.license_raw),
        license_tier=int(тир),
        checked=deps.checked,
        basis=candidate.basis,
        sha256=sha,
        phash=хеш,
        clip_rank=вердикт.rank,
        clip_margin=round(вердикт.margin, 4),
        plates=номера,
        faces=лица,
        bbox_x=float(решение.box.x),
        bbox_y=float(решение.box.y),
        bbox_w=float(решение.box.w),
        bbox_h=float(решение.box.h),
        bbox_area_frac=float(решение.area_frac),
        det_conf=float(решение.box.conf),
        audit="",
    )


def redact_and_crop(
    frame: np.ndarray,
    primary: Detection,
    vehicle_boxes: Sequence[Detection],
    deps: HarvestDeps,
) -> tuple[np.ndarray, int, int, bool]:
    """Очистить весь кадр, затем вырезать кроп главной машины из очищенного.

    Кроп — подмножество очищенного кадра, поэтому номер фоновой машины, попавший в кроп
    краем, уже залит. Зона поиска номеров шире каждого бокса ТС на 15% (§6, `redact.py`).
    """
    итог = redact_frame(frame, vehicle_boxes, deps.plate_detectors, deps.face_detector)
    if not итог.ok:
        return frame, итог.plates, итог.faces, False
    return crop_image(итог.image, primary), итог.plates, итог.faces, True


def harvest_class(
    queries: ClassQueries, deps: HarvestDeps, state: HarvestState, dedup: DedupIndex
) -> ClassOutcome:
    """Собрать один класс до стоп-критерия: 200 просмотренных или 15 принятых."""
    class_id = queries.class_id
    отказы: Counter[str] = Counter()
    начальный_просмотр = state.seen.get(class_id, 0)
    начальный_приём = state.accepted.get(class_id, 0)

    for запрос in _запросы(queries):
        while not is_class_done(state, class_id):
            ключ = f"{class_id}|{basis_of(запрос)}"
            try:
                страница = deps.source.search(
                    SourceQuery(
                        class_id,
                        запрос.category,
                        запрос.text,
                        PAGE_LIMIT,
                        state.tokens.get(ключ),
                    )
                )
            except Exception:
                break
            for кандидат in страница.candidates:
                if is_class_done(state, class_id):
                    break
                метка = f"{кандидат.source}:{кандидат.source_item_id}"
                if метка in state.done_items:
                    continue
                state.done_items.add(метка)
                state.seen[class_id] = state.seen.get(class_id, 0) + 1
                причина, строка = process_candidate(кандидат, deps, dedup)
                if строка is None:
                    отказы[причина] += 1
                    continue
                append_row(deps.manifest_path, строка)
                state.accepted[class_id] = state.accepted.get(class_id, 0) + 1
            if страница.continue_token is None:
                state.tokens.pop(ключ, None)
                break
            state.tokens[ключ] = страница.continue_token

    return ClassOutcome(
        class_id=class_id,
        seen=state.seen.get(class_id, 0) - начальный_просмотр,
        accepted=state.accepted.get(class_id, 0) - начальный_приём,
        rejects=dict(отказы),
    )


def _запросы(queries: ClassQueries) -> list[SourceQuery]:
    """Категории первыми, текстовый поиск добором: категория размечена людьми."""
    из_категорий = [
        SourceQuery(queries.class_id, категория, None, PAGE_LIMIT, None)
        for категория in queries.categories
    ]
    из_текста = [
        SourceQuery(queries.class_id, None, термин, PAGE_LIMIT, None) for термин in queries.terms
    ]
    return из_категорий + из_текста
