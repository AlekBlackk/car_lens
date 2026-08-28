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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.clean import phash_of
from pipeline.crop import Detection, Detector, crop_image
from pipeline.harvest.cache import crop_path, decode_image, digest, store
from pipeline.harvest.dedup import DedupIndex
from pipeline.harvest.filters import ImageEncoder, clip_verdict, vehicle_gate
from pipeline.harvest.licenses import canonical_name, is_allowed, tier_of
from pipeline.harvest.manifest import HarvestRow, append_row, strip_html
from pipeline.harvest.queries import ClassQueries
from pipeline.harvest.redact import redact, redaction_zone
from pipeline.harvest.sources import Candidate, HarvestSource, SourceQuery, basis_of
from pipeline.harvest.state import (
    HarvestState,
    is_class_done,
)

MIN_SIDE = 640
PAGE_LIMIT = 50


@dataclass(frozen=True, slots=True)
class HarvestDeps:
    """Всё внешнее, что нужно прогону. Ничего из этого модуль не создаёт сам."""

    source: HarvestSource
    vehicle_detector: Detector
    plate_detector: Detector
    face_detector: Detector
    encoder: ImageEncoder
    class_matrix: np.ndarray
    class_order: tuple[str, ...]
    negative_matrix: np.ndarray
    allow_sa: bool
    root: Path
    manifest_path: Path
    checked: str


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

    машина = vehicle_gate(кадр, deps.vehicle_detector)
    if машина.box is None:
        assert машина.reason is not None
        return машина.reason, None

    вердикт = clip_verdict(
        кадр, deps.encoder, deps.class_matrix, deps.negative_matrix, индекс_класса
    )
    if not вердикт.is_car_photo:
        return "not_a_car_photo", None

    кроп, номера, лица, замыто = _замытый_кроп(кадр, машина.box, deps)
    if not замыто:
        return "redaction_failed", None

    хеш = phash_of(кроп)
    решение = dedup.check(хеш, candidate.class_id)
    if решение.status != "new":
        return решение.status, None

    sha = digest(данные)
    путь = crop_path(deps.root, candidate.class_id, sha)
    путь.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(путь), кроп)
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
        audit="",
    )


def _замытый_кроп(
    кадр: np.ndarray, бокс: Detection, deps: HarvestDeps
) -> tuple[np.ndarray, int, int, bool]:
    """Замыть зону вокруг машины и вырезать из неё сохраняемый кроп.

    Зона редакции шире сохраняемого кропа (10% против 8%), поэтому кроп целиком лежит
    внутри уже отредактированных пикселей — номер, срезанный границей кропа, замывается
    вместе со всем остальным.
    """
    высота, ширина = кадр.shape[:2]
    x0, y0, x1, y1 = redaction_zone(бокс, ширина, высота)
    зона = кадр[y0:y1, x0:x1]
    итог = redact(зона, deps.plate_detector, deps.face_detector)
    смещённый = Detection(бокс.x - x0, бокс.y - y0, бокс.w, бокс.h, бокс.conf, бокс.cls)
    return crop_image(итог.image, смещённый), итог.plates, итог.faces, итог.ok


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
        SourceQuery(queries.class_id, None, термин, PAGE_LIMIT, None)
        for термин in queries.terms
    ]
    return из_категорий + из_текста
