"""Стейдж галереи сплошного аудита приватности пула.

Спека: `docs/superpowers/specs/2026-09-18-pool-privacy-audit-design.md`. На каждый кадр
готовится миниатюра отредактированного кропа для сетки и контекст-кроп из оригинала.
Отредактированный кроп в родном разрешении не копируется: зум ссылается на сам файл
`crops/<class_id>/<photo_id>.jpg`, то есть человек судит ровно то, что уйдёт в обучение.

Отдельно от `audit_gallery.py`: та обслуживает H3-аудит с другой таксономией, её зум
открывает ту же 480px-миниатюру, а вердикты живут только в памяти вкладки. Здесь вердикты
сразу уходят в `localStorage` под ключом с sha256 сплита и режимом: у всех
`file://`-страниц в Chrome один origin, и без этого пилот и основной проход перетёрли бы
друг друга. Каталог вывода обязан не существовать: стейдж не дописывает поверх старого.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from pipeline.harvest.cache import cache_path
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.pool_audit import (
    FACE_KEYS,
    PLATE_KEYS,
    REGISTRY_COLUMNS,
    assign_sessions,
    file_sha256,
)
from pipeline.schema import read_split
from scripts.stage_plate_leak_views import _resolve_crop

MODES = ("pilot", "main")
LAYOUTS = ("grid", "single")
DEFAULT_THUMB_PX = 480
CONTEXT_LONG_SIDE = 1600
# Поле вокруг бокса машины: номер у самого края бокса без запаса на контексте не найти.
CONTEXT_MARGIN = 0.1


@dataclass(frozen=True, slots=True)
class StagedFrame:
    """Кадр, подготовленный к просмотру. Пути — относительно каталога страниц."""

    photo_id: str
    session: int
    class_id: str
    crop_sha256: str
    thumb: str
    context: str
    zoom: str
    problem: str


STAGE_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(StagedFrame))


def _relative(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def context_box(
    bbox: Sequence[float], size: tuple[int, int], margin: float = CONTEXT_MARGIN
) -> tuple[int, int, int, int]:
    """Бокс сплита `[x, y, w, h]` → `(left, top, right, bottom)` с полем, внутри кадра."""
    x, y, w, h = (float(value) for value in bbox)
    width, height = size
    return (
        max(0, round(x - margin * w)),
        max(0, round(y - margin * h)),
        min(width, round(x + w + margin * w)),
        min(height, round(y + h + margin * h)),
    )


def stage_frame(
    *,
    photo_id: str,
    source: str,
    crop_path: str,
    class_id: str,
    bbox: Sequence[float],
    session: int,
    out: Path,
    crops_root: Path,
    raw_root: Path,
    thumb_px: int = DEFAULT_THUMB_PX,
) -> StagedFrame:
    """Подготовить один кадр. Пропавший кроп или оригинал — `problem`, не исключение.

    Кадр без кропа остаётся на странице плиткой и вердикт не принимает, то есть остаётся
    `pending` и блокирует приёмку — молча пропасть из аудита он не может.
    """
    crop = _resolve_crop(crops_root, crop_path)
    if not crop.is_file():
        return StagedFrame(photo_id, session, class_id, "", "", "", "", "crop_missing")

    thumb = out / "thumbs" / f"{photo_id}.jpg"
    with Image.open(crop) as image:
        preview = image.convert("RGB")
    preview.thumbnail((thumb_px, thumb_px))
    preview.save(thumb, quality=85)

    context = ""
    problem = ""
    original = cache_path(raw_root, source, photo_id, "jpg")
    if original.is_file():
        with Image.open(original) as image:
            # Кроп резался из выпрямленного по EXIF кадра (`harvest/cache.decode_image`).
            upright = (ImageOps.exif_transpose(image) or image).convert("RGB")
        region = upright.crop(context_box(bbox, upright.size))
        region.thumbnail((CONTEXT_LONG_SIDE, CONTEXT_LONG_SIDE))
        context_file = out / "context" / f"{photo_id}.jpg"
        region.save(context_file, quality=90)
        context = _relative(context_file, out)
    else:
        problem = "original_missing"

    return StagedFrame(
        photo_id=photo_id,
        session=session,
        class_id=class_id,
        crop_sha256=file_sha256(crop),
        thumb=_relative(thumb, out),
        context=context,
        zoom=_relative(crop, out),
        problem=problem,
    )


def _frame_config(frame: StagedFrame, layout: str) -> dict[str, Any]:
    # В одиночной раскладке сама лента показывает кроп в родном разрешении.
    image = frame.zoom if layout == "single" else frame.thumb
    return {
        "photo_id": frame.photo_id,
        "session": frame.session,
        "class_id": frame.class_id,
        "crop_sha256": frame.crop_sha256,
        "image": image,
        "context": frame.context,
        "zoom": frame.zoom,
        "problem": frame.problem,
    }


_PAGE = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
#bar{position:sticky;top:0;z-index:1;background:#000;padding:8px;display:flex;gap:16px;
 flex-wrap:wrap;align-items:center}
#warn{display:none;background:#b00;color:#fff;padding:8px}
.grid #frames{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding:4px}
.single #frames{display:grid;grid-template-columns:1fr;gap:12px;padding:4px}
figure{margin:0;background:#1c1c1c;border:3px solid #444}
figure.done{border-color:#2a2}
figure.leak{border-color:#e33}
#frames img{width:100%;display:block;cursor:zoom-in}
.single #frames img{max-height:92vh;object-fit:contain}
.missing{padding:40px;background:#400}
figcaption{padding:4px;font-size:12px}
#zoom{position:fixed;inset:0;z-index:2;background:#000e;display:none;overflow:auto;
 text-align:center}
#zoom img{max-width:98vw;max-height:94vh;cursor:zoom-in}
#zoom.full img{max-width:none;max-height:none;cursor:zoom-out}
#zoom .label{display:none;background:#b00;color:#fff;padding:4px 12px}
#zoom.original .label{display:block}
</style>
<body class="__LAYOUT__">
<div id="bar"><b>__TITLE__</b><span id="progress"></span><span id="legend"></span>
<button onclick="exportCsv()">Выгрузить реестр сеанса</button></div>
<div id="warn">localStorage недоступен: вердикты не сохранятся. Не начинайте сеанс.</div>
<div id="frames"></div>
<div id="zoom" onclick="closeZoom()">
<div class="label">ОРИГИНАЛ — только контекст, не основание вердикта</div>
<img id="zoom-img" alt=""></div>
<script type="application/json" id="config">__CONFIG__</script>
<script>
const CONFIG = JSON.parse(document.getElementById("config").textContent);
const FRAMES = CONFIG.frames;
let current = null;

function storageWorks() {
  try {
    localStorage.setItem(CONFIG.prefix + "__probe", "1");
    localStorage.removeItem(CONFIG.prefix + "__probe");
    return true;
  } catch (error) {
    return false;
  }
}

function load(frame) {
  try {
    return JSON.parse(localStorage.getItem(CONFIG.prefix + frame.photo_id) || "{}");
  } catch (error) {
    return {};
  }
}

function save(frame, verdict) {
  localStorage.setItem(CONFIG.prefix + frame.photo_id, JSON.stringify(verdict));
}

function paint(frame) {
  const verdict = load(frame);
  const plate = verdict.plate || "pending";
  const face = verdict.face || "pending";
  const figure = document.getElementById("f-" + frame.photo_id);
  figure.querySelector(".verdict").textContent = "номер: " + plate + " · лицо: " + face;
  figure.classList.toggle("done", plate !== "pending" && face !== "pending");
  const leak = plate === "leak" || plate === "leak_hist" || face === "leak";
  figure.classList.toggle("leak", leak);
}

function progress() {
  const done = FRAMES.filter(frame => {
    const verdict = load(frame);
    return verdict.plate && verdict.face;
  }).length;
  document.getElementById("progress").textContent = "готово " + done + " / " + FRAMES.length;
}

function openZoom(src, original) {
  const zoom = document.getElementById("zoom");
  zoom.classList.remove("full");
  zoom.classList.toggle("original", original);
  document.getElementById("zoom-img").src = src;
  zoom.style.display = "block";
}

function closeZoom() {
  document.getElementById("zoom").style.display = "none";
}

function exportCsv() {
  const lines = [CONFIG.columns.join(",")];
  for (const frame of FRAMES) {
    const verdict = load(frame);
    const plate = verdict.plate || "pending";
    const face = verdict.face || "pending";
    const sha = verdict.sha || frame.crop_sha256;
    lines.push([frame.photo_id, frame.session, plate, face, sha].join(","));
  }
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = CONFIG.export_name;
  link.click();
}

function render() {
  const container = document.getElementById("frames");
  FRAMES.forEach((frame, index) => {
    const figure = document.createElement("figure");
    figure.id = "f-" + frame.photo_id;
    figure.dataset.index = String(index);
    if (frame.image) {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = frame.image;
      figure.append(image);
    } else {
      const missing = document.createElement("div");
      missing.className = "missing";
      missing.textContent = "нет кропа — кадр остаётся pending";
      figure.append(missing);
    }
    const caption = document.createElement("figcaption");
    const verdict = document.createElement("div");
    verdict.className = "verdict";
    const about = document.createElement("div");
    const problem = frame.problem ? " · " + frame.problem : "";
    about.textContent = frame.class_id + " · " + frame.photo_id.slice(0, 12) + problem;
    caption.append(verdict, about);
    figure.append(caption);
    container.append(figure);
    paint(frame);
  });
  const plates = Object.entries(CONFIG.plate_keys).map(([key, value]) => key + "=" + value);
  const faces = Object.entries(CONFIG.face_keys).map(([key, value]) => key + "=лицо " + value);
  const hint = "клик — кроп, ещё клик — 1:1, shift+клик — оригинал, Esc — закрыть";
  document.getElementById("legend").textContent = [...plates, ...faces, hint].join("  ");
  progress();
}

document.addEventListener("mouseover", event => {
  const figure = event.target.closest("#frames figure");
  if (figure) current = FRAMES[Number(figure.dataset.index)];
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeZoom();
    return;
  }
  if (event.ctrlKey || event.altKey || event.metaKey) return;
  if (!current || current.problem === "crop_missing") return;
  let verdict = load(current);
  // Кроп сменился после прошлого вердикта: старые оси судили другие пиксели.
  if (verdict.sha && verdict.sha !== current.crop_sha256) verdict = {};
  if (event.key in CONFIG.plate_keys) verdict.plate = CONFIG.plate_keys[event.key];
  else if (event.key in CONFIG.face_keys) verdict.face = CONFIG.face_keys[event.key];
  else return;
  verdict.sha = current.crop_sha256;
  save(current, verdict);
  paint(current);
  progress();
});

document.addEventListener("click", event => {
  const figure = event.target.closest("#frames figure");
  if (!figure) return;
  const frame = FRAMES[Number(figure.dataset.index)];
  if (event.shiftKey && frame.context) openZoom(frame.context, true);
  else if (frame.zoom) openZoom(frame.zoom, false);
});

document.getElementById("zoom-img").addEventListener("click", event => {
  event.stopPropagation();
  document.getElementById("zoom").classList.toggle("full");
});

if (!storageWorks()) document.getElementById("warn").style.display = "block";
render();
</script>
"""


def render_page(
    frames: Sequence[StagedFrame],
    *,
    title: str,
    storage_prefix: str,
    layout: str,
    export_name: str,
) -> str:
    """HTML-страница сеанса. Данные и клавиши едут JSON-блоком, а не литералами JS."""
    if layout not in LAYOUTS:
        raise ValueError(f"неизвестная раскладка: {layout}")
    config = {
        "prefix": storage_prefix,
        "plate_keys": dict(PLATE_KEYS),
        "face_keys": dict(FACE_KEYS),
        "columns": list(REGISTRY_COLUMNS),
        "export_name": export_name,
        "frames": [_frame_config(frame, layout) for frame in frames],
    }
    # `</` внутри JSON закрыл бы тег `<script>`; `<\/` — та же строка для `JSON.parse`.
    payload = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    return (
        _PAGE.replace("__TITLE__", html.escape(title))
        .replace("__LAYOUT__", layout)
        .replace("__CONFIG__", payload)
    )


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def select_frames(pool: pd.DataFrame, only_ids: Path | None) -> pd.DataFrame:
    """Весь пул или его подмножество по колонке `sha256` (пилот — 300-выборка)."""
    if only_ids is None:
        return pool
    wanted = set(pd.read_csv(only_ids, dtype={"sha256": str})["sha256"])
    unknown = sorted(wanted - set(pool["photo_id"]))
    if unknown:
        raise ValueError(f"{len(unknown)} photo_id не из пула, например {unknown[0]}")
    return pool.loc[pool["photo_id"].isin(wanted)].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Стейдж галереи аудита приватности пула")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--layout", choices=LAYOUTS, default="grid")
    parser.add_argument("--split", type=Path, default=None, help="умолчание: pool.parquet")
    parser.add_argument("--only-ids", type=Path, default=None, help="CSV с колонкой sha256")
    parser.add_argument("--out", type=Path, default=None, help="умолчание: pool-audit/<mode>")
    parser.add_argument("--crops-root", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None, help="корень кэша с raw/")
    parser.add_argument("--thumb-px", type=int, default=DEFAULT_THUMB_PX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    split: Path = arguments.split or splits_dir() / "pool.parquet"
    out: Path = arguments.out or data_root() / "pool-audit" / arguments.mode
    crops_root: Path = arguments.crops_root or harvest_dir() / "crops"
    raw_root: Path = arguments.raw_root or harvest_dir()
    if out.exists():
        raise FileExistsError(f"{out} уже существует: стейдж пишет только в новый каталог")

    pool = select_frames(read_split(split), arguments.only_ids)
    sessions = assign_sessions(pool["photo_id"])
    (out / "thumbs").mkdir(parents=True)
    (out / "context").mkdir()
    frames: list[StagedFrame] = []
    columns = zip(
        pool["photo_id"],
        pool["source"],
        pool["crop_path"],
        pool["model_id"],
        pool["bbox"],
        strict=True,
    )
    for photo_id, source, crop_path, class_id, bbox in columns:
        frames.append(
            stage_frame(
                photo_id=str(photo_id),
                source=str(source),
                crop_path=str(crop_path),
                class_id=str(class_id),
                bbox=[float(value) for value in bbox],
                session=sessions[str(photo_id)],
                out=out,
                crops_root=crops_root,
                raw_root=raw_root,
                thumb_px=arguments.thumb_px,
            )
        )
    frames.sort(key=lambda frame: (frame.session, frame.photo_id))

    _write_csv(out / "stage.csv", STAGE_COLUMNS, [astuple(frame) for frame in frames])
    problems = [frame for frame in frames if frame.problem]
    _write_csv(
        out / "problems.csv", ("photo_id", "problem"), [(f.photo_id, f.problem) for f in problems]
    )
    for frame in problems:
        print(f"  {frame.problem}: {frame.photo_id}", file=sys.stderr)

    prefix = f"poolaudit:{arguments.mode}:{file_sha256(split)[:16]}:"
    for session in sorted({frame.session for frame in frames}):
        name = f"session-{session:02d}"
        page = render_page(
            [frame for frame in frames if frame.session == session],
            title=f"Аудит пула · {arguments.mode} · {name}",
            storage_prefix=prefix,
            layout=arguments.layout,
            export_name=f"export-{arguments.mode}-{name}.csv",
        )
        (out / f"{name}.html").write_text(page, encoding="utf-8")
    print(f"кадров: {len(frames)}, сеансов: {max(sessions.values())}, проблем: {len(problems)}")
    print(f"каталог: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
