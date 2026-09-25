"""Слепая HTML-галерея для разметки `holdout_ru` по релевантности (спека holdout-relevance §5).

Страница не знает ответов модели: в неё идут кадр, класс и уже поставленные вердикты.
Клавиши ловятся по `event.code`, чтобы работать в русской раскладке: r — relevant,
n — off_street, w — wrong_label, u — unsure, t — флаг «сверху». Прогресс живёт в
`localStorage`, но надёжный результат — только выгруженный CSV: хранилище браузера может
быть недоступно или очищено.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.holdout_relevance import CSV_HEADER, load_relevance
from pipeline.paths import data_root, harvest_dir, splits_dir
from pipeline.schema import read_split

_КЛЮЧ_ХРАНИЛИЩА = "holdout-relevance-v1"

_HEAD = """<!doctype html><meta charset=utf-8><title>Разметка holdout_ru</title><style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
#bar{position:sticky;top:0;z-index:2;background:#000;padding:8px;display:flex;gap:12px;
align-items:center;flex-wrap:wrap}
#warn{color:#fb0}
h2{font-size:16px;margin:16px 8px 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:4px;padding:4px}
figure{margin:0;background:#1c1c1c;border:4px solid #1c1c1c;position:relative}
figure[data-v=""]{border:4px dashed #777}
figure[data-v=relevant]{border-color:#2a2}figure[data-v=off_street]{border-color:#48f}
figure[data-v=wrong_label]{border-color:#e33}figure[data-v=unsure]{border-color:#fb0}
figure[data-t=true]::after{content:"сверху";position:absolute;top:6px;right:6px;
background:#fb0;color:#000;padding:2px 6px;font-weight:bold}
img{width:100%;display:block;cursor:zoom-in}
figcaption{padding:4px;font-size:12px}
#zoom{position:fixed;inset:0;z-index:3;background:#000d;display:none;align-items:center;
justify-content:center}
#zoom img{max-width:96vw;max-height:96vh;width:auto;cursor:zoom-out}
</style>"""

_SCRIPT = """<script>
const K=%(key)s, HEADER=%(header)s, V=%(initial)s;
const CODES={KeyR:'relevant',KeyN:'off_street',KeyW:'wrong_label',KeyU:'unsure'};
function warn(t){document.getElementById('warn').textContent=t;}
try{Object.assign(V,JSON.parse(localStorage.getItem(K)||'{}'));}
catch(e){warn('хранилище браузера недоступно — выгружайте CSV почаще');}
function save(){try{localStorage.setItem(K,JSON.stringify(V));}
 catch(e){warn('хранилище браузера недоступно — выгружайте CSV почаще');}}
function paint(id){const el=document.querySelector('[data-id="'+CSS.escape(id)+'"]');
 const s=V[id]||{v:'',t:false};el.dataset.v=s.v;el.dataset.t=String(s.t);}
function count(){const n=Object.values(V).filter(s=>s.v).length;
 document.getElementById('count').textContent='размечено '+n+' из '+TOTAL;}
function code(e){return e.code||('Key'+(e.key||'').toUpperCase());}
document.addEventListener('keydown',e=>{const id=window._cur;if(!id)return;
 const c=code(e),s=V[id]||{v:'',t:false};
 if(c==='KeyT'){s.t=!s.t;}else if(CODES[c]){s.v=CODES[c];}else{return;}
 V[id]=s;paint(id);save();count();});
document.addEventListener('mouseover',e=>{const f=e.target.closest('figure');
 if(f)window._cur=f.dataset.id;});
function next(){const f=document.querySelector('figure[data-v=""]');
 if(f)f.scrollIntoView({block:'center'});}
function zoom(src){const z=document.getElementById('zoom');z.querySelector('img').src=src;
 z.style.display='flex';}
function dl(){let s=HEADER+'\\n';for(const id in V){const r=V[id];
 if(r.v)s+=id+','+r.v+','+(r.t?'true':'false')+'\\n';}
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));
 a.download='holdout_relevance.csv';a.click();}
const TOTAL=document.querySelectorAll('figure').length;
document.querySelectorAll('figure').forEach(f=>paint(f.dataset.id));count();
</script>"""


@dataclass(frozen=True, slots=True)
class GalleryFrame:
    """Кадр эталона, каким его видит размечающий: без единого следа ответа модели."""

    photo_id: str
    class_id: str
    display: str
    crop_path: str


def frames_from(holdout: pd.DataFrame, meta: Mapping[str, Any]) -> list[GalleryFrame]:
    """Кадры без многомашинных (как в замере), по `class_id`, внутри — по `photo_id`."""
    классы = meta["classes"]
    одиночные = holdout[~holdout["is_multi_car"].astype(bool)]
    кадры = [
        GalleryFrame(
            photo_id=str(pid),
            class_id=str(классы[int(idx)]["class_id"]),
            display=str(классы[int(idx)]["display"]),
            crop_path=str(путь),
        )
        for pid, idx, путь in zip(
            одиночные["photo_id"], одиночные["class_idx"], одиночные["crop_path"], strict=True
        )
    ]
    return sorted(кадры, key=lambda к: (к.class_id, к.photo_id))


def _начальное_состояние(initial: Mapping[str, tuple[str, bool]]) -> str:
    состояние = {pid: {"v": v, "t": t} for pid, (v, t) in initial.items()}
    # `</` внутри <script> закрыл бы тег: экранируем, хотя photo_id — hex.
    return json.dumps(состояние, ensure_ascii=False).replace("</", "<\\/")


def render_gallery(frames: Sequence[GalleryFrame], initial: Mapping[str, tuple[str, bool]]) -> str:
    """Страница: раздел на класс, у кадра — номер в классе и короткий `photo_id`."""
    разделы: list[str] = []
    по_классам: dict[str, list[GalleryFrame]] = {}
    for кадр in frames:
        по_классам.setdefault(кадр.class_id, []).append(кадр)
    for class_id, кадры in по_классам.items():
        плитки = "".join(
            f'<figure data-id="{html.escape(к.photo_id)}" data-v="" data-t="false">'
            f'<img loading=lazy src="thumbs/{html.escape(к.photo_id)}.jpg" '
            f"onclick=\"zoom('full/{html.escape(к.photo_id)}.jpg')\">"
            f"<figcaption>#{номер} · {html.escape(к.photo_id[:8])}</figcaption></figure>"
            for номер, к in enumerate(кадры)
        )
        заголовок = f"{html.escape(кадры[0].display)} · {html.escape(class_id)} · {len(кадры)}"
        разделы.append(f"<h2>{заголовок}</h2><div class=grid>{плитки}</div>")
    скрипт = _SCRIPT % {
        "key": json.dumps(_КЛЮЧ_ХРАНИЛИЩА),
        "header": json.dumps(CSV_HEADER),
        "initial": _начальное_состояние(initial),
    }
    return (
        _HEAD
        + "<div id=bar><b>Разметка holdout_ru</b>"
        + "<span>r — релевантен · n — не с улиц РФ · w — метка неверна · u — не уверен · "
        + "t — сверху</span><span id=count></span>"
        + "<button onclick=next()>следующий неразмеченный</button>"
        + "<button onclick=dl()>Скачать holdout_relevance.csv</button>"
        + "<span>результат — только выгруженный CSV</span><span id=warn></span></div>"
        + "<div id=zoom onclick=\"this.style.display='none'\"><img alt=zoom></div>"
        + "".join(разделы)
        + скрипт
    )


def stage_images(
    frames: Sequence[GalleryFrame], crops_root: Path, out: Path, thumb_px: int
) -> None:
    """Миниатюры и полные кропы рядом со страницей; пропавший кроп — ошибка до записи."""
    import cv2

    пропавшие = [к.photo_id for к in frames if not (crops_root / к.crop_path).is_file()]
    if пропавшие:
        raise FileNotFoundError(f"нет кропов у {len(пропавшие)} кадров: {пропавшие[:10]}")
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    (out / "full").mkdir(parents=True, exist_ok=True)
    for кадр in frames:
        источник = crops_root / кадр.crop_path
        shutil.copyfile(источник, out / "full" / f"{кадр.photo_id}.jpg")
        картинка = cv2.imread(str(источник))
        if картинка is None:
            raise ValueError(f"{источник}: не читается как изображение")
        h, w = картинка.shape[:2]
        k = float(thumb_px) / max(h, w)
        if k < 1.0:
            картинка = cv2.resize(картинка, (round(w * k), round(h * k)))
        cv2.imwrite(
            str(out / "thumbs" / f"{кадр.photo_id}.jpg"),
            картинка,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="Слепая галерея разметки holdout_ru")
    парсер.add_argument("--holdout", type=Path, default=splits_dir() / "holdout_ru.parquet")
    парсер.add_argument("--models", type=Path, default=Path("app/android/models"))
    парсер.add_argument("--crops-root", type=Path, default=harvest_dir() / "crops")
    парсер.add_argument("--out", type=Path, default=data_root() / "relevance-gallery")
    парсер.add_argument("--thumb-px", type=int, default=480)
    парсер.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="выгруженный раньше holdout_relevance.csv: продолжить с места остановки",
    )
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    аргументы = build_parser().parse_args(argv)
    мета = json.loads((аргументы.models / "catalog_meta.json").read_text(encoding="utf-8"))
    кадры = frames_from(read_split(аргументы.holdout), мета)
    начальное: dict[str, tuple[str, bool]] = {}
    if аргументы.verdicts is not None:
        прошлые = load_relevance(аргументы.verdicts)
        начальное = {
            str(pid): (str(v), bool(t))
            for pid, v, t in zip(
                прошлые["photo_id"], прошлые["verdict"], прошлые["top_down"], strict=True
            )
        }
    stage_images(кадры, аргументы.crops_root, аргументы.out, аргументы.thumb_px)
    (аргументы.out / "index.html").write_text(render_gallery(кадры, начальное), encoding="utf-8")
    print(f"галерея: {аргументы.out / 'index.html'} — {len(кадры)} кадров")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
