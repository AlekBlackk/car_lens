# scripts/audit_gallery.py
"""HTML-галерея для ручного аудита H3.

Аудит смотрит сохранённые кропы, не оригиналы: едет кроп. Порядок — обязательные кадры
первыми (номера, лица, ядровые классы), внутри — по убыванию `clip_rank` (непохожее
дешевле ловить). Клавиши: o = ok, p = drop:plate, f = drop:face, m = drop:wrong_model,
c = conflict. Кнопка «Скачать verdicts.csv» отдаёт файл в формате
`pipeline.harvest.audit.load_audit`.

Галерея самодостаточна: миниатюры лежат рядом относительными путями, скрипт их
предварительно ужимает, чтобы страница на 2000 кадров открывалась.
"""

from __future__ import annotations

import argparse
import html
from collections.abc import Callable, Sequence
from pathlib import Path

from pipeline.harvest.audit import required_audit_files
from pipeline.harvest.manifest import HarvestRow, load_manifest

_HEAD = """<!doctype html><meta charset=utf-8><title>Аудит H3</title><style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
#bar{position:sticky;top:0;background:#000;padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:4px;padding:4px}
figure{margin:0;background:#1c1c1c;border:3px solid transparent}
figure.req{border-color:#c50}
figure.v-ok{border-color:#2a2}figure.v-drop{border-color:#e33}figure.v-conflict{border-color:#fb0}
img{width:100%;display:block;cursor:zoom-in}
figcaption{padding:4px;font-size:12px}
#zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center}
#zoom img{max-width:96vw;max-height:96vh;width:auto;cursor:zoom-out}
</style>"""

_SCRIPT = """<script>
const V={};
function mark(f,v){V[f]=v;const el=document.querySelector('[data-f="'+CSS.escape(f)+'"]');
 el.className=el.className.replace(/ v-\\S+/g,'')+' v-'+(v.indexOf('drop')===0?'drop':v);}
document.addEventListener('keydown',e=>{const f=window._cur;if(!f)return;
 const k={o:'ok',p:'drop:plate',f:'drop:face',m:'drop:wrong_model',c:'conflict'}[e.key];
 if(k)mark(f,k);});
document.addEventListener('mouseover',e=>{const fig=e.target.closest('figure');
 if(fig)window._cur=fig.dataset.f;});
function zoom(src){const z=document.getElementById('zoom');z.querySelector('img').src=src;
 z.style.display='flex';}
function dl(){let s='file,verdict\\n';for(const k in V)s+=k+','+V[k]+'\\n';
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));
 a.download='verdicts.csv';a.click();}
</script>"""


def render_gallery(
    rows: Sequence[HarvestRow],
    required: set[str],
    thumb_rel: Callable[[HarvestRow], str],
) -> str:
    """Собрать HTML: обязательные кадры первыми, внутри — по убыванию clip_rank."""
    порядок = sorted(rows, key=lambda с: (с.file not in required, -с.clip_rank))
    плитки: list[str] = []
    for с in порядок:
        клеймо = "req" if с.file in required else ""
        подпись = (
            f"{html.escape(с.class_id)} · rank {с.clip_rank} · plates {с.plates} · faces {с.faces}"
        )
        плитки.append(
            f'<figure class="{клеймо}" data-f="{html.escape(с.file)}">'
            f'<img loading=lazy src="{html.escape(thumb_rel(с))}" onclick="zoom(this.src)">'
            f"<figcaption>{подпись}</figcaption></figure>"
        )
    return (
        _HEAD
        + "<div id=bar><b>Аудит H3</b>"
        + "<span>o=ok p=drop:plate f=drop:face m=drop:wrong_model c=conflict</span>"
        + "<button onclick=dl()>Скачать verdicts.csv</button>"
        + f"<span>всего {len(rows)}, обязательных {len(required)}</span></div>"
        + "<div id=zoom onclick=\"this.style.display='none'\"><img alt=zoom></div>"
        + "<div class=grid>"
        + "".join(плитки)
        + "</div>"
        + _SCRIPT
    )


def build_parser() -> argparse.ArgumentParser:
    парсер = argparse.ArgumentParser(description="HTML-галерея аудита H3")
    парсер.add_argument("--manifest", type=Path, required=True)
    парсер.add_argument("--data-root", type=Path, required=True)
    парсер.add_argument("--thumb-px", type=int, default=480)
    парсер.add_argument("--seed", type=int, default=0)
    return парсер


def main(argv: Sequence[str] | None = None) -> int:
    """Ужать миниатюры и записать `audit/index.html` в каталог данных."""
    import cv2

    аргументы = build_parser().parse_args(argv)
    строки = load_manifest(аргументы.manifest)
    надо = required_audit_files(строки, seed=аргументы.seed)

    каталог = аргументы.data_root / "audit"
    (каталог / "thumbs").mkdir(parents=True, exist_ok=True)
    for с in строки:
        источник = аргументы.data_root / "crops" / с.class_id / f"{с.sha256}.jpg"
        цель = каталог / "thumbs" / с.file
        if цель.exists() or not источник.exists():
            continue
        картинка = cv2.imread(str(источник))
        if картинка is None:
            continue
        h, w = картинка.shape[:2]
        k = float(аргументы.thumb_px) / max(h, w)
        if k < 1.0:
            картинка = cv2.resize(картинка, (int(w * k), int(h * k)))
        cv2.imwrite(str(цель), картинка, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    (каталог / "index.html").write_text(
        render_gallery(строки, надо, lambda с: f"thumbs/{с.file}"), encoding="utf-8"
    )
    print(f"галерея: {каталог / 'index.html'} — {len(надо)} обязательных кадров")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
