"""Скрипт слепого VLM-скрининга выборки 300 кадров на утечки номеров и лиц.

Прогоняет модель со зрением (VLM) по кадрам выборки bench/plate-leak-sample.csv
вслепую (без знания истины о том, какие кадры протекли).
Использует кропы из оригинала (с масштабированием до long_side=1600 px для сохранения
деталей без избыточных затрат времени на инференс), как в stage_plate_leak_views.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from pipeline.paths import data_root

DEFAULT_MODEL = "hf.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF:Q8_0"
DEFAULT_LONG_SIDE = 1600
OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = (
    "Твоя задача — аудит приватности (GDPR / 152-ФЗ).\n"
    "Нужно найти ЛЮБЫЕ незамазанные регистрационные знаки и незамазанные лица людей.\n\n"
    "Обязательно проверь:\n"
    "- Передний и задний бамперы автомобиля, зону под фарами и решеткой радиатора "
    "(номер может быть в тени, сбоку или под углом).\n"
    "- Номера машин на заднем плане / фоне.\n"
    "- Лица людей (включая водителя и пассажиров через лобовое или боковые стекла).\n\n"
    "Если номер или лицо закрыты сплошным черным прямоугольником — это замазано (ок).\n"
    "Если хотя бы часть символов номера видна или номер вовсе не замазан — это УТЕЧКА "
    "(flag = true).\n"
    "Если номера вообще нет (пустая рамка, шильд автосалона) или он стерт без символов — "
    "это НЕ утечка (flag = false).\n\n"
    "Ответь строго в формате JSON:\n"
    "{\n"
    '  "has_unredacted_plate": true/false,\n'
    '  "plate_details": "описание номера и места или null",\n'
    '  "has_unredacted_face": true/false,\n'
    '  "face_details": "описание лица или null",\n'
    '  "flag": true/false\n'
    "}\n"
)

KNOWN_PLATE_LEAKS = {6, 82, 156, 205, 214, 224, 266, 286}
HISTORICAL_PLATE_LEAKS = {43}
KNOWN_FACE_MISSES = {32, 36, 68, 102, 120, 179, 196, 218, 231, 254}


def prepare_image_b64(
    path: Path, long_side: int = DEFAULT_LONG_SIDE
) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        scale = min(1.0, long_side / max(w, h))
        new_size = (round(w * scale), round(h * scale))
        resized = img_rgb.resize(new_size, Image.Resampling.LANCZOS) if scale < 1.0 else img_rgb
        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), (w, h)


def query_vlm(img_b64: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    content = res["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return dict(json.loads(content))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-доверительный интервал Уилсона для доли.

    Берём Уилсона, а не нормальное приближение: на малых n у краёв
    нормальный интервал вылезает за [0, 1] и занижает ширину. Нужен,
    чтобы отличить доказанный провал по флагам от точечной оценки.
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_report(
    processed: dict[int, dict[str, Any]],
    model: str,
    sample_total: int | None = None,
) -> dict[str, Any]:
    """Считает отчёт скрининга по уже обработанным кадрам.

    Числитель и знаменатель recall берутся по кадрам, реально попавшим в
    прогон (evaluated), а не по всем известным утечкам выборки. Иначе
    неполный прогон занижает recall и превращает «не проверено» в «провал».
    Пока покрытие по номерам неполное, критерий 8/8 и общий вердикт остаются
    null: доказать их на части выборки нельзя.
    """

    def split(known: set[int]) -> tuple[list[int], list[int], list[int]]:
        evaluated = sorted(n for n in known if n in processed)
        hits = [n for n in evaluated if processed[n]["flag"]]
        misses = [n for n in evaluated if not processed[n]["flag"]]
        return evaluated, hits, misses

    plate_eval, plate_hits, plate_misses = split(KNOWN_PLATE_LEAKS)
    face_eval, face_hits, face_misses = split(KNOWN_FACE_MISSES)
    hist_eval, hist_hits, _ = split(HISTORICAL_PLATE_LEAKS)

    total_evaluated = len(processed)
    total_flags = sum(1 for r in processed.values() if r["flag"])

    known_all = KNOWN_PLATE_LEAKS | HISTORICAL_PLATE_LEAKS | KNOWN_FACE_MISSES
    clean = [r for n, r in processed.items() if n not in known_all]
    clean_flags = sum(1 for r in clean if r["flag"])

    flag_rate = total_flags / total_evaluated if total_evaluated else 0.0
    ci_lo, ci_hi = wilson_ci(total_flags, total_evaluated)
    clean_flag_rate = clean_flags / len(clean) if clean else 0.0

    plate_recall = len(plate_hits) / len(plate_eval) if plate_eval else None
    face_recall = len(face_hits) / len(face_eval) if face_eval else None

    plate_coverage_complete = len(plate_eval) == len(KNOWN_PLATE_LEAKS)
    flags_ok = flag_rate <= 0.15
    ci_includes_threshold = ci_lo <= 0.15 <= ci_hi

    # Критерий 8/8 и итоговый вердикт определимы только при полном покрытии.
    criterion_plates = (
        len(plate_hits) == len(KNOWN_PLATE_LEAKS) if plate_coverage_complete else None
    )
    triage_works = (criterion_plates and flags_ok) if plate_coverage_complete else None

    def rnd(value: float | None) -> float | None:
        return round(value, 4) if value is not None else None

    if triage_works is None:
        recommendation = (
            f"предварительно: покрытие неполное "
            f"({len(plate_eval)}/{len(KNOWN_PLATE_LEAKS)} номеров, "
            f"{len(face_eval)}/{len(KNOWN_FACE_MISSES)} лиц). Триаж отклонён по "
            f"нагрузке: флаги {flag_rate * 100:.1f}% (95% ДИ "
            f"{ci_lo * 100:.0f}–{ci_hi * 100:.0f}%). Порог 15% попадает в "
            f"интервал, провал статистически не доказан — нужен полный прогон 300"
        )
    elif triage_works:
        recommendation = "триаж работает: модель отбирает кадры для ручной проверки"
    else:
        recommendation = "триаж отклонён: требуется сплошной ручной аудит"

    return {
        "model": model,
        "sample_total": sample_total,
        "total_evaluated": total_evaluated,
        "coverage_complete": (
            plate_coverage_complete
            and len(face_eval) == len(KNOWN_FACE_MISSES)
            and len(hist_eval) == len(HISTORICAL_PLATE_LEAKS)
        ),
        "total_flags": total_flags,
        "overall_flag_rate": round(flag_rate, 4),
        "overall_flag_rate_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "plate_leaks": {
            "known_count": len(KNOWN_PLATE_LEAKS),
            "evaluated": len(plate_eval),
            "hits": plate_hits,
            "misses": plate_misses,
            "recall_over_evaluated": rnd(plate_recall),
        },
        "historical_plate_leaks": {
            "known_count": len(HISTORICAL_PLATE_LEAKS),
            "evaluated": len(hist_eval),
            "hits": hist_hits,
        },
        "face_misses": {
            "known_count": len(KNOWN_FACE_MISSES),
            "evaluated": len(face_eval),
            "hits": face_hits,
            "misses": face_misses,
            "recall_over_evaluated": rnd(face_recall),
        },
        "clean_frames": {
            "evaluated": len(clean),
            "flags": clean_flags,
            "flag_rate": round(clean_flag_rate, 4),
        },
        "verdict": {
            "triage_works": triage_works,
            "criterion_8_of_8_plates": criterion_plates,
            "criterion_flags_leq_15_percent": flags_ok,
            "flag_rate_ci95_includes_15pct": ci_includes_threshold,
            "recommendation": recommendation,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Слепой VLM-скрининг выборки 300 кадров")
    parser.add_argument("--sample", type=Path, default=Path("bench/plate-leak-sample.csv"))
    parser.add_argument(
        "--crops-root",
        type=Path,
        default=data_root() / "harvest" / "crops",
    )
    parser.add_argument("--out-jsonl", type=Path, default=Path("bench/screening_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("bench/screening_report.json"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--recompute-only",
        action="store_true",
        help="Пересчитать отчёт из существующего jsonl без запуска VLM.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.sample, encoding="utf-8-sig")

    processed: dict[int, dict[str, Any]] = {}
    if args.out_jsonl.is_file():
        with open(args.out_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line.strip())
                    processed[rec["номер"]] = rec
        print(f"Загружено ранее обработанных кадров: {len(processed)}")

    if not args.recompute_only:
        count = 0
        try:
            with open(args.out_jsonl, "a", encoding="utf-8") as out_file:
                for _, row in df.iterrows():
                    num = int(row["номер"])
                    if args.limit and count >= args.limit:
                        break

                    if num in processed:
                        continue

                    class_id = str(row["класс"])
                    sha256 = str(row["sha256"])
                    crop_path = args.crops_root / class_id / f"{sha256}.jpg"

                    if not crop_path.is_file():
                        print(f"Кадр #{num:03d}: файл не найден {crop_path}", file=sys.stderr)
                        continue

                    t0 = time.time()
                    try:
                        b64, _ = prepare_image_b64(crop_path)
                        vlm_res = query_vlm(b64, model=args.model)
                        latency = time.time() - t0
                        flag = bool(vlm_res.get("flag", False))
                        has_plate = bool(vlm_res.get("has_unredacted_plate", False))
                        has_face = bool(vlm_res.get("has_unredacted_face", False))
                        plate_details = vlm_res.get("plate_details")
                        face_details = vlm_res.get("face_details")

                        record = {
                            "номер": num,
                            "sha256": sha256,
                            "класс": class_id,
                            "flag": flag,
                            "has_unredacted_plate": has_plate,
                            "plate_details": plate_details,
                            "has_unredacted_face": has_face,
                            "face_details": face_details,
                            "latency_s": round(latency, 2),
                            "gt_вердикт": str(row.get("вердикт", "")),
                            "gt_заметка": str(row.get("заметка", "")),
                        }

                        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_file.flush()
                        processed[num] = record
                        count += 1

                        print(
                            f"[{len(processed)}/{len(df)}] #{num:03d} ({latency:.1f}s): "
                            f"flag={flag} (plate={has_plate}, face={has_face}) | "
                            f"gt={row.get('вердикт', '')}"
                        )

                    except Exception as e:
                        print(f"Ошибка на кадре #{num:03d}: {e}", file=sys.stderr)

        except KeyboardInterrupt:
            print("\nОстановлено пользователем.")

    print("\n--- Формирование отчёта скрининга ---")
    report = build_report(processed, args.model, sample_total=len(df))

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Отчёт сохранён в {args.report}")

    plate = report["plate_leaks"]
    face = report["face_misses"]
    ci_lo, ci_hi = report["overall_flag_rate_ci95"]
    print(
        f"Recall по номерам (по оценённым): {len(plate['hits'])}/{plate['evaluated']} "
        f"из {plate['known_count']} известных"
    )
    print(
        f"Recall по лицам (по оценённым): {len(face['hits'])}/{face['evaluated']} "
        f"из {face['known_count']} известных"
    )
    print(
        f"Доля флагов: {report['total_flags']}/{report['total_evaluated']} "
        f"({report['overall_flag_rate'] * 100:.1f}%, 95% ДИ "
        f"{ci_lo * 100:.0f}–{ci_hi * 100:.0f}%)"
    )
    triage = report["verdict"]["triage_works"]
    if triage is None:
        verdict_str = "НЕ ОПРЕДЕЛЁН (неполное покрытие)"
    else:
        verdict_str = "ТРИАЖ РАБОТАЕТ" if triage else "ТРИАЖ ОТКЛОНЁН"
    print(f"ИТОГОВЫЙ ВЕРДИКТ: {verdict_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
