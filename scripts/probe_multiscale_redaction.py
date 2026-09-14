"""Ранний зонд мультимасштабной редакции номеров на бюджет отсева (Задача 3).

Проверяет выполнимость мультимасштабной нарезки на случайной выборке корпуса (500 строк),
не обогащённой известными утечками и не пересекающейся с калибровочными 300 кадрами.

Кандидаты:
1. baseline: SSD 1280 (conf 0.50) + YOLO 1280 (conf 0.25)
2. yolo_1280_640: YOLO 1280 (conf 0.25) + YOLO 640 (conf 0.25)
3. yolo_1280_640_320: YOLO 1280 + 640 + 320 (conf 0.25 у всех трёх)
4. yolo_1280_640_ssd_1280: YOLO 1280 + 640 (conf 0.25) + SSD 1280 (conf 0.50)

Бюджет отсева: ~3.8% (37 попаданий top-3 запаса до границы Клоппера–Пирсона).
Если нижняя граница интервала Клоппера–Пирсона превышает бюджет — план закрывается.

Примеры запуска:
    uv run python -m scripts.probe_multiscale_redaction --step all
    uv run python -m scripts.probe_multiscale_redaction --step sample
    uv run python -m scripts.probe_multiscale_redaction --step run --candidates baseline
    uv run python -m scripts.probe_multiscale_redaction --step evaluate
    uv run python -m scripts.probe_multiscale_redaction --step diagnose
    python scripts/probe_multiscale_redaction.py --help
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Бутстрап sys.path для прямого вызова: python scripts/probe_multiscale_redaction.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.harvest.cache import cache_path, decode_image  # noqa: E402
from pipeline.harvest.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    HarvestRow,
    load_manifest,
    write_manifest,
)
from pipeline.harvest.redaction_profile import (  # noqa: E402
    CURRENT_REDACTION_PROFILE,
    DetectorEntry,
    DetectorProfile,
    RedactionProfile,
)
from pipeline.paths import data_root, harvest_dir  # noqa: E402
from pipeline.plate_leak import clopper_pearson  # noqa: E402

DEFAULT_SEED = 20260914
DEFAULT_SAMPLE_SIZE = 500
DROPOUT_BUDGET = 0.038  # ~3.8% (37 попаданий top-3)


def make_probe_candidates() -> dict[str, RedactionProfile]:
    """Сформировать четыре профиля зонда согласно спеке Задачи 3."""
    c1_baseline = CURRENT_REDACTION_PROFILE

    c2_yolo_1280_640 = RedactionProfile(
        (
            DetectorEntry(
                instance="yolo_1280",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=1280, overlap=0.25),
            ),
            DetectorEntry(
                instance="yolo_640",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=640, overlap=0.25),
            ),
        )
    )

    c3_yolo_1280_640_320 = RedactionProfile(
        (
            DetectorEntry(
                instance="yolo_1280",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=1280, overlap=0.25),
            ),
            DetectorEntry(
                instance="yolo_640",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=640, overlap=0.25),
            ),
            DetectorEntry(
                instance="yolo_320",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=320, overlap=0.25),
            ),
        )
    )

    c4_yolo_1280_640_ssd_1280 = RedactionProfile(
        (
            DetectorEntry(
                instance="ssd_1280",
                weights="plate",
                params=DetectorProfile(confidence=0.50, tile=1280, overlap=0.25),
            ),
            DetectorEntry(
                instance="yolo_1280",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=1280, overlap=0.25),
            ),
            DetectorEntry(
                instance="yolo_640",
                weights="plate_hires",
                params=DetectorProfile(confidence=0.25, tile=640, overlap=0.25),
            ),
        )
    )

    return {
        "baseline": c1_baseline,
        "yolo_1280_640": c2_yolo_1280_640,
        "yolo_1280_640_320": c3_yolo_1280_640_320,
        "yolo_1280_640_ssd_1280": c4_yolo_1280_640_ssd_1280,
    }


def default_probe_dir() -> Path:
    """Каталог зонда мультимасштаба под ~/carlens-data/multiscale/probe."""
    return data_root() / "multiscale" / "probe"


def default_calibration_sample_path() -> Path:
    """Путь к калибровочной выборке 300 кадров."""
    return Path(__file__).resolve().parent.parent / "bench" / "plate-leak-sample.csv"


def default_canonical_geometry_path() -> Path:
    """Путь к 15 каноническим номерам геометрии."""
    return data_root() / "redaction-calibration" / "geometry" / "canonical.csv"


def select_probe_sample(
    manifest_path: Path,
    calibration_sample_path: Path,
    canonical_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    cache_root: Path | None = None,
) -> tuple[list[HarvestRow], set[str]]:
    """Выбрать случайную выборку корпуса без пересечения с калибровочными и каноническими кадрами.

    Сортировка по sha256 перед сэмплированием гарантирует стабильность независимо от
    случайных перестановок строк входного файла.
    """
    rows = load_manifest(manifest_path)
    excluded_hashes: set[str] = set()

    if calibration_sample_path.exists():
        sample_df = pd.read_csv(calibration_sample_path, dtype={"sha256": str})
        if "sha256" in sample_df.columns:
            excluded_hashes.update(sample_df["sha256"].dropna().astype(str).unique())
    else:
        print(f"Предупреждение: файл калибровочной выборки не найден: {calibration_sample_path}")

    if canonical_path.exists():
        canonical_df = pd.read_csv(canonical_path, dtype={"sha256": str})
        if "sha256" in canonical_df.columns:
            excluded_hashes.update(canonical_df["sha256"].dropna().astype(str).unique())
    else:
        print(f"Предупреждение: файл канонических номеров не найден: {canonical_path}")

    # Фильтруем корпус
    candidates = [r for r in rows if r.sha256 not in excluded_hashes]
    candidates.sort(key=lambda r: r.sha256)

    if len(candidates) < sample_size:
        raise ValueError(
            f"Недостаточно строк для выборки: доступно {len(candidates)}, запрошено {sample_size}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(candidates, sample_size)
    sampled.sort(key=lambda r: r.sha256)

    # Проверяем наличие кэша
    if cache_root is not None:
        missing_cache = [
            r.sha256
            for r in sampled
            if not cache_path(cache_root, r.source, r.sha256, "jpg").exists()
        ]
        if missing_cache:
            raise ValueError(
                f"Отсутствует кэш оригиналов для {len(missing_cache)} кадров выборки, "
                f"например {missing_cache[:3]}"
            )

    return sampled, excluded_hashes


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    """Результаты прогона одного кандидата."""

    name: str
    input_count: int
    accepted_count: int
    rejected_count: int
    redaction_failed_count: int
    redaction_failed_share: float
    ci_low: float
    ci_high: float
    exceeds_budget: bool
    reject_reasons: dict[str, int]


def parse_candidate_report(
    name: str, report_path: Path, budget: float = DROPOUT_BUDGET
) -> CandidateMetrics:
    """Разобрать report.json и рассчитать интервал Клоппера–Пирсона."""
    data = json.loads(report_path.read_text(encoding="utf-8"))
    counts = data.get("counts", {})
    input_count = int(counts.get("input", 0))
    accepted_count = int(counts.get("accepted", 0))
    rejected_count = int(counts.get("rejected", 0))
    reject_reasons: dict[str, int] = {
        str(k): int(v) for k, v in data.get("reject_reasons", {}).items()
    }

    rf_count = reject_reasons.get("redaction_failed", 0)
    rf_share = rf_count / input_count if input_count > 0 else 0.0
    ci_low, ci_high = (
        clopper_pearson(rf_count, input_count, confidence=0.95) if input_count > 0 else (0.0, 1.0)
    )
    exceeds = ci_low > budget

    return CandidateMetrics(
        name=name,
        input_count=input_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        redaction_failed_count=rf_count,
        redaction_failed_share=rf_share,
        ci_low=ci_low,
        ci_high=ci_high,
        exceeds_budget=exceeds,
        reject_reasons=reject_reasons,
    )


def save_profile_document(profile: RedactionProfile, path: Path) -> None:
    """Сохранить профиль в JSON для передачи в rebuild_redaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = asdict(profile)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_rebuild_subprocess(
    manifest_path: Path,
    out_root: Path,
    profile_path: Path | None,
    *,
    data_root_dir: Path,
    trace: bool = True,
) -> None:
    """Выполнить rebuild_redaction.py через изолированный подпроцесс."""
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "scripts.rebuild_redaction",
        "--manifest",
        str(manifest_path),
        "--out-root",
        str(out_root),
        "--data-root",
        str(data_root_dir),
    ]
    if profile_path is not None:
        cmd.extend(["--profile", str(profile_path)])
    if trace:
        cmd.append("--trace")

    print(f"Запуск: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"Команда завершилась с кодом {result.returncode}")


def render_diagnostic_frame(
    raw_image: np.ndarray,
    photo_id: str,
    candidate_name: str,
    observations: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Отрендерить кадр с вжжёнными боксами детектора и аннотацией ошибки.

    Зелёный: боксы первого прохода (найденные номера).
    Красный: боксы второго прохода (остатки, вызвавшие отсев).
    """
    canvas = raw_image.copy()
    height, width = canvas.shape[:2]

    first_pass_merged = [o for o in observations if o.get("pass") == "first" and o.get("merged")]
    second_pass_merged = [o for o in observations if o.get("pass") == "second" and o.get("merged")]

    for obs in first_pass_merged:
        bx = round(float(obs["x"]))
        by = round(float(obs["y"]))
        bw = round(float(obs["w"]))
        bh = round(float(obs["h"]))
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)

    for obs in second_pass_merged:
        bx = round(float(obs["x"]))
        by = round(float(obs["y"]))
        bw = round(float(obs["w"]))
        bh = round(float(obs["h"]))
        conf = float(obs.get("confidence", 0.0))
        det = str(obs.get("detector", ""))
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 0, 255), 4)
        cv2.putText(
            canvas,
            f"{det} {conf:.2f}",
            (bx, max(20, by - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    banner_height = 80
    new_w, new_h = pil_img.width, pil_img.height + banner_height
    banner_img = Image.new("RGB", (new_w, new_h), "white")
    banner_img.paste(pil_img, (0, banner_height))

    draw = ImageDraw.Draw(banner_img)
    font = ImageFont.load_default()

    header_text = (
        f"ID: {photo_id} | Candidate: {candidate_name} | Fails: redaction_failed\n"
        f"Dimensions: {width}x{height} | 1st pass boxes: {len(first_pass_merged)} | "
        f"2nd pass residuals (RED): {len(second_pass_merged)}"
    )
    draw.text((20, 20), header_text, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner_img.save(output_path, quality=90)


def create_diagnostics_for_candidate(
    candidate_name: str,
    candidate_dir: Path,
    harvest_data_dir: Path,
    out_diagnostics_dir: Path,
    max_frames: int = 10,
) -> list[str]:
    """Сгенерировать диагностические изображения для отсеянных кадров."""
    rejects_file = candidate_dir / "rejects.csv"
    obs_file = candidate_dir / "observations.csv"
    if not rejects_file.exists() or not obs_file.exists():
        return []

    rejects_df = pd.read_csv(rejects_file)
    rf_df = rejects_df[rejects_df["reason"] == "redaction_failed"]
    if rf_df.empty:
        return []

    obs_df = pd.read_csv(obs_file)

    inspected_ids: list[str] = []
    sample_rfs = rf_df.head(max_frames)

    for _, row in sample_rfs.iterrows():
        photo_id = str(row["sha256"])
        source = str(row["source"])
        raw_cache = cache_path(harvest_data_dir, source, photo_id, "jpg")
        if not raw_cache.exists():
            continue

        raw_data = raw_cache.read_bytes()
        raw_img = decode_image(raw_data)

        raw_rows = obs_df[obs_df["photo_id"] == photo_id].to_dict(orient="records")
        frame_obs: list[dict[str, Any]] = [{str(k): v for k, v in r.items()} for r in raw_rows]
        out_path = out_diagnostics_dir / f"{candidate_name}_{photo_id[:12]}.jpg"
        render_diagnostic_frame(raw_img, photo_id, candidate_name, frame_obs, out_path)
        inspected_ids.append(photo_id)
    return inspected_ids


def print_residual_analysis(candidate_name: str, candidate_dir: Path, max_frames: int = 10) -> None:
    """Вывести статистику и детальный разбор причин срабатывания 2-го прохода."""
    rejects_file = candidate_dir / "rejects.csv"
    obs_file = candidate_dir / "observations.csv"
    if not rejects_file.exists() or not obs_file.exists():
        return

    rejects_df = pd.read_csv(rejects_file)
    rf_df = rejects_df[rejects_df["reason"] == "redaction_failed"]
    if rf_df.empty:
        print(f"\n--- {candidate_name}: 0 отсеянных кадров ---")
        return

    obs_df = pd.read_csv(obs_file)
    rf_ids = set(rf_df["sha256"].dropna().astype(str))

    sec_raw = obs_df[
        (obs_df["photo_id"].isin(rf_ids)) & (obs_df["pass"] == "second") & (obs_df["raw"])
    ]
    sec_merged = obs_df[
        (obs_df["photo_id"].isin(rf_ids)) & (obs_df["pass"] == "second") & (obs_df["merged"])
    ]

    print("\n" + "=" * 65)
    print(f"Анализ остатков второго прохода: {candidate_name}")
    print(f"Всего отсеяно с redaction_failed: {len(rf_ids)} кадров")
    print("Распределение сырых срабатываний (raw) по детекторам:")
    for det, cnt in sec_raw["detector"].value_counts().items():
        print(f"  {det}: {cnt}")

    print("Распределение объединённых остатков (merged):")
    print(f"  Всего объединённых остатков: {len(sec_merged)}")
    if not sec_raw.empty:
        c_min = sec_raw["confidence"].min()
        c_mean = sec_raw["confidence"].mean()
        c_max = sec_raw["confidence"].max()
        print(f"  Уверенность (conf): min={c_min:.3f}, mean={c_mean:.3f}, max={c_max:.3f}")

    sample_size = min(max_frames, len(rf_df))
    print(f"\nДетальный разбор первых {sample_size} отсеянных кадров:")
    for idx, (_, row) in enumerate(rf_df.head(max_frames).iterrows(), 1):
        pid = str(row["sha256"])
        f_obs = obs_df[obs_df["photo_id"] == pid]
        p1 = f_obs[(f_obs["pass"] == "first") & (f_obs["merged"])]
        p2 = f_obs[(f_obs["pass"] == "second") & (f_obs["merged"])]
        p2_raw = f_obs[(f_obs["pass"] == "second") & (f_obs["raw"])]

        dets_p2 = [str(d) for d in p2_raw["detector"].unique()]
        res_info: list[str] = []
        for _, r in p2.iterrows():
            w, h, conf = float(r["w"]), float(r["h"]), float(r["confidence"])
            aspect = w / max(h, 1.0)
            res_info.append(f"{w:.0f}x{h:.0f} (aspect={aspect:.2f}, conf={conf:.2f})")

        print(
            f"  #{idx:02d} ID: {pid[:12]} | 1st pass: {len(p1)} заливок | "
            f"2nd pass: {len(p2)} остатков [{', '.join(dets_p2)}] -> {'; '.join(res_info)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Зонд мультимасштабной редакции на бюджет отсева")
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=default_probe_dir(),
        help="Корень результатов зонда (~/carlens-data/multiscale/probe)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Путь к harvest_manifest.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=harvest_dir(),
        help="Корень харвеста (~/carlens-data/harvest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Фиксированный seed выборки",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Размер выборки (по умолчанию 500)",
    )
    parser.add_argument(
        "--step",
        choices=["sample", "run", "evaluate", "diagnose", "all"],
        default="all",
        help="Шаг выполнения (по умолчанию all)",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=["baseline", "yolo_1280_640", "yolo_1280_640_320", "yolo_1280_640_ssd_1280"],
        help="Список кандидатов для прогона",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probe_dir: Path = args.probe_dir
    probe_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest_path = probe_dir / "sample_manifest.csv"
    sample_ids_path = probe_dir / "sample_ids.json"
    profiles_dir = probe_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # 1. Сэмплирование (Step 1)
    if args.step in ("sample", "all"):
        print(f"=== Шаг 1: Формирование выборки {args.sample_size} строк (seed={args.seed}) ===")
        sample_rows, excluded_hashes = select_probe_sample(
            manifest_path=args.manifest,
            calibration_sample_path=default_calibration_sample_path(),
            canonical_path=default_canonical_geometry_path(),
            seed=args.seed,
            sample_size=args.sample_size,
            cache_root=args.data_root,
        )
        write_manifest(sample_manifest_path, sample_rows)
        sample_metadata = {
            "seed": args.seed,
            "sample_size": len(sample_rows),
            "excluded_hashes_count": len(excluded_hashes),
            "sha256_list": [r.sha256 for r in sample_rows],
        }
        sample_ids_path.write_text(
            json.dumps(sample_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Выборка сохранена в {sample_manifest_path} ({len(sample_rows)} строк)")

    if args.step == "sample":
        return 0

    if not sample_manifest_path.exists():
        print(
            f"Ошибка: файл выборки {sample_manifest_path} не найден. "
            "Запустите сначала с --step sample"
        )
        return 1

    all_candidates = make_probe_candidates()
    selected_candidates = {
        name: all_candidates[name] for name in args.candidates if name in all_candidates
    }

    # 2. Прогон кандидатов (Step 2)
    if args.step in ("run", "all"):
        print(f"=== Шаг 2: Прогон кандидатов ({', '.join(selected_candidates.keys())}) ===")
        for name, profile in selected_candidates.items():
            candidate_out = probe_dir / name
            if candidate_out.exists():
                print(
                    f"Кандидат {name}: каталог {candidate_out} уже существует, пропускаем прогон."
                )
                continue

            profile_path = profiles_dir / f"{name}.json"
            save_profile_document(profile, profile_path)
            print(f"\n--- Прогон кандидата {name} ---")
            run_rebuild_subprocess(
                manifest_path=sample_manifest_path,
                out_root=candidate_out,
                profile_path=profile_path,
                data_root_dir=args.data_root,
                trace=True,
            )

    if args.step == "run":
        return 0

    # 3. Оценка отсева (Step 3)
    metrics_list: list[CandidateMetrics] = []
    if args.step in ("evaluate", "all"):
        print("\n=== Шаг 3: Сравнение отсева с бюджетом (~3.8%) ===")
        print(
            f"{'Кандидат':<25} | {'Принято':<8} | {'Отсев':<6} | {'Redact Fail':<11} | "
            f"{'Доля RF':<8} | {'95% CI Клоппера–Пирсона':<24} | {'Бюджет':<10}"
        )
        print("-" * 105)
        for name in selected_candidates:
            report_path = probe_dir / name / "report.json"
            if not report_path.exists():
                print(f"Внимание: {report_path} не существует!")
                continue
            metrics = parse_candidate_report(name, report_path)
            metrics_list.append(metrics)
            verdict_str = "ПРЕВЫШЕН!" if metrics.exceeds_budget else "В норме"
            ci_str = f"[{metrics.ci_low * 100:.2f}%, {metrics.ci_high * 100:.2f}%]"
            share_pct_str = f"{metrics.redaction_failed_share * 100:>6.2f}%"
            print(
                f"{name:<25} | {metrics.accepted_count:<8} | {metrics.rejected_count:<6} | "
                f"{metrics.redaction_failed_count:<11} | {share_pct_str} | "
                f"{ci_str:<24} | {verdict_str:<10}"
            )

    # 4. Диагностика отсеянных кадров (Step 4)
    if args.step in ("diagnose", "all"):
        print("\n=== Шаг 4: Визуальная диагностика отсеянных кадров ===")
        diagnostics_dir = probe_dir / "diagnostics"
        for name in selected_candidates:
            candidate_dir = probe_dir / name
            inspected = create_diagnostics_for_candidate(
                candidate_name=name,
                candidate_dir=candidate_dir,
                harvest_data_dir=args.data_root,
                out_diagnostics_dir=diagnostics_dir,
                max_frames=10,
            )
            print(
                f"Кандидат {name}: сохранено {len(inspected)} "
                f"диагностических кадров в {diagnostics_dir}"
            )
            print_residual_analysis(name, candidate_dir, max_frames=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
