"""Сравнение дообучения с few-shot базисом: чтение базиса, правила решения, отчёт."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.fewshot.compare import (
    RULE_MARGIN_HIGH,
    RULE_MARGIN_LOW,
    RunProvenance,
    decision_rule,
    read_fewshot_metrics,
    render_comparison,
)
from pipeline.fewshot.gate import G_FS2_CORE_CLASSES, check_gate
from pipeline.fewshot.metrics import FewShotMetrics

КАТАЛОГ = 150
BASELINE_TOP1 = 0.4412
BASELINE_TOP3 = 0.6290
ПРЕДРЕГИСТРАЦИЯ = "bench/finetune-ru-preregistration.md"

# Двадцать один неядровой класс: у дообучения их дельты идут от −0.10 до +0.10 шагом 0.01,
# то есть десять лучших — `класс011`…`класс020`, десять худших — `класс000`…`класс009`, а
# `класс010` с нулевой дельтой не попадает ни в одну таблицу и служит проверкой отсечки.
ПРОЧИЕ = tuple(f"класс{номер:03d}" for номер in range(21))
КЛАССЫ = (*G_FS2_CORE_CLASSES, *ПРОЧИЕ)
ОПОРА = {класс: 5 + номер for номер, класс in enumerate(КЛАССЫ)}
# Класс есть у дообучения и отсутствует в базисе: сравнивать его не с чем.
ЛИШНИЙ = "chevrolet__lanos"


def _базис() -> FewShotMetrics:
    """Базис: у всех классов top-3 ровно на пороге G_FS3."""
    return FewShotMetrics(
        top1=BASELINE_TOP1,
        top3=BASELINE_TOP3,
        per_class_top3=dict.fromkeys(КЛАССЫ, 0.50),
        per_class_support=dict(ОПОРА),
    )


def _дообучение() -> FewShotMetrics:
    """Главная трасса: прирост +0.0510 по top-3, поклассовые дельты веером."""
    поклассово = dict.fromkeys(G_FS2_CORE_CLASSES, 0.50)
    for номер, класс in enumerate(ПРОЧИЕ):
        поклассово[класс] = 0.50 + (номер - 10) / 100
    поклассово[ЛИШНИЙ] = 0.90
    return FewShotMetrics(
        top1=0.45,
        top3=0.68,
        per_class_top3=поклассово,
        per_class_support={**ОПОРА, ЛИШНИЙ: 4},
    )


def _голова() -> FewShotMetrics:
    """Справочная трасса: те же классы, числа ниже главной.

    Поклассовые доли нарочно другие: если бы дельты считались по справочной трассе, все они
    стали бы одинаковыми и таблицы дельт это показали бы.
    """
    дообучение = _дообучение()
    return FewShotMetrics(
        top1=0.40,
        top3=0.60,
        per_class_top3=dict.fromkeys(дообучение.per_class_top3, 0.30),
        per_class_support=dict(дообучение.per_class_support),
    )


def _провенанс() -> RunProvenance:
    return RunProvenance(
        git_sha="abc1234",
        run_date="2026-09-06",
        backbone="mobilenetv4_conv_medium.e500_r256_in1k",
        device="cuda",
        epochs=60,
        seed=1337,
        train_minutes=14.5,
        weights_sha256="aaaa1111",
        config_sha256="bbbb2222",
        index_sha256="cccc3333",
        pool_sha256="dddd4444",
        holdout_sha256="eeee5555",
        pool_frames=2863,
        holdout_frames=1523,
        catalog_classes=КАТАЛОГ,
        preregistration=ПРЕДРЕГИСТРАЦИЯ,
    )


def _отчёт() -> str:
    """Отчёт по одному набору фикстур: три замера, гейт по главной трассе, провенанс."""
    дообучение = _дообучение()
    return render_comparison(
        finetune=дообучение,
        finetune_head=_голова(),
        baseline=_базис(),
        gate=check_gate(дообучение, catalog_classes=КАТАЛОГ),
        provenance=_провенанс(),
    )


def _строка_класса(отчёт: str, класс: str) -> str:
    """Единственная строка таблицы дельт про этот класс."""
    строки = [строка for строка in отчёт.splitlines() if f"`{класс}`" in строка]
    assert len(строки) == 1, f"{класс}: строк {len(строки)}, ожидалась одна"
    return строки[0]


def test_чтение_базиса_из_json(tmp_path: Path) -> None:
    путь = tmp_path / "fewshot_metrics.json"
    путь.write_text(
        json.dumps(
            {
                "top1": BASELINE_TOP1,
                "top3": BASELINE_TOP3,
                "per_class_top3": {"lada__granta": 0.30},
                "per_class_support": {"lada__granta": 10},
                # Базис носит рядом вердикт гейта: чтение не обязано с ним спорить.
                "gate_passed": False,
                "gate_failures": ["G_FS1: top-1 0.4412 < 0.7"],
                "deficient_classes": ["lada__granta"],
                "refined_classes": ["lada__granta"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    базис = read_fewshot_metrics(путь)

    assert базис.top1 == pytest.approx(BASELINE_TOP1)
    assert базис.top3 == pytest.approx(BASELINE_TOP3)
    assert базис.per_class_top3 == {"lada__granta": 0.30}
    assert базис.per_class_support == {"lada__granta": 10}


def test_чтение_базиса_роняет_и_называет_поле(tmp_path: Path) -> None:
    путь = tmp_path / "fewshot_metrics.json"
    путь.write_text(
        json.dumps({"top1": BASELINE_TOP1, "top3": BASELINE_TOP3, "per_class_top3": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="per_class_support"):
        read_fewshot_metrics(путь)


def test_правило_решения_на_границах() -> None:
    """Пять случаев поимённо: операторы `≤` и `>` иначе некому удержать.

    Разность считать нельзя: `0.6290 + 0.05 - 0.6290` в двоичных float даёт
    0.050000000000000044, и «ровно +0.05» ушло бы в правило 3.
    """

    def правило(прирост: float) -> str:
        return decision_rule(baseline_top3=BASELINE_TOP3, finetune_top3=BASELINE_TOP3 + прирост)

    assert правило(RULE_MARGIN_LOW).startswith("Правило 1")
    assert правило(0.0501).startswith("Правило 3")
    assert правило(RULE_MARGIN_HIGH).startswith("Правило 3")
    assert правило(0.1001).startswith("Правило 2")
    assert правило(-0.05).startswith("Правило 1")


def test_отчёт_содержит_обе_трассы_и_дельту() -> None:
    отчёт = _отчёт()

    assert "через индекс" in отчёт
    assert "ArcFace" in отчёт
    assert "+0.0510" in отчёт, "дельта top-3 «дообучение минус базис» со знаком"
    assert "0.6000" in отчёт, "top-3 справочной трассы"
    # Справочную линию порогов нельзя прочитать как приёмку.
    assert "§5.2" in отчёт and "не проверяется" in отчёт


def test_отчёт_печатает_поклассовые_дельты_с_опорой() -> None:
    отчёт = _отчёт()

    лучший = _строка_класса(отчёт, "класс020")
    худший = _строка_класса(отчёт, "класс000")
    assert лучший.rstrip().endswith(f"| {ОПОРА['класс020']} |")
    assert худший.rstrip().endswith(f"| {ОПОРА['класс000']} |")
    assert "+0.1000" in лучший and "-0.1000" in худший
    # Отсечка на десяти: класс с нулевой дельтой не попал ни в одну таблицу.
    assert "`класс010`" not in отчёт
    # Класса, которого нет в базисе, в дельтах нет — и в отчёте тоже.
    assert ЛИШНИЙ not in отчёт


def test_отчёт_называет_искажения_и_предрегистрацию() -> None:
    отчёт = _отчёт()

    assert "Четыре искажения" in отчёт
    assert "Утечка номеров" in отчёт
    assert "Заливка" in отчёт
    assert "Чужие модели" in отчёт
    assert "pHash" in отчёт
    assert "видело в обучении" in отчёт, "оговорка искажения 4"
    assert ПРЕДРЕГИСТРАЦИЯ in отчёт
