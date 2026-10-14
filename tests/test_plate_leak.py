"""Тесты замера утечки номера в обучающей выборке (§7 `bench/plate-leak-bound.md`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.plate_leak import (
    LEAK_STRICT,
    LEAK_WITH_HISTORICAL,
    LEAK_WITH_PARTIAL,
    SAMPLE_COLUMNS,
    PopulationSizes,
    clopper_pearson,
    compute_bound,
    merge_sample_verdicts,
    project_to_sample_schema,
    strict_leak_verdicts,
)

# Золотые числа из bound_report.json (посчитаны scipy.stats.beta.ppf, 2026-09-05):
# k=8 из n=300 -> (1.16%, 5.19%); k=2 из n=52 -> (0.47%, 13.21%).
GOLDEN = [
    (8, 300, (0.0116, 0.0519)),
    (9, 300, (0.0138, 0.0562)),
    (10, 300, (0.0161, 0.0604)),
    (2, 52, (0.0047, 0.1321)),
    (6, 248, (0.0089, 0.0519)),
]


@pytest.mark.parametrize("k,n,ожидание", GOLDEN)
def test_клоппер_пирсон_совпадает_со_scipy(k: int, n: int, ожидание: tuple[float, float]) -> None:
    """Точный биномиальный интервал без scipy — числа сверены с scipy.stats.beta.ppf."""
    низ, верх = clopper_pearson(k, n)
    assert низ == pytest.approx(ожидание[0], abs=5e-5)
    assert верх == pytest.approx(ожидание[1], abs=5e-5)


def test_клоппер_пирсон_нулевых_успехов() -> None:
    """k=0: нижняя граница строго 0.0, а не результат бисекции на вырожденной бете."""
    низ, верх = clopper_pearson(0, 100)
    assert низ == 0.0
    assert верх > 0.0


def test_клоппер_пирсон_всех_успехов() -> None:
    """k=n: верхняя граница строго 1.0."""
    низ, верх = clopper_pearson(100, 100)
    assert низ < 1.0
    assert верх == 1.0


def test_клоппер_пирсон_отвергает_k_вне_диапазона() -> None:
    with pytest.raises(ValueError, match="вне диапазона"):
        clopper_pearson(11, 10)


def test_вердикты_вширь_вложены_друг_в_друга() -> None:
    """Три определения утечки — строго нарастающие подмножества, а не независимые наборы."""
    assert LEAK_STRICT < LEAK_WITH_HISTORICAL < LEAK_WITH_PARTIAL


def сделать_выборку() -> pd.DataFrame:
    """Шесть строк, без вердикта: черновик до просмотра человеком."""
    return pd.DataFrame(
        {
            "номер": [1, 2, 3, 4, 5, 6],
            "sha256": [f"sha{n}" for n in range(1, 7)],
            "pageid": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "класс": [
                "lada__2107",
                "lada__2107",
                "kia__rio",
                "kia__rio",
                "uaz__patriot",
                "uaz__patriot",
            ],
            "plates": [1, 0, 1, 1, 0, 1],
            "faces": [0, 0, 0, 0, 0, 0],
            "штамп": ["ok"] * 6,
        }
    )


def сделать_вердикты() -> pd.DataFrame:
    """Два строгих leak, один исторический, один частичный, два чистых."""
    return pd.DataFrame(
        {
            "номер": [1, 2, 3, 4, 5, 6],
            "вердикт": [
                "leak",
                "leak",
                "leak_hist",
                "clean_partial",
                "clean_no_plate",
                "clean_filled",
            ],
            "заметка": ["", "", "", "", "", ""],
        }
    )


def test_свод_роняет_на_кадре_без_вердикта() -> None:
    выборка = сделать_выборку()
    вердикты = сделать_вердикты().iloc[:-1]  # номер 6 без вердикта

    with pytest.raises(ValueError, match="без вердикта"):
        merge_sample_verdicts(выборка, вердикты)


def test_свод_и_подсчёт_утечек_по_трём_определениям() -> None:
    сведено = merge_sample_verdicts(сделать_выборку(), сделать_вердикты())
    отчёт = compute_bound(сведено, PopulationSizes(pool_frames=3000, accepted_frames=6000))

    по_имени = {б.definition: б for б in отчёт.bounds}
    assert по_имени["строго"].leaks == 2
    assert по_имени["с_историческими"].leaks == 3
    assert по_имени["с_частичными"].leaks == 4
    # Перенос на популяцию — доля выборки, умноженная на размер популяции.
    assert по_имени["строго"].pool_estimate == round(3000 * 2 / 6)


def test_поклассовая_страта_по_наличию_номера_в_детекции() -> None:
    сведено = merge_sample_verdicts(сделать_выборку(), сделать_вердикты())
    отчёт = compute_bound(сведено, PopulationSizes(pool_frames=3000, accepted_frames=6000))

    по_страте = {с.stratum: с for с in отчёт.strata}
    # plates=0: номера 2 и 5, среди них один leak (номер 2).
    assert по_страте["plates_0"].seen == 2
    assert по_страте["plates_0"].leaks == 1
    # plates>=1: номера 1,3,4,6, среди них один строгий leak (номер 1).
    assert по_страте["plates_ge1"].seen == 4
    assert по_страте["plates_ge1"].leaks == 1


def test_кадры_утечек_несут_историю_но_не_частичные() -> None:
    """Список кадров для ручной прополки — leak и leak_hist, без clean_partial."""
    сведено = merge_sample_verdicts(сделать_выборку(), сделать_вердикты())
    отчёт = compute_bound(сведено, PopulationSizes(pool_frames=3000, accepted_frames=6000))

    номера = {к.number for к in отчёт.leak_frames}
    assert номера == {1, 2, 3}
    первый = next(к for к in отчёт.leak_frames if к.number == 1)
    assert первый.class_id == "lada__2107"
    assert первый.sha256 == "sha1"


def test_проекция_отбрасывает_рабочие_колонки_черновика() -> None:
    """`project_to_sample_schema` оставляет ровно девять колонок, `crop_path` уходит."""
    сведено = merge_sample_verdicts(сделать_выборку(), сделать_вердикты())
    сведено = сведено.assign(crop_path="lada__2107/a.jpg")  # рабочая колонка черновика

    проекция = project_to_sample_schema(сведено)

    assert list(проекция.columns) == list(SAMPLE_COLUMNS)


def test_проекция_роняет_если_не_хватает_колонки() -> None:
    урезано = сделать_выборку().drop(columns=["класс"])
    with pytest.raises(ValueError, match="не хватает колонок"):
        project_to_sample_schema(урезано)


def test_строгие_вердикты_дают_формат_манифеста() -> None:
    """`strict_leak_verdicts` — вход для `file,verdict` манифеста, только `leak`."""
    сведено = merge_sample_verdicts(сделать_выборку(), сделать_вердикты())
    строгие = strict_leak_verdicts(сведено)

    assert sorted(строгие["sha256"]) == ["sha1", "sha2"]


_ВЫБОРКА = Path(__file__).resolve().parent.parent / "bench" / "plate-leak-sample.csv"


# bench/ не публикуется (см. .gitignore): тест читает реальную выборку, если она лежит локально.
@pytest.mark.skipif(not _ВЫБОРКА.exists(), reason="нет bench/plate-leak-sample.csv")
def test_настоящий_замер_2026_09_05_совпадает_с_golden_отчётом() -> None:
    """Смоук по реальным данным: `bench/plate-leak-sample.csv` уже несёт вердикты.

    Числа сверены с `bound_report.json` того же прогона (посчитан scipy) — не только
    юнит на выдуманных шести строках, а воспроизведение опубликованного результата.
    """
    путь = _ВЫБОРКА
    выборка = pd.read_csv(путь)
    отчёт = compute_bound(выборка, PopulationSizes(pool_frames=2863, accepted_frames=4386))

    assert отчёт.sample_size == 300
    по_имени = {б.definition: б for б in отчёт.bounds}
    строго = по_имени["строго"]
    assert строго.leaks == 8
    assert строго.pool_estimate == 76
    assert строго.pool_interval == (33, 148)
    assert по_имени["с_историческими"].leaks == 9
    assert по_имени["с_частичными"].leaks == 10
    assert len(отчёт.leak_frames) == 9  # leak + leak_hist, без clean_partial
