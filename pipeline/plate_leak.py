"""Оценка частоты утечки номера в обучающей выборке (§7 `bench/plate-leak-bound.md`).

Замер отвечает на один вопрос: сколько кадров с читаемым регистрационным номером
попадёт в обучение, если учить по сплиту как есть. Ответ — не точное число, а доверительный
интервал: выборка в 300 кадров не смотрит всю популяцию, и точечная оценка без интервала
читалась бы как гарантия там, где её нет.

Схема данных — `номер, sha256, pageid, класс, plates, faces, штамп, вердикт, заметка`,
как в уже собранном `bench/plate-leak-sample.csv` (§11 отчёта): `штамп` — отметка `audit`
манифеста на момент замера, `вердикт` — суждение человека по одному из шести значений,
`plates`/`faces` — числа детектора, по которым строятся страты (штамп известен и без
просмотра, а плашка — нет).

Три вердикта вширь, а не один: `LEAK_STRICT` — только однозначно читаемый номер,
`LEAK_WITH_HISTORICAL` добавляет исторические/архивные снимки, `LEAK_WITH_PARTIAL` —
ещё и частично заретушированные. Каждое следующее множество — надмножество предыдущего, и
дальше по конвейеру идёт только строгое: историческую утечку решает человек, а не эта оценка.

Интервал Клоппера–Пирсона считается без scipy: он не в базовых зависимостях проекта, а
переизобретать регуляризованную неполную бета-функцию стоит одного дня и проверяется точно —
`_regularized_incomplete_beta`/`_beta_ppf` сверены с `scipy.stats.beta.ppf` (числа замера
2026-09-05, `tests/test_plate_leak.py`) и совпадают до пятого знака. Модуль торч не
импортирует и тестируется в базовом окружении.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "LEAK_DEFINITIONS",
    "LEAK_STRICT",
    "LEAK_WITH_HISTORICAL",
    "LEAK_WITH_PARTIAL",
    "SAMPLE_COLUMNS",
    "BoundReport",
    "LeakBound",
    "LeakFrame",
    "PopulationSizes",
    "StratumBound",
    "clopper_pearson",
    "compute_bound",
    "merge_sample_verdicts",
    "project_to_sample_schema",
    "strict_leak_verdicts",
]

# Схема `bench/plate-leak-sample.csv` (§11 отчёта). Черновик до просмотра (из
# `sample_plate_leak.py`) может нести и другие колонки (например `crop_path` — как найти
# файл на диске), но зафиксированный артефакт несёт ровно эти девять.
SAMPLE_COLUMNS: tuple[str, ...] = (
    "номер",
    "sha256",
    "pageid",
    "класс",
    "plates",
    "faces",
    "штамп",
    "вердикт",
    "заметка",
)

LEAK_STRICT = frozenset({"leak"})
LEAK_WITH_HISTORICAL = LEAK_STRICT | {"leak_hist"}
LEAK_WITH_PARTIAL = LEAK_WITH_HISTORICAL | {"clean_partial"}

LEAK_DEFINITIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("строго", LEAK_STRICT),
    ("с_историческими", LEAK_WITH_HISTORICAL),
    ("с_частичными", LEAK_WITH_PARTIAL),
)


def _continued_fraction(x: float, a: float, b: float, iterations: int = 200) -> float:
    """Непрерывная дробь Лентца для `I_x(a, b)` (Numerical Recipes §6.4)."""
    tiny = 1e-300
    eps = 1e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        числитель = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + числитель * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + числитель / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        числитель = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + числитель * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + числитель / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        приращение = d * c
        h *= приращение
        if abs(приращение - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """`I_x(a, b)` — регуляризованная неполная бета-функция, CDF бета-распределения."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_бета = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    множитель = math.exp(ln_бета + a * math.log(x) + b * math.log(1.0 - x))
    # Дробь Лентца сходится быстро только на одной половине области; для другой
    # используется симметрия I_x(a,b) = 1 - I_{1-x}(b,a).
    if x < (a + 1.0) / (a + b + 2.0):
        return множитель * _continued_fraction(x, a, b) / a
    return 1.0 - множитель * _continued_fraction(1.0 - x, b, a) / b


def _beta_ppf(p: float, a: float, b: float, tol: float = 1e-12) -> float:
    """Квантиль бета-распределения: бисекция по CDF, монотонной на `[0, 1]`."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    низ, верх = 0.0, 1.0
    for _ in range(200):
        середина = (низ + верх) / 2.0
        if _regularized_incomplete_beta(середина, a, b) < p:
            низ = середина
        else:
            верх = середина
        if верх - низ < tol:
            break
    return (низ + верх) / 2.0


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Точный биномиальный интервал (доли `0..1`) для `k` успехов из `n`.

    `k=0` и `k=n` — вырожденные концы: бисекция на границе бета-распределения теряет
    точность, а точный ответ известен без неё (`0.0` и `1.0` соответственно).
    """
    if not 0 <= k <= n:
        raise ValueError(f"k={k} вне диапазона 0..{n}")
    if n <= 0:
        raise ValueError(f"n={n} должно быть положительным")
    альфа = 1.0 - confidence
    низ = 0.0 if k == 0 else _beta_ppf(альфа / 2.0, k, n - k + 1)
    верх = 1.0 if k == n else _beta_ppf(1.0 - альфа / 2.0, k + 1, n - k)
    return низ, верх


@dataclass(frozen=True, slots=True)
class PopulationSizes:
    """Размер сплита и принятого манифеста на момент замера.

    Числа читаются из файлов при запуске, а не зашиваются литералом: зашитое число —
    та самая мина, из-за которой прежний замер молча путал сплит `pool` до и после
    прополки (`bench/plate-leak-bound.md`, §11).
    """

    pool_frames: int
    accepted_frames: int


@dataclass(frozen=True, slots=True)
class LeakBound:
    """Оценка по одному определению утечки: доля в выборке и её перенос на популяцию."""

    definition: str
    leaks: int
    sample_size: int
    share: float
    ci: tuple[float, float]
    pool_estimate: int
    pool_interval: tuple[int, int]
    accepted_interval: tuple[int, int]


@dataclass(frozen=True, slots=True)
class StratumBound:
    """Оценка внутри страты — по признаку, доступному до просмотра (здесь: `plates`)."""

    stratum: str
    seen: int
    leaks: int
    share: float
    ci: tuple[float, float]


@dataclass(frozen=True, slots=True)
class LeakFrame:
    """Один найденный кадр утечки — для ручной прополки и для сайдкара провенанса."""

    number: int
    verdict: str
    pageid: str
    class_id: str
    plates: int
    faces: int
    sha256: str
    note: str


@dataclass(frozen=True, slots=True)
class BoundReport:
    """Отчёт замера: сколько посмотрено, что нашли, что это значит для популяции."""

    sample_size: int
    verdict_counts: dict[str, int]
    bounds: tuple[LeakBound, ...]
    strata: tuple[StratumBound, ...]
    leak_frames: tuple[LeakFrame, ...]


def merge_sample_verdicts(sample: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Свести черновик выборки (без `вердикт`) с суждениями человека по `номер`.

    `validate="1:1"` — своя строка вердикта на каждый номер выборки, ни дублей, ни
    пропажи. Пропущенный вердикт называется по номеру: тихая утечка `NaN` дальше по
    конвейеру превратилась бы в строку `"nan"`, которая ни в одно множество не попадает
    и просто не считается утечкой — то есть занижает оценку молча.
    """
    объединено = sample.merge(verdicts, on="номер", how="left", validate="1:1")
    без_вердикта = объединено[объединено["вердикт"].isna()]
    if not без_вердикта.empty:
        номера = ", ".join(str(n) for n in без_вердикта["номер"].tolist()[:10])
        raise ValueError(f"кадры без вердикта: {номера}")
    return объединено


def compute_bound(
    merged: pd.DataFrame, population: PopulationSizes, *, confidence: float = 0.95
) -> BoundReport:
    """Посчитать доверительные интервалы по трём определениям утечки и по стратам.

    `merged` — одна строка на кадр выборки с колонками `номер, sha256, pageid, класс,
    plates, faces, вердикт` и опционально `заметка`: ровно схема
    `bench/plate-leak-sample.csv` (§11 отчёта).
    """
    n = len(merged)
    if n == 0:
        raise ValueError("выборка пуста: интервал не по чему считать")

    границы = []
    for имя, набор in LEAK_DEFINITIONS:
        k = int(merged["вердикт"].isin(набор).sum())
        низ, верх = clopper_pearson(k, n, confidence)
        границы.append(
            LeakBound(
                definition=имя,
                leaks=k,
                sample_size=n,
                share=k / n,
                ci=(низ, верх),
                pool_estimate=round(population.pool_frames * k / n),
                pool_interval=(
                    round(population.pool_frames * низ),
                    round(population.pool_frames * верх),
                ),
                accepted_interval=(
                    round(population.accepted_frames * низ),
                    round(population.accepted_frames * верх),
                ),
            )
        )

    страты = []
    for метка, срез in (
        ("plates_0", merged[merged["plates"] == 0]),
        ("plates_ge1", merged[merged["plates"] > 0]),
    ):
        пусто = len(срез) == 0
        k = int(срез["вердикт"].isin(LEAK_STRICT).sum())
        низ, верх = (0.0, 0.0) if пусто else clopper_pearson(k, len(срез), confidence)
        страты.append(
            StratumBound(
                stratum=метка,
                seen=len(срез),
                leaks=k,
                share=0.0 if пусто else k / len(срез),
                ci=(низ, верх),
            )
        )

    утечки = merged[merged["вердикт"].isin(LEAK_WITH_HISTORICAL)]
    заметки = утечки["заметка"] if "заметка" in merged.columns else [None] * len(утечки)
    кадры = tuple(
        LeakFrame(
            number=int(номер),
            verdict=str(вердикт),
            pageid=str(pageid),
            class_id=str(класс),
            plates=int(plates),
            faces=int(faces),
            sha256=str(sha256),
            note=str(заметка) if pd.notna(заметка) else "",
        )
        for номер, sha256, pageid, класс, plates, faces, вердикт, заметка in zip(
            утечки["номер"],
            утечки["sha256"],
            утечки["pageid"],
            утечки["класс"],
            утечки["plates"],
            утечки["faces"],
            утечки["вердикт"],
            заметки,
            strict=True,
        )
    )

    return BoundReport(
        sample_size=n,
        verdict_counts={str(k): int(v) for k, v in merged["вердикт"].value_counts().items()},
        bounds=tuple(границы),
        strata=tuple(страты),
        leak_frames=кадры,
    )


def project_to_sample_schema(merged: pd.DataFrame) -> pd.DataFrame:
    """Свести к ровно девяти колонкам `SAMPLE_COLUMNS` — тому, что уходит в git.

    Черновик (`sample_plate_leak.py`) несёт `crop_path` для поиска файла на диске;
    итоговый закоммиченный `bench/plate-leak-sample.csv` его не несёт вовсе — колонка
    не в §11 отчёта, и лишнее поле в зафиксированном артефакте плодило бы вопрос,
    зачем оно там, при следующем чтении.
    """
    отсутствуют = [к for к in SAMPLE_COLUMNS if к not in merged.columns]
    if отсутствуют:
        raise ValueError(f"после свода не хватает колонок: {', '.join(отсутствуют)}")
    return merged[list(SAMPLE_COLUMNS)]


def strict_leak_verdicts(merged: pd.DataFrame) -> pd.DataFrame:
    """Строки со строгим вердиктом утечки — вход для `file,verdict` манифеста.

    Только `LEAK_STRICT`: историческую и частичную утечку решает человек отдельно
    (§7 `bench/plate-leak-bound.md`), автоматической прополке они не подлежат.
    """
    return merged[merged["вердикт"].isin(LEAK_STRICT)]
