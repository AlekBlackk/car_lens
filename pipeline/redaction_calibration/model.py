"""Типизированные результаты одного профильного прогона и его сводка."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

FrameOutcome: TypeAlias = Literal["accepted", "rejected"]
HardGate: TypeAlias = Literal[
    "provenance_workspace_freshness",
    "unresolved_empty",
    "current_geometry_witness",
    "structural_actual_tile_preflight",
    "full_canonical_coverage",
    "strict_calibration_review",
    "report_arithmetic",
]

# Предрегистрированный порядок семи гейтов: отчёт называет первый провалившийся по нему.
HARD_GATES: tuple[HardGate, ...] = (
    "provenance_workspace_freshness",
    "unresolved_empty",
    "current_geometry_witness",
    "structural_actual_tile_preflight",
    "full_canonical_coverage",
    "strict_calibration_review",
    "report_arithmetic",
)

# §7.3 спеки: блокирующие говорят, что числам профиля нельзя верить либо что их нет вовсе.
# Структурный префлайт здесь потому, что негодный по геометрии кандидат отвергается до
# инференса и измеренных чисел у него не появляется, а не потому, что он «хуже».
BLOCKING_GATES: tuple[HardGate, ...] = (
    "provenance_workspace_freshness",
    "unresolved_empty",
    "current_geometry_witness",
    "structural_actual_tile_preflight",
    "report_arithmetic",
)

# Допускающие: числа достоверны, профиль измеряется и попадает на фронт, но непродвигаем.
PERMITTING_GATES: tuple[HardGate, ...] = (
    "full_canonical_coverage",
    "strict_calibration_review",
)


@dataclass(frozen=True, slots=True)
class FrameProfileResult:
    """Пиксельный результат одной входной строки изолированной пересборки."""

    photo_id: str
    class_id: str
    outcome: FrameOutcome
    reject_reason: str | None
    old_black_frac: float | None
    new_black_frac: float | None
    delta_black: float | None
    pixels_equal: bool | None
    audit_preserved: bool


@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    """Стоимость профиля по новым чёрным пикселям, изменениям и отказам."""

    median_delta_black: float
    p95_delta_black: float
    max_delta_black: float
    above_quarter: int
    new_over_redacted: int
    changed_crops: int
    rejects: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ConfidenceMargin:
    """Запас порога детектора до соседней границы и слабейшего истинного номера."""

    to_next_lower: float
    to_min_true: float


@dataclass(frozen=True, slots=True)
class PlateCoverage:
    """Три исхода измерения приватности по каноническим номерам.

    `covered` — номер накрыт объединением заливок своего кадра; `uncovered` — кадр не
    отсеян, но номер не накрыт; `rejected` — кадр отсеян, поэтому номер не относится ни к
    накрытым, ни к непокрытым. Инвариант: `covered + uncovered + rejected` равно числу
    канонических номеров с `verdict == "true_plate"`.
    """

    covered: int
    uncovered: int
    rejected: int

    @property
    def complete(self) -> bool:
        """Единственное определение гейта `full_canonical_coverage`: непокрытых нет.

        Отсеянный номер гейт не роняет: он не утечка, потому что кадра в корпусе нет.
        Поэтому `covered` среди прошедших гейт профилей всё же различается — ровно на
        число отсеянных канонических кадров, и ось приватности это видит.
        """
        return self.uncovered == 0


@dataclass(frozen=True, slots=True)
class ProfileScore:
    """Раздельные Pareto-оси профиля: запасы, приватность и издержки, без скалярной свёртки.

    Ось `covered_plates` считает накрытие истинных канонических номеров заливками профиля
    и сама по себе не доказывает отсутствие строгих утечек: за это отвечает отдельный гейт.
    Профиль с проваленным допускающим гейтом теперь до фронта доходит (§7.3), поэтому ось
    различает и число утечек, а не только число отсеянных канонических кадров.

    Ось `rejected_frames` — предрегистрированная `rejects_by_reason`, свёрнутая в скаляр.
    Без неё отсев не виден ни одной оси издержек: у отсеянной строки `delta_black`,
    `pixels_equal` и `new_black_frac` пусты, и профиль, отсеявший весь корпус, оказался бы
    на фронте с нулями по всем издержкам.
    """

    profile_id: str
    ssd_margin: ConfidenceMargin
    yolo_margin: ConfidenceMargin
    covered_plates: int
    median_delta: float
    p95_delta: float
    above_quarter: int
    new_over_redacted: int
    changed_crops: int
    rejected_frames: int


@dataclass(frozen=True, slots=True)
class FrontierEntry:
    """Строка фронта Pareto вместе с правом на продвижение.

    Провал допускающего гейта не убирает профиль с фронта: его числа достоверны и кампания
    обязана их показать. Продвигать такой профиль по-прежнему нельзя, поэтому право названо
    явно, а не выводится читателем отчёта из отсутствия гейтового поля.
    """

    score: ProfileScore
    failed_permitting_gates: tuple[HardGate, ...]

    @property
    def profile_id(self) -> str:
        return self.score.profile_id

    @property
    def promotable(self) -> bool:
        return not self.failed_permitting_gates


@dataclass(frozen=True, slots=True)
class ProfileGateReport:
    """Факты семи упорядоченных калибровочных гейтов одного профиля."""

    profile_id: str
    provenance_fresh: bool
    unresolved_count: int
    geometry_current: bool
    geometry_preflight_passed: bool
    canonical_coverage_complete: bool
    strict_leaks: int
    unreviewable_accepted: int
    report_arithmetic_closed: bool
    operational_errors: int = 0
    canonical_sha256: str | None = None
    w_max: int | None = None
    h_max: int | None = None
    oversized_blind_zone_count: int = 0
    geometry_preflight_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GeometryGateVerdict:
    """Разделённый вердикт геометрического префлайта подтайловых и надтайловых номеров."""

    passed: bool
    subtile_passed: bool
    oversized_passed: bool
    oversized_count: int
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.passed


GeometryPreflightVerdict = GeometryGateVerdict


@dataclass(frozen=True, slots=True)
class VisualVeto:
    """Трассируемая дисквалификация профиля по уничтоженному признаку модели."""

    profile_id: str
    sha256: str | None
    plate_id: str | None
    destroyed_feature: str
    reason: str
