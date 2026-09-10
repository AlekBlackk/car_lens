"""Явные автоматы состояний рекалибровки с неизменяемыми свидетелями."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class CalibrationStateError(ValueError):
    """Переход запрещён автоматом или не подтверждён свидетелем."""


class CampaignState(StrEnum):
    DRAFT = "draft"
    PREREGISTERED = "preregistered"
    TOOLING_VERIFIED = "tooling_verified"
    CONTROL_VERIFIED = "control_verified"
    ANNOTATIONS_OPEN = "annotations_open"
    ANNOTATIONS_RESOLVED = "annotations_resolved"
    PROFILES_EVALUATED = "profiles_evaluated"
    SELECTION_REQUIRED = "selection_required"
    CAMPAIGN_COMPLETE = "campaign_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ProfileState(StrEnum):
    DECLARED = "declared"
    BUILT = "built"
    BUILD_FAILED = "build_failed"
    GEOMETRY_FRESH = "geometry_fresh"
    GEOMETRY_STALE = "geometry_stale"
    CALIBRATION_PASSED = "calibration_passed"
    CALIBRATION_FAILED = "calibration_failed"
    VISUAL_VETO_PASSED = "visual_veto_passed"
    VISUAL_VETO_FAILED = "visual_veto_failed"
    QUALITY_GUARDRAIL_PASSED = "quality_guardrail_passed"
    QUALITY_GUARDRAIL_FAILED = "quality_guardrail_failed"
    INDEPENDENT_PRIVACY_PASSED = "independent_privacy_passed"
    INDEPENDENT_PRIVACY_FAILED = "independent_privacy_failed"


class PromotionState(StrEnum):
    NOT_SELECTED = "not_selected"
    PROMOTABLE = "promotable"
    PROMOTED = "promoted"
    PROMOTION_FAILED = "promotion_failed"
    H3_REAUDIT_PENDING = "h3_reaudit_pending"
    H3_REAUDITED = "h3_reaudited"
    H3_REAUDIT_FAILED = "h3_reaudit_failed"
    END_TO_END_REPORTED = "end_to_end_reported"
    END_TO_END_FAILED = "end_to_end_failed"
    FULL_POOL_AUDITED = "full_pool_audited"


class BatchState(StrEnum):
    SEALED = "sealed"
    OPENED = "opened"
    REVIEW_IN_PROGRESS = "review_in_progress"
    COMPLETED_PASS = "completed_pass"
    COMPLETED_FAIL = "completed_fail"


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    """Текущее состояние кампании."""

    state: CampaignState


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """Состояние профиля и геометрический свидетель, который его разрешил."""

    profile_id: str
    state: ProfileState
    profile_sha256: str
    canonical_sha256: str | None = None
    geometry_witness_sha256: str | None = None
    w_max: int | None = None
    h_max: int | None = None


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """Состояние продвижения выбранного профиля."""

    state: PromotionState
    profile_id: str | None = None
    profile_sha256: str | None = None
    geometry_witness_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class BatchRecord:
    """Неизменяемое исполнение партии с профилем и свидетелем открытия."""

    number: int
    state: BatchState
    batch_sha256: str
    profile_id: str | None = None
    profile_sha256: str | None = None
    geometry_witness_sha256: str | None = None


_CAMPAIGN_TRANSITIONS = {
    CampaignState.DRAFT: {CampaignState.PREREGISTERED},
    CampaignState.PREREGISTERED: {CampaignState.TOOLING_VERIFIED},
    CampaignState.TOOLING_VERIFIED: {CampaignState.CONTROL_VERIFIED},
    CampaignState.CONTROL_VERIFIED: {CampaignState.ANNOTATIONS_OPEN},
    CampaignState.ANNOTATIONS_OPEN: {CampaignState.ANNOTATIONS_RESOLVED},
    CampaignState.ANNOTATIONS_RESOLVED: {
        CampaignState.ANNOTATIONS_OPEN,
        CampaignState.PROFILES_EVALUATED,
    },
    CampaignState.PROFILES_EVALUATED: {
        CampaignState.SELECTION_REQUIRED,
        CampaignState.CAMPAIGN_COMPLETE,
    },
    CampaignState.SELECTION_REQUIRED: {
        CampaignState.CAMPAIGN_COMPLETE,
        CampaignState.BUDGET_EXHAUSTED,
    },
}

_PROFILE_TRANSITIONS = {
    ProfileState.DECLARED: {ProfileState.BUILT, ProfileState.BUILD_FAILED},
    ProfileState.BUILT: {ProfileState.GEOMETRY_FRESH, ProfileState.GEOMETRY_STALE},
    ProfileState.GEOMETRY_FRESH: {
        ProfileState.GEOMETRY_STALE,
        ProfileState.CALIBRATION_PASSED,
        ProfileState.CALIBRATION_FAILED,
    },
    ProfileState.CALIBRATION_PASSED: {
        ProfileState.VISUAL_VETO_PASSED,
        ProfileState.VISUAL_VETO_FAILED,
    },
    ProfileState.VISUAL_VETO_PASSED: {
        ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.QUALITY_GUARDRAIL_FAILED,
    },
    ProfileState.QUALITY_GUARDRAIL_PASSED: {
        ProfileState.INDEPENDENT_PRIVACY_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_FAILED,
    },
}

_PROMOTION_TRANSITIONS = {
    PromotionState.NOT_SELECTED: {PromotionState.PROMOTABLE},
    PromotionState.PROMOTABLE: {
        PromotionState.PROMOTED,
        PromotionState.PROMOTION_FAILED,
    },
    PromotionState.PROMOTED: {PromotionState.H3_REAUDIT_PENDING},
    PromotionState.H3_REAUDIT_PENDING: {
        PromotionState.H3_REAUDITED,
        PromotionState.H3_REAUDIT_FAILED,
    },
    PromotionState.H3_REAUDITED: {
        PromotionState.END_TO_END_REPORTED,
        PromotionState.END_TO_END_FAILED,
    },
    PromotionState.END_TO_END_REPORTED: {PromotionState.FULL_POOL_AUDITED},
}

_BATCH_TRANSITIONS = {
    BatchState.SEALED: {BatchState.OPENED},
    BatchState.OPENED: {BatchState.REVIEW_IN_PROGRESS},
    BatchState.REVIEW_IN_PROGRESS: {
        BatchState.COMPLETED_PASS,
        BatchState.COMPLETED_FAIL,
    },
}


def transition_campaign(record: CampaignRecord, target: CampaignState) -> CampaignRecord:
    """Выполнить один явно разрешённый переход кампании."""
    if target not in _CAMPAIGN_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход кампании {record.state.value} → {target.value}"
        )
    return replace(record, state=target)


def transition_profile(
    record: ProfileRecord,
    target: ProfileState,
    *,
    canonical_sha256: str | None = None,
    geometry_witness_sha256: str | None = None,
    w_max: int | None = None,
    h_max: int | None = None,
) -> ProfileRecord:
    """Выполнить переход профиля, проверяя новый геометрический свидетель."""
    if target not in _PROFILE_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход профиля {record.state.value} → {target.value}"
        )
    if target is not ProfileState.GEOMETRY_FRESH:
        return replace(record, state=target)
    if (
        canonical_sha256 != record.canonical_sha256
        or w_max != record.w_max
        or h_max != record.h_max
        or geometry_witness_sha256 is None
    ):
        raise CalibrationStateError(
            "геометрический свидетель не совпадает с canonical SHA256 и точными W_max/H_max"
        )
    return replace(
        record,
        state=target,
        geometry_witness_sha256=geometry_witness_sha256,
    )


def transition_promotion(
    record: PromotionRecord,
    target: PromotionState,
    *,
    profile: ProfileRecord | None = None,
) -> PromotionRecord:
    """Выполнить переход продвижения и привязать разрешающий профиль."""
    if target not in _PROMOTION_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход продвижения {record.state.value} → {target.value}"
        )
    if target is not PromotionState.PROMOTABLE:
        return replace(record, state=target)
    if profile is None or profile.state is not ProfileState.INDEPENDENT_PRIVACY_PASSED:
        raise CalibrationStateError(
            "promotable разрешён только профилем в independent_privacy_passed"
        )
    if profile.geometry_witness_sha256 is None:
        raise CalibrationStateError("у профиля нет геометрического свидетеля")
    return replace(
        record,
        state=target,
        profile_id=profile.profile_id,
        profile_sha256=profile.profile_sha256,
        geometry_witness_sha256=profile.geometry_witness_sha256,
    )


def transition_batch(
    record: BatchRecord,
    target: BatchState,
    *,
    profile: ProfileRecord | None = None,
) -> BatchRecord:
    """Выполнить переход партии и запечатать свидетеля при раскрытии."""
    if target not in _BATCH_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход партии {record.state.value} → {target.value}"
        )
    if target is not BatchState.OPENED:
        return replace(record, state=target)
    if profile is None or profile.geometry_witness_sha256 is None:
        raise CalibrationStateError("для открытия партии нужен профиль со свидетелем")
    return replace(
        record,
        state=target,
        profile_id=profile.profile_id,
        profile_sha256=profile.profile_sha256,
        geometry_witness_sha256=profile.geometry_witness_sha256,
    )
