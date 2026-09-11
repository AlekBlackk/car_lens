"""Явные автоматы состояний рекалибровки с неизменяемыми свидетелями."""

from __future__ import annotations

import re
from collections.abc import Sequence
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


def _require_sha256(name: str, value: str | None) -> str:
    if value is None or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise CalibrationStateError(f"{name}: нужен полный шестнадцатеричный SHA256")
    return value


def _require_id(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise CalibrationStateError(f"{name}: нужен непустой идентификатор")
    return value


def _require_dimension(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationStateError(f"{name}: нужен конкретный неотрицательный int")
    return value


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    """Состояние кампании и SHA256 результатов её доказательных этапов."""

    state: CampaignState
    preregistration_result_sha256: str | None = None
    tooling_result_sha256: str | None = None
    control_result_sha256: str | None = None
    annotations_result_sha256: str | None = None
    profile_evaluation_result_sha256: str | None = None
    selection_result_sha256: str | None = None
    budget_result_sha256: str | None = None
    unresolved_count: int | None = None
    evaluated_survivor_ids: tuple[str, ...] = ()
    selected_profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """Состояние профиля, геометрия и SHA256 результатов проверок."""

    profile_id: str
    state: ProfileState
    profile_sha256: str
    canonical_sha256: str | None = None
    geometry_witness_sha256: str | None = None
    w_max: int | None = None
    h_max: int | None = None
    build_result_sha256: str | None = None
    geometry_result_sha256: str | None = None
    calibration_result_sha256: str | None = None
    visual_veto_result_sha256: str | None = None
    quality_guardrail_result_sha256: str | None = None
    independent_privacy_result_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_id("profile_id", self.profile_id)
        _require_sha256("profile_sha256", self.profile_sha256)


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """Состояние продвижения и SHA256 внешних результатов."""

    state: PromotionState
    profile_id: str | None = None
    profile_sha256: str | None = None
    geometry_witness_sha256: str | None = None
    snapshot_result_sha256: str | None = None
    promotion_result_sha256: str | None = None
    h3_result_sha256: str | None = None
    end_to_end_result_sha256: str | None = None
    full_pool_result_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class BatchRecord:
    """Неизменяемое исполнение партии со свидетелями каждого этапа."""

    number: int
    state: BatchState
    batch_sha256: str
    profile_id: str | None = None
    profile_sha256: str | None = None
    geometry_witness_sha256: str | None = None
    open_result_sha256: str | None = None
    review_result_sha256: str | None = None
    completion_result_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 1:
            raise CalibrationStateError("number: нужен положительный int")
        _require_sha256("batch_sha256", self.batch_sha256)


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
        ProfileState.GEOMETRY_STALE,
        ProfileState.VISUAL_VETO_PASSED,
        ProfileState.VISUAL_VETO_FAILED,
    },
    ProfileState.VISUAL_VETO_PASSED: {
        ProfileState.GEOMETRY_STALE,
        ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.QUALITY_GUARDRAIL_FAILED,
    },
    ProfileState.QUALITY_GUARDRAIL_PASSED: {
        ProfileState.GEOMETRY_STALE,
        ProfileState.INDEPENDENT_PRIVACY_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_FAILED,
    },
    ProfileState.INDEPENDENT_PRIVACY_PASSED: {ProfileState.GEOMETRY_STALE},
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


def _validate_survivors(values: Sequence[str]) -> tuple[str, ...]:
    survivors = tuple(values)
    if not survivors or any(not value.strip() for value in survivors):
        raise CalibrationStateError("survivor IDs должны быть непустыми")
    if len(set(survivors)) != len(survivors):
        raise CalibrationStateError("survivor IDs не должны повторяться")
    return survivors


def transition_campaign(
    record: CampaignRecord,
    target: CampaignState,
    *,
    preregistration_result_sha256: str | None = None,
    tooling_result_sha256: str | None = None,
    control_result_sha256: str | None = None,
    annotations_result_sha256: str | None = None,
    profile_evaluation_result_sha256: str | None = None,
    selection_result_sha256: str | None = None,
    budget_result_sha256: str | None = None,
    unresolved_count: int | None = None,
    evaluated_survivor_ids: Sequence[str] | None = None,
    selected_profile_id: str | None = None,
) -> CampaignRecord:
    """Выполнить один разрешённый переход с контекстом и SHA256 результата."""
    if target not in _CAMPAIGN_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход кампании {record.state.value} → {target.value}"
        )
    if target is CampaignState.PREREGISTERED:
        return replace(
            record,
            state=target,
            preregistration_result_sha256=_require_sha256(
                "preregistration_result_sha256", preregistration_result_sha256
            ),
        )
    if target is CampaignState.TOOLING_VERIFIED:
        return replace(
            record,
            state=target,
            tooling_result_sha256=_require_sha256("tooling_result_sha256", tooling_result_sha256),
        )
    if target is CampaignState.CONTROL_VERIFIED:
        return replace(
            record,
            state=target,
            control_result_sha256=_require_sha256("control_result_sha256", control_result_sha256),
        )
    if target is CampaignState.ANNOTATIONS_RESOLVED:
        if isinstance(unresolved_count, bool) or unresolved_count != 0:
            raise CalibrationStateError("annotations_resolved требует unresolved_count = 0")
        return replace(
            record,
            state=target,
            unresolved_count=0,
            annotations_result_sha256=_require_sha256(
                "annotations_result_sha256", annotations_result_sha256
            ),
        )
    if target is CampaignState.PROFILES_EVALUATED:
        return replace(
            record,
            state=target,
            evaluated_survivor_ids=_validate_survivors(evaluated_survivor_ids or ()),
            profile_evaluation_result_sha256=_require_sha256(
                "profile_evaluation_result_sha256", profile_evaluation_result_sha256
            ),
        )
    if target is CampaignState.SELECTION_REQUIRED:
        survivors = record.evaluated_survivor_ids
        if evaluated_survivor_ids is not None:
            survivors = _validate_survivors(evaluated_survivor_ids)
        if len(survivors) < 2:
            raise CalibrationStateError("selection_required требует несколько survivors")
        return replace(record, state=target, evaluated_survivor_ids=survivors)
    if target is CampaignState.CAMPAIGN_COMPLETE:
        survivors = record.evaluated_survivor_ids
        if record.state is CampaignState.PROFILES_EVALUATED:
            if evaluated_survivor_ids is not None:
                survivors = _validate_survivors(evaluated_survivor_ids)
            if len(survivors) != 1:
                raise CalibrationStateError("прямое завершение требует ровно один survivor")
            selected_profile_id = survivors[0]
        elif selected_profile_id is None or selected_profile_id not in survivors:
            raise CalibrationStateError("выбранный профиль должен входить в survivors")
        return replace(
            record,
            state=target,
            evaluated_survivor_ids=survivors,
            selected_profile_id=selected_profile_id,
            selection_result_sha256=_require_sha256(
                "selection_result_sha256", selection_result_sha256
            ),
        )
    if target is CampaignState.BUDGET_EXHAUSTED:
        return replace(
            record,
            state=target,
            budget_result_sha256=_require_sha256("budget_result_sha256", budget_result_sha256),
        )
    return replace(record, state=target)


def _matches_current_geometry(
    profile: ProfileRecord,
    *,
    current_canonical_sha256: str | None,
    current_geometry_witness_sha256: str | None,
    current_w_max: int | None,
    current_h_max: int | None,
) -> bool:
    canonical = _require_sha256("current_canonical_sha256", current_canonical_sha256)
    witness = _require_sha256("current_geometry_witness_sha256", current_geometry_witness_sha256)
    w_max = _require_dimension("current_w_max", current_w_max)
    h_max = _require_dimension("current_h_max", current_h_max)
    return (
        profile.canonical_sha256 == canonical
        and profile.geometry_witness_sha256 == witness
        and profile.w_max == w_max
        and profile.h_max == h_max
    )


def _require_current_geometry(
    profile: ProfileRecord,
    *,
    current_canonical_sha256: str | None,
    current_geometry_witness_sha256: str | None,
    current_w_max: int | None,
    current_h_max: int | None,
) -> None:
    if not _matches_current_geometry(
        profile,
        current_canonical_sha256=current_canonical_sha256,
        current_geometry_witness_sha256=current_geometry_witness_sha256,
        current_w_max=current_w_max,
        current_h_max=current_h_max,
    ):
        raise CalibrationStateError("профиль не соответствует текущей геометрии")


def transition_profile(
    record: ProfileRecord,
    target: ProfileState,
    *,
    canonical_sha256: str | None = None,
    geometry_witness_sha256: str | None = None,
    w_max: int | None = None,
    h_max: int | None = None,
    current_canonical_sha256: str | None = None,
    current_geometry_witness_sha256: str | None = None,
    current_w_max: int | None = None,
    current_h_max: int | None = None,
    build_result_sha256: str | None = None,
    geometry_result_sha256: str | None = None,
    calibration_result_sha256: str | None = None,
    visual_veto_result_sha256: str | None = None,
    quality_guardrail_result_sha256: str | None = None,
    independent_privacy_result_sha256: str | None = None,
) -> ProfileRecord:
    """Выполнить переход профиля с явной проверкой геометрии и результата."""
    if target not in _PROFILE_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход профиля {record.state.value} → {target.value}"
        )
    if target is ProfileState.BUILT or target is ProfileState.BUILD_FAILED:
        return replace(
            record,
            state=target,
            build_result_sha256=_require_sha256("build_result_sha256", build_result_sha256),
        )
    if target is ProfileState.GEOMETRY_FRESH:
        return replace(
            record,
            state=target,
            canonical_sha256=_require_sha256("canonical_sha256", canonical_sha256),
            geometry_witness_sha256=_require_sha256(
                "geometry_witness_sha256", geometry_witness_sha256
            ),
            w_max=_require_dimension("w_max/h_max", w_max),
            h_max=_require_dimension("w_max/h_max", h_max),
            geometry_result_sha256=_require_sha256(
                "geometry_result_sha256", geometry_result_sha256
            ),
        )
    if target is ProfileState.GEOMETRY_STALE:
        return replace(record, state=target)
    if target is ProfileState.CALIBRATION_PASSED:
        _require_current_geometry(
            record,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
    if target is ProfileState.CALIBRATION_PASSED or target is ProfileState.CALIBRATION_FAILED:
        return replace(
            record,
            state=target,
            calibration_result_sha256=_require_sha256(
                "calibration_result_sha256", calibration_result_sha256
            ),
        )
    if target is ProfileState.VISUAL_VETO_PASSED:
        _require_current_geometry(
            record,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
    if target is ProfileState.VISUAL_VETO_PASSED or target is ProfileState.VISUAL_VETO_FAILED:
        return replace(
            record,
            state=target,
            visual_veto_result_sha256=_require_sha256(
                "visual_veto_result_sha256", visual_veto_result_sha256
            ),
        )
    if target is ProfileState.QUALITY_GUARDRAIL_PASSED:
        _require_current_geometry(
            record,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
    if (
        target is ProfileState.QUALITY_GUARDRAIL_PASSED
        or target is ProfileState.QUALITY_GUARDRAIL_FAILED
    ):
        return replace(
            record,
            state=target,
            quality_guardrail_result_sha256=_require_sha256(
                "quality_guardrail_result_sha256", quality_guardrail_result_sha256
            ),
        )
    if target is ProfileState.INDEPENDENT_PRIVACY_PASSED:
        _require_current_geometry(
            record,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
    return replace(
        record,
        state=target,
        independent_privacy_result_sha256=_require_sha256(
            "independent_privacy_result_sha256", independent_privacy_result_sha256
        ),
    )


def transition_promotion(
    record: PromotionRecord,
    target: PromotionState,
    *,
    profile: ProfileRecord | None = None,
    current_canonical_sha256: str | None = None,
    current_geometry_witness_sha256: str | None = None,
    current_w_max: int | None = None,
    current_h_max: int | None = None,
    snapshot_result_sha256: str | None = None,
    promotion_result_sha256: str | None = None,
    h3_result_sha256: str | None = None,
    end_to_end_result_sha256: str | None = None,
    full_pool_result_sha256: str | None = None,
) -> PromotionRecord:
    """Выполнить переход продвижения и связать проверенный профиль."""
    if target not in _PROMOTION_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход продвижения {record.state.value} → {target.value}"
        )
    if target is PromotionState.PROMOTABLE:
        snapshot = _require_sha256("snapshot_result_sha256", snapshot_result_sha256)
        if profile is None or profile.state is not ProfileState.INDEPENDENT_PRIVACY_PASSED:
            raise CalibrationStateError(
                "promotable разрешён только профилем в independent_privacy_passed"
            )
        _require_current_geometry(
            profile,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
        return replace(
            record,
            state=target,
            profile_id=profile.profile_id,
            profile_sha256=profile.profile_sha256,
            geometry_witness_sha256=profile.geometry_witness_sha256,
            snapshot_result_sha256=snapshot,
        )
    if target is PromotionState.PROMOTED or target is PromotionState.PROMOTION_FAILED:
        return replace(
            record,
            state=target,
            promotion_result_sha256=_require_sha256(
                "promotion_result_sha256", promotion_result_sha256
            ),
        )
    if target is PromotionState.H3_REAUDITED or target is PromotionState.H3_REAUDIT_FAILED:
        return replace(
            record,
            state=target,
            h3_result_sha256=_require_sha256("h3_result_sha256", h3_result_sha256),
        )
    if target is PromotionState.END_TO_END_REPORTED or target is PromotionState.END_TO_END_FAILED:
        return replace(
            record,
            state=target,
            end_to_end_result_sha256=_require_sha256(
                "end_to_end_result_sha256", end_to_end_result_sha256
            ),
        )
    if target is PromotionState.FULL_POOL_AUDITED:
        return replace(
            record,
            state=target,
            full_pool_result_sha256=_require_sha256(
                "full_pool_result_sha256", full_pool_result_sha256
            ),
        )
    return replace(record, state=target)


def _validate_batch_open(
    record: BatchRecord,
    batches: Sequence[BatchRecord] | None,
) -> None:
    if batches is None:
        raise CalibrationStateError("для открытия нужна полная история партий")
    numbers = [batch.number for batch in batches]
    matching = [batch for batch in batches if batch.number == record.number]
    if len(numbers) != len(set(numbers)) or len(matching) != 1 or matching[0] != record:
        raise CalibrationStateError("история партий не содержит ровно одну целевую запись")
    if any(batch.state is BatchState.COMPLETED_PASS for batch in batches):
        raise CalibrationStateError("кампания уже прошла: новые партии не открываются")
    if any(
        batch.number != record.number
        and batch.state in {BatchState.OPENED, BatchState.REVIEW_IN_PROGRESS}
        for batch in batches
    ):
        raise CalibrationStateError("уже существует активная партия")
    sealed = sorted(batch.number for batch in batches if batch.state is BatchState.SEALED)
    if not sealed or record.number != sealed[0]:
        actual = "нет" if not sealed else str(sealed[0])
        raise CalibrationStateError(f"следующая sealed-партия {actual}")
    if record.number > 1 and {
        batch.number for batch in batches if batch.number < record.number
    } != set(range(1, record.number)):
        raise CalibrationStateError("предыдущие партии должны присутствовать без пропусков")
    if any(
        batch.number < record.number and batch.state is not BatchState.COMPLETED_FAIL
        for batch in batches
    ):
        raise CalibrationStateError("предыдущие партии должны завершиться completed_fail")


def transition_batch(
    record: BatchRecord,
    target: BatchState,
    *,
    batches: Sequence[BatchRecord] | None = None,
    profile: ProfileRecord | None = None,
    current_canonical_sha256: str | None = None,
    current_geometry_witness_sha256: str | None = None,
    current_w_max: int | None = None,
    current_h_max: int | None = None,
    open_result_sha256: str | None = None,
    review_result_sha256: str | None = None,
    completion_result_sha256: str | None = None,
) -> BatchRecord:
    """Выполнить переход партии с проверкой общей неизменяемой истории."""
    if target not in _BATCH_TRANSITIONS.get(record.state, set()):
        raise CalibrationStateError(
            f"недопустимый переход партии {record.state.value} → {target.value}"
        )
    if target is BatchState.OPENED:
        open_result = _require_sha256("open_result_sha256", open_result_sha256)
        _validate_batch_open(record, batches)
        # Партия и есть источник independent-вердикта (спека §10.2), поэтому требовать здесь
        # `independent_privacy_passed` значило бы требовать результат самой этой партии.
        if profile is None or profile.state is not ProfileState.QUALITY_GUARDRAIL_PASSED:
            raise CalibrationStateError(
                "партия открывается только профилем в quality_guardrail_passed"
            )
        _require_current_geometry(
            profile,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
        return replace(
            record,
            state=target,
            profile_id=profile.profile_id,
            profile_sha256=profile.profile_sha256,
            geometry_witness_sha256=profile.geometry_witness_sha256,
            open_result_sha256=open_result,
        )
    if target is BatchState.REVIEW_IN_PROGRESS:
        return replace(
            record,
            state=target,
            review_result_sha256=_require_sha256("review_result_sha256", review_result_sha256),
        )
    return replace(
        record,
        state=target,
        completion_result_sha256=_require_sha256(
            "completion_result_sha256", completion_result_sha256
        ),
    )
