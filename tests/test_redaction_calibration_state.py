"""Контекстные переходы рекалибровки и неизменяемые свидетели."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest

from pipeline.redaction_calibration.state import (
    BatchRecord,
    BatchState,
    CalibrationStateError,
    CampaignRecord,
    CampaignState,
    ProfileRecord,
    ProfileState,
    PromotionRecord,
    PromotionState,
    campaign_budget_state,
    transition_batch,
    transition_campaign,
    transition_profile,
    transition_promotion,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
INVALID_HASHES = (None, "", "a" * 63, "g" * 64)

CAMPAIGN_EDGES = {
    (CampaignState.DRAFT, CampaignState.PREREGISTERED),
    (CampaignState.PREREGISTERED, CampaignState.TOOLING_VERIFIED),
    (CampaignState.TOOLING_VERIFIED, CampaignState.CONTROL_VERIFIED),
    (CampaignState.CONTROL_VERIFIED, CampaignState.ANNOTATIONS_OPEN),
    (CampaignState.ANNOTATIONS_OPEN, CampaignState.ANNOTATIONS_RESOLVED),
    (CampaignState.ANNOTATIONS_RESOLVED, CampaignState.ANNOTATIONS_OPEN),
    (CampaignState.ANNOTATIONS_RESOLVED, CampaignState.PROFILES_EVALUATED),
    (CampaignState.PROFILES_EVALUATED, CampaignState.SELECTION_REQUIRED),
    (CampaignState.PROFILES_EVALUATED, CampaignState.CAMPAIGN_COMPLETE),
    (CampaignState.SELECTION_REQUIRED, CampaignState.CAMPAIGN_COMPLETE),
    (CampaignState.SELECTION_REQUIRED, CampaignState.BUDGET_EXHAUSTED),
    # Единственный выживший профиль тоже способен сжечь бюджет: без этого ребра кампания
    # уходила бы в campaign_complete с проваленным профилем в selected_profile_id.
    (CampaignState.PROFILES_EVALUATED, CampaignState.BUDGET_EXHAUSTED),
    # Дефект Д-3: нулевой исход не имел законного ребра, и кампания 2026-09-07 осталась в
    # preregistered, закрытая отчётом. Бюджет партий здесь нетронут.
    (CampaignState.PROFILES_EVALUATED, CampaignState.NO_ELIGIBLE_PROFILES),
}

PROFILE_EDGES = {
    (ProfileState.DECLARED, ProfileState.BUILT),
    (ProfileState.DECLARED, ProfileState.BUILD_FAILED),
    (ProfileState.BUILT, ProfileState.GEOMETRY_FRESH),
    (ProfileState.BUILT, ProfileState.GEOMETRY_STALE),
    (ProfileState.GEOMETRY_FRESH, ProfileState.GEOMETRY_STALE),
    (ProfileState.GEOMETRY_FRESH, ProfileState.CALIBRATION_PASSED),
    (ProfileState.GEOMETRY_FRESH, ProfileState.CALIBRATION_FAILED),
    (ProfileState.CALIBRATION_PASSED, ProfileState.GEOMETRY_STALE),
    (ProfileState.CALIBRATION_PASSED, ProfileState.VISUAL_VETO_PASSED),
    (ProfileState.CALIBRATION_PASSED, ProfileState.VISUAL_VETO_FAILED),
    (ProfileState.VISUAL_VETO_PASSED, ProfileState.GEOMETRY_STALE),
    (ProfileState.VISUAL_VETO_PASSED, ProfileState.QUALITY_GUARDRAIL_PASSED),
    (ProfileState.VISUAL_VETO_PASSED, ProfileState.QUALITY_GUARDRAIL_FAILED),
    (ProfileState.QUALITY_GUARDRAIL_PASSED, ProfileState.GEOMETRY_STALE),
    (ProfileState.QUALITY_GUARDRAIL_PASSED, ProfileState.INDEPENDENT_PRIVACY_PASSED),
    (ProfileState.QUALITY_GUARDRAIL_PASSED, ProfileState.INDEPENDENT_PRIVACY_FAILED),
    (ProfileState.INDEPENDENT_PRIVACY_PASSED, ProfileState.GEOMETRY_STALE),
}

PROMOTION_EDGES = {
    (PromotionState.NOT_SELECTED, PromotionState.PROMOTABLE),
    (PromotionState.PROMOTABLE, PromotionState.PROMOTED),
    (PromotionState.PROMOTABLE, PromotionState.PROMOTION_FAILED),
    (PromotionState.PROMOTED, PromotionState.H3_REAUDIT_PENDING),
    (PromotionState.H3_REAUDIT_PENDING, PromotionState.H3_REAUDITED),
    (PromotionState.H3_REAUDIT_PENDING, PromotionState.H3_REAUDIT_FAILED),
    (PromotionState.H3_REAUDITED, PromotionState.END_TO_END_REPORTED),
    (PromotionState.H3_REAUDITED, PromotionState.END_TO_END_FAILED),
    (PromotionState.END_TO_END_REPORTED, PromotionState.FULL_POOL_AUDITED),
}

BATCH_EDGES = {
    (BatchState.SEALED, BatchState.OPENED),
    (BatchState.OPENED, BatchState.REVIEW_IN_PROGRESS),
    (BatchState.REVIEW_IN_PROGRESS, BatchState.COMPLETED_PASS),
    (BatchState.REVIEW_IN_PROGRESS, BatchState.COMPLETED_FAIL),
}

PROFILE_RESULT_FIELDS = {
    ProfileState.BUILT: "build_result_sha256",
    ProfileState.BUILD_FAILED: "build_result_sha256",
    ProfileState.GEOMETRY_FRESH: "geometry_result_sha256",
    ProfileState.CALIBRATION_PASSED: "calibration_result_sha256",
    ProfileState.CALIBRATION_FAILED: "calibration_result_sha256",
    ProfileState.VISUAL_VETO_PASSED: "visual_veto_result_sha256",
    ProfileState.VISUAL_VETO_FAILED: "visual_veto_result_sha256",
    ProfileState.QUALITY_GUARDRAIL_PASSED: "quality_guardrail_result_sha256",
    ProfileState.QUALITY_GUARDRAIL_FAILED: "quality_guardrail_result_sha256",
    ProfileState.INDEPENDENT_PRIVACY_PASSED: "independent_privacy_result_sha256",
    ProfileState.INDEPENDENT_PRIVACY_FAILED: "independent_privacy_result_sha256",
}

CAMPAIGN_RESULT_FIELDS = {
    CampaignState.PREREGISTERED: "preregistration_result_sha256",
    CampaignState.TOOLING_VERIFIED: "tooling_result_sha256",
    CampaignState.CONTROL_VERIFIED: "control_result_sha256",
    CampaignState.ANNOTATIONS_RESOLVED: "annotations_result_sha256",
    CampaignState.PROFILES_EVALUATED: "profile_evaluation_result_sha256",
    CampaignState.CAMPAIGN_COMPLETE: "selection_result_sha256",
    CampaignState.BUDGET_EXHAUSTED: "budget_result_sha256",
    CampaignState.NO_ELIGIBLE_PROFILES: "no_eligible_profiles_result_sha256",
}

PROMOTION_RESULT_FIELDS = {
    PromotionState.PROMOTABLE: "snapshot_result_sha256",
    PromotionState.PROMOTED: "promotion_result_sha256",
    PromotionState.PROMOTION_FAILED: "promotion_result_sha256",
    PromotionState.H3_REAUDITED: "h3_result_sha256",
    PromotionState.H3_REAUDIT_FAILED: "h3_result_sha256",
    PromotionState.END_TO_END_REPORTED: "end_to_end_result_sha256",
    PromotionState.END_TO_END_FAILED: "end_to_end_result_sha256",
    PromotionState.FULL_POOL_AUDITED: "full_pool_result_sha256",
}

BATCH_RESULT_FIELDS = {
    BatchState.OPENED: "open_result_sha256",
    BatchState.REVIEW_IN_PROGRESS: "review_result_sha256",
    BatchState.COMPLETED_PASS: "completion_result_sha256",
    BatchState.COMPLETED_FAIL: "completion_result_sha256",
}


def _campaign(state: CampaignState) -> CampaignRecord:
    survivors = {
        CampaignState.PROFILES_EVALUATED: ("profile-a", "profile-b"),
        CampaignState.SELECTION_REQUIRED: ("profile-a", "profile-b"),
    }.get(state, ())
    return CampaignRecord(
        state=state,
        preregistration_result_sha256=SHA_A,
        tooling_result_sha256=SHA_A,
        control_result_sha256=SHA_A,
        annotations_result_sha256=SHA_A,
        profile_evaluation_result_sha256=SHA_A,
        selection_result_sha256=SHA_A,
        budget_result_sha256=SHA_A,
        no_eligible_profiles_result_sha256=SHA_A,
        evaluated_survivor_ids=survivors,
    )


def _profile(state: ProfileState, *, profile_id: str = "profile-a") -> ProfileRecord:
    return ProfileRecord(
        profile_id=profile_id,
        state=state,
        profile_sha256=SHA_A,
        canonical_sha256=SHA_B,
        geometry_witness_sha256=SHA_C,
        w_max=17,
        h_max=11,
        build_result_sha256=SHA_D,
        geometry_result_sha256=SHA_E,
        calibration_result_sha256=SHA_F,
        visual_veto_result_sha256=SHA_D,
        quality_guardrail_result_sha256=SHA_E,
        independent_privacy_result_sha256=SHA_F,
    )


def _promotion(state: PromotionState) -> PromotionRecord:
    selected = state is not PromotionState.NOT_SELECTED
    return PromotionRecord(
        state=state,
        profile_id="profile-a" if selected else None,
        profile_sha256=SHA_A if selected else None,
        geometry_witness_sha256=SHA_C if selected else None,
        snapshot_result_sha256=SHA_D if selected else None,
        promotion_result_sha256=SHA_E if selected else None,
        h3_result_sha256=SHA_F if selected else None,
        end_to_end_result_sha256=SHA_D if selected else None,
        full_pool_result_sha256=SHA_E if selected else None,
    )


def _batch(state: BatchState, number: int = 1) -> BatchRecord:
    opened = state is not BatchState.SEALED
    return BatchRecord(
        number=number,
        state=state,
        batch_sha256=SHA_A,
        profile_id="profile-a" if opened else None,
        profile_sha256=SHA_A if opened else None,
        geometry_witness_sha256=SHA_C if opened else None,
        open_result_sha256=SHA_D if opened else None,
        review_result_sha256=SHA_E if opened else None,
        completion_result_sha256=SHA_F if opened else None,
    )


def _assert_campaign_transition(record: CampaignRecord, target: CampaignState) -> CampaignRecord:
    if target is CampaignState.PREREGISTERED:
        return transition_campaign(record, target, preregistration_result_sha256=SHA_B)
    if target is CampaignState.TOOLING_VERIFIED:
        return transition_campaign(record, target, tooling_result_sha256=SHA_B)
    if target is CampaignState.CONTROL_VERIFIED:
        return transition_campaign(record, target, control_result_sha256=SHA_B)
    if target is CampaignState.ANNOTATIONS_RESOLVED:
        return transition_campaign(
            record,
            target,
            unresolved_count=0,
            annotations_result_sha256=SHA_B,
        )
    if target is CampaignState.PROFILES_EVALUATED:
        return transition_campaign(
            record,
            target,
            evaluated_survivor_ids=("profile-a", "profile-b"),
            profile_evaluation_result_sha256=SHA_B,
        )
    if target is CampaignState.SELECTION_REQUIRED:
        return transition_campaign(
            record,
            target,
            evaluated_survivor_ids=("profile-a", "profile-b"),
        )
    if target is CampaignState.CAMPAIGN_COMPLETE:
        if record.state is CampaignState.PROFILES_EVALUATED:
            return transition_campaign(
                record,
                target,
                evaluated_survivor_ids=("profile-a",),
                selection_result_sha256=SHA_B,
            )
        return transition_campaign(
            record,
            target,
            selected_profile_id=(
                record.evaluated_survivor_ids[0] if record.evaluated_survivor_ids else "profile-a"
            ),
            selection_result_sha256=SHA_B,
        )
    if target is CampaignState.BUDGET_EXHAUSTED:
        return transition_campaign(
            record,
            target,
            batches=_exhausted_batches(),
            budget_result_sha256=SHA_B,
        )
    if target is CampaignState.NO_ELIGIBLE_PROFILES:
        # Предусловие терминала — пустая запись, а не пустой аргумент: подменять им список
        # выживших ребро больше не даёт
        return transition_campaign(
            replace(record, evaluated_survivor_ids=()),
            target,
            evaluated_survivor_ids=(),
            no_eligible_profiles_result_sha256=SHA_B,
        )
    return transition_campaign(record, target)


def _exhausted_batches() -> tuple[BatchRecord, ...]:
    return tuple(_batch(BatchState.COMPLETED_FAIL, number) for number in range(1, 9))


def _assert_profile_transition(record: ProfileRecord, target: ProfileState) -> ProfileRecord:
    if target is ProfileState.BUILT or target is ProfileState.BUILD_FAILED:
        return transition_profile(record, target, build_result_sha256=SHA_D)
    if target is ProfileState.GEOMETRY_FRESH:
        return transition_profile(
            record,
            target,
            canonical_sha256=SHA_B,
            geometry_witness_sha256=SHA_C,
            w_max=17,
            h_max=11,
            geometry_result_sha256=SHA_D,
        )
    if target is ProfileState.CALIBRATION_PASSED:
        return transition_profile(
            record,
            target,
            calibration_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.CALIBRATION_FAILED:
        return transition_profile(record, target, calibration_result_sha256=SHA_D)
    if target is ProfileState.VISUAL_VETO_PASSED:
        return transition_profile(
            record,
            target,
            visual_veto_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.VISUAL_VETO_FAILED:
        return transition_profile(record, target, visual_veto_result_sha256=SHA_D)
    if target is ProfileState.QUALITY_GUARDRAIL_PASSED:
        return transition_profile(
            record,
            target,
            quality_guardrail_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.QUALITY_GUARDRAIL_FAILED:
        return transition_profile(record, target, quality_guardrail_result_sha256=SHA_D)
    if target is ProfileState.INDEPENDENT_PRIVACY_PASSED:
        return transition_profile(
            record,
            target,
            independent_privacy_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.INDEPENDENT_PRIVACY_FAILED:
        return transition_profile(record, target, independent_privacy_result_sha256=SHA_D)
    return transition_profile(record, target)


def _assert_promotion_transition(
    record: PromotionRecord, target: PromotionState
) -> PromotionRecord:
    if target is PromotionState.PROMOTABLE:
        return transition_promotion(
            record,
            target,
            profile=_profile(ProfileState.INDEPENDENT_PRIVACY_PASSED),
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
            snapshot_result_sha256=SHA_B,
        )
    if target is PromotionState.PROMOTED or target is PromotionState.PROMOTION_FAILED:
        return transition_promotion(record, target, promotion_result_sha256=SHA_B)
    if target is PromotionState.H3_REAUDITED or target is PromotionState.H3_REAUDIT_FAILED:
        return transition_promotion(record, target, h3_result_sha256=SHA_B)
    if target in {PromotionState.END_TO_END_REPORTED, PromotionState.END_TO_END_FAILED}:
        return transition_promotion(record, target, end_to_end_result_sha256=SHA_B)
    if target is PromotionState.FULL_POOL_AUDITED:
        return transition_promotion(record, target, full_pool_result_sha256=SHA_B)
    return transition_promotion(record, target)


def _assert_batch_transition(record: BatchRecord, target: BatchState) -> BatchRecord:
    if target is BatchState.OPENED:
        return transition_batch(
            record,
            target,
            batches=(record,),
            profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
            open_result_sha256=SHA_B,
        )
    if target is BatchState.REVIEW_IN_PROGRESS:
        return transition_batch(record, target, review_result_sha256=SHA_B)
    if target is BatchState.COMPLETED_PASS or target is BatchState.COMPLETED_FAIL:
        return transition_batch(record, target, completion_result_sha256=SHA_B)
    return transition_batch(record, target)


@pytest.mark.parametrize(("source", "target"), sorted(CAMPAIGN_EDGES))
def test_campaign_allows_declared_edges_with_context(
    source: CampaignState, target: CampaignState
) -> None:
    record = _campaign(source)

    result = _assert_campaign_transition(record, target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(CampaignState, repeat=2) if edge not in CAMPAIGN_EDGES],
)
def test_campaign_rejects_every_other_enum_edge(
    source: CampaignState, target: CampaignState
) -> None:
    record = _campaign(source)
    with pytest.raises(CalibrationStateError, match="недопустимый переход кампании"):
        _assert_campaign_transition(record, target)


def test_budget_exhausted_requires_fully_failed_batch_history() -> None:
    """`budget_exhausted` — вывод из истории партий, а не ярлык на усмотрение вызывающего."""
    record = _campaign(CampaignState.PROFILES_EVALUATED)
    exhausted = _exhausted_batches()

    with pytest.raises(CalibrationStateError, match="историю партий"):
        transition_campaign(record, CampaignState.BUDGET_EXHAUSTED, budget_result_sha256=SHA_B)
    for incomplete in (
        (*exhausted[:7], _batch(BatchState.SEALED, 8)),
        (*exhausted[:7], _batch(BatchState.COMPLETED_PASS, 8)),
        (*exhausted[:7], _batch(BatchState.REVIEW_IN_PROGRESS, 8)),
    ):
        with pytest.raises(CalibrationStateError, match="бюджет не исчерпан"):
            transition_campaign(
                record,
                CampaignState.BUDGET_EXHAUSTED,
                batches=incomplete,
                budget_result_sha256=SHA_B,
            )

    result = transition_campaign(
        record,
        CampaignState.BUDGET_EXHAUSTED,
        batches=exhausted,
        budget_result_sha256=SHA_B,
    )

    assert result.state is CampaignState.BUDGET_EXHAUSTED
    assert result.selected_profile_id is None


def test_campaign_budget_state_derives_outcome_from_batches() -> None:
    sealed = tuple(_batch(BatchState.SEALED, number) for number in range(1, 9))
    exhausted = _exhausted_batches()
    passed = (*exhausted[:3], _batch(BatchState.COMPLETED_PASS, 4), *sealed[4:])

    assert campaign_budget_state(()) == "available"
    assert campaign_budget_state(sealed) == "available"
    assert campaign_budget_state((*exhausted[:7], sealed[7])) == "available"
    assert campaign_budget_state(passed) == "passed"
    assert campaign_budget_state(exhausted) == "budget_exhausted"


@pytest.mark.parametrize(
    "broken",
    [
        (2, 3),  # нумерация не с единицы: потеряна первая партия
        (1, 3),  # дыра в середине
        (1, 1),  # дубль
    ],
)
def test_campaign_budget_state_rejects_broken_numbering(broken: tuple[int, ...]) -> None:
    batches = tuple(_batch(BatchState.COMPLETED_FAIL, number) for number in broken)

    with pytest.raises(CalibrationStateError, match="подряд с 1"):
        campaign_budget_state(batches)


@pytest.mark.parametrize(("source", "target"), sorted(PROFILE_EDGES))
def test_profile_allows_declared_edges_with_context(
    source: ProfileState, target: ProfileState
) -> None:
    result = _assert_profile_transition(_profile(source), target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(ProfileState, repeat=2) if edge not in PROFILE_EDGES],
)
def test_profile_rejects_every_other_enum_edge(source: ProfileState, target: ProfileState) -> None:
    with pytest.raises(CalibrationStateError, match="недопустимый переход профиля"):
        _assert_profile_transition(_profile(source), target)


@pytest.mark.parametrize(("source", "target"), sorted(PROMOTION_EDGES))
def test_promotion_allows_declared_edges_with_context(
    source: PromotionState, target: PromotionState
) -> None:
    record = _promotion(source)

    result = _assert_promotion_transition(record, target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(PromotionState, repeat=2) if edge not in PROMOTION_EDGES],
)
def test_promotion_rejects_every_other_enum_edge(
    source: PromotionState, target: PromotionState
) -> None:
    record = _promotion(source)
    with pytest.raises(CalibrationStateError, match="недопустимый переход продвижения"):
        _assert_promotion_transition(record, target)


@pytest.mark.parametrize(("source", "target"), sorted(BATCH_EDGES))
def test_batch_allows_declared_edges_with_context(source: BatchState, target: BatchState) -> None:
    record = _batch(source)

    result = _assert_batch_transition(record, target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(BatchState, repeat=2) if edge not in BATCH_EDGES],
)
def test_batch_rejects_every_other_enum_edge(source: BatchState, target: BatchState) -> None:
    record = _batch(source)
    with pytest.raises(CalibrationStateError, match="недопустимый переход партии"):
        _assert_batch_transition(record, target)


@pytest.mark.parametrize("active", [BatchState.OPENED, BatchState.REVIEW_IN_PROGRESS])
def test_batch_open_rejects_another_active_batch(active: BatchState) -> None:
    first = _batch(active, 1)
    second = _batch(BatchState.SEALED, 2)

    with pytest.raises(CalibrationStateError, match="активная партия"):
        transition_batch(
            second,
            BatchState.OPENED,
            batches=(first, second),
            profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
            open_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ((BatchState.SEALED, BatchState.SEALED), "следующая sealed-партия 1"),
        ((BatchState.OPENED, BatchState.SEALED), "активная партия"),
        ((BatchState.REVIEW_IN_PROGRESS, BatchState.SEALED), "активная партия"),
        ((BatchState.COMPLETED_PASS, BatchState.SEALED), "кампания уже прошла"),
    ],
)
def test_second_batch_open_requires_failed_prior_history(
    history: tuple[BatchState, BatchState], message: str
) -> None:
    first = _batch(history[0], 1)
    second = _batch(history[1], 2)

    with pytest.raises(CalibrationStateError, match=message):
        transition_batch(
            second,
            BatchState.OPENED,
            batches=(first, second),
            profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
            open_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )


def test_second_batch_opens_after_prior_failure_without_mutating_history() -> None:
    first = _batch(BatchState.COMPLETED_FAIL, 1)
    second = _batch(BatchState.SEALED, 2)

    opened = transition_batch(
        second,
        BatchState.OPENED,
        batches=(first, second),
        profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
        open_result_sha256=SHA_D,
        current_canonical_sha256=SHA_B,
        current_geometry_witness_sha256=SHA_C,
        current_w_max=17,
        current_h_max=11,
    )

    assert first == _batch(BatchState.COMPLETED_FAIL, 1)
    assert opened.number == 2
    assert opened.state is BatchState.OPENED
    assert opened.open_result_sha256 == SHA_D


def test_batch_open_rejects_nonfailed_lower_number_even_when_it_is_not_prior_in_order() -> None:
    target = _batch(BatchState.SEALED, 2)
    future_active = _batch(BatchState.OPENED, 3)

    with pytest.raises(CalibrationStateError, match="активная партия"):
        transition_batch(
            target,
            BatchState.OPENED,
            batches=(target, future_active),
            profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
            open_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )


def test_batch_open_rejects_missing_lower_batch_number() -> None:
    target = _batch(BatchState.SEALED, 2)

    with pytest.raises(CalibrationStateError, match="предыдущие партии"):
        transition_batch(
            target,
            BatchState.OPENED,
            batches=(target,),
            profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
            open_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )


def test_batch_open_rejects_missing_target_or_duplicate_numbers() -> None:
    target = _batch(BatchState.SEALED, 2)
    profile = _profile(ProfileState.QUALITY_GUARDRAIL_PASSED)

    for batches in ((_batch(BatchState.SEALED, 1),), (target, target)):
        with pytest.raises(CalibrationStateError, match="история партий"):
            transition_batch(
                target,
                BatchState.OPENED,
                batches=batches,
                profile=profile,
                open_result_sha256=SHA_D,
                current_canonical_sha256=SHA_B,
                current_geometry_witness_sha256=SHA_C,
                current_w_max=17,
                current_h_max=11,
            )


@pytest.mark.parametrize("survivors", [(), ("profile-a", "profile-b")])
def test_campaign_direct_completion_requires_exactly_one_survivor(
    survivors: tuple[str, ...],
) -> None:
    record = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=survivors,
    )

    with pytest.raises(CalibrationStateError, match="ровно один survivor"):
        transition_campaign(
            record,
            CampaignState.CAMPAIGN_COMPLETE,
            selection_result_sha256=SHA_D,
        )


def test_campaign_direct_completion_retains_single_survivor_and_witness() -> None:
    record = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=("profile-a",),
        profile_evaluation_result_sha256=SHA_C,
    )

    result = transition_campaign(
        record,
        CampaignState.CAMPAIGN_COMPLETE,
        selection_result_sha256=SHA_D,
    )

    assert result.evaluated_survivor_ids == ("profile-a",)
    assert result.selected_profile_id == "profile-a"
    assert result.profile_evaluation_result_sha256 == SHA_C
    assert result.selection_result_sha256 == SHA_D


def test_selection_required_needs_multiple_survivors_and_selected_member() -> None:
    evaluated = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=("profile-a", "profile-b"),
    )
    selection = transition_campaign(evaluated, CampaignState.SELECTION_REQUIRED)
    assert selection.evaluated_survivor_ids == ("profile-a", "profile-b")

    for selected in (None, "profile-c"):
        with pytest.raises(CalibrationStateError, match="выбранный профиль"):
            transition_campaign(
                selection,
                CampaignState.CAMPAIGN_COMPLETE,
                selected_profile_id=selected,
                selection_result_sha256=SHA_D,
            )


def test_campaign_profiles_evaluated_persists_unique_survivors() -> None:
    record = _campaign(CampaignState.ANNOTATIONS_RESOLVED)

    result = transition_campaign(
        record,
        CampaignState.PROFILES_EVALUATED,
        evaluated_survivor_ids=("profile-b", "profile-a"),
        profile_evaluation_result_sha256=SHA_D,
    )

    assert result.evaluated_survivor_ids == ("profile-b", "profile-a")
    assert result.profile_evaluation_result_sha256 == SHA_D
    # Пустой список выживших больше не ошибка: его фиксирует отдельный терминал Д-3.
    for survivors in (("",), ("profile-a", "profile-a")):
        with pytest.raises(CalibrationStateError, match="survivor"):
            transition_campaign(
                record,
                CampaignState.PROFILES_EVALUATED,
                evaluated_survivor_ids=survivors,
                profile_evaluation_result_sha256=SHA_D,
            )


def test_profiles_evaluated_accepts_empty_survivors() -> None:
    """Нулевой исход оценивания — законное состояние: его фиксирует терминал Д-3."""
    record = _campaign(CampaignState.ANNOTATIONS_RESOLVED)

    result = transition_campaign(
        record,
        CampaignState.PROFILES_EVALUATED,
        evaluated_survivor_ids=(),
        profile_evaluation_result_sha256=SHA_D,
    )

    assert result.state is CampaignState.PROFILES_EVALUATED
    assert result.evaluated_survivor_ids == ()
    assert result.profile_evaluation_result_sha256 == SHA_D


def test_profiles_evaluated_requires_explicit_survivor_list() -> None:
    """Забытый аргумент не должен молча означать «выживших нет»."""
    with pytest.raises(CalibrationStateError, match="evaluated_survivor_ids"):
        transition_campaign(
            _campaign(CampaignState.ANNOTATIONS_RESOLVED),
            CampaignState.PROFILES_EVALUATED,
            profile_evaluation_result_sha256=SHA_D,
        )


def test_no_eligible_profiles_reaches_terminal_from_empty_survivors() -> None:
    # `selected_profile_id` заполнен намеренно: законным путём в `profiles_evaluated` он не
    # попадает, и без метки проверка «переход его не трогает» была бы вакуумной.
    record = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=(),
        profile_evaluation_result_sha256=SHA_C,
        selected_profile_id="profile-a",
    )

    result = transition_campaign(
        record,
        CampaignState.NO_ELIGIBLE_PROFILES,
        no_eligible_profiles_result_sha256=SHA_D,
    )

    assert result.state is CampaignState.NO_ELIGIBLE_PROFILES
    assert result.evaluated_survivor_ids == ()
    assert result.no_eligible_profiles_result_sha256 == SHA_D
    assert result.profile_evaluation_result_sha256 == SHA_C
    assert result.selected_profile_id == "profile-a"


@pytest.mark.parametrize("survivors", [("profile-a",), ("profile-a", "profile-b")])
def test_no_eligible_profiles_rejects_nonempty_survivors(
    survivors: tuple[str, ...],
) -> None:
    carried = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=survivors,
    )
    empty = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=(),
    )

    with pytest.raises(CalibrationStateError, match="survivors"):
        transition_campaign(
            carried,
            CampaignState.NO_ELIGIBLE_PROFILES,
            no_eligible_profiles_result_sha256=SHA_D,
        )
    with pytest.raises(CalibrationStateError, match="survivors"):
        transition_campaign(
            empty,
            CampaignState.NO_ELIGIBLE_PROFILES,
            evaluated_survivor_ids=survivors,
            no_eligible_profiles_result_sha256=SHA_D,
        )


@pytest.mark.parametrize("survivors", [("",), ("profile-a", "profile-a")])
def test_no_eligible_profiles_validates_explicit_survivor_override(
    survivors: tuple[str, ...],
) -> None:
    """Явный список проходит ту же проверку, что и на входе в `profiles_evaluated`.

    Шаблон различает две ветви намеренно: сообщение о непустом списке тоже содержит слово
    «survivors», и совпадение по нему означало бы, что проверка идентификаторов не сработала.
    """
    record = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=(),
    )

    with pytest.raises(CalibrationStateError, match="survivor IDs"):
        transition_campaign(
            record,
            CampaignState.NO_ELIGIBLE_PROFILES,
            evaluated_survivor_ids=survivors,
            no_eligible_profiles_result_sha256=SHA_D,
        )


def test_no_eligible_profiles_is_terminal() -> None:
    assert not any(source is CampaignState.NO_ELIGIBLE_PROFILES for source, _ in CAMPAIGN_EDGES)

    record = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=(),
    )
    terminal = transition_campaign(
        record,
        CampaignState.NO_ELIGIBLE_PROFILES,
        no_eligible_profiles_result_sha256=SHA_D,
    )

    for target in CampaignState:
        with pytest.raises(CalibrationStateError, match="недопустимый переход кампании"):
            _assert_campaign_transition(terminal, target)


@pytest.mark.parametrize("unresolved_count", [1, -1, True])
def test_annotations_resolved_requires_zero_unresolved_rows(unresolved_count: int) -> None:
    with pytest.raises(CalibrationStateError, match="unresolved"):
        transition_campaign(
            _campaign(CampaignState.ANNOTATIONS_OPEN),
            CampaignState.ANNOTATIONS_RESOLVED,
            unresolved_count=unresolved_count,
            annotations_result_sha256=SHA_D,
        )


def test_annotations_resolved_retains_empty_evidence() -> None:
    result = transition_campaign(
        _campaign(CampaignState.ANNOTATIONS_OPEN),
        CampaignState.ANNOTATIONS_RESOLVED,
        unresolved_count=0,
        annotations_result_sha256=SHA_D,
    )

    assert result.unresolved_count == 0
    assert result.annotations_result_sha256 == SHA_D


def test_annotations_resolved_requires_explicit_unresolved_evidence() -> None:
    with pytest.raises(CalibrationStateError, match="unresolved"):
        transition_campaign(
            _campaign(CampaignState.ANNOTATIONS_OPEN),
            CampaignState.ANNOTATIONS_RESOLVED,
            annotations_result_sha256=SHA_D,
        )


def _campaign_witness_call(
    target: CampaignState, witness: str | None
) -> Callable[[], CampaignRecord]:
    sources = {
        CampaignState.PREREGISTERED: CampaignState.DRAFT,
        CampaignState.TOOLING_VERIFIED: CampaignState.PREREGISTERED,
        CampaignState.CONTROL_VERIFIED: CampaignState.TOOLING_VERIFIED,
        CampaignState.ANNOTATIONS_RESOLVED: CampaignState.ANNOTATIONS_OPEN,
        CampaignState.PROFILES_EVALUATED: CampaignState.ANNOTATIONS_RESOLVED,
        CampaignState.CAMPAIGN_COMPLETE: CampaignState.PROFILES_EVALUATED,
        CampaignState.BUDGET_EXHAUSTED: CampaignState.SELECTION_REQUIRED,
        CampaignState.NO_ELIGIBLE_PROFILES: CampaignState.PROFILES_EVALUATED,
    }
    record = _campaign(sources[target])
    if target is CampaignState.PREREGISTERED:
        return lambda: transition_campaign(record, target, preregistration_result_sha256=witness)
    if target is CampaignState.TOOLING_VERIFIED:
        return lambda: transition_campaign(record, target, tooling_result_sha256=witness)
    if target is CampaignState.CONTROL_VERIFIED:
        return lambda: transition_campaign(record, target, control_result_sha256=witness)
    if target is CampaignState.ANNOTATIONS_RESOLVED:
        return lambda: transition_campaign(
            record,
            target,
            unresolved_count=0,
            annotations_result_sha256=witness,
        )
    if target is CampaignState.PROFILES_EVALUATED:
        return lambda: transition_campaign(
            record,
            target,
            evaluated_survivor_ids=("profile-a", "profile-b"),
            profile_evaluation_result_sha256=witness,
        )
    if target is CampaignState.CAMPAIGN_COMPLETE:
        record = replace(record, evaluated_survivor_ids=("profile-a",))
        return lambda: transition_campaign(record, target, selection_result_sha256=witness)
    if target is CampaignState.NO_ELIGIBLE_PROFILES:
        record = replace(record, evaluated_survivor_ids=())
        return lambda: transition_campaign(
            record,
            target,
            no_eligible_profiles_result_sha256=witness,
        )
    return lambda: transition_campaign(
        record,
        target,
        batches=_exhausted_batches(),
        budget_result_sha256=witness,
    )


def _profile_witness_call(target: ProfileState, witness: str | None) -> Callable[[], ProfileRecord]:
    sources = {
        ProfileState.BUILT: ProfileState.DECLARED,
        ProfileState.BUILD_FAILED: ProfileState.DECLARED,
        ProfileState.GEOMETRY_FRESH: ProfileState.BUILT,
        ProfileState.CALIBRATION_PASSED: ProfileState.GEOMETRY_FRESH,
        ProfileState.CALIBRATION_FAILED: ProfileState.GEOMETRY_FRESH,
        ProfileState.VISUAL_VETO_PASSED: ProfileState.CALIBRATION_PASSED,
        ProfileState.VISUAL_VETO_FAILED: ProfileState.CALIBRATION_PASSED,
        ProfileState.QUALITY_GUARDRAIL_PASSED: ProfileState.VISUAL_VETO_PASSED,
        ProfileState.QUALITY_GUARDRAIL_FAILED: ProfileState.VISUAL_VETO_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_PASSED: ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_FAILED: ProfileState.QUALITY_GUARDRAIL_PASSED,
    }
    record = _profile(sources[target])
    if target is ProfileState.BUILT or target is ProfileState.BUILD_FAILED:
        return lambda: transition_profile(record, target, build_result_sha256=witness)
    if target is ProfileState.GEOMETRY_FRESH:
        return lambda: transition_profile(
            record,
            target,
            canonical_sha256=SHA_B,
            geometry_witness_sha256=SHA_C,
            w_max=17,
            h_max=11,
            geometry_result_sha256=witness,
        )
    if target is ProfileState.CALIBRATION_PASSED:
        return lambda: transition_profile(
            record,
            target,
            calibration_result_sha256=witness,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.CALIBRATION_FAILED:
        return lambda: transition_profile(record, target, calibration_result_sha256=witness)
    if target is ProfileState.VISUAL_VETO_PASSED:
        return lambda: transition_profile(
            record,
            target,
            visual_veto_result_sha256=witness,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.VISUAL_VETO_FAILED:
        return lambda: transition_profile(record, target, visual_veto_result_sha256=witness)
    if target is ProfileState.QUALITY_GUARDRAIL_PASSED:
        return lambda: transition_profile(
            record,
            target,
            quality_guardrail_result_sha256=witness,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.QUALITY_GUARDRAIL_FAILED:
        return lambda: transition_profile(record, target, quality_guardrail_result_sha256=witness)
    if target is ProfileState.INDEPENDENT_PRIVACY_PASSED:
        return lambda: transition_profile(
            record,
            target,
            independent_privacy_result_sha256=witness,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    return lambda: transition_profile(record, target, independent_privacy_result_sha256=witness)


def _promotion_witness_call(
    target: PromotionState, witness: str | None
) -> Callable[[], PromotionRecord]:
    sources = {
        PromotionState.PROMOTABLE: PromotionState.NOT_SELECTED,
        PromotionState.PROMOTED: PromotionState.PROMOTABLE,
        PromotionState.PROMOTION_FAILED: PromotionState.PROMOTABLE,
        PromotionState.H3_REAUDITED: PromotionState.H3_REAUDIT_PENDING,
        PromotionState.H3_REAUDIT_FAILED: PromotionState.H3_REAUDIT_PENDING,
        PromotionState.END_TO_END_REPORTED: PromotionState.H3_REAUDITED,
        PromotionState.END_TO_END_FAILED: PromotionState.H3_REAUDITED,
        PromotionState.FULL_POOL_AUDITED: PromotionState.END_TO_END_REPORTED,
    }
    record = _promotion(sources[target])
    if target is PromotionState.PROMOTABLE:
        return lambda: transition_promotion(
            record,
            target,
            profile=_profile(ProfileState.INDEPENDENT_PRIVACY_PASSED),
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
            snapshot_result_sha256=witness,
        )
    if target is PromotionState.PROMOTED or target is PromotionState.PROMOTION_FAILED:
        return lambda: transition_promotion(record, target, promotion_result_sha256=witness)
    if target is PromotionState.H3_REAUDITED or target is PromotionState.H3_REAUDIT_FAILED:
        return lambda: transition_promotion(record, target, h3_result_sha256=witness)
    if target in {PromotionState.END_TO_END_REPORTED, PromotionState.END_TO_END_FAILED}:
        return lambda: transition_promotion(record, target, end_to_end_result_sha256=witness)
    return lambda: transition_promotion(record, target, full_pool_result_sha256=witness)


def _batch_witness_call(target: BatchState, witness: str | None) -> Callable[[], BatchRecord]:
    sources = {
        BatchState.OPENED: BatchState.SEALED,
        BatchState.REVIEW_IN_PROGRESS: BatchState.OPENED,
        BatchState.COMPLETED_PASS: BatchState.REVIEW_IN_PROGRESS,
        BatchState.COMPLETED_FAIL: BatchState.REVIEW_IN_PROGRESS,
    }
    record = _batch(sources[target])
    if target is BatchState.OPENED:
        return lambda: transition_batch(
            record,
            target,
            batches=(record,),
            profile=_profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
            open_result_sha256=witness,
        )
    if target is BatchState.REVIEW_IN_PROGRESS:
        return lambda: transition_batch(record, target, review_result_sha256=witness)
    return lambda: transition_batch(record, target, completion_result_sha256=witness)


WITNESS_CASES = [
    *(
        pytest.param(_campaign_witness_call, target, id=f"campaign-{target.value}")
        for target in CAMPAIGN_RESULT_FIELDS
    ),
    *(
        pytest.param(_profile_witness_call, target, id=f"profile-{target.value}")
        for target in PROFILE_RESULT_FIELDS
    ),
    *(
        pytest.param(_promotion_witness_call, target, id=f"promotion-{target.value}")
        for target in PROMOTION_RESULT_FIELDS
    ),
    *(
        pytest.param(_batch_witness_call, target, id=f"batch-{target.value}")
        for target in BATCH_RESULT_FIELDS
    ),
]


def _call_invalid_witness(
    factory: object, target: object, witness: str | None
) -> CampaignRecord | ProfileRecord | PromotionRecord | BatchRecord:
    if factory is _campaign_witness_call and isinstance(target, CampaignState):
        return _campaign_witness_call(target, witness)()
    if factory is _profile_witness_call and isinstance(target, ProfileState):
        return _profile_witness_call(target, witness)()
    if factory is _promotion_witness_call and isinstance(target, PromotionState):
        return _promotion_witness_call(target, witness)()
    if factory is _batch_witness_call and isinstance(target, BatchState):
        return _batch_witness_call(target, witness)()
    raise AssertionError("несогласованный witness case")


@pytest.mark.parametrize(("factory", "target"), WITNESS_CASES)
@pytest.mark.parametrize("witness", INVALID_HASHES)
def test_every_evidence_milestone_rejects_missing_empty_or_malformed_sha256(
    factory: object,
    target: object,
    witness: str | None,
) -> None:
    with pytest.raises(CalibrationStateError, match="SHA256"):
        _call_invalid_witness(factory, target, witness)


def _positive_profile_transition(
    record: ProfileRecord, target: ProfileState, result_sha256: str = SHA_D
) -> ProfileRecord:
    if target is ProfileState.CALIBRATION_PASSED:
        return transition_profile(
            record,
            target,
            calibration_result_sha256=result_sha256,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.VISUAL_VETO_PASSED:
        return transition_profile(
            record,
            target,
            visual_veto_result_sha256=result_sha256,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.QUALITY_GUARDRAIL_PASSED:
        return transition_profile(
            record,
            target,
            quality_guardrail_result_sha256=result_sha256,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    if target is ProfileState.INDEPENDENT_PRIVACY_PASSED:
        return transition_profile(
            record,
            target,
            independent_privacy_result_sha256=result_sha256,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    raise AssertionError(f"{target.value} не является положительным переходом")


@pytest.mark.parametrize(
    ("target", "field"),
    [
        (ProfileState.CALIBRATION_PASSED, "calibration_result_sha256"),
        (ProfileState.VISUAL_VETO_PASSED, "visual_veto_result_sha256"),
        (ProfileState.QUALITY_GUARDRAIL_PASSED, "quality_guardrail_result_sha256"),
        (ProfileState.INDEPENDENT_PRIVACY_PASSED, "independent_privacy_result_sha256"),
    ],
)
def test_positive_profile_transitions_require_current_geometry_and_retain_results(
    target: ProfileState, field: str
) -> None:
    sources = {
        ProfileState.CALIBRATION_PASSED: ProfileState.GEOMETRY_FRESH,
        ProfileState.VISUAL_VETO_PASSED: ProfileState.CALIBRATION_PASSED,
        ProfileState.QUALITY_GUARDRAIL_PASSED: ProfileState.VISUAL_VETO_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_PASSED: ProfileState.QUALITY_GUARDRAIL_PASSED,
    }
    record = _profile(sources[target])
    result = _positive_profile_transition(record, target)

    assert getattr(result, field) == SHA_D
    assert result.geometry_witness_sha256 == SHA_C
    assert result.canonical_sha256 == SHA_B


@pytest.mark.parametrize(
    (
        "current_canonical_sha256",
        "current_geometry_witness_sha256",
        "current_w_max",
        "current_h_max",
    ),
    [
        (SHA_D, SHA_C, 17, 11),
        (SHA_B, SHA_D, 17, 11),
        (SHA_B, SHA_C, 18, 11),
        (SHA_B, SHA_C, 17, 12),
    ],
)
def test_later_profile_pass_rejects_current_geometry_mismatch(
    current_canonical_sha256: str,
    current_geometry_witness_sha256: str,
    current_w_max: int,
    current_h_max: int,
) -> None:
    record = _profile(ProfileState.CALIBRATION_PASSED)

    with pytest.raises(CalibrationStateError, match="текущей геометрии"):
        transition_profile(
            record,
            ProfileState.VISUAL_VETO_PASSED,
            visual_veto_result_sha256=SHA_D,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )
    assert record == _profile(ProfileState.CALIBRATION_PASSED)


def test_failed_profile_milestones_do_not_require_current_geometry() -> None:
    calibration = transition_profile(
        _profile(ProfileState.GEOMETRY_FRESH),
        ProfileState.CALIBRATION_FAILED,
        calibration_result_sha256=SHA_D,
    )
    visual = transition_profile(
        _profile(ProfileState.CALIBRATION_PASSED),
        ProfileState.VISUAL_VETO_FAILED,
        visual_veto_result_sha256=SHA_D,
    )
    quality = transition_profile(
        _profile(ProfileState.VISUAL_VETO_PASSED),
        ProfileState.QUALITY_GUARDRAIL_FAILED,
        quality_guardrail_result_sha256=SHA_D,
    )
    privacy = transition_profile(
        _profile(ProfileState.QUALITY_GUARDRAIL_PASSED),
        ProfileState.INDEPENDENT_PRIVACY_FAILED,
        independent_privacy_result_sha256=SHA_D,
    )

    assert (
        calibration.state,
        visual.state,
        quality.state,
        privacy.state,
    ) == (
        ProfileState.CALIBRATION_FAILED,
        ProfileState.VISUAL_VETO_FAILED,
        ProfileState.QUALITY_GUARDRAIL_FAILED,
        ProfileState.INDEPENDENT_PRIVACY_FAILED,
    )


@pytest.mark.parametrize(
    "source",
    [
        ProfileState.GEOMETRY_FRESH,
        ProfileState.CALIBRATION_PASSED,
        ProfileState.VISUAL_VETO_PASSED,
        ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_PASSED,
    ],
)
def test_every_geometry_dependent_positive_state_can_be_explicitly_invalidated(
    source: ProfileState,
) -> None:
    record = _profile(source)

    stale = transition_profile(record, ProfileState.GEOMETRY_STALE)

    assert stale.state is ProfileState.GEOMETRY_STALE
    assert stale.geometry_witness_sha256 == SHA_C
    assert record.state is source


@pytest.mark.parametrize(
    (
        "current_canonical_sha256",
        "current_geometry_witness_sha256",
        "current_w_max",
        "current_h_max",
    ),
    [
        (SHA_D, SHA_C, 17, 11),
        (SHA_B, SHA_D, 17, 11),
        (SHA_B, SHA_C, 18, 11),
        (SHA_B, SHA_C, 17, 12),
    ],
)
def test_promotion_authorization_rejects_current_geometry_mismatch(
    current_canonical_sha256: str,
    current_geometry_witness_sha256: str,
    current_w_max: int,
    current_h_max: int,
) -> None:
    record = _promotion(PromotionState.NOT_SELECTED)

    with pytest.raises(CalibrationStateError, match="текущей геометрии"):
        transition_promotion(
            record,
            PromotionState.PROMOTABLE,
            profile=_profile(ProfileState.INDEPENDENT_PRIVACY_PASSED),
            snapshot_result_sha256=SHA_D,
            current_canonical_sha256=current_canonical_sha256,
            current_geometry_witness_sha256=current_geometry_witness_sha256,
            current_w_max=current_w_max,
            current_h_max=current_h_max,
        )


def test_promotion_requires_independent_privacy_passed() -> None:
    for state in ProfileState:
        if state is ProfileState.INDEPENDENT_PRIVACY_PASSED:
            continue
        with pytest.raises(CalibrationStateError, match="independent_privacy_passed"):
            transition_promotion(
                _promotion(PromotionState.NOT_SELECTED),
                PromotionState.PROMOTABLE,
                profile=_profile(state),
                snapshot_result_sha256=SHA_D,
                current_canonical_sha256=SHA_B,
                current_geometry_witness_sha256=SHA_C,
                current_w_max=17,
                current_h_max=11,
            )


def _open_first_batch(profile: ProfileRecord) -> BatchRecord:
    sealed = _batch(BatchState.SEALED)
    return transition_batch(
        sealed,
        BatchState.OPENED,
        batches=(sealed,),
        profile=profile,
        open_result_sha256=SHA_D,
        current_canonical_sha256=SHA_B,
        current_geometry_witness_sha256=SHA_C,
        current_w_max=17,
        current_h_max=11,
    )


def test_batch_open_requires_quality_guardrail_passed_profile() -> None:
    """Партия — источник independent-вердикта, поэтому открывает её guardrail-профиль."""
    opened = _open_first_batch(_profile(ProfileState.QUALITY_GUARDRAIL_PASSED))

    assert opened.state is BatchState.OPENED
    assert opened.profile_id == "profile-a"

    for state in ProfileState:
        if state is ProfileState.QUALITY_GUARDRAIL_PASSED:
            continue
        with pytest.raises(CalibrationStateError, match="quality_guardrail_passed"):
            _open_first_batch(_profile(state))


def test_promotion_and_batch_open_require_profile_argument() -> None:
    with pytest.raises(CalibrationStateError, match="independent_privacy_passed"):
        transition_promotion(
            _promotion(PromotionState.NOT_SELECTED),
            PromotionState.PROMOTABLE,
            snapshot_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )
    sealed = _batch(BatchState.SEALED)
    with pytest.raises(CalibrationStateError, match="quality_guardrail_passed"):
        transition_batch(
            sealed,
            BatchState.OPENED,
            batches=(sealed,),
            open_result_sha256=SHA_D,
            current_canonical_sha256=SHA_B,
            current_geometry_witness_sha256=SHA_C,
            current_w_max=17,
            current_h_max=11,
        )


@pytest.mark.parametrize("profile_id", ["", "   "])
def test_profile_rejects_empty_id(profile_id: str) -> None:
    with pytest.raises(CalibrationStateError, match="profile_id"):
        _profile(ProfileState.DECLARED, profile_id=profile_id)


@pytest.mark.parametrize("profile_sha256", INVALID_HASHES)
def test_profile_rejects_missing_or_malformed_identity_hash(
    profile_sha256: str | None,
) -> None:
    with pytest.raises(CalibrationStateError, match="profile_sha256"):
        ProfileRecord(
            profile_id="profile-a",
            state=ProfileState.DECLARED,
            profile_sha256=profile_sha256 or "",
        )


@pytest.mark.parametrize("number", [0, -1, True])
def test_batch_rejects_invalid_number(number: int) -> None:
    with pytest.raises(CalibrationStateError, match="number"):
        BatchRecord(number=number, state=BatchState.SEALED, batch_sha256=SHA_A)


@pytest.mark.parametrize("batch_sha256", INVALID_HASHES)
def test_batch_rejects_missing_or_malformed_identity_hash(batch_sha256: str | None) -> None:
    with pytest.raises(CalibrationStateError, match="batch_sha256"):
        BatchRecord(number=1, state=BatchState.SEALED, batch_sha256=batch_sha256 or "")


@pytest.mark.parametrize(
    ("w_max", "h_max"),
    [(-1, 11), (17, -1), (True, 11), (17, False), (None, 11), (17, None)],
)
def test_geometry_fresh_rejects_invalid_dimensions(w_max: int | None, h_max: int | None) -> None:
    with pytest.raises(CalibrationStateError, match="w_max/h_max"):
        transition_profile(
            _profile(ProfileState.BUILT),
            ProfileState.GEOMETRY_FRESH,
            canonical_sha256=SHA_B,
            geometry_witness_sha256=SHA_C,
            geometry_result_sha256=SHA_D,
            w_max=w_max,
            h_max=h_max,
        )


def test_records_are_frozen_slotted_and_transition_retains_prior_witnesses() -> None:
    records = (
        _campaign(CampaignState.DRAFT),
        _profile(ProfileState.GEOMETRY_FRESH),
        _promotion(PromotionState.PROMOTABLE),
        _batch(BatchState.OPENED),
    )
    for record in records:
        assert not hasattr(record, "__dict__")
        with pytest.raises(FrozenInstanceError):
            record.state = record.state  # type: ignore[misc]

    passed = transition_profile(
        records[1],
        ProfileState.CALIBRATION_PASSED,
        calibration_result_sha256=SHA_D,
        current_canonical_sha256=SHA_B,
        current_geometry_witness_sha256=SHA_C,
        current_w_max=17,
        current_h_max=11,
    )
    assert passed.build_result_sha256 == SHA_D
    assert passed.geometry_result_sha256 == SHA_E
    assert passed.calibration_result_sha256 == SHA_D
    assert records[1].state is ProfileState.GEOMETRY_FRESH


def test_no_eligible_profiles_override_сверяется_а_не_подменяет() -> None:
    """Запись авторитетна: иначе «не осталось» превращается в «мы решили, что не осталось»."""
    с_выжившими = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=("profile-a", "profile-b"),
    )

    with pytest.raises(CalibrationStateError, match="survivors"):
        transition_campaign(
            с_выжившими,
            CampaignState.NO_ELIGIBLE_PROFILES,
            evaluated_survivor_ids=(),
            no_eligible_profiles_result_sha256=SHA_D,
        )


def test_no_eligible_profiles_принимает_совпадающий_override() -> None:
    пустая = replace(
        _campaign(CampaignState.PROFILES_EVALUATED),
        evaluated_survivor_ids=(),
    )

    result = transition_campaign(
        пустая,
        CampaignState.NO_ELIGIBLE_PROFILES,
        evaluated_survivor_ids=(),
        no_eligible_profiles_result_sha256=SHA_D,
    )

    assert result.state is CampaignState.NO_ELIGIBLE_PROFILES
    assert result.evaluated_survivor_ids == ()
