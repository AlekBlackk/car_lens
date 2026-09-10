"""Переходы состояний рекалибровки и их свидетели."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
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
    transition_batch,
    transition_campaign,
    transition_profile,
    transition_promotion,
)

PROFILE_SHA = "a" * 64
CANONICAL_SHA = "b" * 64
WITNESS_SHA = "c" * 64
NEW_WITNESS_SHA = "d" * 64
BATCH_SHA = "e" * 64

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
}

PROFILE_EDGES = {
    (ProfileState.DECLARED, ProfileState.BUILT),
    (ProfileState.DECLARED, ProfileState.BUILD_FAILED),
    (ProfileState.BUILT, ProfileState.GEOMETRY_FRESH),
    (ProfileState.BUILT, ProfileState.GEOMETRY_STALE),
    (ProfileState.GEOMETRY_FRESH, ProfileState.GEOMETRY_STALE),
    (ProfileState.GEOMETRY_FRESH, ProfileState.CALIBRATION_PASSED),
    (ProfileState.GEOMETRY_FRESH, ProfileState.CALIBRATION_FAILED),
    (ProfileState.CALIBRATION_PASSED, ProfileState.VISUAL_VETO_PASSED),
    (ProfileState.CALIBRATION_PASSED, ProfileState.VISUAL_VETO_FAILED),
    (ProfileState.VISUAL_VETO_PASSED, ProfileState.QUALITY_GUARDRAIL_PASSED),
    (ProfileState.VISUAL_VETO_PASSED, ProfileState.QUALITY_GUARDRAIL_FAILED),
    (
        ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_PASSED,
    ),
    (
        ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_FAILED,
    ),
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


def _campaign(state: CampaignState) -> CampaignRecord:
    return CampaignRecord(state=state)


def _profile(state: ProfileState) -> ProfileRecord:
    return ProfileRecord(
        profile_id="profile-a",
        state=state,
        profile_sha256=PROFILE_SHA,
        canonical_sha256=CANONICAL_SHA,
        geometry_witness_sha256=(
            WITNESS_SHA
            if state
            not in {
                ProfileState.DECLARED,
                ProfileState.BUILT,
                ProfileState.BUILD_FAILED,
            }
            else None
        ),
        w_max=17,
        h_max=11,
    )


def _promotion(state: PromotionState) -> PromotionRecord:
    return PromotionRecord(
        state=state,
        profile_id="profile-a" if state is not PromotionState.NOT_SELECTED else None,
        profile_sha256=PROFILE_SHA if state is not PromotionState.NOT_SELECTED else None,
        geometry_witness_sha256=(WITNESS_SHA if state is not PromotionState.NOT_SELECTED else None),
    )


def _batch(state: BatchState) -> BatchRecord:
    return BatchRecord(
        number=1,
        state=state,
        batch_sha256=BATCH_SHA,
        profile_id="profile-a" if state is not BatchState.SEALED else None,
        profile_sha256=PROFILE_SHA if state is not BatchState.SEALED else None,
        geometry_witness_sha256=WITNESS_SHA if state is not BatchState.SEALED else None,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    sorted(CAMPAIGN_EDGES),
)
def test_campaign_allows_declared_edges(source: CampaignState, target: CampaignState) -> None:
    result = transition_campaign(_campaign(source), target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(CampaignState, repeat=2) if edge not in CAMPAIGN_EDGES],
)
def test_campaign_rejects_every_other_edge(source: CampaignState, target: CampaignState) -> None:
    with pytest.raises(CalibrationStateError, match="недопустимый переход кампании"):
        transition_campaign(_campaign(source), target)


@pytest.mark.parametrize(
    ("source", "target"),
    sorted(PROFILE_EDGES),
)
def test_profile_allows_declared_edges(source: ProfileState, target: ProfileState) -> None:
    record = _profile(source)
    if target is ProfileState.GEOMETRY_FRESH:
        result = transition_profile(
            record,
            target,
            canonical_sha256=CANONICAL_SHA,
            geometry_witness_sha256=WITNESS_SHA,
            w_max=17,
            h_max=11,
        )
    else:
        result = transition_profile(record, target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(ProfileState, repeat=2) if edge not in PROFILE_EDGES],
)
def test_profile_rejects_every_other_edge(source: ProfileState, target: ProfileState) -> None:
    with pytest.raises(CalibrationStateError, match="недопустимый переход профиля"):
        transition_profile(_profile(source), target)


@pytest.mark.parametrize(
    ("source", "target"),
    sorted(PROMOTION_EDGES),
)
def test_promotion_allows_declared_edges(source: PromotionState, target: PromotionState) -> None:
    profile = _profile(ProfileState.INDEPENDENT_PRIVACY_PASSED)
    result = transition_promotion(_promotion(source), target, profile=profile)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(PromotionState, repeat=2) if edge not in PROMOTION_EDGES],
)
def test_promotion_rejects_every_other_edge(source: PromotionState, target: PromotionState) -> None:
    profile = _profile(ProfileState.INDEPENDENT_PRIVACY_PASSED)
    with pytest.raises(CalibrationStateError, match="недопустимый переход продвижения"):
        transition_promotion(_promotion(source), target, profile=profile)


@pytest.mark.parametrize(
    ("source", "target"),
    sorted(BATCH_EDGES),
)
def test_batch_allows_declared_edges(source: BatchState, target: BatchState) -> None:
    record = _batch(source)
    if target is BatchState.OPENED:
        result = transition_batch(
            record,
            target,
            profile=_profile(ProfileState.INDEPENDENT_PRIVACY_PASSED),
        )
    else:
        result = transition_batch(record, target)

    assert result.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [edge for edge in product(BatchState, repeat=2) if edge not in BATCH_EDGES],
)
def test_batch_rejects_every_other_edge(source: BatchState, target: BatchState) -> None:
    with pytest.raises(CalibrationStateError, match="недопустимый переход партии"):
        transition_batch(_batch(source), target)


def test_geometry_fresh_requires_matching_canonical_hash_exact_extents_and_witness() -> None:
    built = _profile(ProfileState.BUILT)

    for canonical_sha256, geometry_witness_sha256, w_max, h_max in [
        (NEW_WITNESS_SHA, WITNESS_SHA, 17, 11),
        (CANONICAL_SHA, WITNESS_SHA, 18, 11),
        (CANONICAL_SHA, WITNESS_SHA, 17, 12),
        (CANONICAL_SHA, None, 17, 11),
    ]:
        with pytest.raises(CalibrationStateError, match="геометрический свидетель"):
            transition_profile(
                built,
                ProfileState.GEOMETRY_FRESH,
                canonical_sha256=canonical_sha256,
                geometry_witness_sha256=geometry_witness_sha256,
                w_max=w_max,
                h_max=h_max,
            )


def test_profile_pass_states_retain_geometry_witness() -> None:
    fresh = _profile(ProfileState.GEOMETRY_FRESH)

    passed = transition_profile(fresh, ProfileState.CALIBRATION_PASSED)

    assert passed.canonical_sha256 == CANONICAL_SHA
    assert passed.geometry_witness_sha256 == WITNESS_SHA
    assert (passed.w_max, passed.h_max) == (17, 11)


def test_stale_profile_cannot_advance_to_any_pass_state() -> None:
    stale = transition_profile(_profile(ProfileState.GEOMETRY_FRESH), ProfileState.GEOMETRY_STALE)

    for target in (
        ProfileState.CALIBRATION_PASSED,
        ProfileState.VISUAL_VETO_PASSED,
        ProfileState.QUALITY_GUARDRAIL_PASSED,
        ProfileState.INDEPENDENT_PRIVACY_PASSED,
    ):
        with pytest.raises(CalibrationStateError, match="недопустимый переход профиля"):
            transition_profile(stale, target)


def test_promotion_authorization_requires_independent_pass_and_copies_hashes() -> None:
    promotion = _promotion(PromotionState.NOT_SELECTED)
    passed = _profile(ProfileState.INDEPENDENT_PRIVACY_PASSED)

    result = transition_promotion(promotion, PromotionState.PROMOTABLE, profile=passed)

    assert result.profile_id == passed.profile_id
    assert result.profile_sha256 == passed.profile_sha256
    assert result.geometry_witness_sha256 == passed.geometry_witness_sha256
    with pytest.raises(CalibrationStateError, match="independent_privacy_passed"):
        transition_promotion(promotion, PromotionState.PROMOTABLE)
    for state in ProfileState:
        if state is ProfileState.INDEPENDENT_PRIVACY_PASSED:
            continue
        with pytest.raises(CalibrationStateError, match="independent_privacy_passed"):
            transition_promotion(promotion, PromotionState.PROMOTABLE, profile=_profile(state))


def test_opening_batch_captures_profile_and_witness_hashes() -> None:
    profile = _profile(ProfileState.INDEPENDENT_PRIVACY_PASSED)

    opened = transition_batch(_batch(BatchState.SEALED), BatchState.OPENED, profile=profile)

    assert opened.profile_id == profile.profile_id
    assert opened.profile_sha256 == profile.profile_sha256
    assert opened.geometry_witness_sha256 == profile.geometry_witness_sha256


def test_opening_batch_requires_profile_with_geometry_witness() -> None:
    sealed = _batch(BatchState.SEALED)
    missing_witness = _profile(ProfileState.BUILT)

    with pytest.raises(CalibrationStateError, match="профиль со свидетелем"):
        transition_batch(sealed, BatchState.OPENED)
    with pytest.raises(CalibrationStateError, match="профиль со свидетелем"):
        transition_batch(sealed, BatchState.OPENED, profile=missing_witness)


def test_completed_batch_outcomes_are_terminal_and_retain_witness() -> None:
    for completed in (BatchState.COMPLETED_PASS, BatchState.COMPLETED_FAIL):
        record = _batch(completed)
        assert record.geometry_witness_sha256 == WITNESS_SHA
        for target in BatchState:
            with pytest.raises(CalibrationStateError, match="недопустимый переход партии"):
                transition_batch(record, target)


def test_records_are_frozen_and_slotted() -> None:
    records = (
        _campaign(CampaignState.DRAFT),
        _profile(ProfileState.DECLARED),
        _promotion(PromotionState.NOT_SELECTED),
        _batch(BatchState.SEALED),
    )

    for record in records:
        assert not hasattr(record, "__dict__")
        with pytest.raises(FrozenInstanceError):
            record.state = record.state  # type: ignore[misc]
