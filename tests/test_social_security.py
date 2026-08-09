"""Tests for the Social Security benefit estimator (fintracker/social_security.py)."""
import pytest

from fintracker.constants import SS_WAGE_BASE
from fintracker.social_security import (
    estimate_aime, claim_factor, estimate_annual_benefit,
    FULL_RETIREMENT_AGE, CREDITS_TO_QUALIFY,
)


def test_aime_caps_earnings_at_wage_base():
    # Earnings above the wage base earn no extra benefit.
    assert estimate_aime(500_000, 35) == pytest.approx(SS_WAGE_BASE / 12)
    assert estimate_aime(60_000, 35) == pytest.approx(5_000.0)


def test_aime_partial_career_averages_in_zeros():
    # Half the 35-year window worked → half the AIME (missing years count as $0).
    assert estimate_aime(60_000, 17.5) == pytest.approx(2_500.0)
    assert estimate_aime(60_000, 0) == 0.0


def test_pia_uses_bend_point_percentages():
    # AIME 14,050 → 90% of first 1,174 + 32% up to 7,078 + 15% above (monthly PIA),
    # ×12 at full retirement age. Confirms the progressive_tax reuse is wired right.
    expected_monthly = 0.90 * 1_174 + 0.32 * (7_078 - 1_174) + 0.15 * (14_050 - 7_078)
    got = estimate_annual_benefit(SS_WAGE_BASE, years_worked=35, claim_age=FULL_RETIREMENT_AGE)
    assert got == pytest.approx(expected_monthly * 12)


@pytest.mark.parametrize("age, factor", [(62, 0.70), (67, 1.00), (70, 1.24)])
def test_claim_factor_early_full_delayed(age, factor):
    assert claim_factor(age) == pytest.approx(factor, abs=1e-9)


def test_claim_age_is_clamped_to_62_70():
    assert claim_factor(55) == claim_factor(62)   # can't claim before 62
    assert claim_factor(75) == claim_factor(70)   # no credits past 70


def test_under_40_credits_yields_nothing():
    assert estimate_annual_benefit(150_000, credits=CREDITS_TO_QUALIFY - 1) == 0.0
    assert estimate_annual_benefit(150_000, credits=CREDITS_TO_QUALIFY) > 0.0


def test_haircut_scales_the_benefit():
    full = estimate_annual_benefit(150_000, claim_age=67)
    assert estimate_annual_benefit(150_000, claim_age=67, haircut=0.75) == pytest.approx(0.75 * full)


def test_high_earner_at_fra_is_near_the_real_max():
    # A full 35-year max-wage career, claimed at FRA, lands near the real-world
    # maximum benefit (~$4k/mo for this wage-base year) — a sanity anchor.
    got = estimate_annual_benefit(SS_WAGE_BASE, years_worked=35, claim_age=67)
    assert 45_000 < got < 50_000
