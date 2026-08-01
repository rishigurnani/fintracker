"""
Regression tests for the age-aware 401k elective-deferral limit.

The projection engine caps 401k contributions at the base limit ($23,000) for
people under 50 and the catch-up limit ($30,500) at 50+, using the primary's age
derived from the RetirementProfile.  When no RetirementProfile is configured the
age is unknown and the catch-up (most permissive) ceiling is used so age-agnostic
plans are unaffected.
"""
import pytest

from fintracker.constants import LIMIT_401K, LIMIT_401K_CATCHUP, limit_401k
from fintracker.models import (
    FilingStatus, State, IncomeProfile, RetirementProfile, StrategyToggles,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments


def _retirement(current_age):
    return RetirementProfile(current_age=current_age, retirement_age=65,
                             desired_annual_income=80_000, years_in_retirement=30)


def _contribs(plan):
    return [s.annual_retirement_contributions for s in ProjectionEngine(plan).run_deterministic()]


# ── The pure limit helper ───────────────────────────────────────────────────

class TestLimit401kHelper:

    def test_under_50_uses_base_limit(self):
        assert limit_401k(35) == LIMIT_401K
        assert limit_401k(49) == LIMIT_401K

    def test_exactly_50_uses_catchup(self):
        assert limit_401k(50) == LIMIT_401K_CATCHUP

    def test_over_50_uses_catchup(self):
        assert limit_401k(65) == LIMIT_401K_CATCHUP

    def test_unknown_age_falls_back_to_catchup(self):
        """No age info → most permissive ceiling, so age-agnostic plans are unchanged."""
        assert limit_401k(None) == LIMIT_401K_CATCHUP


# ── Engine integration ──────────────────────────────────────────────────────

class TestAgeAware401kInProjection:

    def test_under_50_caps_stated_contribution_to_base(self):
        """A 35-year-old stating $30,500 is capped to the $23,000 base limit."""
        plan = make_plan(
            investments=investments(current_liquid_cash=500_000,
                                    annual_401k_contribution=30_500),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            retirement=_retirement(35),
            projection_years=3,
        )
        assert all(c == pytest.approx(LIMIT_401K, abs=1) for c in _contribs(plan))

    def test_over_50_allows_catchup_contribution(self):
        """A 55-year-old stating $30,500 keeps the full catch-up amount."""
        plan = make_plan(
            investments=investments(current_liquid_cash=500_000,
                                    annual_401k_contribution=30_500),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            retirement=_retirement(55),
            projection_years=3,
        )
        assert all(c == pytest.approx(LIMIT_401K_CATCHUP, abs=1) for c in _contribs(plan))

    def test_catchup_unlocks_the_year_the_primary_turns_50(self):
        """Crossing 50 mid-projection: base limit before, catch-up from age 50 on."""
        # current_age 48 → ages 48, 49, 50, 51, 52 across projection years 1..5
        plan = make_plan(
            investments=investments(current_liquid_cash=500_000,
                                    annual_401k_contribution=30_500),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            retirement=_retirement(48),
            projection_years=5,
        )
        contribs = _contribs(plan)
        assert contribs[0] == pytest.approx(LIMIT_401K, abs=1)          # age 48
        assert contribs[1] == pytest.approx(LIMIT_401K, abs=1)          # age 49
        assert contribs[2] == pytest.approx(LIMIT_401K_CATCHUP, abs=1)  # age 50
        assert contribs[3] == pytest.approx(LIMIT_401K_CATCHUP, abs=1)  # age 51
        assert contribs[4] == pytest.approx(LIMIT_401K_CATCHUP, abs=1)  # age 52

    def test_no_retirement_profile_keeps_catchup_ceiling(self):
        """Backward compat: without a RetirementProfile the catch-up cap applies."""
        plan = make_plan(
            investments=investments(current_liquid_cash=500_000,
                                    annual_401k_contribution=30_500),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            projection_years=2,
        )
        assert all(c == pytest.approx(LIMIT_401K_CATCHUP, abs=1) for c in _contribs(plan))

    def test_below_base_limit_is_never_raised(self):
        """A stated amount under the base limit is honored exactly, regardless of age."""
        plan = make_plan(
            investments=investments(current_liquid_cash=500_000,
                                    annual_401k_contribution=15_000),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            retirement=_retirement(60),
            projection_years=2,
        )
        assert all(c == pytest.approx(15_000, abs=1) for c in _contribs(plan))

    def test_partner_contribution_shares_the_primary_age_basis(self):
        """Partner 401k is capped on the same age basis (no separate partner age)."""
        plan = make_plan(
            income=IncomeProfile(120_000, FilingStatus.MARRIED_FILING_JOINTLY,
                                 State.TEXAS, spouse_gross_annual_income=120_000),
            investments=investments(current_liquid_cash=500_000,
                                    annual_401k_contribution=0,
                                    partner_annual_401k_contribution=30_500),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            retirement=_retirement(35),   # under 50 → partner capped to base too
            projection_years=2,
        )
        # Only the partner contributes; total retirement contribution == base cap.
        assert all(c == pytest.approx(LIMIT_401K, abs=1) for c in _contribs(plan))
