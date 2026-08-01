"""
Regression tests for the salary real-growth plateau.

After ``salary_growth_peak_age`` the projection stops granting real raises:
nominal salary growth is capped at inflation minus ``salary_real_decline_rate``
and is never lifted above the underlying rate.  Age comes from the
RetirementProfile; without one the salary grows unchanged.
"""
import pytest

from fintracker.models import (
    FilingStatus, State, IncomeProfile, RetirementProfile, InvestmentProfile,
    StrategyToggles,
)
from fintracker.projections import ProjectionEngine
from fintracker.config import save_plan, load_plan
from tests.builders import make_plan, investments


def _plan(current_age=50, years=7, growth=0.04, inflation=0.03, peak=55, decline=0.0):
    return make_plan(
        income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
        investments=investments(
            current_liquid_cash=500_000,
            annual_salary_growth_rate=growth,
            annual_inflation_rate=inflation,
            salary_growth_peak_age=peak,
            salary_real_decline_rate=decline,
        ),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        retirement=RetirementProfile(current_age=current_age, retirement_age=65,
                                     desired_annual_income=80_000, years_in_retirement=30),
        projection_years=years,
    )


def _incomes(plan):
    return [s.gross_income for s in ProjectionEngine(plan).run_deterministic()]


class TestSalaryPlateau:

    def test_full_growth_before_peak_inflation_only_after(self):
        # current_age 50, peak 55: the primary turns 55 entering projection year 6,
        # so growth is 4% through year 5 and inflation (3%) from year 6 on.
        inc = _incomes(_plan(current_age=50, peak=55, growth=0.04, inflation=0.03))
        assert inc[4] / inc[3] == pytest.approx(1.04, rel=1e-6)   # pre-peak: real raise
        assert inc[5] / inc[4] == pytest.approx(1.03, rel=1e-6)   # post-peak: inflation only
        assert inc[6] / inc[5] == pytest.approx(1.03, rel=1e-6)

    def test_peak_age_is_configurable(self):
        # peak 53 (vs default 55): the plateau starts two transitions earlier.
        inc = _incomes(_plan(current_age=50, peak=53))
        assert inc[2] / inc[1] == pytest.approx(1.04, rel=1e-6)   # still pre-peak
        assert inc[3] / inc[2] == pytest.approx(1.03, rel=1e-6)   # plateaus earlier

    def test_real_decline_rate_reduces_below_inflation(self):
        inc = _incomes(_plan(current_age=50, peak=55, growth=0.04, inflation=0.03, decline=0.02))
        # post-peak nominal growth = inflation - decline = 0.03 - 0.02 = 0.01
        assert inc[5] / inc[4] == pytest.approx(1.01, rel=1e-6)

    def test_plateau_never_raises_growth_above_the_underlying_rate(self):
        # base growth (0) is already below inflation — the plateau must not lift it.
        inc = _incomes(_plan(current_age=50, peak=55, growth=0.0, inflation=0.03))
        assert inc[5] / inc[4] == pytest.approx(1.0, rel=1e-9)

    def test_no_retirement_profile_grows_unchanged(self):
        plan = make_plan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
            investments=investments(current_liquid_cash=500_000,
                                    annual_salary_growth_rate=0.04, annual_inflation_rate=0.03),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=7,   # no retirement=... → age unknown → no plateau
        )
        inc = _incomes(plan)
        assert inc[6] / inc[5] == pytest.approx(1.04, rel=1e-6)

    def test_defaults_are_backward_compatible(self):
        inv = InvestmentProfile()
        assert inv.salary_growth_peak_age == 55
        assert inv.salary_real_decline_rate == 0.0

    def test_config_round_trip_preserves_plateau_fields(self, tmp_path):
        plan = _plan(peak=52, decline=0.015)
        path = tmp_path / "plan.yaml"
        save_plan(plan, path)
        loaded = load_plan(path)
        assert loaded.investments.salary_growth_peak_age == 52
        assert loaded.investments.salary_real_decline_rate == pytest.approx(0.015)
