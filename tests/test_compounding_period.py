"""
Tunable compounding period (InvestmentProfile.compounding_period_months).

Annual rates are converted geometrically, so a lump starting balance grows by
exactly (1 + annual) at any period — only DEPOSITS made during the year change,
earning partial-year growth when compounding is finer than annual.
"""
import pytest

from fintracker.models import (
    FilingStatus, HousingProfile, IncomeProfile, State, StrategyToggles,
)
from fintracker.projections import ProjectionEngine, _deposit_growth_factor
from tests.builders import make_plan, investments, zero_lifestyle, renting_housing


def _annuity_factor_bruteforce(r, period_months):
    """FV of n equal end-of-period deposits summing to $1, computed by summation —
    an independent check of the closed form."""
    n = int(round(12 / period_months))
    rp = (1 + r) ** (1 / n) - 1
    return sum((1 / n) * (1 + rp) ** (n - k) for k in range(1, n + 1))


class TestDepositGrowthFactor:

    def test_annual_period_is_identity(self):
        assert _deposit_growth_factor(0.08, 12) == 1.0

    def test_monthly_factor_matches_expected(self):
        # 8% annual, monthly: ordinary-annuity factor ≈ 1.03616.
        assert _deposit_growth_factor(0.08, 1) == pytest.approx(1.03616, abs=1e-5)

    @pytest.mark.parametrize("period", [1, 3, 6, 12])
    def test_closed_form_matches_bruteforce_annuity(self, period):
        assert _deposit_growth_factor(0.08, period) == pytest.approx(
            _annuity_factor_bruteforce(0.08, period), rel=1e-9)

    def test_finer_period_earns_more(self):
        vals = [_deposit_growth_factor(0.08, p) for p in (0.5, 1, 3, 6, 12, 24)]
        assert vals == sorted(vals, reverse=True)   # strictly decreasing with period
        assert vals[-2] == 1.0                       # period 12 → exactly 1

    def test_zero_rate_deposits_do_not_grow(self):
        assert _deposit_growth_factor(0.0, 1) == 1.0


def _lump_plan(period, income=0, k401=0, maximize=False):
    return make_plan(
        income=IncomeProfile(income, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=0),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0, current_brokerage_balance=100_000,
                                current_retirement_balance=100_000,
                                annual_401k_contribution=k401, annual_market_return=0.08,
                                compounding_period_months=period),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=maximize),
        projection_years=10,
    )


class TestCompoundingInProjection:

    def test_lump_sum_growth_is_period_independent(self):
        """No deposits → the balance is a pure lump sum, so monthly and annual
        compounding must give the identical result (the geometric-conversion
        property). This is the invariant that keeps the change principled."""
        annual = ProjectionEngine(_lump_plan(12)).run_deterministic()[-1].brokerage_balance
        monthly = ProjectionEngine(_lump_plan(1)).run_deterministic()[-1].brokerage_balance
        assert monthly == pytest.approx(annual, rel=1e-9)
        assert annual == pytest.approx(100_000 * 1.08 ** 10, rel=1e-9)

    def test_contributions_grow_more_under_monthly_compounding(self):
        """With yearly 401k contributions, monthly compounding lets each year's
        deposit earn part of a year's return, so the balance ends higher."""
        annual = ProjectionEngine(_lump_plan(12, income=200_000, k401=23_000, maximize=True)
                                  ).run_deterministic()[-1].retirement_balance
        monthly = ProjectionEngine(_lump_plan(1, income=200_000, k401=23_000, maximize=True)
                                   ).run_deterministic()[-1].retirement_balance
        assert monthly > annual
