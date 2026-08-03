"""
The projection must grow assets at the accumulation market return while working
and de-risk to ``expected_post_retirement_return`` once retired — the same rate
the retirement-readiness calc discounts at. A single Monte-Carlo sampled return
still overrides both (variability is modelled there).
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, RetirementProfile, State,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, renting_housing, zero_lifestyle


def _idle_plan(current_age, retirement_age, market, post_ret, years=2):
    """A household with no income/expenses/contributions, so balances change only
    by the investment-return regime under test."""
    return make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=0),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0, current_retirement_balance=1_000_000,
                                annual_market_return=market),
        retirement=RetirementProfile(current_age=current_age, retirement_age=retirement_age,
                                     medicare_start_age=200,  # keep Medicare out of it
                                     expected_post_retirement_return=post_ret),
        projection_years=years,
    )


def test_working_years_grow_at_market_return():
    plan = _idle_plan(current_age=40, retirement_age=65, market=0.08, post_ret=0.05)
    y1 = ProjectionEngine(plan).run_deterministic()[0]
    assert y1.retirement_balance == pytest.approx(1_080_000, rel=1e-6)


def test_retired_years_grow_at_post_retirement_return():
    plan = _idle_plan(current_age=66, retirement_age=65, market=0.08, post_ret=0.05)
    y1 = ProjectionEngine(plan).run_deterministic()[0]
    assert y1.retirement_balance == pytest.approx(1_050_000, rel=1e-6)  # 5%, not 8%


def test_monte_carlo_sampled_return_overrides_regime():
    """A sampled return passed as override wins even in a retired year."""
    plan = _idle_plan(current_age=66, retirement_age=65, market=0.08, post_ret=0.05)
    eng = ProjectionEngine(plan)
    state = eng._initial_state()
    eng._apply_timeline_events(state, 1)
    snap = eng._compute_year(state, 1, market_return_override=0.20)
    assert snap.retirement_balance == pytest.approx(1_200_000, rel=1e-6)  # 20% override
