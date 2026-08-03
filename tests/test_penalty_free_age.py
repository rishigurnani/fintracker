"""
Penalty-free access to the 401k/IRA and Roth earnings is gated on age 59½,
decoupled from ``retirement_age``:

  * an EARLY retiree (retired before 59½) still cannot tap the 401k without
    penalty — the deficit falls to brokerage/negative, not the 401k;
  * a STILL-WORKING 60-year-old past 59½ can tap it.

This isolates the gate from the retirement event (income stop / de-risking).
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, RetirementProfile, State,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, renting_housing, zero_lifestyle


def _plan(current_age, retirement_age, annual_rent, brokerage=0.0, retirement=1_000_000.0):
    return make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=annual_rent / 12),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0, current_brokerage_balance=brokerage,
                                current_retirement_balance=retirement),
        retirement=RetirementProfile(current_age=current_age, retirement_age=retirement_age,
                                     medicare_start_age=200, expected_post_retirement_return=0.0),
        projection_years=2,
    )


def _y1(plan):
    return ProjectionEngine(plan).run_deterministic()[0]


def test_early_retiree_below_59_cannot_tap_401k():
    """Retired at 55: at age 55 the 401k is penalty-locked, so a deficit with no
    brokerage drives brokerage negative rather than drawing the 401k."""
    s = _y1(_plan(current_age=55, retirement_age=55, annual_rent=40_000,
                  brokerage=0.0, retirement=1_000_000))
    assert s.annual_retirement_withdrawal == pytest.approx(0.0, abs=1.0)
    assert s.retirement_balance == pytest.approx(1_000_000, rel=1e-9)  # untouched
    assert s.brokerage_balance == pytest.approx(-40_000, rel=1e-4)     # last resort


def test_still_working_past_59_can_tap_401k():
    """Age 60 but not retired until 65: past 59½ the 401k is penalty-free, so it
    funds a deficit the brokerage can't cover."""
    s = _y1(_plan(current_age=60, retirement_age=65, annual_rent=40_000,
                  brokerage=0.0, retirement=1_000_000))
    assert s.annual_retirement_withdrawal == pytest.approx(40_000, rel=1e-6)  # 0% wd tax
    assert s.retirement_balance == pytest.approx(960_000, rel=1e-6)
    assert s.brokerage_balance == pytest.approx(0.0, abs=1.0)          # not driven negative
