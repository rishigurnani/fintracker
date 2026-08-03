"""
The Roth waterfall must offer the *whole* balance (contributions + earnings) once
retired — a qualified, tax-free distribution after 59½ — not only the contribution
basis. Before retirement only the basis is penalty-free, so earnings stay locked.
The draw is reported in annual_roth_withdrawal.
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, RetirementProfile, State,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, renting_housing, zero_lifestyle


def _plan(current_age, retirement_age, annual_rent, roth_balance, roth_basis):
    """Retired/working household whose ONLY deficit is rent, with no brokerage,
    cash, or 401k — so any funding must come from the Roth. Roth is seeded with
    earnings above basis to test that earnings are (in)accessible by age."""
    plan = make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=annual_rent / 12),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0, current_brokerage_balance=0,
                                current_retirement_balance=0),
        retirement=RetirementProfile(current_age=current_age, retirement_age=retirement_age,
                                     medicare_start_age=200, expected_post_retirement_return=0.0),
        projection_years=2,
    )
    eng = ProjectionEngine(plan)
    return eng, roth_balance, roth_basis


def _run(eng, roth_balance, roth_basis):
    state = eng._initial_state()
    state.roth_ira_balance        = roth_balance
    state.roth_contribution_basis = roth_basis
    state.roth_vested_basis       = roth_basis
    snaps = []
    for year in range(1, eng._horizon() + 1):
        eng._apply_timeline_events(state, year)
        snap = eng._compute_year(state, year)
        snaps.append(snap)
        eng._advance_state(state, snap)
    return snaps


class TestRothWaterfall:

    def test_retired_full_roth_including_earnings_is_spendable(self):
        """Retired: a $300k deficit is fully covered from a Roth with only $100k
        basis but $500k total — earnings are drawn tax-free, not left stranded."""
        eng, bal, basis = _plan(current_age=66, retirement_age=65, annual_rent=300_000,
                                roth_balance=500_000, roth_basis=100_000)
        s = _run(eng, bal, basis)[0]
        assert s.annual_roth_withdrawal == pytest.approx(300_000, rel=1e-6)
        assert s.roth_ira_balance == pytest.approx(200_000, rel=1e-6)
        assert s.brokerage_balance == pytest.approx(0.0, abs=1.0)  # not driven negative

    def test_pre_retirement_only_basis_is_accessible(self):
        """Before retirement only the $100k basis is penalty-free; the remaining
        $200k of the deficit can't touch earnings and books as negative brokerage."""
        eng, bal, basis = _plan(current_age=40, retirement_age=65, annual_rent=300_000,
                                roth_balance=500_000, roth_basis=100_000)
        s = _run(eng, bal, basis)[0]
        assert s.annual_roth_withdrawal == pytest.approx(100_000, rel=1e-6)  # basis only
        assert s.roth_ira_balance == pytest.approx(400_000, rel=1e-6)        # earnings locked
        assert s.brokerage_balance == pytest.approx(-200_000, rel=1e-4)      # last resort
