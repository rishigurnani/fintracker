"""
Engine-wide invariants — the guardrail for the account/cash-flow settlement layer.

These assert properties that must hold for EVERY projection regardless of the
funding path taken, so any regression in how money moves between accounts is
caught here rather than in a single feature test:

  * accounts that represent real savings never go negative (only the brokerage
    may, and only as a last-resort insolvency signal);
  * net worth always equals the sum of its components.

They run over the shipped configs plus a retired, HSA-funded, car-owning plan
that actually exercises the waterfall.
"""
import pytest

from fintracker.config import load_plan
from fintracker.models import (
    FilingStatus, IncomeProfile, LifestyleProfile, RetirementProfile, State,
    TimelineEvent,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, renting_housing


def _plans():
    plans = {
        "personal": load_plan("config/personal.yaml"),
        "sample": load_plan("config/sample.yaml"),
    }
    # A retired household with real medical costs, an HSA, and a replacing car —
    # exercises deficit funding, HSA medical spend, and purchase funding together.
    plans["retired_medical"] = make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=1_000),
        lifestyle=LifestyleProfile(annual_medical_oop=8_000, medical_auto_scale=False,
                                   annual_vacation=20_000),
        investments=investments(current_liquid_cash=100_000,
                                current_brokerage_balance=200_000,
                                current_retirement_balance=3_000_000,
                                annual_market_return=0.06, annual_inflation_rate=0.03,
                                annual_hsa_contribution=0),
        retirement=RetirementProfile(current_age=66, retirement_age=65,
                                     retirement_withdrawal_tax_rate=0.22),
        timeline_events=[TimelineEvent(year=3, description="big expense",
                                       extra_one_time_expense=250_000)],
        projection_years=20,
    )
    return plans


NON_NEGATIVE_ACCOUNTS = (
    "retirement_balance", "hsa_balance", "roth_ira_balance",
    "college_529_balance", "uninvested_cash", "cash_buffer",
)


@pytest.mark.parametrize("name", ["personal", "sample", "retired_medical"])
def test_savings_accounts_never_negative(name):
    """Every real savings account stays >= 0 across the whole projection. Only
    brokerage may dip below zero (last-resort insolvency), so it is excluded."""
    plan = _plans()[name]
    for s in ProjectionEngine(plan).run_deterministic():
        for field in NON_NEGATIVE_ACCOUNTS:
            assert getattr(s, field) >= -1.0, \
                f"{name} year {s.year}: {field} = {getattr(s, field):,.2f} < 0"


@pytest.mark.parametrize("name", ["personal", "sample", "retired_medical"])
def test_net_worth_equals_sum_of_components(name):
    """Net worth is exactly the sum of the balance-sheet components every year
    (the life-insurance payout lands only in the final/death year)."""
    plan = _plans()[name]
    snaps = ProjectionEngine(plan).run_deterministic()
    for s in snaps:
        components = (
            s.retirement_balance + s.brokerage_balance + s.hsa_balance
            + s.roth_ira_balance + s.college_529_balance + s.home_equity
            + s.uninvested_cash + s.cash_buffer + s.business_equity
        )
        # The final year may include a one-off life-insurance death benefit.
        if s is snaps[-1]:
            assert s.net_worth >= components - 1.0
        else:
            assert abs(s.net_worth - components) < 1.0, \
                f"{name} year {s.year}: nw={s.net_worth:,.2f} vs components={components:,.2f}"
