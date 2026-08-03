"""
Tests for the lump-sum purchase funding waterfall (_fund_purchase).

Lump-sum outflows (one-time expenses, car/home down payments, business
investment, weddings) must be funded from the full asset waterfall — brokerage
→ cash → 401k (once retired, grossed up for tax) → Roth — instead of debiting
brokerage unconditionally. The old behaviour drove an empty brokerage into an
ever-compounding negative balance even with a large 401k available.

Zero market/inflation and (mostly) zero income isolate a single purchase so its
funding source is exact.
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, RetirementProfile, State, TimelineEvent,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, zero_lifestyle, renting_housing


def _project(plan):
    return ProjectionEngine(plan).run_deterministic()


def _retired_plan(brokerage=0.0, retirement=1_000_000.0, wd_tax=0.25,
                  events=None, years=2):
    # current_age > retirement_age so the household is retired from year 1;
    # medicare_start_age kept above the ages tested so no Medicare cost intrudes.
    return make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=0),
        lifestyle=zero_lifestyle(),
        investments=investments(
            current_liquid_cash=0, current_brokerage_balance=brokerage,
            current_retirement_balance=retirement,
        ),
        retirement=RetirementProfile(
            current_age=61, retirement_age=60, medicare_start_age=90,
            retirement_withdrawal_tax_rate=wd_tax,
            expected_post_retirement_return=0.0,  # isolate purchase funding from growth
        ),
        timeline_events=events or [],
        projection_years=years,
    )


class TestRetiredPurchaseFunding:

    def test_empty_brokerage_purchase_draws_401k_not_negative(self):
        """Retired, brokerage empty: a lump-sum expense is funded from the 401k
        (grossed up for tax), and brokerage never goes negative."""
        plan = _retired_plan(brokerage=0, retirement=1_000_000, wd_tax=0.25,
                             events=[TimelineEvent(year=1, description="big expense",
                                                   extra_one_time_expense=100_000)])
        s = _project(plan)[0]
        # $100k net funded from the 401k, grossed up: 100k / (1-0.25) = 133,333.
        assert s.brokerage_balance == pytest.approx(0.0, abs=1.0)   # NOT negative
        assert s.retirement_balance == pytest.approx(1_000_000 - 133_333.33, rel=1e-4)
        assert s.annual_retirement_withdrawal == pytest.approx(133_333.33, rel=1e-4)

    def test_brokerage_covers_purchase_leaves_401k_untouched(self):
        """When brokerage can cover the outflow, behaviour is unchanged: it comes
        from brokerage and the 401k is not tapped."""
        plan = _retired_plan(brokerage=200_000, retirement=1_000_000,
                             events=[TimelineEvent(year=1, description="expense",
                                                   extra_one_time_expense=100_000)])
        s = _project(plan)[0]
        assert s.brokerage_balance == pytest.approx(100_000, rel=1e-6)
        assert s.retirement_balance == pytest.approx(1_000_000, rel=1e-9)
        assert s.annual_retirement_withdrawal == pytest.approx(0.0, abs=1.0)

    def test_brokerage_never_negative_across_repeated_purchases(self):
        """Repeated lump-sum expenses that outstrip brokerage keep drawing the
        401k; brokerage floors at zero every year."""
        events = [TimelineEvent(year=y, description="expense",
                                extra_one_time_expense=100_000) for y in (1, 2, 3)]
        plan = _retired_plan(brokerage=50_000, retirement=2_000_000, events=events, years=4)
        snaps = _project(plan)
        assert all(s.brokerage_balance >= -1.0 for s in snaps), \
            [(s.year, round(s.brokerage_balance)) for s in snaps]
        # The 401k shrank to fund them.
        assert snaps[-1].retirement_balance < 2_000_000


class TestPreRetirementPurchaseFunding:

    def test_pre_retirement_purchase_does_not_tap_401k(self):
        """Before 59½/retirement the 401k is penalty-locked, so a purchase that
        exhausts brokerage is NOT funded from it — it books as negative brokerage
        (genuine insolvency) rather than silently raiding retirement savings."""
        plan = make_plan(
            income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
            housing=renting_housing(monthly_rent=0),
            lifestyle=zero_lifestyle(),
            investments=investments(
                current_liquid_cash=0, current_brokerage_balance=0,
                current_retirement_balance=1_000_000,
            ),
            retirement=RetirementProfile(current_age=40, retirement_age=65),
            timeline_events=[TimelineEvent(year=1, description="expense",
                                           extra_one_time_expense=50_000)],
            projection_years=2,
        )
        s = _project(plan)[0]
        assert s.retirement_balance == pytest.approx(1_000_000, rel=1e-9)  # untouched
        assert s.annual_retirement_withdrawal == pytest.approx(0.0, abs=1.0)
        assert s.brokerage_balance == pytest.approx(-50_000, rel=1e-4)      # last resort
