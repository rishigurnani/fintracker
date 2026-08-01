"""
Regression tests for the wedding sinking fund (save-then-spend).

Wedding contributions are held in brokerage and grow at the market rate, then
the accrued fund is paid out for the wedding when the child turns 26.  Before
this, the yearly savings were subtracted from cash flow but never accumulated —
the money silently vanished from net worth.
"""
import pytest

from fintracker.models import (
    FilingStatus, State, IncomeProfile, StrategyToggles, TimelineEvent,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, zero_lifestyle


def _snaps(rate, mkt=0.0, years=30, children=1, events=None):
    plan = make_plan(
        income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=zero_lifestyle(num_children=children, annual_wedding_fund_per_child=rate),
        investments=investments(current_liquid_cash=1_000_000, annual_market_return=mkt),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        timeline_events=events or [],
        projection_years=years,
    )
    return ProjectionEngine(plan).run_deterministic()


class TestWeddingSinkingFund:

    def test_yearly_savings_figure_is_unchanged(self):
        # Child (born year 0) saves through age 25 → years 1..25, then stops.
        saves = [s.annual_wedding_save for s in _snaps(5_000)]
        assert all(s == 5_000 for s in saves[:25])   # years 1..25
        assert saves[25] == 0.0                       # year 26: saving has stopped

    def test_savings_are_held_not_vanished_during_accumulation(self):
        # The old bug reduced net worth every saving year (money vanished). Now the
        # money is retained (held in brokerage), so during accumulation net worth
        # matches the no-fund baseline instead of falling below it.
        with_fund = _snaps(5_000, mkt=0.0)
        without   = _snaps(0.0, mkt=0.0)
        assert with_fund[4].net_worth == pytest.approx(without[4].net_worth, abs=1)
        assert with_fund[24].net_worth == pytest.approx(without[24].net_worth, abs=1)

    def test_fund_is_spent_at_the_wedding_then_consumed(self):
        with_fund = _snaps(5_000, mkt=0.0)
        without   = _snaps(0.0, mkt=0.0)
        # Year 26 (age 26): the full accrued fund is paid out, and only then.
        assert with_fund[25].annual_wedding_spend == pytest.approx(125_000, abs=1)
        assert all(s.annual_wedding_spend == 0.0 for i, s in enumerate(with_fund) if i != 25)
        # From the wedding on, net worth is the baseline minus the wedding cost.
        assert with_fund[26].net_worth == pytest.approx(without[26].net_worth - 125_000, abs=1)

    def test_fund_grows_while_invested(self):
        # With a positive market return the payout exceeds the nominal contributions.
        grown = _snaps(5_000, mkt=0.06)[25].annual_wedding_spend
        assert grown > 125_000

    def test_todays_dollars_conversion_uses_cumulative_inflation(self):
        # cumulative_inflation is the engine's own price-level factor: (1+inf)^(year-1).
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(num_children=1, annual_wedding_fund_per_child=5_000),
            investments=investments(current_liquid_cash=1_000_000,
                                    annual_market_return=0.0, annual_inflation_rate=0.03),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=30,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].cumulative_inflation == pytest.approx(1.0)   # year 1 baseline
        wedding = snaps[25]                                          # year 26 payout
        assert wedding.cumulative_inflation == pytest.approx(1.03 ** 25, rel=1e-9)
        # The nominal payout is worth materially less in today's dollars.
        today = wedding.to_todays_dollars(wedding.annual_wedding_spend)
        assert today == pytest.approx(wedding.annual_wedding_spend / 1.03 ** 25, rel=1e-9)
        assert today < wedding.annual_wedding_spend

    def test_no_fund_means_no_saving_or_spending(self):
        snaps = _snaps(0.0)
        assert all(s.annual_wedding_save == 0.0 for s in snaps)
        assert all(s.annual_wedding_spend == 0.0 for s in snaps)

    def test_child_born_midway_gets_their_own_wedding(self):
        # No children at start; one is born in year 2 → wedding 26 years later.
        events = [TimelineEvent(year=2, description="Child", new_child=True)]
        snaps = _snaps(5_000, mkt=0.0, years=30, children=0, events=events)
        # Born in year 2 at age 0, saves ages 0..25 → 26 contributions of 5k = 130k.
        assert snaps[1].annual_wedding_save == 5_000     # year 2, first saving year
        wedding = [i for i, s in enumerate(snaps) if s.annual_wedding_spend > 0]
        assert wedding == [27]                            # year 28 = age 26
        assert snaps[27].annual_wedding_spend == pytest.approx(130_000, abs=1)
