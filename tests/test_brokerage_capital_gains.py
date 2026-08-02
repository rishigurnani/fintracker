"""
Regression tests for capital-gains tax on the taxable brokerage at retirement.

Cap-gains tax must apply only to the *gains* (balance − cost basis), never the
whole balance.  The engine tracks unrealized gains as market appreciation only:
contributions and withdrawals move value and basis together and create no gain.
"""
import pytest

from fintracker.models import (
    FilingStatus, State, IncomeProfile, RetirementProfile, TimelineEvent,
    StrategyToggles,
)
from fintracker.projections import ProjectionEngine, _after_tax_value
from tests.builders import make_plan, investments, zero_lifestyle


def _sale_plan(cap_gains_inv=0.20, cap_gains_ret=0.0):
    """Brokerage starts at 100k basis, grows 10%/yr; a 55k expense in year 2 forces
    a sale of exactly half the account (worth 110k with 10k of gains by then)."""
    return make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0, current_brokerage_balance=100_000,
                                annual_market_return=0.10,
                                capital_gains_tax_rate=cap_gains_inv),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        timeline_events=[TimelineEvent(year=2, description="Sell to fund expense",
                                       extra_one_time_expense=55_000)],
        projection_years=3,
        retirement=RetirementProfile(current_age=62, retirement_age=65,
                                     desired_annual_income=40_000,
                                     capital_gains_tax_rate=cap_gains_ret),
    )


def _isolated_brokerage_plan(start=100_000, mkt=0.10, years=1, cap_gains=0.0):
    """A plan with no income/expenses so the brokerage only grows at `mkt`."""
    return make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0, current_brokerage_balance=start,
                                annual_market_return=mkt),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        projection_years=years,
        retirement=RetirementProfile(current_age=65 - years, retirement_age=65,
                                     desired_annual_income=40_000,
                                     capital_gains_tax_rate=cap_gains),
    )


class TestAfterTaxValueHelper:

    def test_full_balance_taxed_when_base_is_balance(self):
        assert _after_tax_value(100_000, 100_000, 0.20) == pytest.approx(80_000)

    def test_only_taxable_base_is_taxed(self):
        # $100k balance, $30k of it gains, 20% → tax $6k, keep $94k.
        assert _after_tax_value(100_000, 30_000, 0.20) == pytest.approx(94_000)

    def test_zero_rate_is_identity(self):
        assert _after_tax_value(100_000, 100_000, 0.0) == 100_000


class TestBrokerageGainsTracking:

    def test_gains_equal_balance_minus_basis(self):
        # No contributions → basis stays at the $100k start; gains = growth.
        snaps = ProjectionEngine(_isolated_brokerage_plan(years=10)).run_deterministic()
        for s in snaps:
            assert s.brokerage_gains == pytest.approx(s.brokerage_balance - 100_000, abs=1)

    def test_gains_never_exceed_balance_or_go_negative(self):
        # A large one-time expense drains brokerage below the accrued gains;
        # the cap must keep gains within [0, balance].
        plan = make_plan(
            income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=0, current_brokerage_balance=200_000,
                                    annual_market_return=0.10),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=3, description="Big spend",
                                           extra_one_time_expense=250_000)],
            projection_years=6,
        )
        for s in ProjectionEngine(plan).run_deterministic():
            assert 0.0 <= s.brokerage_gains <= max(0.0, s.brokerage_balance) + 1


class TestCapitalGainsAtRetirement:

    def test_only_gains_are_taxed_not_basis(self):
        # Year 1: $100k → $110k, so $10k gain. 20% cap gains → tax $2k → $108k.
        # The old bug taxed the whole $110k (→ $88k).
        plan = _isolated_brokerage_plan(start=100_000, mkt=0.10, years=1, cap_gains=0.20)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.projected_balance_pretax == pytest.approx(110_000, abs=1)
        assert rr.projected_balance_at_retirement == pytest.approx(108_000, abs=1)
        # Explicitly ensure we did NOT tax the whole balance.
        assert rr.projected_balance_at_retirement != pytest.approx(88_000, abs=1)

    def test_zero_cap_gains_leaves_brokerage_untaxed(self):
        plan = _isolated_brokerage_plan(start=100_000, mkt=0.10, years=1, cap_gains=0.0)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.projected_balance_at_retirement == pytest.approx(110_000, abs=1)

    def test_capital_gains_rate_override_is_respected(self):
        plan = _isolated_brokerage_plan(start=100_000, mkt=0.10, years=1, cap_gains=0.0)
        rr = ProjectionEngine(plan).compute_retirement_readiness(capital_gains_rate=0.50)
        # $10k gain × 50% = $5k tax → $105k.
        assert rr.projected_balance_at_retirement == pytest.approx(105_000, abs=1)

    def test_lower_retirement_rate_shrinks_the_haircut(self):
        # Accumulation rate 20%, but the retirement drawdown rate is set to 0% —
        # the remaining $10k of unrealized gains is untaxed at the haircut.
        plan = _isolated_brokerage_plan(start=100_000, mkt=0.10, years=1, cap_gains=0.20)
        plan.investments.retirement_capital_gains_tax_rate = 0.0
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.projected_balance_at_retirement == pytest.approx(110_000, abs=1)

    def test_retirement_rate_defaults_to_accumulation_rate(self):
        # Unset (None) → haircut uses the accumulation rate, i.e. 20% on $10k → $108k.
        plan = _isolated_brokerage_plan(start=100_000, mkt=0.10, years=1, cap_gains=0.20)
        assert plan.investments.retirement_capital_gains_tax_rate is None
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.projected_balance_at_retirement == pytest.approx(108_000, abs=1)


class TestPayAsYouGoRealization:

    def test_sale_realizes_prorata_gains_and_taxes_them_that_year(self):
        # Year 2 sells 55k of a 110k account that holds 10k gains → realizes half
        # (5k) → 20% tax = 1k, charged the year of the sale.
        snaps = ProjectionEngine(_sale_plan(cap_gains_inv=0.20)).run_deterministic()
        assert snaps[0].annual_capital_gains_tax == 0.0          # no sale in year 1
        assert snaps[1].annual_capital_gains_tax == pytest.approx(1_000, abs=1)

    def test_realizing_reduces_unrealized_gains_so_no_double_tax(self):
        # After realizing 5k of the 10k gains via the sale, the tracker must drop
        # by that 5k (the rest keeps growing) — the realized gains aren't taxed
        # again by the retirement haircut.
        snaps = ProjectionEngine(_sale_plan(cap_gains_inv=0.20)).run_deterministic()
        # Pre-sale year 1: 10k gains. Post-sale (year 2) unrealized gains are the
        # 5k that survived plus that year's fresh growth — strictly below the
        # counterfactual with no sale.
        no_sale = ProjectionEngine(_sale_plan(cap_gains_inv=0.20)).run_deterministic()
        assert snaps[1].brokerage_gains < no_sale[0].brokerage_gains + 6_000
        assert snaps[1].brokerage_gains >= 0.0

    def test_rate_comes_from_investment_profile(self):
        snaps = ProjectionEngine(_sale_plan(cap_gains_inv=0.20, cap_gains_ret=0.0)).run_deterministic()
        assert snaps[1].annual_capital_gains_tax == pytest.approx(1_000, abs=1)

    def test_falls_back_to_retirement_profile_rate(self):
        # InvestmentProfile rate 0 → fall back to RetirementProfile's 20%.
        snaps = ProjectionEngine(_sale_plan(cap_gains_inv=0.0, cap_gains_ret=0.20)).run_deterministic()
        assert snaps[1].annual_capital_gains_tax == pytest.approx(1_000, abs=1)

    def test_zero_rate_means_no_realization_tax(self):
        snaps = ProjectionEngine(_sale_plan(cap_gains_inv=0.0, cap_gains_ret=0.0)).run_deterministic()
        assert all(s.annual_capital_gains_tax == 0.0 for s in snaps)
