"""Regression tests for income tax on pass-through business profit.

Business net profit is ordinary income on top of W-2 wages: it owes federal +
state income tax (on top of self-employment tax), its above-the-line / QBI
deductions reduce that tax, and a loss offsets W-2 income in-year (capped by
§461(l)). The old engine charged SE tax but *zero* income tax on business
profit, discarded the health/QBI deductions, and double-subtracted SE tax and
the solo-401k from cash flow.
"""
import pytest

from fintracker.models import (
    FilingStatus, State, IncomeProfile, BusinessProfile,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, zero_lifestyle


def _no_income_plan(w2):
    """Baseline: a W-2-only plan with no business, for differencing."""
    return make_plan(
        income=IncomeProfile(w2, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0),
        projection_years=1,
    )


def _business_plan(revenue, expense_ratio=0.0, w2=0, brokerage=0,
                   dividend_yield=0.0, **biz):
    biz.setdefault("use_qbi_deduction", False)
    biz.setdefault("equity_multiple", 0.0)
    return make_plan(
        income=IncomeProfile(w2, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=0,
                                current_brokerage_balance=brokerage,
                                taxable_dividend_yield=dividend_yield),
        business=BusinessProfile(annual_revenue=revenue, expense_ratio=expense_ratio,
                                 start_year=1, **biz),
        projection_years=1,
    )


class TestProfitIsIncomeTaxed:

    def test_business_profit_is_income_taxed(self):
        # $100k of business profit with no W-2 income: the engine must now levy
        # federal income tax on it. The old engine charged $0 income tax here.
        no_biz = ProjectionEngine(_no_income_plan(0)).run_deterministic()[0]
        biz = ProjectionEngine(_business_plan(100_000)).run_deterministic()[0]
        assert no_biz.annual_tax_total == pytest.approx(0.0, abs=1)   # truly no income
        assert biz.annual_tax_total > 10_000                          # profit is taxed

    def test_cash_flow_nets_se_tax_and_solo_401k_exactly_once(self):
        # With no W-2 income, spendable = business take-home − income tax. The old
        # line double-subtracted SE tax (and the solo-401k), which would drop
        # net_income well below this identity.
        snap = ProjectionEngine(_business_plan(100_000)).run_deterministic()[0]
        assert snap.net_income == pytest.approx(
            snap.annual_business_income - snap.annual_tax_total, abs=1)
        assert snap.net_income > 0

    def test_qbi_deduction_lowers_the_tax(self):
        # Turning QBI on shelters 20% of the profit, so income tax must fall.
        off = ProjectionEngine(_business_plan(100_000)).run_deterministic()[0]
        on = ProjectionEngine(
            _business_plan(100_000, use_qbi_deduction=True)).run_deterministic()[0]
        assert on.annual_tax_total < off.annual_tax_total


class TestLossHandling:

    def test_loss_offsets_w2_and_se_tax_is_floored(self):
        # A $50k business loss against a $150k salary. SE tax is floored at 0 on a
        # loss, so take-home equals the loss exactly (the old code produced a bogus
        # positive SE "credit" that shrank the loss by ~$1,339).
        loss = ProjectionEngine(
            _business_plan(50_000, expense_ratio=2.0, w2=150_000)).run_deterministic()[0]
        no_biz = ProjectionEngine(_no_income_plan(150_000)).run_deterministic()[0]
        assert loss.annual_business_income == pytest.approx(-50_000, abs=1)
        # The loss shelters W-2 income (tax falls) but by less than the full loss.
        assert loss.annual_tax_total < no_biz.annual_tax_total
        assert 0 < (no_biz.annual_tax_total - loss.annual_tax_total) < 50_000

    def test_excess_business_loss_is_capped(self):
        # High W-2 so a large loss can't merely zero out taxable income. A loss beyond
        # the §461(l) limit ($305k single, 2024) shelters no more than one at the cap.
        def tax(net_loss):
            plan = _business_plan(abs(net_loss), expense_ratio=2.0, w2=1_200_000)
            return ProjectionEngine(plan).run_deterministic()[0].annual_tax_total
        at_cap = tax(-305_000)
        over_cap = tax(-700_000)
        under_cap = tax(-100_000)
        assert over_cap == pytest.approx(at_cap, abs=50)   # past the cap: no more shelter
        assert under_cap > at_cap                          # a bigger (capped) loss shelters more


class TestBusinessLiftsLTCGStackingFloor:

    def test_business_income_pushes_dividends_out_of_the_0pct_band(self):
        # $30k of qualified dividends (10% on $300k). Alone, they sit in the 0% LTCG
        # band → untaxed. $60k of business profit lifts the ordinary-income floor the
        # dividends stack on, pushing them into the 15% band. The old code stacked
        # preferential income on W-2 wages only, so business income never raised it.
        no_biz = ProjectionEngine(
            _business_plan(0, brokerage=300_000, dividend_yield=0.10)
        ).run_deterministic()[0]
        # revenue=0 means net_profit 0; use a separate plan WITH profit:
        with_biz = ProjectionEngine(
            _business_plan(60_000, brokerage=300_000, dividend_yield=0.10)
        ).run_deterministic()[0]
        assert no_biz.annual_capital_gains_tax == pytest.approx(0.0, abs=1)
        assert with_biz.annual_capital_gains_tax > 1_000
