"""
Selling a primary residence realizes a capital gain = amount realized (sale price
net of selling costs) minus the cost basis (purchase price). The IRC §121 exclusion
shelters the first $250k (single) / $500k (married); the rest is taxed at the
capital-gains rate. Proceeds still land in brokerage, but the gain is no longer
added tax-free.

Market return is pinned to 0 so the brokerage carries no gains of its own — the
only realized gain in the sale year is the home sale, making the tax exact.
"""
import pytest

from fintracker.models import (
    FilingStatus, HousingProfile, IncomeProfile, State, StrategyToggles, TimelineEvent,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, zero_lifestyle


def _home_sale_plan(filing_status, appreciation, cap_gains=0.20, basis=400_000, income=100_000):
    """Owns a `basis` home outright, sells and downsizes in year 10."""
    return make_plan(
        income=IncomeProfile(income, filing_status, State.TEXAS),
        housing=HousingProfile(basis, basis, 0.0),           # owned, cash, no mortgage
        lifestyle=zero_lifestyle(),
        investments=investments(current_liquid_cash=800_000, current_brokerage_balance=0,
                                annual_market_return=0.0, annual_home_appreciation_rate=appreciation,
                                capital_gains_tax_rate=cap_gains),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        timeline_events=[TimelineEvent(year=10, description="downsize", buy_home=True,
                                       new_home_price=100_000, new_home_down_payment=100_000,
                                       new_home_interest_rate=0.0, sell_current_home=True,
                                       seller_closing_cost_rate=0.06, buyer_closing_cost_rate=0.0)],
        projection_years=10,
    )


def _sale_year_tax(plan):
    return ProjectionEngine(plan).run_deterministic()[9].annual_capital_gains_tax  # year 10


class TestHomeSaleCapGains:

    def test_gain_under_exclusion_is_untaxed(self):
        """A modest gain fully within the $250k exclusion produces no tax."""
        tax = _sale_year_tax(_home_sale_plan(FilingStatus.SINGLE, appreciation=0.02))
        assert tax == pytest.approx(0.0, abs=1.0)

    def test_large_gain_is_taxed(self):
        """A gain well above the exclusion is taxed at the cap-gains rate."""
        tax = _sale_year_tax(_home_sale_plan(FilingStatus.SINGLE, appreciation=0.15))
        assert tax > 0

    def test_married_exclusion_is_double_the_single(self):
        """Married shelters $500k vs $250k. With high income the whole gain sits in
        the 20% LTCG band, so the single filer pays exactly (500k − 250k) × 20% more
        — the extra $250k of exclusion at the top marginal rate."""
        single = _sale_year_tax(_home_sale_plan(FilingStatus.SINGLE, 0.15, income=1_000_000))
        married = _sale_year_tax(_home_sale_plan(FilingStatus.MARRIED_FILING_JOINTLY, 0.15, income=1_000_000))
        assert single - married == pytest.approx((500_000 - 250_000) * 0.20, rel=1e-4)
