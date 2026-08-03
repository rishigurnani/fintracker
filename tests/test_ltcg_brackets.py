"""
Long-term capital-gains brackets (0/15/20%), stacked on ordinary taxable income
and inflation-indexed — so the same gain is taxed differently depending on the
year's income and price level, with no hand-set flat rate.
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, InvestmentProfile, State, StrategyToggles,
)
from fintracker.tax_engine import TaxEngine, _federal_ltcg_tax

# 2024 single breakpoints: 0% ≤ 47,025 ; 15% ≤ 518,900 ; 20% above.


class TestFederalBracketMath:
    """Direct tests of the stacking function with known ordinary income."""

    def test_gain_entirely_in_zero_band(self):
        # $30k gain on top of $0 ordinary income stays under the 47,025 top → 0%.
        assert _federal_ltcg_tax(30_000, 0, FilingStatus.SINGLE) == pytest.approx(0.0)

    def test_gain_entirely_in_15_band(self):
        # Ordinary income already above the 0% top; a $50k gain sits in the 15% band.
        assert _federal_ltcg_tax(50_000, 100_000, FilingStatus.SINGLE) == pytest.approx(7_500)

    def test_gain_entirely_in_20_band(self):
        # Ordinary income above the 15% top (518,900) → the whole gain is 20%.
        assert _federal_ltcg_tax(100_000, 600_000, FilingStatus.SINGLE) == pytest.approx(20_000)

    def test_gain_spanning_all_three_bands(self):
        # $0 ordinary income, $600k gain: 0% on 0–47,025, 15% on 47,025–518,900,
        # 20% on 518,900–600,000.
        expected = (518_900 - 47_025) * 0.15 + (600_000 - 518_900) * 0.20
        assert _federal_ltcg_tax(600_000, 0, FilingStatus.SINGLE) == pytest.approx(expected)

    def test_inflation_indexes_the_breakpoints(self):
        # At 2× the price level the 0% band doubles to ~94,050, so a $90k gain on
        # $0 income is still fully in the 0% band — untaxed.
        assert _federal_ltcg_tax(90_000, 0, FilingStatus.SINGLE, inflation_factor=2.0) == pytest.approx(0.0)


class TestEngineCapitalGainsTax:

    def _args(self, income, filing=FilingStatus.SINGLE, state=State.TEXAS):
        return (
            IncomeProfile(income, filing, state),
            InvestmentProfile(),
            StrategyToggles(maximize_hsa=False, maximize_401k=False),
        )

    def test_same_gain_taxed_less_in_a_low_income_year(self):
        """The retirement case: a $200k gain realized in a zero-income year is taxed
        far more lightly than in a high-earning year — automatically, via brackets."""
        eng = TaxEngine()
        low = eng.capital_gains_tax(200_000, *self._args(0))
        high = eng.capital_gains_tax(200_000, *self._args(600_000))
        assert low < high
        # Low year: entirely within the 15% band (200k < 518,900). High year: 20%.
        assert low == pytest.approx((200_000 - 47_025) * 0.15, rel=1e-3)
        assert high == pytest.approx(200_000 * 0.20, rel=1e-3)

    def test_state_adds_ordinary_rate_tax(self):
        """In a state that taxes gains as ordinary income (GA), the state piece is
        added on top of the federal bracket tax; a no-income-tax state (TX) has none."""
        eng = TaxEngine()
        tx = eng.capital_gains_tax(200_000, *self._args(100_000, state=State.TEXAS))
        ga = eng.capital_gains_tax(200_000, *self._args(100_000, state=State.GEORGIA))
        assert ga > tx
