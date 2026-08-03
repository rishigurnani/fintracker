"""
Tests for the retirement withdrawal waterfall:

  * the 401k/IRA is drawn to fund a retirement deficit (ordinary income, grossed
    up for tax) but never before retirement_age;
  * the draw order is configurable and has a balance-derived default;
  * IRMAA MAGI is driven by the actual withdrawal;
  * funding from the 401k stops the false "insolvency" of a negative brokerage.

Plans pin market/inflation/salary/rent growth to 0 so a single deficit resolves
to exact, hand-checkable withdrawals.
"""
import pytest

from fintracker.models import (
    FilingStatus, FinancialPlan, HousingProfile, IncomeProfile, InvestmentProfile,
    LifestyleProfile, RetirementProfile, State, StrategyToggles,
)
from fintracker.projections import (
    ProjectionEngine, _complete_order, _fund_deficit, WITHDRAWAL_SOURCES,
    _ORDER_BRACKET_FILL, _ORDER_CONVENTIONAL,
)
from tests.builders import investments, zero_lifestyle


# --- Plan builder: renter with a fixed rent deficit, retiring at 65 ------------

# Default straddles the 59½ penalty-free boundary: year 1 = age 58 (401k locked),
# year 2 = age 59 (penalty-free), which is what the drawdown tests below exercise.
def _ret_plan(rent_month=1_000, brokerage=0.0, retirement=0.0, liquid=0.0,
              wd_rate=0.0, order=None, current_age=58, married=False,
              medicare_premium=0.0, self_ltc=0.0, years=3):
    return FinancialPlan(
        income=IncomeProfile(
            0.0,
            FilingStatus.MARRIED_FILING_JOINTLY if married else FilingStatus.SINGLE,
            State.TEXAS),
        housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=rent_month,
                               annual_rent_increase_rate=0.0),
        lifestyle=zero_lifestyle(annual_self_ltc_cost=self_ltc, self_ltc_start_age=66),
        investments=investments(current_liquid_cash=liquid,
                                current_brokerage_balance=brokerage,
                                current_retirement_balance=retirement),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False,
                                   retirement_withdrawal_order=order),
        retirement=RetirementProfile(current_age=current_age, retirement_age=65,
                                     retirement_withdrawal_tax_rate=wd_rate,
                                     annual_medicare_premium=medicare_premium,
                                     # isolate withdrawal mechanics from growth
                                     expected_post_retirement_return=0.0),
        projection_years=years,
    )


def _project(plan):
    return ProjectionEngine(plan).run_deterministic()


# ══════════════════════════════════════════════════════════════════════════════
# Pure helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestFundDeficitHelper:

    def test_completes_and_sanitises_order(self):
        got = _complete_order(["brokerage", "bogus", "cash_buffer"], WITHDRAWAL_SOURCES)
        assert got[:2] == ["brokerage", "cash_buffer"]          # known kept, in order
        assert "bogus" not in got                               # unknown dropped
        assert set(got) == set(WITHDRAWAL_SOURCES)              # every source present

    def test_tax_free_draw_dollar_for_dollar(self):
        available = {k: 0.0 for k in WITHDRAWAL_SOURCES}
        available["brokerage"] = 100.0
        reductions, taxable, tax, short = _fund_deficit(
            available, ["brokerage"], 40.0, 0.25)
        assert reductions["brokerage"] == pytest.approx(40.0)
        assert taxable == 0.0 and tax == 0.0 and short == 0.0

    def test_401k_draw_is_grossed_up_for_tax(self):
        available = {k: 0.0 for k in WITHDRAWAL_SOURCES}
        available["retirement_401k"] = 1_000.0
        reductions, taxable, tax, short = _fund_deficit(
            available, ["retirement_401k"], 30.0, 0.25)
        # Need $30 net → withdraw 30/0.75 = 40 pre-tax, $10 tax.
        assert reductions["retirement_401k"] == pytest.approx(40.0)
        assert taxable == pytest.approx(40.0)
        assert tax == pytest.approx(10.0)
        assert short == 0.0

    def test_shortfall_when_sources_exhausted(self):
        available = {k: 0.0 for k in WITHDRAWAL_SOURCES}
        available["cash_buffer"] = 3.0
        _, _, _, short = _fund_deficit(available, ["cash_buffer"], 10.0, 0.0)
        assert short == pytest.approx(7.0)


# ══════════════════════════════════════════════════════════════════════════════
# 401k drawdown in retirement
# ══════════════════════════════════════════════════════════════════════════════

class TestRetirementDrawdown:

    def test_401k_not_touched_before_retirement(self):
        # Year 1 = age 58 (pre-59½): deficit funded from brokerage only, 401k locked.
        snaps = _project(_ret_plan(brokerage=50_000, retirement=500_000,
                                   order=list(_ORDER_BRACKET_FILL)))
        y1 = snaps[0]
        assert y1.annual_retirement_withdrawal == 0.0
        assert y1.retirement_balance == pytest.approx(500_000)      # untouched
        assert y1.brokerage_balance == pytest.approx(38_000)        # 50k - 12k rent

    def test_401k_funds_deficit_in_retirement(self):
        # Year 2 = age 59 (penalty-free): 401k-first order draws the deficit from the 401k.
        snaps = _project(_ret_plan(brokerage=50_000, retirement=500_000,
                                   order=list(_ORDER_BRACKET_FILL)))
        y2 = snaps[1]
        assert y2.annual_retirement_withdrawal == pytest.approx(12_000)
        assert y2.annual_retirement_withdrawal_tax == 0.0
        assert y2.retirement_balance == pytest.approx(488_000)      # 500k - 12k
        assert y2.brokerage_balance == pytest.approx(38_000)        # brokerage untouched

    def test_withdrawal_grossed_up_for_ordinary_income_tax(self):
        snaps = _project(_ret_plan(brokerage=50_000, retirement=500_000,
                                   wd_rate=0.25, order=list(_ORDER_BRACKET_FILL)))
        y2 = snaps[1]
        # $12k net need → 12k/0.75 = 16k pre-tax withdrawal, $4k tax.
        assert y2.annual_retirement_withdrawal == pytest.approx(16_000)
        assert y2.annual_retirement_withdrawal_tax == pytest.approx(4_000)
        assert y2.retirement_balance == pytest.approx(484_000)

    def test_conventional_order_spends_brokerage_first(self):
        snaps = _project(_ret_plan(brokerage=50_000, retirement=500_000,
                                   order=list(_ORDER_CONVENTIONAL)))
        y2 = snaps[1]
        assert y2.annual_retirement_withdrawal == 0.0               # 401k not reached
        assert y2.retirement_balance == pytest.approx(500_000)      # untouched
        assert y2.brokerage_balance == pytest.approx(26_000)        # 38k - 12k


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic default from starting balances
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultWithdrawalOrder:

    def test_large_401k_defaults_to_bracket_fill(self):
        # retirement >> brokerage → default draws the 401k before the brokerage.
        snaps = _project(_ret_plan(brokerage=50_000, retirement=500_000, order=None))
        y2 = snaps[1]
        assert y2.annual_retirement_withdrawal == pytest.approx(12_000)
        assert y2.brokerage_balance == pytest.approx(38_000)        # brokerage preserved

    def test_large_brokerage_defaults_to_conventional(self):
        # brokerage >> retirement → default spends the brokerage, 401k untouched.
        snaps = _project(_ret_plan(brokerage=500_000, retirement=50_000, order=None))
        y2 = snaps[1]
        assert y2.annual_retirement_withdrawal == 0.0
        assert y2.retirement_balance == pytest.approx(50_000)
        assert y2.brokerage_balance == pytest.approx(476_000)       # 488k - 12k


# ══════════════════════════════════════════════════════════════════════════════
# IRMAA driven by the real withdrawal; no false insolvency
# ══════════════════════════════════════════════════════════════════════════════

class TestWithdrawalDrivenIrmaaAndLiquidity:

    def test_irmaa_fires_on_large_withdrawal(self):
        # A big self-LTC deficit at 66 forces a large 401k withdrawal → high MAGI
        # → IRMAA on top of the base premium (married ⇒ base = 2 × premium).
        snaps = _project(_ret_plan(brokerage=0.0, retirement=5_000_000, liquid=0.0,
                                   married=True, medicare_premium=2_000,
                                   self_ltc=300_000, current_age=65, years=3))
        y3 = snaps[2]  # age 67 → self_ltc active (start age 66)
        assert y3.annual_retirement_withdrawal > 300_000
        base_premium = 2_000 * 2 * y3.cumulative_inflation
        assert y3.annual_medicare_cost > base_premium + 1.0        # IRMAA surcharge present

    def test_small_withdrawal_keeps_irmaa_zero(self):
        # Modest rent-only deficit → small withdrawal → MAGI under the threshold.
        snaps = _project(_ret_plan(retirement=500_000, married=True,
                                   medicare_premium=2_000, current_age=65))
        y1 = snaps[0]  # age 65, retired
        assert y1.annual_medicare_cost == pytest.approx(2_000 * 2)  # base only, no IRMAA

    def test_401k_funding_prevents_negative_brokerage(self):
        # Tiny brokerage, big 401k: before 59½ the deficit would drive brokerage
        # negative; once penalty-free the 401k covers it and brokerage stays put.
        snaps = _project(_ret_plan(brokerage=5_000, retirement=500_000,
                                   order=list(_ORDER_BRACKET_FILL), years=3))
        y1, y2 = snaps[0], snaps[1]
        assert y1.brokerage_balance < 0                # age 58 (pre-59½): brokerage overdrawn
        assert y2.brokerage_balance == pytest.approx(y1.brokerage_balance)  # not drawn further
        assert y2.annual_retirement_withdrawal == pytest.approx(12_000)     # 401k covered it
