"""
Tests for the tax-modeling gap fixes:

  #3  Ordinary Roth IRA contributions happen without the backdoor toggle, and
      phase out over the MAGI income limit (backdoor bypasses the limit).
  #5  Itemized deductions (mortgage interest + capped SALT) beat the standard
      deduction for a homeowner with a large mortgage.
  #6  Self-employment tax caps the 12.4% Social Security portion at the wage base
      (coordinated with W-2 wages); the 2.9% Medicare portion continues.
  #7  NIIT (3.8%), taxable-brokerage dividend drag, and taxation of Social
      Security benefits in the retirement-readiness calc.
  #12 A negative taxable-brokerage balance is reachable (the app surfaces a
      warning for it).
"""
import pytest

from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, InvestmentProfile, StrategyToggles, BusinessProfile,
    RetirementProfile, TimelineEvent, LifestyleProfile,
)
from fintracker.tax_engine import TaxEngine, DeductionInputs
from fintracker.projections import ProjectionEngine
from fintracker.constants import (
    ROTH_IRA_LIMIT, SS_WAGE_BASE, NIIT_THRESHOLD_SINGLE,
)

from tests.builders import make_plan, investments, zero_lifestyle, renting_housing


# ---------------------------------------------------------------------------
# #5 — Itemized deductions
# ---------------------------------------------------------------------------

class TestItemizedDeductions:
    def setup_method(self):
        self.engine = TaxEngine()
        # Texas (no state income tax) so SALT is property tax only — keeps the math
        # about the federal itemize-vs-standard choice, not state tax.
        self.income = IncomeProfile(gross_annual_income=200_000,
                                    filing_status=FilingStatus.SINGLE, state=State.TEXAS)
        self.inv = InvestmentProfile()
        self.strat = StrategyToggles(maximize_hsa=False, maximize_401k=False)

    def test_large_mortgage_itemizes_below_standard(self):
        """A big mortgage + property tax itemizes, lowering federal tax vs standard."""
        std = self.engine.calculate(self.income, self.inv, self.strat)
        itemized = self.engine.calculate(
            self.income, self.inv, self.strat,
            deductions=DeductionInputs(mortgage_interest=30_000, property_tax=8_000))
        assert itemized.federal_income_tax < std.federal_income_tax

    def test_salt_is_capped_at_10k(self):
        """Property tax beyond the $10k SALT cap yields no extra deduction."""
        at_cap = self.engine.calculate(
            self.income, self.inv, self.strat,
            deductions=DeductionInputs(mortgage_interest=20_000, property_tax=10_000))
        over_cap = self.engine.calculate(
            self.income, self.inv, self.strat,
            deductions=DeductionInputs(mortgage_interest=20_000, property_tax=50_000))
        assert at_cap.federal_income_tax == pytest.approx(over_cap.federal_income_tax)

    def test_small_deductions_keep_standard(self):
        """Itemizable total below the standard deduction changes nothing."""
        std = self.engine.calculate(self.income, self.inv, self.strat)
        tiny = self.engine.calculate(
            self.income, self.inv, self.strat,
            deductions=DeductionInputs(mortgage_interest=2_000, property_tax=1_000))
        assert tiny.federal_income_tax == pytest.approx(std.federal_income_tax)

    def test_none_deductions_is_standard_only(self):
        """Passing no deductions is identical to the pre-change standard-only path."""
        a = self.engine.calculate(self.income, self.inv, self.strat)
        b = self.engine.calculate(self.income, self.inv, self.strat, deductions=None)
        assert a.federal_income_tax == pytest.approx(b.federal_income_tax)


# ---------------------------------------------------------------------------
# #7 — Net Investment Income Tax (unit)
# ---------------------------------------------------------------------------

class TestNIIT:
    def setup_method(self):
        self.engine = TaxEngine()

    def test_below_threshold_is_zero(self):
        assert self.engine.net_investment_income_tax(
            magi=150_000, net_investment_income=50_000,
            filing_status=FilingStatus.SINGLE) == 0.0

    def test_charged_on_full_nii_when_magi_far_above(self):
        # MAGI - threshold ($100k) exceeds NII ($50k) → 3.8% on the full NII.
        tax = self.engine.net_investment_income_tax(
            magi=NIIT_THRESHOLD_SINGLE + 100_000, net_investment_income=50_000,
            filing_status=FilingStatus.SINGLE)
        assert tax == pytest.approx(0.038 * 50_000)

    def test_limited_by_magi_excess(self):
        # Only $10k of MAGI clears the threshold → tax on the lesser ($10k).
        tax = self.engine.net_investment_income_tax(
            magi=NIIT_THRESHOLD_SINGLE + 10_000, net_investment_income=50_000,
            filing_status=FilingStatus.SINGLE)
        assert tax == pytest.approx(0.038 * 10_000)

    def test_zero_nii_is_zero(self):
        assert self.engine.net_investment_income_tax(
            magi=1_000_000, net_investment_income=0.0,
            filing_status=FilingStatus.SINGLE) == 0.0


# ---------------------------------------------------------------------------
# #6 — Self-employment tax cap
# ---------------------------------------------------------------------------

class TestSelfEmploymentTaxCap:
    def _se_tax(self, revenue, primary_income=0.0):
        """SE tax the engine charges on a sole-prop with the given net profit."""
        plan = make_plan(
            income=IncomeProfile(primary_income, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=100_000),
            business=BusinessProfile(annual_revenue=revenue, expense_ratio=0.0,
                                     use_qbi_deduction=False, equity_multiple=0.0,
                                     start_year=1),
        )
        eng = ProjectionEngine(plan)
        state = eng._initial_state()
        state.income_primary = primary_income
        _net, se_tax, _eq, _solo = eng._business(state, 1, 1.0)
        return se_tax

    def test_below_wage_base_matches_full_rate(self):
        # Net profit under the wage base → SS portion applies to all of it, so the
        # capped math equals the old flat 15.3% × 92.35%.
        base = 100_000 * 0.9235
        assert self._se_tax(100_000) == pytest.approx(base * 0.153)

    def test_above_wage_base_is_capped(self):
        # Well above the wage base: SS portion stops at the base, Medicare continues.
        se_base = 300_000 * 0.9235
        expected = min(se_base, SS_WAGE_BASE) * 0.124 + se_base * 0.029
        se = self._se_tax(300_000)
        assert se == pytest.approx(expected)
        assert se < se_base * 0.153   # strictly less than the old uncapped charge

    def test_w2_wages_consume_the_wage_base(self):
        # Owner already earns the full wage base in W-2 wages → no SS portion left
        # for SE income; only the 2.9% Medicare portion is charged.
        se_base = 100_000 * 0.9235
        assert self._se_tax(100_000, primary_income=SS_WAGE_BASE) == pytest.approx(se_base * 0.029)


# ---------------------------------------------------------------------------
# #3 — Roth contribution + MAGI phase-out
# ---------------------------------------------------------------------------

class TestRothContribution:
    def _roth_contrib(self, gross, backdoor=False):
        plan = make_plan(
            income=IncomeProfile(gross, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=300_000,
                                    annual_roth_ira_contribution=ROTH_IRA_LIMIT),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False,
                                       use_backdoor_roth=backdoor),
            projection_years=1,
        )
        return ProjectionEngine(plan).run_deterministic()[0].annual_roth_contribution

    def test_ordinary_contribution_without_backdoor(self):
        """Under the income limit, a plain Roth contribution is made even with the
        backdoor toggle OFF — the bug was contributing $0 here."""
        assert self._roth_contrib(gross=100_000, backdoor=False) == pytest.approx(ROTH_IRA_LIMIT)

    def test_phased_out_above_income_limit(self):
        """Above the MAGI band, a direct contribution phases to $0."""
        assert self._roth_contrib(gross=250_000, backdoor=False) == pytest.approx(0.0)

    def test_backdoor_bypasses_income_limit(self):
        """The backdoor route contributes the full amount regardless of income."""
        assert self._roth_contrib(gross=250_000, backdoor=True) == pytest.approx(ROTH_IRA_LIMIT)

    def test_partial_phase_out_in_band(self):
        """Inside the phase-out band the contribution is a fraction of the limit."""
        # Single band is 146k–161k; 153.5k is roughly the midpoint → ~half.
        contrib = self._roth_contrib(gross=153_500, backdoor=False)
        assert 0.0 < contrib < ROTH_IRA_LIMIT


# ---------------------------------------------------------------------------
# #7 — Brokerage dividend drag
# ---------------------------------------------------------------------------

class TestBrokerageDividendDrag:
    def _final_brokerage(self, dividend_yield, gross):
        plan = make_plan(
            income=IncomeProfile(gross, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=0, current_brokerage_balance=500_000,
                                    annual_market_return=0.07,
                                    taxable_dividend_yield=dividend_yield),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=15,
        )
        return ProjectionEngine(plan).run_deterministic()[-1].brokerage_balance

    def test_dividend_drag_lowers_growth(self):
        """With ordinary income (so qualified dividends leave the 0% band and are
        taxed at 15%), a dividend yield leaks tax every year — the account ends lower
        than the same account with no drag. Income sweeps identically into both, so
        the difference is purely the dividend tax."""
        assert self._final_brokerage(0.02, gross=150_000) < self._final_brokerage(0.0, gross=150_000)

    def test_zero_yield_grows_at_pure_market(self):
        """With no income and no dividend yield the account is isolated and compounds
        at exactly the market return — the pre-change behaviour."""
        no_drag = self._final_brokerage(0.0, gross=0)
        assert no_drag == pytest.approx(500_000 * (1.07 ** 15), rel=1e-6)

    def test_no_drag_in_zero_percent_bracket(self):
        """A retiree with no other income pays 0% on qualified dividends (0% LTCG
        band), so even a 2% yield produces no drag — the model gets this right."""
        assert self._final_brokerage(0.02, gross=0) == pytest.approx(
            self._final_brokerage(0.0, gross=0), rel=1e-9)


# ---------------------------------------------------------------------------
# #7 — Social Security benefits are taxed in the readiness calc
# ---------------------------------------------------------------------------

class TestSocialSecurityTaxation:
    def _required(self, withdrawal_tax_rate):
        # Real retirement-year living costs above the SS benefit, so the size of the
        # (now taxed) SS offset actually moves the required nest egg.
        plan = make_plan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=5_000,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=investments(current_liquid_cash=100_000),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=6,
            retirement=RetirementProfile(current_age=60, retirement_age=63,
                                         desired_annual_income=50_000,
                                         years_in_retirement=3,
                                         estimated_social_security_annual=30_000),
        )
        rr = ProjectionEngine(plan).compute_retirement_readiness(
            withdrawal_tax_rate=withdrawal_tax_rate)
        return rr.required_balance

    def test_taxing_ss_raises_required_balance(self):
        """Taxing SS benefits shrinks the offset, so more is needed than when SS is
        treated as tax-free (withdrawal rate 0)."""
        assert self._required(0.30) > self._required(0.0)


# ---------------------------------------------------------------------------
# #12 — Negative brokerage is reachable (app warns on it)
# ---------------------------------------------------------------------------

class TestNegativeBrokerageReachable:
    def test_large_purchase_drives_brokerage_negative(self):
        """A lump-sum outflow larger than every account leaves a negative taxable
        brokerage — the insolvency signal the app surfaces as a warning."""
        plan = make_plan(
            income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            housing=renting_housing(),
            investments=investments(current_liquid_cash=20_000, current_brokerage_balance=0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=2, description="Huge spend",
                                           extra_one_time_expense=200_000)],
            projection_years=3,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert any(s.brokerage_balance < 0 for s in snaps)
