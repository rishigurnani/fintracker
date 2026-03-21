"""
Tests for TaxEngine.

We test:
  - Known bracket calculations against hand-computed values
  - FICA math (SS wage base cap, additional Medicare tax)
  - State tax for supported states
  - HSA / 401k deduction effects
  - MFJ vs Single filing status differences
  - Edge cases: zero income, very high income
"""
import pytest
from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, InvestmentProfile, StrategyToggles,
)
from fintracker.tax_engine import TaxEngine, _apply_brackets


# ---------------------------------------------------------------------------
# Unit tests for bracket helper
# ---------------------------------------------------------------------------

class TestApplyBrackets:
    SIMPLE_BRACKETS = [(10_000, 0.10), (50_000, 0.20), (float("inf"), 0.30)]

    def test_zero_income(self):
        assert _apply_brackets(0, self.SIMPLE_BRACKETS) == 0.0

    def test_within_first_bracket(self):
        assert _apply_brackets(5_000, self.SIMPLE_BRACKETS) == pytest.approx(500.0)

    def test_exactly_at_bracket_boundary(self):
        # 10k at 10% = 1000
        assert _apply_brackets(10_000, self.SIMPLE_BRACKETS) == pytest.approx(1_000.0)

    def test_spans_two_brackets(self):
        # 10k @10% + 10k @20% = 1000 + 2000 = 3000
        assert _apply_brackets(20_000, self.SIMPLE_BRACKETS) == pytest.approx(3_000.0)

    def test_spans_all_brackets(self):
        # 10k @10% + 40k @20% + 10k @30% = 1000 + 8000 + 3000 = 12000
        assert _apply_brackets(60_000, self.SIMPLE_BRACKETS) == pytest.approx(12_000.0)

    def test_empty_brackets_returns_zero(self):
        assert _apply_brackets(50_000, []) == 0.0


# ---------------------------------------------------------------------------
# TaxEngine integration tests
# ---------------------------------------------------------------------------

class TestFederalTax:

    def setup_method(self):
        self.engine = TaxEngine()
        self.no_strategies = StrategyToggles(
            maximize_hsa=False, maximize_401k=False,
            use_529_state_deduction=False, use_roth_ladder=False,
        )
        self.no_contributions = InvestmentProfile()

    def _result(self, gross, filing_status, state=State.TEXAS):
        """Helper: calculate with no deductions or strategies, in a no-income-tax state."""
        income = IncomeProfile(gross_annual_income=gross, filing_status=filing_status, state=state)
        return self.engine.calculate(income, self.no_contributions, self.no_strategies)

    def test_zero_income_single(self):
        r = self._result(0, FilingStatus.SINGLE)
        assert r.federal_income_tax == 0.0
        assert r.social_security_tax == 0.0
        assert r.medicare_tax == 0.0
        assert r.state_income_tax == 0.0

    def test_low_income_single_10pct_bracket(self):
        # $20k gross: std deduction $14,600 → taxable = $5,400 → 10% bracket → $540
        r = self._result(20_000, FilingStatus.SINGLE)
        assert r.federal_income_tax == pytest.approx(540.0, abs=1.0)

    def test_single_crosses_22pct_bracket(self):
        # $80k gross: taxable = 80k - 14.6k = 65.4k
        # 10%: 0-11.6k = $1,160
        # 12%: 11.6-47.15k = $4,266
        # 22%: 47.15-65.4k = $4,015
        # Total ≈ $9,441
        r = self._result(80_000, FilingStatus.SINGLE)
        assert r.federal_income_tax == pytest.approx(9_441.0, abs=50.0)

    def test_mfj_standard_deduction_is_double(self):
        # Same gross income, MFJ has higher std deduction → lower tax
        single = self._result(100_000, FilingStatus.SINGLE)
        mfj = self._result(100_000, FilingStatus.MARRIED_FILING_JOINTLY)
        assert mfj.federal_income_tax < single.federal_income_tax

    def test_social_security_wage_base_cap(self):
        # Above SS wage base ($168,600): SS doesn't increase further
        r_below = self._result(160_000, FilingStatus.SINGLE)
        r_above = self._result(200_000, FilingStatus.SINGLE)
        # SS should be capped for the higher income
        assert r_above.social_security_tax == pytest.approx(168_600 * 0.062, abs=1.0)
        assert r_below.social_security_tax < r_above.social_security_tax

    def test_additional_medicare_tax_single(self):
        # Only kicks in above $200k for single
        r_below = self._result(199_000, FilingStatus.SINGLE)
        r_above = self._result(250_000, FilingStatus.SINGLE)
        assert r_below.additional_medicare_tax == 0.0
        assert r_above.additional_medicare_tax > 0.0
        assert r_above.additional_medicare_tax == pytest.approx(
            (250_000 - 200_000) * 0.009, abs=1.0
        )

    def test_additional_medicare_tax_mfj_higher_threshold(self):
        # MFJ threshold is $250k
        r = self._result(240_000, FilingStatus.MARRIED_FILING_JOINTLY)
        assert r.additional_medicare_tax == 0.0

    def test_high_income_reaches_37pct_bracket(self):
        r = self._result(700_000, FilingStatus.SINGLE)
        # Should be in the 37% bracket territory
        assert r.federal_income_tax > 200_000

    def test_total_tax_increases_with_income(self):
        r1 = self._result(50_000, FilingStatus.SINGLE)
        r2 = self._result(100_000, FilingStatus.SINGLE)
        r3 = self._result(200_000, FilingStatus.SINGLE)
        assert r1.total_annual_tax < r2.total_annual_tax < r3.total_annual_tax


class TestHSADeduction:

    def setup_method(self):
        self.engine = TaxEngine()
        self.income = IncomeProfile(
            gross_annual_income=100_000,
            filing_status=FilingStatus.SINGLE,
            state=State.TEXAS,  # No state tax to isolate federal effect
        )

    def test_hsa_reduces_federal_tax(self):
        no_hsa = InvestmentProfile(annual_hsa_contribution=0)
        with_hsa = InvestmentProfile(annual_hsa_contribution=4_150)
        strat_off = StrategyToggles(maximize_hsa=False, maximize_401k=False)
        strat_on = StrategyToggles(maximize_hsa=True, maximize_401k=False)

        r_no_hsa = self.engine.calculate(self.income, no_hsa, strat_off)
        r_with_hsa = self.engine.calculate(self.income, with_hsa, strat_on)

        assert r_with_hsa.federal_income_tax < r_no_hsa.federal_income_tax

    def test_hsa_reduces_fica(self):
        no_hsa = InvestmentProfile(annual_hsa_contribution=0)
        with_hsa = InvestmentProfile(annual_hsa_contribution=4_150)
        strat_off = StrategyToggles(maximize_hsa=False, maximize_401k=False)
        strat_on = StrategyToggles(maximize_hsa=True, maximize_401k=False)

        r_no_hsa = self.engine.calculate(self.income, no_hsa, strat_off)
        r_with_hsa = self.engine.calculate(self.income, with_hsa, strat_on)

        assert r_with_hsa.total_fica < r_no_hsa.total_fica

    def test_hsa_deduction_amount_matches_contribution(self):
        with_hsa = InvestmentProfile(annual_hsa_contribution=4_150)
        strat_on = StrategyToggles(maximize_hsa=True, maximize_401k=False)
        r = self.engine.calculate(self.income, with_hsa, strat_on)
        assert r.hsa_deduction == pytest.approx(4_150.0)

    def test_disabled_strategy_means_zero_hsa_deduction(self):
        with_hsa = InvestmentProfile(annual_hsa_contribution=4_150)
        strat_off = StrategyToggles(maximize_hsa=False)
        r = self.engine.calculate(self.income, with_hsa, strat_off)
        assert r.hsa_deduction == 0.0


class Test401kDeduction:

    def setup_method(self):
        self.engine = TaxEngine()
        self.income = IncomeProfile(
            gross_annual_income=120_000,
            filing_status=FilingStatus.SINGLE,
            state=State.TEXAS,
        )

    def test_401k_reduces_federal_tax(self):
        no_401k = InvestmentProfile(annual_401k_contribution=0)
        with_401k = InvestmentProfile(annual_401k_contribution=23_000)
        strat_off = StrategyToggles(maximize_401k=False)
        strat_on = StrategyToggles(maximize_401k=True)

        r_no = self.engine.calculate(self.income, no_401k, strat_off)
        r_with = self.engine.calculate(self.income, with_401k, strat_on)

        assert r_with.federal_income_tax < r_no.federal_income_tax

    def test_401k_does_not_reduce_fica(self):
        """Traditional 401k reduces income tax but NOT FICA."""
        no_401k = InvestmentProfile(annual_401k_contribution=0)
        with_401k = InvestmentProfile(annual_401k_contribution=23_000)
        strat_off = StrategyToggles(maximize_401k=False)
        strat_on = StrategyToggles(maximize_401k=True)

        r_no = self.engine.calculate(self.income, no_401k, strat_off)
        r_with = self.engine.calculate(self.income, with_401k, strat_on)

        assert r_with.social_security_tax == pytest.approx(r_no.social_security_tax, abs=1.0)


class TestStateTax:

    def setup_method(self):
        self.engine = TaxEngine()
        self.inv = InvestmentProfile()
        self.strat = StrategyToggles(maximize_hsa=False, maximize_401k=False)

    def _calc(self, gross, state, filing=FilingStatus.SINGLE):
        income = IncomeProfile(gross_annual_income=gross, filing_status=filing, state=state)
        return self.engine.calculate(income, self.inv, self.strat)

    def test_texas_no_state_tax(self):
        r = self._calc(100_000, State.TEXAS)
        assert r.state_income_tax == 0.0

    def test_florida_no_state_tax(self):
        r = self._calc(100_000, State.FLORIDA)
        assert r.state_income_tax == 0.0

    def test_georgia_flat_tax(self):
        # GA flat rate 5.39%; taxable = gross - GA std deduction ($12k for single)
        gross = 100_000
        r = self._calc(gross, State.GEORGIA)
        expected = max(0, gross - 12_000) * 0.0539
        assert r.state_income_tax == pytest.approx(expected, abs=1.0)

    def test_georgia_mfj_std_deduction_doubles(self):
        gross = 200_000
        single = self._calc(gross, State.GEORGIA, FilingStatus.SINGLE)
        mfj = self._calc(gross, State.GEORGIA, FilingStatus.MARRIED_FILING_JOINTLY)
        # MFJ deduction is $24k vs $12k → lower state tax
        assert mfj.state_income_tax < single.state_income_tax

    def test_california_higher_rate_than_georgia(self):
        gross = 200_000
        ga = self._calc(gross, State.GEORGIA)
        ca = self._calc(gross, State.CALIFORNIA)
        assert ca.state_income_tax > ga.state_income_tax

    def test_georgia_529_deduction_single(self):
        gross = 100_000
        income = IncomeProfile(gross_annual_income=gross, filing_status=FilingStatus.SINGLE, state=State.GEORGIA)
        inv = InvestmentProfile(annual_529_contribution=8_000)
        strat = StrategyToggles(maximize_hsa=False, maximize_401k=False, use_529_state_deduction=True)
        r = self.engine.calculate(income, inv, strat, num_children=1)
        # GA allows $8k deduction per beneficiary (single filer)
        no_529_strat = StrategyToggles(maximize_hsa=False, maximize_401k=False, use_529_state_deduction=False)
        r_no529 = self.engine.calculate(income, inv, no_529_strat, num_children=1)
        assert r.state_income_tax < r_no529.state_income_tax

    def test_california_does_not_allow_hsa_state_deduction(self):
        """California does not recognize HSA deductions at the state level."""
        gross = 100_000
        income = IncomeProfile(gross_annual_income=gross, filing_status=FilingStatus.SINGLE, state=State.CALIFORNIA)
        inv_hsa = InvestmentProfile(annual_hsa_contribution=4_150)
        strat_on = StrategyToggles(maximize_hsa=True)
        strat_off = StrategyToggles(maximize_hsa=False)
        r_on = self.engine.calculate(income, inv_hsa, strat_on)
        r_off = self.engine.calculate(income, inv_hsa, strat_off)
        # State tax should be identical because CA ignores HSA
        assert r_on.state_income_tax == pytest.approx(r_off.state_income_tax, abs=1.0)

    def test_other_state_uses_flat_rate(self):
        gross = 100_000
        flat_rate = 0.04
        income = IncomeProfile(
            gross_annual_income=gross,
            filing_status=FilingStatus.SINGLE,
            state=State.OTHER,
            other_state_flat_rate=flat_rate,
        )
        r = self.engine.calculate(income, self.inv, self.strat)
        assert r.state_income_tax == pytest.approx(gross * flat_rate, abs=1.0)


class TestMarginalRate:

    def test_marginal_rate_increases_with_income(self):
        engine = TaxEngine()
        inv = InvestmentProfile()
        strat = StrategyToggles(maximize_hsa=False, maximize_401k=False)
        inc_low = IncomeProfile(gross_annual_income=40_000, filing_status=FilingStatus.SINGLE, state=State.TEXAS)
        inc_high = IncomeProfile(gross_annual_income=400_000, filing_status=FilingStatus.SINGLE, state=State.TEXAS)
        assert engine.marginal_rate(inc_low, inv, strat) < engine.marginal_rate(inc_high, inv, strat)

    def test_no_state_tax_state_has_lower_marginal_rate(self):
        engine = TaxEngine()
        inv = InvestmentProfile()
        strat = StrategyToggles(maximize_hsa=False, maximize_401k=False)
        inc_tx = IncomeProfile(gross_annual_income=100_000, filing_status=FilingStatus.SINGLE, state=State.TEXAS)
        inc_ga = IncomeProfile(gross_annual_income=100_000, filing_status=FilingStatus.SINGLE, state=State.GEORGIA)
        assert engine.marginal_rate(inc_tx, inv, strat) < engine.marginal_rate(inc_ga, inv, strat)
