"""
Tests for MortgageCalculator.

We test:
  - Monthly P&I payment formula correctness
  - Full amortization schedule integrity (balance reaches zero, all principal sums to loan)
  - PMI removal at 80% LTV
  - Edge cases: zero down payment, zero interest rate, full cash purchase
"""
import pytest
from fintracker.models import HousingProfile
from fintracker.mortgage import MortgageCalculator


class TestMonthlyPayment:

    def test_known_payment_6pct_30yr(self):
        """$400k loan (500k home, 100k down) @ 6.5% / 30yr → ~$2,528/mo."""
        profile = HousingProfile(home_price=500_000, down_payment=100_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        payment = calc.monthly_pi_payment()
        assert payment == pytest.approx(2_528.27, abs=1.0)

    def test_known_payment_320k_loan(self):
        """$320k loan (400k home, 80k down) @ 6.5% / 30yr → ~$2,022.62/mo."""
        profile = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        payment = calc.monthly_pi_payment()
        assert payment == pytest.approx(2_022.62, abs=1.0)

    def test_zero_interest_rate(self):
        """Zero interest: payment = loan / n_months."""
        profile = HousingProfile(home_price=300_000, down_payment=0, interest_rate=0.0)
        calc = MortgageCalculator(profile)
        expected = 300_000 / 360
        assert calc.monthly_pi_payment() == pytest.approx(expected, abs=0.01)

    def test_no_loan_zero_payment(self):
        """Full cash purchase: no monthly payment."""
        profile = HousingProfile(home_price=400_000, down_payment=400_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        assert calc.monthly_pi_payment() == 0.0

    def test_payment_increases_with_rate(self):
        low_rate = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.04)
        high_rate = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.08)
        assert MortgageCalculator(low_rate).monthly_pi_payment() < MortgageCalculator(high_rate).monthly_pi_payment()

    def test_payment_increases_with_loan_size(self):
        small = HousingProfile(home_price=200_000, down_payment=40_000, interest_rate=0.065)
        large = HousingProfile(home_price=600_000, down_payment=120_000, interest_rate=0.065)
        assert MortgageCalculator(small).monthly_pi_payment() < MortgageCalculator(large).monthly_pi_payment()

    def test_15yr_payment_higher_than_30yr(self):
        p30 = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065, loan_term_years=30)
        p15 = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065, loan_term_years=15)
        assert MortgageCalculator(p15).monthly_pi_payment() > MortgageCalculator(p30).monthly_pi_payment()


class TestAmortizationSchedule:

    @pytest.fixture
    def standard_calc(self):
        profile = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065)
        return MortgageCalculator(profile)

    def test_schedule_length_is_360_months(self, standard_calc):
        schedule = standard_calc.full_schedule()
        assert len(schedule) == 360

    def test_final_balance_is_zero(self, standard_calc):
        schedule = standard_calc.full_schedule()
        assert schedule[-1].balance == pytest.approx(0.0, abs=1.0)

    def test_principal_sums_to_loan_amount(self, standard_calc):
        schedule = standard_calc.full_schedule()
        total_principal = sum(r.principal for r in schedule)
        assert total_principal == pytest.approx(320_000.0, abs=5.0)

    def test_early_payments_mostly_interest(self, standard_calc):
        """First payment: interest > principal (front-loaded interest)."""
        schedule = standard_calc.full_schedule()
        first = schedule[0]
        assert first.interest > first.principal

    def test_late_payments_mostly_principal(self, standard_calc):
        """Last payment: principal > interest."""
        schedule = standard_calc.full_schedule()
        last = schedule[-1]
        assert last.principal > last.interest

    def test_months_are_sequential(self, standard_calc):
        schedule = standard_calc.full_schedule()
        for i, row in enumerate(schedule):
            assert row.month == i + 1

    def test_balance_monotonically_decreases(self, standard_calc):
        schedule = standard_calc.full_schedule()
        for i in range(1, len(schedule)):
            assert schedule[i].balance <= schedule[i - 1].balance

    def test_cumulative_interest_monotonically_increases(self, standard_calc):
        schedule = standard_calc.full_schedule()
        for i in range(1, len(schedule)):
            assert schedule[i].cumulative_interest >= schedule[i - 1].cumulative_interest

    def test_equity_increases_over_time(self, standard_calc):
        schedule = standard_calc.full_schedule()
        year1_dec = next(r for r in schedule if r.month == 12)
        year10_dec = next(r for r in schedule if r.month == 120)
        assert year10_dec.equity > year1_dec.equity

    def test_empty_schedule_for_zero_loan(self):
        profile = HousingProfile(home_price=400_000, down_payment=400_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        assert calc.full_schedule() == []


class TestPMI:

    @pytest.fixture
    def pmi_calc(self):
        """10% down → PMI required."""
        profile = HousingProfile(
            home_price=400_000,
            down_payment=40_000,  # 10%
            interest_rate=0.065,
            pmi_annual_rate=0.005,
        )
        return MortgageCalculator(profile)

    def test_pmi_charged_in_early_months(self, pmi_calc):
        schedule = pmi_calc.full_schedule()
        assert schedule[0].pmi > 0.0

    def test_pmi_stops_at_80pct_ltv(self, pmi_calc):
        schedule = pmi_calc.full_schedule()
        # Find first month with no PMI
        no_pmi_months = [r for r in schedule if r.pmi == 0.0]
        assert len(no_pmi_months) > 0
        first_no_pmi = no_pmi_months[0]
        # After that month, all should be zero
        for row in schedule[first_no_pmi.month - 1:]:
            assert row.pmi == 0.0

    def test_no_pmi_at_20pct_down(self):
        profile = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        schedule = calc.full_schedule()
        assert all(r.pmi == 0.0 for r in schedule)

    def test_summary_pmi_removal_month_positive(self, pmi_calc):
        summary = pmi_calc.summary()
        assert summary.pmi_removal_month > 0
        assert summary.monthly_pmi_initial > 0


class TestMortgageSummary:

    def test_summary_total_interest_positive(self):
        profile = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        summary = calc.summary()
        assert summary.total_interest_paid > 0

    def test_summary_annual_rows_length(self):
        profile = HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        summary = calc.summary()
        # Should have roughly 30 annual snapshots
        assert len(summary.annual_rows) >= 29

    def test_zero_loan_summary(self):
        profile = HousingProfile(home_price=400_000, down_payment=400_000, interest_rate=0.065)
        calc = MortgageCalculator(profile)
        summary = calc.summary()
        assert summary.loan_amount == 0.0
        assert summary.monthly_pi == 0.0
