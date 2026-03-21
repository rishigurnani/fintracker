"""
Mortgage calculator with full amortization schedule.

Unlike simplified approximations, this produces exact monthly payment schedules
including PMI removal, interest/principal split, and cumulative equity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from fintracker.models import HousingProfile


@dataclass
class AmortizationRow:
    """One month in the amortization schedule."""
    month: int
    year: int           # 1-based year
    payment: float      # Total payment (P&I + PMI)
    principal: float
    interest: float
    pmi: float
    balance: float      # Remaining loan balance after this payment
    cumulative_interest: float
    home_value: float   # Estimated (appreciate at HousingProfile rate)
    equity: float       # home_value - balance


@dataclass
class MortgageSummary:
    """High-level mortgage statistics."""
    monthly_pi: float
    monthly_pmi_initial: float
    total_interest_paid: float
    pmi_removal_month: int          # Month PMI drops off (0 = never / not applicable)
    payoff_month: int
    loan_amount: float
    # Annual snapshots for charting
    annual_rows: list[AmortizationRow]


class MortgageCalculator:
    """Computes full amortization schedule for a HousingProfile."""

    def __init__(self, profile: HousingProfile, annual_appreciation_rate: float = 0.035):
        self._p = profile
        self._appreciation_rate = annual_appreciation_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def monthly_pi_payment(self) -> float:
        """Principal + Interest payment only (no PMI, taxes, insurance)."""
        return self._compute_monthly_pi(self._p.loan_amount, self._p.interest_rate, self._p.loan_term_years)

    def monthly_total_payment(self, include_taxes_insurance: bool = True) -> float:
        """Full PITI payment for year 1."""
        pi = self.monthly_pi_payment()
        pmi = self._pmi_payment(self._p.loan_amount)
        taxes_insurance = self._monthly_taxes_insurance() if include_taxes_insurance else 0.0
        return pi + pmi + taxes_insurance

    def full_schedule(self) -> list[AmortizationRow]:
        """Generate month-by-month amortization table."""
        loan = self._p.loan_amount
        if loan <= 0:
            return []

        monthly_rate = self._p.interest_rate / 12
        n_payments = self._p.loan_term_years * 12
        monthly_pi = self._compute_monthly_pi(loan, self._p.interest_rate, self._p.loan_term_years)

        balance = loan
        cumulative_interest = 0.0
        pmi_active = self._p.requires_pmi
        pmi_removal_month = 0
        home_value = self._p.home_price
        monthly_appreciation = (1 + self._appreciation_rate) ** (1 / 12)

        rows: list[AmortizationRow] = []
        for month in range(1, n_payments + 1):
            year = math.ceil(month / 12)

            if month % 12 == 1 and month > 1:
                home_value *= (1 + self._appreciation_rate)

            interest = balance * monthly_rate
            principal = monthly_pi - interest

            # Guard against floating-point overpayment on last payment
            if balance - principal < 0:
                principal = balance
            balance = max(0.0, balance - principal)
            cumulative_interest += interest

            # PMI: charged while LTV > 80%
            pmi = 0.0
            if pmi_active:
                ltv = balance / self._p.home_price
                if ltv <= 0.80:
                    pmi_active = False
                    pmi_removal_month = month
                else:
                    pmi = self._pmi_payment(balance)

            equity = home_value - balance

            rows.append(AmortizationRow(
                month=month,
                year=year,
                payment=monthly_pi + pmi,
                principal=principal,
                interest=interest,
                pmi=pmi,
                balance=balance,
                cumulative_interest=cumulative_interest,
                home_value=home_value,
                equity=equity,
            ))

            if balance == 0:
                break

        return rows

    def summary(self) -> MortgageSummary:
        """Return high-level stats derived from the full schedule."""
        schedule = self.full_schedule()
        if not schedule:
            return MortgageSummary(
                monthly_pi=0.0,
                monthly_pmi_initial=0.0,
                total_interest_paid=0.0,
                pmi_removal_month=0,
                payoff_month=0,
                loan_amount=0.0,
                annual_rows=[],
            )

        annual_rows = [row for row in schedule if row.month % 12 == 0 or row.month == len(schedule)]

        return MortgageSummary(
            monthly_pi=self.monthly_pi_payment(),
            monthly_pmi_initial=self._pmi_payment(self._p.loan_amount) if self._p.requires_pmi else 0.0,
            total_interest_paid=schedule[-1].cumulative_interest,
            pmi_removal_month=next((r.month for r in schedule if r.pmi == 0 and self._p.requires_pmi), 0),
            payoff_month=schedule[-1].month,
            loan_amount=self._p.loan_amount,
            annual_rows=annual_rows,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_monthly_pi(loan_amount: float, annual_rate: float, term_years: int) -> float:
        if loan_amount <= 0:
            return 0.0
        if annual_rate == 0:
            return loan_amount / (term_years * 12)
        r = annual_rate / 12
        n = term_years * 12
        return loan_amount * r / (1 - (1 + r) ** -n)

    def _pmi_payment(self, current_balance: float) -> float:
        if not self._p.requires_pmi:
            return 0.0
        return current_balance * self._p.pmi_annual_rate / 12

    def _monthly_taxes_insurance(self) -> float:
        annual = (
            self._p.home_price * self._p.annual_property_tax_rate
            + self._p.annual_insurance
            + self._p.home_price * self._p.annual_maintenance_rate
        )
        return annual / 12
