"""
Tax calculation engine.

Covers:
  - 2024/2025 Federal income tax (progressive brackets)
  - FICA (Social Security + Medicare, including Additional Medicare Tax)
  - State income tax for supported states
  - HSA, 401k, and 529 deduction effects

All bracket amounts are for the *tax year* embedded in each bracket table.
Brackets should be updated annually; the year is noted in each table name.
"""
from __future__ import annotations

from dataclasses import dataclass
from fintracker.models import FilingStatus, IncomeProfile, InvestmentProfile, StrategyToggles, State


# ---------------------------------------------------------------------------
# Federal bracket tables — 2024 tax year
# Format: list of (upper_bound, marginal_rate) tuples; last bound = infinity
# ---------------------------------------------------------------------------

_FEDERAL_BRACKETS_2024: dict[FilingStatus, list[tuple[float, float]]] = {
    FilingStatus.SINGLE: [
        (11_600,  0.10),
        (47_150,  0.12),
        (100_525, 0.22),
        (191_950, 0.24),
        (243_725, 0.32),
        (609_350, 0.35),
        (float("inf"), 0.37),
    ],
    FilingStatus.MARRIED_FILING_JOINTLY: [
        (23_200,  0.10),
        (94_300,  0.12),
        (201_050, 0.22),
        (383_900, 0.24),
        (487_450, 0.32),
        (731_200, 0.35),
        (float("inf"), 0.37),
    ],
    FilingStatus.HEAD_OF_HOUSEHOLD: [
        (16_550,  0.10),
        (63_100,  0.12),
        (100_500, 0.22),
        (191_950, 0.24),
        (243_700, 0.32),
        (609_350, 0.35),
        (float("inf"), 0.37),
    ],
}

_STANDARD_DEDUCTIONS_2024: dict[FilingStatus, float] = {
    FilingStatus.SINGLE: 14_600,
    FilingStatus.MARRIED_FILING_JOINTLY: 29_200,
    FilingStatus.HEAD_OF_HOUSEHOLD: 21_900,
}

# FICA constants (2024)
_SS_WAGE_BASE_2024 = 168_600
_SS_RATE = 0.062
_MEDICARE_RATE = 0.0145
_ADDITIONAL_MEDICARE_RATE = 0.009
_ADDITIONAL_MEDICARE_THRESHOLD_SINGLE = 200_000
_ADDITIONAL_MEDICARE_THRESHOLD_MFJ = 250_000

# ---------------------------------------------------------------------------
# State tax configurations
# ---------------------------------------------------------------------------

@dataclass
class _StateTaxConfig:
    """Minimal state tax descriptor."""
    name: str
    brackets: list[tuple[float, float]]   # (upper_bound, rate)  — empty = no tax
    standard_deduction_single: float = 0.0
    standard_deduction_mfj: float = 0.0
    # Some states don't allow 401k/HSA pre-tax treatment
    allows_401k_deduction: bool = True
    allows_hsa_deduction: bool = True
    allows_529_deduction: bool = False
    # GA-style: per-beneficiary 529 deduction cap
    per_beneficiary_529_deduction: float = 0.0


_STATE_TAX_CONFIGS: dict[State, _StateTaxConfig] = {
    State.GEORGIA: _StateTaxConfig(
        name="Georgia",
        brackets=[(float("inf"), 0.0539)],  # Flat 5.39% (2024+)
        standard_deduction_single=12_000,
        standard_deduction_mfj=24_000,
        allows_529_deduction=True,
        per_beneficiary_529_deduction=8_000,  # $4k single / $8k MFJ (as of 2024)
    ),
    State.CALIFORNIA: _StateTaxConfig(
        name="California",
        brackets=[
            (10_412,  0.01),
            (24_684,  0.02),
            (38_959,  0.04),
            (54_081,  0.06),
            (68_350,  0.08),
            (349_137, 0.093),
            (418_961, 0.103),
            (698_274, 0.113),
            (float("inf"), 0.123),
        ],
        standard_deduction_single=5_202,
        standard_deduction_mfj=10_404,
        allows_hsa_deduction=False,  # CA does not recognize HSA deduction
        allows_529_deduction=False,
    ),
    State.NEW_YORK: _StateTaxConfig(
        name="New York",
        brackets=[
            (17_150,  0.04),
            (23_600,  0.045),
            (27_900,  0.0525),
            (161_550, 0.0585),
            (323_200, 0.0625),
            (2_155_350, 0.0685),
            (5_000_000, 0.0965),
            (25_000_000, 0.103),
            (float("inf"), 0.109),
        ],
        standard_deduction_single=8_000,
        standard_deduction_mfj=16_050,
        allows_529_deduction=True,
        per_beneficiary_529_deduction=5_000,
    ),
    State.TEXAS: _StateTaxConfig(name="Texas", brackets=[]),   # No income tax
    State.FLORIDA: _StateTaxConfig(name="Florida", brackets=[]),
    State.WASHINGTON: _StateTaxConfig(name="Washington", brackets=[]),
    State.ILLINOIS: _StateTaxConfig(
        name="Illinois",
        brackets=[(float("inf"), 0.0495)],  # Flat 4.95%
        standard_deduction_single=0,
        standard_deduction_mfj=0,
        allows_529_deduction=True,
        per_beneficiary_529_deduction=10_000,
    ),
    State.NORTH_CAROLINA: _StateTaxConfig(
        name="North Carolina",
        brackets=[(float("inf"), 0.045)],
        standard_deduction_single=12_750,
        standard_deduction_mfj=25_500,
        allows_529_deduction=False,
    ),
    State.VIRGINIA: _StateTaxConfig(
        name="Virginia",
        brackets=[
            (3_000,  0.02),
            (5_000,  0.03),
            (17_000, 0.05),
            (float("inf"), 0.0575),
        ],
        standard_deduction_single=8_000,
        standard_deduction_mfj=16_000,
        allows_529_deduction=False,
    ),
    State.COLORADO: _StateTaxConfig(
        name="Colorado",
        brackets=[(float("inf"), 0.044)],
        standard_deduction_single=14_600,  # Piggybacks federal standard deduction
        standard_deduction_mfj=29_200,
        allows_529_deduction=True,
        per_beneficiary_529_deduction=20_000,
    ),
    # OTHER: configured dynamically
}


@dataclass
class TaxResult:
    """Full annual tax breakdown."""
    federal_income_tax: float
    social_security_tax: float
    medicare_tax: float
    additional_medicare_tax: float
    state_income_tax: float

    # Deductions applied
    hsa_deduction: float
    retirement_401k_deduction: float
    state_529_deduction: float

    # Derived
    @property
    def total_fica(self) -> float:
        return self.social_security_tax + self.medicare_tax + self.additional_medicare_tax

    @property
    def total_annual_tax(self) -> float:
        return self.federal_income_tax + self.total_fica + self.state_income_tax

    @property
    def total_monthly_tax(self) -> float:
        return self.total_annual_tax / 12

    @property
    def effective_rate(self) -> float:
        """Effective rate against FICA base (gross earned income)."""
        return 0.0  # Calculated in TaxEngine; placeholder here


def _apply_brackets(taxable_income: float, brackets: list[tuple[float, float]]) -> float:
    """Apply progressive brackets to taxable income.  Returns total tax."""
    tax = 0.0
    prev_bound = 0.0
    for upper_bound, rate in brackets:
        if taxable_income <= prev_bound:
            break
        taxable_slice = min(taxable_income, upper_bound) - prev_bound
        tax += taxable_slice * rate
        prev_bound = upper_bound
    return tax


class TaxEngine:
    """Stateless tax calculator.

    Usage::

        engine = TaxEngine()
        result = engine.calculate(income, investments, strategies, num_children=2)
    """

    def calculate(
        self,
        income: IncomeProfile,
        investments: InvestmentProfile,
        strategies: StrategyToggles,
        num_children: int = 0,
        filing_status_override: FilingStatus | None = None,
        gross_income_override: float | None = None,
    ) -> TaxResult:
        """Calculate full annual tax liability.

        Args:
            income: Income and filing configuration.
            investments: Contribution amounts used for deductions.
            strategies: Which strategies are active.
            num_children: Used for state 529 deduction calculation.
            filing_status_override: Override filing status (used in projections).
            gross_income_override: Override gross income (used in projections).
        """
        filing_status = filing_status_override or income.filing_status
        gross = gross_income_override if gross_income_override is not None else income.total_gross_income

        # --- Pre-tax deductions ---
        hsa_deduction = investments.annual_hsa_contribution if strategies.maximize_hsa else 0.0
        k401_deduction = investments.annual_401k_contribution if strategies.maximize_401k else 0.0

        # Federal standard deduction
        std_deduction = _STANDARD_DEDUCTIONS_2024.get(filing_status, 14_600)

        # Federal taxable income: gross - 401k - HSA - standard deduction
        fed_taxable = max(0.0, gross - k401_deduction - hsa_deduction - std_deduction)

        brackets = _FEDERAL_BRACKETS_2024.get(filing_status, _FEDERAL_BRACKETS_2024[FilingStatus.SINGLE])
        federal_tax = _apply_brackets(fed_taxable, brackets)

        # --- FICA ---
        ss_base = min(gross - hsa_deduction, _SS_WAGE_BASE_2024)
        ss_tax = max(0.0, ss_base) * _SS_RATE

        medicare_base = max(0.0, gross - hsa_deduction)
        medicare_tax = medicare_base * _MEDICARE_RATE

        add_medicare_threshold = (
            _ADDITIONAL_MEDICARE_THRESHOLD_MFJ
            if filing_status == FilingStatus.MARRIED_FILING_JOINTLY
            else _ADDITIONAL_MEDICARE_THRESHOLD_SINGLE
        )
        add_medicare_tax = max(0.0, medicare_base - add_medicare_threshold) * _ADDITIONAL_MEDICARE_RATE

        # --- State tax ---
        state = income.state
        if state == State.OTHER:
            state_tax = max(0.0, gross - k401_deduction - hsa_deduction) * income.other_state_flat_rate
            state_529_deduction = 0.0
        else:
            config = _STATE_TAX_CONFIGS.get(state)
            if config is None or not config.brackets:
                state_tax = 0.0
                state_529_deduction = 0.0
            else:
                state_std = (
                    config.standard_deduction_mfj
                    if filing_status == FilingStatus.MARRIED_FILING_JOINTLY
                    else config.standard_deduction_single
                )
                state_hsa = hsa_deduction if config.allows_hsa_deduction else 0.0
                state_401k = k401_deduction if config.allows_401k_deduction else 0.0

                if config.allows_529_deduction and strategies.use_529_state_deduction:
                    state_529_deduction = min(
                        investments.annual_529_contribution * num_children,
                        config.per_beneficiary_529_deduction * num_children,
                    )
                else:
                    state_529_deduction = 0.0

                state_taxable = max(
                    0.0,
                    gross - state_401k - state_hsa - state_529_deduction - state_std,
                )
                state_tax = _apply_brackets(state_taxable, config.brackets)

        return TaxResult(
            federal_income_tax=federal_tax,
            social_security_tax=ss_tax,
            medicare_tax=medicare_tax,
            additional_medicare_tax=add_medicare_tax,
            state_income_tax=state_tax,
            hsa_deduction=hsa_deduction,
            retirement_401k_deduction=k401_deduction,
            state_529_deduction=state_529_deduction,
        )

    def marginal_rate(
        self,
        income: IncomeProfile,
        investments: InvestmentProfile,
        strategies: StrategyToggles,
        filing_status_override: FilingStatus | None = None,
        gross_income_override: float | None = None,
    ) -> float:
        """Return the combined marginal federal + state rate (useful for strategy analysis)."""
        gross = gross_income_override if gross_income_override is not None else income.total_gross_income
        filing_status = filing_status_override or income.filing_status
        hsa_deduction = investments.annual_hsa_contribution if strategies.maximize_hsa else 0.0
        k401_deduction = investments.annual_401k_contribution if strategies.maximize_401k else 0.0
        std_deduction = _STANDARD_DEDUCTIONS_2024.get(filing_status, 14_600)
        fed_taxable = max(0.0, gross - k401_deduction - hsa_deduction - std_deduction)

        brackets = _FEDERAL_BRACKETS_2024.get(filing_status, _FEDERAL_BRACKETS_2024[FilingStatus.SINGLE])
        # Find applicable bracket
        fed_marginal = brackets[-1][1]
        prev = 0.0
        for upper, rate in brackets:
            if fed_taxable <= upper:
                fed_marginal = rate
                break
            prev = upper

        state = income.state
        state_marginal = 0.0
        if state != State.OTHER and state in _STATE_TAX_CONFIGS:
            config = _STATE_TAX_CONFIGS[state]
            if config.brackets:
                state_marginal = config.brackets[-1][1]
                for upper, rate in config.brackets:
                    if fed_taxable <= upper:
                        state_marginal = rate
                        break

        return fed_marginal + state_marginal
