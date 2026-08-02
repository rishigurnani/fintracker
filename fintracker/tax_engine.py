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
from fintracker.finance_math import progressive_tax, marginal_rate_at
from fintracker.models import (
    FilingStatus, IncomeProfile, InvestmentProfile, StrategyToggles, State,
    by_filing_status,
)


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
        # Flat 4.99% for 2026 (down from 5.19%). GA is on a scheduled decline of
        # ~0.125pp/yr toward a 3.99% floor, contingent on state revenue targets;
        # modelled here as a static current-year rate. Verify against the GA DOR.
        name="Georgia",
        brackets=[(float("inf"), 0.0499)],  # Flat 4.99% (2026)
        standard_deduction_single=15_000,
        standard_deduction_mfj=30_000,
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


def state_display_name(state: State) -> str:
    """Human-readable state name for UI labels (e.g. 'Georgia', 'New York')."""
    cfg = _STATE_TAX_CONFIGS.get(state)
    if cfg:
        return cfg.name
    return "Custom flat rate" if state == State.OTHER else state.value


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


# Progressive-bracket math lives in finance_math; kept under the historical
# module-private name so existing importers (and tests) are unaffected.
_apply_brackets = progressive_tax


def _scale_brackets(brackets: list[tuple[float, float]], factor: float) -> list[tuple[float, float]]:
    """Inflation-index bracket bounds by ``factor`` (rates unchanged; inf stays inf).

    Real IRS/state brackets re-index to inflation each year; scaling the 2024
    bounds by the projection year's cumulative inflation keeps a taxpayer in the
    same *real* bracket instead of creeping into higher nominal ones.
    """
    if factor == 1.0:
        return brackets
    return [(bound * factor, rate) for bound, rate in brackets]


def _federal_income_tax(gross: float, filing_status: FilingStatus,
                        k401_deduction: float, hsa_deduction: float,
                        inflation_factor: float = 1.0) -> float:
    """Federal income tax after pre-tax 401k/HSA and the standard deduction.

    Brackets and the standard deduction are inflation-indexed by inflation_factor
    (1.0 = the base tax year).
    """
    std_deduction = _STANDARD_DEDUCTIONS_2024.get(filing_status, 14_600) * inflation_factor
    fed_taxable = max(0.0, gross - k401_deduction - hsa_deduction - std_deduction)
    brackets = _FEDERAL_BRACKETS_2024.get(filing_status, _FEDERAL_BRACKETS_2024[FilingStatus.SINGLE])
    return _apply_brackets(fed_taxable, _scale_brackets(brackets, inflation_factor))


def _fica_taxes(gross: float, hsa_deduction: float,
                filing_status: FilingStatus,
                inflation_factor: float = 1.0) -> tuple[float, float, float]:
    """FICA: (social security, medicare, additional medicare). HSA is FICA-exempt.

    The Social Security wage base is inflation-indexed (it rises annually in
    reality); the Additional Medicare Tax threshold is NOT — it is fixed by
    statute and does not index.
    """
    ss_base = min(gross - hsa_deduction, _SS_WAGE_BASE_2024 * inflation_factor)
    ss_tax = max(0.0, ss_base) * _SS_RATE

    medicare_base = max(0.0, gross - hsa_deduction)
    medicare_tax = medicare_base * _MEDICARE_RATE

    add_medicare_threshold = by_filing_status(
        filing_status,
        _ADDITIONAL_MEDICARE_THRESHOLD_SINGLE,
        _ADDITIONAL_MEDICARE_THRESHOLD_MFJ,
    )
    add_medicare_tax = max(0.0, medicare_base - add_medicare_threshold) * _ADDITIONAL_MEDICARE_RATE
    return ss_tax, medicare_tax, add_medicare_tax


def _state_income_tax(income: IncomeProfile, filing_status: FilingStatus, gross: float,
                      k401_deduction: float, hsa_deduction: float,
                      strategies: StrategyToggles, investments: InvestmentProfile,
                      num_children: int, inflation_factor: float = 1.0) -> tuple[float, float]:
    """State income tax and the state 529 deduction applied: (state_tax, state_529_deduction).

    State brackets and standard deduction are inflation-indexed by inflation_factor.
    (Simplification: a few states don't index in reality; flat-tax states have no
    bounds to index.)
    """
    state = income.state
    if state == State.OTHER:
        state_tax = max(0.0, gross - k401_deduction - hsa_deduction) * income.other_state_flat_rate
        return state_tax, 0.0

    config = _STATE_TAX_CONFIGS.get(state)
    if config is None or not config.brackets:
        return 0.0, 0.0  # no state income tax

    state_std = by_filing_status(
        filing_status,
        config.standard_deduction_single,
        config.standard_deduction_mfj,
    ) * inflation_factor
    state_hsa = hsa_deduction if config.allows_hsa_deduction else 0.0
    state_401k = k401_deduction if config.allows_401k_deduction else 0.0

    if config.allows_529_deduction and strategies.use_529_state_deduction:
        state_529_deduction = min(
            investments.annual_529_contribution * num_children,
            config.per_beneficiary_529_deduction * num_children,
        )
    else:
        state_529_deduction = 0.0

    state_taxable = max(0.0, gross - state_401k - state_hsa - state_529_deduction - state_std)
    return _apply_brackets(state_taxable, _scale_brackets(config.brackets, inflation_factor)), state_529_deduction


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
        inflation_factor: float = 1.0,
    ) -> TaxResult:
        """Calculate full annual tax liability.

        Args:
            income: Income and filing configuration.
            investments: Contribution amounts used for deductions.
            strategies: Which strategies are active.
            num_children: Used for state 529 deduction calculation.
            filing_status_override: Override filing status (used in projections).
            gross_income_override: Override gross income (used in projections).
            inflation_factor: Cumulative price level vs the base tax year; scales
                brackets, standard deductions, and the SS wage base so nominal
                income in a future projection year is taxed against inflation-
                indexed thresholds (1.0 = base year; the default keeps single-year
                callers unchanged).
        """
        filing_status = filing_status_override or income.filing_status
        gross = gross_income_override if gross_income_override is not None else income.total_gross_income

        # --- Pre-tax deductions (used by federal and, where allowed, state) ---
        hsa_deduction = investments.annual_hsa_contribution if strategies.maximize_hsa else 0.0
        k401_deduction = investments.annual_401k_contribution if strategies.maximize_401k else 0.0

        federal_tax = _federal_income_tax(gross, filing_status, k401_deduction, hsa_deduction,
                                          inflation_factor)
        ss_tax, medicare_tax, add_medicare_tax = _fica_taxes(gross, hsa_deduction, filing_status,
                                                             inflation_factor)
        state_tax, state_529_deduction = _state_income_tax(
            income, filing_status, gross, k401_deduction, hsa_deduction,
            strategies, investments, num_children, inflation_factor,
        )

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
        fed_marginal = marginal_rate_at(fed_taxable, brackets)

        state = income.state
        state_marginal = 0.0
        if state != State.OTHER and state in _STATE_TAX_CONFIGS:
            state_marginal = marginal_rate_at(fed_taxable, _STATE_TAX_CONFIGS[state].brackets)

        return fed_marginal + state_marginal
