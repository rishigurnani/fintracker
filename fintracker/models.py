"""
Data models for fintracker.

All dollar amounts are nominal (current-year) values unless otherwise noted.
Rates are expressed as decimals (0.05 = 5%).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FilingStatus(str, Enum):
    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"
    HEAD_OF_HOUSEHOLD = "head_of_household"


class State(str, Enum):
    """States with income tax modelled. 'OTHER' uses a user-supplied flat rate."""
    GEORGIA       = "GA"
    CALIFORNIA    = "CA"
    NEW_YORK      = "NY"
    TEXAS         = "TX"
    FLORIDA       = "FL"
    WASHINGTON    = "WA"
    ILLINOIS      = "IL"
    NORTH_CAROLINA = "NC"
    VIRGINIA      = "VA"
    COLORADO      = "CO"
    OTHER         = "OTHER"


# ---------------------------------------------------------------------------
# Income
# ---------------------------------------------------------------------------

@dataclass
class IncomeProfile:
    """Gross earned income and tax-filing configuration."""
    gross_annual_income: float
    filing_status: FilingStatus = FilingStatus.SINGLE
    state: State = State.GEORGIA
    other_state_flat_rate: float = 0.05
    spouse_gross_annual_income: float = 0.0

    @property
    def total_gross_income(self) -> float:
        return self.gross_annual_income + self.spouse_gross_annual_income


# ---------------------------------------------------------------------------
# Housing
# ---------------------------------------------------------------------------

@dataclass
class HousingProfile:
    """Home-purchase / mortgage parameters, or renting configuration."""
    home_price: float
    down_payment: float
    interest_rate: float
    loan_term_years: int = 30
    annual_property_tax_rate: float = 0.012
    annual_insurance: float = 2_000
    annual_maintenance_rate: float = 0.01
    pmi_annual_rate: float = 0.005
    is_renting: bool = False
    monthly_rent: float = 0.0
    annual_rent_increase_rate: float = 0.03

    @property
    def down_payment_pct(self) -> float:
        return self.down_payment / self.home_price if self.home_price else 0.0

    @property
    def loan_amount(self) -> float:
        return max(0.0, self.home_price - self.down_payment)

    @property
    def requires_pmi(self) -> bool:
        return self.down_payment_pct < 0.20 and not self.is_renting


# ---------------------------------------------------------------------------
# Lifestyle
# ---------------------------------------------------------------------------

@dataclass
class LifestyleProfile:
    """Recurring lifestyle expenses, scaled automatically with family size."""
    monthly_childcare: float = 0.0
    num_children: int = 0
    num_pets: int = 0
    annual_pet_cost: float = 0.0

    # Healthcare — baseline for a single adult; auto-scaled by family size when
    # medical_auto_scale=True.  Set False to pin the raw value.
    annual_medical_oop: float = 0.0
    medical_auto_scale: bool = True
    medical_spouse_multiplier: float = 1.8   # family plan ~80% more than single
    medical_per_child_annual: float = 1_500  # paediatric OOP per child

    annual_vacation: float = 0.0
    monthly_other_recurring: float = 0.0

    # Wedding fund — annual savings per child toward their wedding, held in
    # brokerage.  Contributions stop when each child reaches age 25.
    # Separate from college costs and configured here, not in CollegeProfile.
    annual_wedding_fund_per_child: float = 0.0

    # Parent care — annual cost to support ageing parents.
    # Activated / deactivated via start_parent_care / stop_parent_care events.
    annual_parent_care_cost: float = 0.0

    @property
    def annual_total(self) -> float:
        return (
            self.monthly_childcare * 12
            + self.annual_pet_cost
            + self.annual_medical_oop
            + self.annual_vacation
            + self.monthly_other_recurring * 12
            + self.annual_parent_care_cost
        )

    def scaled_medical_oop(self, is_married: bool, num_children: int) -> float:
        """Healthcare OOP scaled to current family size."""
        if not self.medical_auto_scale:
            return self.annual_medical_oop
        base = self.annual_medical_oop
        if is_married:
            base *= self.medical_spouse_multiplier
        base += num_children * self.medical_per_child_annual
        return base


# ---------------------------------------------------------------------------
# Investments & savings
# ---------------------------------------------------------------------------

@dataclass
class InvestmentProfile:
    """Savings balances, annual contribution targets, and economic assumptions."""

    # Current balances
    current_liquid_cash: float = 0.0
    current_retirement_balance: float = 0.0
    current_brokerage_balance: float = 0.0
    one_time_upcoming_expenses: float = 0.0   # deducted from liquid cash on day 1

    # Annual contributions (nominal dollars; honored exactly, capped at IRS limits)
    annual_401k_contribution: float = 0.0
    partner_annual_401k_contribution: float = 0.0   # independent IRS limit
    annual_roth_ira_contribution: float = 0.0
    annual_hsa_contribution: float = 0.0
    annual_529_contribution: float = 0.0             # per child
    annual_brokerage_contribution: float = 0.0       # earmarked taxable investment

    # Economic assumptions
    annual_market_return: float = 0.08
    annual_inflation_rate: float = 0.03
    annual_salary_growth_rate: float = 0.04
    partner_salary_growth_rate: float = 0.04
    annual_home_appreciation_rate: float = 0.035

    # Projection behaviour — whether surplus breathing room is swept into
    # brokerage (earns market return) or left as uninvested cash (0% return).
    # Stored here rather than StrategyToggles because it is a cash-flow routing
    # decision, not a tax-optimisation strategy.
    auto_invest_surplus: bool = True

    @property
    def investable_cash(self) -> float:
        return max(0.0, self.current_liquid_cash - self.one_time_upcoming_expenses)


# ---------------------------------------------------------------------------
# Strategy toggles (tax-optimisation only)
# ---------------------------------------------------------------------------

@dataclass
class StrategyToggles:
    """Which tax-optimisation strategies are active."""
    maximize_hsa: bool = True
    use_529_state_deduction: bool = True
    maximize_401k: bool = True
    use_roth_ladder: bool = False
    roth_conversion_annual_amount: float = 0.0


# ---------------------------------------------------------------------------
# Optional plan extensions
# ---------------------------------------------------------------------------

@dataclass
class RetirementProfile:
    """
    Retirement readiness analysis parameters.

    The engine inflates `desired_annual_income` (today's dollars) to nominal
    retirement-year dollars and computes the lump-sum required to fund that
    income for `years_in_retirement` years at `expected_post_retirement_return`,
    after subtracting Social Security.
    """
    current_age: int = 35
    retirement_age: int = 65
    desired_annual_income: float = 80_000     # today's dollars
    years_in_retirement: int = 30
    expected_post_retirement_return: float = 0.05
    estimated_social_security_annual: float = 0.0  # today's dollars

    @property
    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.current_age)


@dataclass
class CollegeProfile:
    """
    College-cost modelling parameters.

    529 balances draw down tax-free in college years; any shortfall comes from
    brokerage.  The AOTC credit (up to $2,500/student/year, first 4 years) is
    applied as a direct tax reduction where income qualifies.

    529 Glide Path
    --------------
    The 529 grows at `early_529_return` for the first `glide_path_years` of the
    projection, then shifts to `late_529_return`.  This is independent of the
    general `annual_market_return` in InvestmentProfile.
    """
    annual_cost_per_child: float = 35_000    # today's dollars
    years_per_child: int = 4
    start_age: int = 18
    use_aotc_credit: bool = True
    early_529_return: float = 0.08           # equity-heavy early years
    late_529_return: float = 0.04            # bond-heavy near college
    glide_path_years: int = 10               # switch point


@dataclass
class CarProfile:
    """
    Car-purchase and financing parameters.

    Cars are purchased every `replace_every_years` years.  Each purchase is
    financed with a down payment (from brokerage) plus an amortising loan whose
    annual P&I reduces breathing room.  When a new car is bought, the old one is
    handed down to any child who has reached `hand_down_age`, or sold for
    `residual_value`.  All dollar amounts are today's dollars; the engine inflates
    them.  For a two-car household, set `num_cars=2`; the engine staggers
    purchases by one year to smooth cash-flow spikes.
    """
    car_price: float = 25_000
    down_payment: float = 5_000
    loan_rate: float = 0.065
    loan_term_years: int = 5
    replace_every_years: int = 10
    residual_value: float = 5_000
    hand_down_age: int = 16
    num_cars: int = 1


# ---------------------------------------------------------------------------
# Timeline events
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A discrete life event that changes financial inputs in a given projection year."""
    year: int
    description: str

    # Income
    income_change: Optional[float] = None          # new gross income for primary person
    partner_income_change: Optional[float] = None

    # Work continuity
    stop_working: bool = False
    resume_working: bool = False
    partner_stop_working: bool = False
    partner_resume_working: bool = False

    # Family
    new_child: bool = False
    child_birth_year_override: Optional[int] = None  # for college cost timing
    new_pet: bool = False
    marriage: bool = False

    # Parent care activation
    start_parent_care: bool = False
    stop_parent_care: bool = False

    # Home purchase
    buy_home: bool = False
    new_home_price: Optional[float] = None
    new_home_down_payment: Optional[float] = None
    new_home_interest_rate: Optional[float] = None
    sell_current_home: bool = True
    buyer_closing_cost_rate: float = 0.02
    seller_closing_cost_rate: float = 0.06
    home_price_override: Optional[float] = None  # back-compat alias

    # One-off cash flows
    extra_one_time_expense: float = 0.0
    extra_one_time_income: float = 0.0


# ---------------------------------------------------------------------------
# Top-level plan
# ---------------------------------------------------------------------------

@dataclass
class FinancialPlan:
    """Everything needed to run a full projection."""
    income: IncomeProfile
    housing: HousingProfile
    lifestyle: LifestyleProfile
    investments: InvestmentProfile
    strategies: StrategyToggles = field(default_factory=StrategyToggles)
    timeline_events: list[TimelineEvent] = field(default_factory=list)
    projection_years: int = 30
    retirement: Optional[RetirementProfile] = None
    college: Optional[CollegeProfile] = None
    car: Optional[CarProfile] = None

    def events_for_year(self, year: int) -> list[TimelineEvent]:
        return [e for e in self.timeline_events if e.year == year]