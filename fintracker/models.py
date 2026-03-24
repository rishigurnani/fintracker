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
    """States with meaningful individual income tax.  'OTHER' uses a flat rate supplied by user."""
    GEORGIA = "GA"
    CALIFORNIA = "CA"
    NEW_YORK = "NY"
    TEXAS = "TX"
    FLORIDA = "FL"
    WASHINGTON = "WA"
    ILLINOIS = "IL"
    NORTH_CAROLINA = "NC"
    VIRGINIA = "VA"
    COLORADO = "CO"
    OTHER = "OTHER"


@dataclass
class IncomeProfile:
    """Gross earned income and filing configuration."""
    gross_annual_income: float
    filing_status: FilingStatus = FilingStatus.SINGLE
    state: State = State.GEORGIA
    other_state_flat_rate: float = 0.05
    spouse_gross_annual_income: float = 0.0

    @property
    def total_gross_income(self) -> float:
        return self.gross_annual_income + self.spouse_gross_annual_income


@dataclass
class HousingProfile:
    """Home purchase and mortgage parameters."""
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
        if self.home_price == 0:
            return 0.0
        return self.down_payment / self.home_price

    @property
    def loan_amount(self) -> float:
        return max(0.0, self.home_price - self.down_payment)

    @property
    def requires_pmi(self) -> bool:
        return self.down_payment_pct < 0.20 and not self.is_renting


@dataclass
class LifestyleProfile:
    """Recurring lifestyle expenses."""
    monthly_childcare: float = 0.0
    num_children: int = 0
    num_pets: int = 0
    annual_pet_cost: float = 0.0

    # --- Healthcare ---
    # annual_medical_oop is the baseline cost for a SINGLE adult with no kids.
    # When medical_auto_scale=True the projection engine scales this automatically:
    #   - Marriage:    base * medical_spouse_multiplier  (insurance premium + higher OOP)
    #   - Per child:   + medical_per_child_annual        (pediatric visits, prescriptions)
    # Set medical_auto_scale=False to pin the raw value and manage it yourself.
    annual_medical_oop: float = 0.0
    medical_auto_scale: bool = True
    medical_spouse_multiplier: float = 1.8   # family plan ~80% more than single
    medical_per_child_annual: float = 1_500  # ~$1,500/yr per child OOP (pediatric avg)

    annual_vacation: float = 0.0
    monthly_other_recurring: float = 0.0

    @property
    def annual_total(self) -> float:
        return (
            self.monthly_childcare * 12
            + self.annual_pet_cost
            + self.annual_medical_oop
            + self.annual_vacation
            + self.monthly_other_recurring * 12
        )

    def scaled_medical_oop(self, is_married: bool, num_children: int) -> float:
        """Return healthcare OOP scaled to the current family size."""
        if not self.medical_auto_scale:
            return self.annual_medical_oop
        base = self.annual_medical_oop
        if is_married:
            base *= self.medical_spouse_multiplier
        base += num_children * self.medical_per_child_annual
        return base


@dataclass
class InvestmentProfile:
    """Savings, retirement, and liquid asset configuration."""
    current_liquid_cash: float = 0.0
    current_retirement_balance: float = 0.0
    current_brokerage_balance: float = 0.0
    one_time_upcoming_expenses: float = 0.0

    annual_401k_contribution: float = 0.0          # Primary person's 401k
    partner_annual_401k_contribution: float = 0.0  # Partner's 401k (separate IRS limit)
    annual_roth_ira_contribution: float = 0.0
    annual_hsa_contribution: float = 0.0
    annual_529_contribution: float = 0.0
    annual_brokerage_contribution: float = 0.0

    annual_market_return: float = 0.08
    annual_inflation_rate: float = 0.03
    annual_salary_growth_rate: float = 0.04         # Primary person's salary growth
    partner_salary_growth_rate: float = 0.04        # Partner's salary growth (can differ)
    annual_home_appreciation_rate: float = 0.035

    @property
    def investable_cash(self) -> float:
        return max(0.0, self.current_liquid_cash - self.one_time_upcoming_expenses)


@dataclass
class StrategyToggles:
    """Which tax-optimization strategies are active."""
    maximize_hsa: bool = True
    use_529_state_deduction: bool = True
    maximize_401k: bool = True
    use_roth_ladder: bool = False
    roth_conversion_annual_amount: float = 0.0


@dataclass
class TimelineEvent:
    """A discrete life event that changes financial inputs in a given year."""
    year: int
    description: str
    income_change: Optional[float] = None          # New gross income for primary person
    partner_income_change: Optional[float] = None  # New gross income for partner
    new_child: bool = False
    new_pet: bool = False
    marriage: bool = False

    # Home purchase: set buy_home=True and provide the new home's details.
    # The engine will sell the current home (capturing equity as cash), then
    # open a fresh mortgage on the new home.
    buy_home: bool = False
    new_home_price: Optional[float] = None         # purchase price of the new home
    new_home_down_payment: Optional[float] = None  # cash down payment
    new_home_interest_rate: Optional[float] = None # mortgage rate; defaults to current plan rate
    sell_current_home: bool = True                 # add current equity to brokerage on sale
    # Closing cost rates — applied automatically; override per-event if your deal differs
    buyer_closing_cost_rate: float = 0.02    # % of purchase price (title, lender fees, escrow, taxes)
    seller_closing_cost_rate: float = 0.06   # % of sale price (agent commissions, transfer tax, etc.)

    # Legacy field kept for YAML back-compat
    home_price_override: Optional[float] = None

    extra_one_time_expense: float = 0.0
    extra_one_time_income: float = 0.0


@dataclass
class FinancialPlan:
    """Top-level container — everything needed to run a full projection."""
    income: IncomeProfile
    housing: HousingProfile
    lifestyle: LifestyleProfile
    investments: InvestmentProfile
    strategies: StrategyToggles = field(default_factory=StrategyToggles)
    timeline_events: list[TimelineEvent] = field(default_factory=list)
    projection_years: int = 30

    def events_for_year(self, year: int) -> list[TimelineEvent]:
        return [e for e in self.timeline_events if e.year == year]