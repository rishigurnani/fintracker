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
    annual_medical_oop: float = 0.0
    medical_auto_scale: bool = True
    medical_spouse_multiplier: float = 1.8
    medical_per_child_annual: float = 1_500

    annual_vacation: float = 0.0
    monthly_other_recurring: float = 0.0

    # Wedding fund: annual amount saved per child toward their future wedding.
    # Deducted from breathing room and held in brokerage; separate from college costs.
    annual_wedding_fund_per_child: float = 0.0

    # --- Parent care ---
    # Annual cost to support aging parents (e.g. in-home care, assisted living contribution).
    # Set parent_care_start_year / parent_care_end_year in TimelineEvents to bound it.
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

    annual_401k_contribution: float = 0.0
    partner_annual_401k_contribution: float = 0.0
    annual_roth_ira_contribution: float = 0.0
    annual_hsa_contribution: float = 0.0
    annual_529_contribution: float = 0.0
    # Fixed annual amount earmarked for taxable brokerage (separate from organic surplus).
    # This is deducted from breathing room and tracked separately in the projection.
    annual_brokerage_contribution: float = 0.0

    annual_market_return: float = 0.08
    annual_inflation_rate: float = 0.03
    annual_salary_growth_rate: float = 0.04
    partner_salary_growth_rate: float = 0.04
    annual_home_appreciation_rate: float = 0.035

    @property
    def investable_cash(self) -> float:
        return max(0.0, self.current_liquid_cash - self.one_time_upcoming_expenses)


@dataclass
class RetirementProfile:
    """
    Parameters for retirement readiness analysis.

    The engine computes whether the projected retirement balance at
    `retirement_age` is sufficient to fund `desired_annual_income`
    (expressed in today's dollars) for `years_in_retirement` years.
    """
    current_age: int = 35
    retirement_age: int = 65
    # Desired annual income in retirement, in TODAY'S dollars.
    # The engine inflates this to nominal dollars at retirement.
    desired_annual_income: float = 80_000
    years_in_retirement: int = 30
    # Conservative post-retirement portfolio return (less equity-heavy than accumulation)
    expected_post_retirement_return: float = 0.05
    # Optional Social Security estimate (today's dollars)
    estimated_social_security_annual: float = 0.0

    @property
    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.current_age)


@dataclass
class CarProfile:
    """
    Parameters for modelling car purchases and financing.

    Cars are purchased every `replace_every_years`. Each purchase is financed
    with a down payment (cash from brokerage) and a loan (P&I reduces breathing
    room each year). When a new car is bought, the old one is either handed down
    to a child who has reached `hand_down_age`, or sold for `residual_value`.

    All dollar amounts are in TODAY's dollars — the engine inflates them.
    `num_cars` allows a two-car household to model both cars simultaneously,
    assuming they were purchased at different points so payments are staggered
    across the projection (first car starts yr 1, second car starts yr 2 by default).
    """
    car_price: float = 25_000           # purchase price in today's dollars
    down_payment: float = 5_000         # cash down payment per car
    loan_rate: float = 0.065            # annual interest rate (e.g. 0.065 = 6.5%)
    loan_term_years: int = 5            # loan repayment period
    replace_every_years: int = 10       # years between new car purchases
    residual_value: float = 5_000       # sale price of old car when not handing down
    hand_down_age: int = 16             # minimum child age to receive a handed-down car
    num_cars: int = 1                   # cars in the household


@dataclass
class CollegeProfile:
    """
    Parameters for modelling college costs per child.

    The projection engine draws down 529 balances in college years
    and applies the American Opportunity Tax Credit (AOTC) where eligible.

    529 Growth Glide Path
    ---------------------
    529 accounts use an age-based growth rate rather than the portfolio's
    general market return:
      - early_529_return: aggressive rate for the first `glide_path_years`
        years of saving (default 8% — equity-heavy)
      - late_529_return:  conservative rate after that (default 4% — bond-heavy)
    This reflects the standard practice of shifting 529s to safer assets
    as college approaches.
    """
    # Annual college cost per child in TODAY'S dollars (tuition + room/board)
    annual_cost_per_child: float = 35_000
    years_per_child: int = 4
    # Age at which each child starts college
    start_age: int = 18
    # 529 balances are drawn down first (tax-free); remainder comes from brokerage
    # AOTC: up to $2,500 credit per student per year, first 4 years, income-limited
    use_aotc_credit: bool = True
    # Income phase-out: single $80k-$90k, MFJ $160k-$180k (2024 figures)
    # NOTE: wedding fund savings are configured in LifestyleProfile,
    # not here — they are separate from college costs.
    # 529 glide path: aggressive return for first N years, then conservative
    early_529_return: float = 0.08   # years 1–glide_path_years
    late_529_return:  float = 0.04   # after glide_path_years
    glide_path_years: int   = 10     # switch point


@dataclass
class StrategyToggles:
    """Which tax-optimization strategies are active."""
    maximize_hsa: bool = True
    use_529_state_deduction: bool = True
    maximize_401k: bool = True
    use_roth_ladder: bool = False
    roth_conversion_annual_amount: float = 0.0
    # When True: all surplus breathing room is swept into brokerage each year,
    # earning the full market return. When False: surplus sits in cash (0% return).
    # Toggle ON to see the compounding benefit of investing every spare dollar.
    auto_invest_surplus: bool = True


@dataclass
class TimelineEvent:
    """A discrete life event that changes financial inputs in a given year."""
    year: int
    description: str
    income_change: Optional[float] = None          # New gross income for primary person
    partner_income_change: Optional[float] = None  # New gross income for partner

    # Work start/stop: stops income (sabbatical, caregiving, retirement).
    # Resume restores income to the last value * (1+salary_growth)^years_off.
    stop_working: bool = False           # Primary stops working (income → 0)
    resume_working: bool = False         # Primary resumes (income_change sets new salary)
    partner_stop_working: bool = False
    partner_resume_working: bool = False

    new_child: bool = False
    # birth_year is set automatically to event.year when new_child=True.
    # Override only if you need a specific birth year for college cost timing.
    child_birth_year_override: Optional[int] = None

    new_pet: bool = False
    marriage: bool = False

    # Parent care: start/stop the annual_parent_care_cost from LifestyleProfile
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
    home_price_override: Optional[float] = None  # back-compat

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
    retirement: Optional[RetirementProfile] = None
    college: Optional[CollegeProfile] = None
    car: Optional[CarProfile] = None

    def events_for_year(self, year: int) -> list[TimelineEvent]:
        return [e for e in self.timeline_events if e.year == year]