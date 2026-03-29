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
class ChildcarePhase:
    """
    Cost of childcare for a child in a given age range.

    age_start and age_end are inclusive (e.g. age_start=0, age_end=2 covers ages 0, 1, 2).
    monthly_cost is in today's dollars — the engine inflates it each year.
    Any age not covered by a phase costs $0 (child is self-sufficient or at college).
    """
    age_start: int          # first age this phase applies to (inclusive)
    age_end:   int          # last age this phase applies to (inclusive)
    monthly_cost: float     # today's dollars per month per child at this age


@dataclass
class ChildcareProfile:
    """
    Age-bracketed childcare cost schedule.

    Replaces the flat monthly_childcare field in LifestyleProfile with a
    realistic cost curve that tracks what childcare actually costs at each
    life stage.  The engine looks up each child's current age each year and
    applies the matching phase.

    Example YAML::

        childcare_profile:
          phases:
            - age_start: 0
              age_end:   2
              monthly_cost: 2500   # infant/toddler — full-time daycare or nanny
            - age_start: 3
              age_end:   4
              monthly_cost: 1500   # preschool
            - age_start: 5
              age_end:  12
              monthly_cost: 600    # before/after school + summer camps
            - age_start: 13
              age_end:  17
              monthly_cost: 150    # activities, minimal supervision
            # age 18+ → handled by CollegeProfile; defaults to $0 here

    Backward compatibility: if childcare_profile is None (the default),
    the engine falls back to LifestyleProfile.monthly_childcare × num_children,
    preserving all existing plans unchanged.
    """
    phases: list = field(default_factory=list)  # list[ChildcarePhase]

    def monthly_cost_at_age(self, age: int) -> float:
        """Return the monthly cost for a child of the given age. 0 if not covered."""
        for phase in self.phases:
            if phase.age_start <= age <= phase.age_end:
                return phase.monthly_cost
        return 0.0


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

    # Age-bracketed childcare schedule. When set, overrides monthly_childcare.
    # monthly_childcare is retained for backward compatibility.
    childcare_profile: Optional[ChildcareProfile] = None

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
class MatchTier:
    """One tier of an employer 401k match formula.

    Examples::

        # 50% match on first 6% of salary
        MatchTier(match_pct=0.50, up_to_pct_of_salary=0.06)

        # Second tier: 25% match on next 4%
        MatchTier(match_pct=0.25, up_to_pct_of_salary=0.04)
    """
    match_pct: float             # employer matches this fraction of employee contribution
    up_to_pct_of_salary: float   # up to this percentage of gross salary per tier


@dataclass
class EmployerMatch:
    """
    Employer 401k matching formula. Supports any combination of:

    * Tiered match (list of MatchTier) — handles simple and complex structures
    * Absolute annual dollar cap (annual_cap)
    * Cliff vesting schedule (vesting_years; 0 = immediate)
    * Profit sharing (flat employer contribution regardless of employee amount)

    The total employer match is:
        sum over tiers of (employee_contrib_in_tier × match_pct)
        + profit_sharing_annual
        capped at annual_cap (if set)
        zeroed if projection_year < vesting_years (cliff vesting)

    Common configurations::

        # Simple: 50% match on first 6% of salary (most common)
        EmployerMatch(tiers=[MatchTier(0.50, 0.06)])

        # Dollar-for-dollar on first 3%
        EmployerMatch(tiers=[MatchTier(1.00, 0.03)])

        # Tiered: 100% on first 3%, 50% on next 2%
        EmployerMatch(tiers=[MatchTier(1.00, 0.03), MatchTier(0.50, 0.02)])

        # Any tier structure capped at $5,000/yr
        EmployerMatch(tiers=[MatchTier(1.00, 0.10)], annual_cap=5000.0)

        # 3-year cliff vesting, dollar-for-dollar on 4%
        EmployerMatch(tiers=[MatchTier(1.00, 0.04)], vesting_years=3)

        # Profit sharing only ($3k/yr, no tier match)
        EmployerMatch(tiers=[], profit_sharing_annual=3000.0)
    """
    tiers: list = field(default_factory=list)  # list[MatchTier]
    annual_cap: Optional[float] = None         # absolute $ ceiling on total match
    vesting_years: int = 0                     # cliff: forfeit if leaving before this year
    profit_sharing_annual: float = 0.0         # flat employer add regardless of employee contrib

    def compute_match(self, employee_contribution: float, gross_salary: float,
                      projection_year: int) -> float:
        """
        Compute employer match for one year.

        projection_year counts from 1 (i.e. the vesting clock starts at employment
        start, which we approximate as projection year 1).
        """
        if self.vesting_years > 0 and projection_year < self.vesting_years:
            return 0.0

        match = self.profit_sharing_annual
        employee_remaining = employee_contribution   # track how much contrib is "used up"

        for tier in self.tiers:
            tier_ceiling = gross_salary * tier.up_to_pct_of_salary
            contrib_in_tier = min(employee_remaining, tier_ceiling)
            match += contrib_in_tier * tier.match_pct
            employee_remaining -= contrib_in_tier
            if employee_remaining <= 0:
                break

        if self.annual_cap is not None:
            match = min(match, self.annual_cap)

        return match


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

    # Employer 401k match — set to None if your employer offers no match
    employer_match: Optional[EmployerMatch] = None

    # Cash buffer: target number of months of total expenses to keep as
    # liquid cash (0% return) before sweeping surplus to brokerage.
    # e.g. cash_buffer_months=3 → always keep 3 months of expenses accessible.
    # This is separate from uninvested_cash (the auto_invest_surplus toggle):
    # the buffer is intentional and maintained even when auto_invest_surplus=True.
    cash_buffer_months: float = 0.0

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
class BusinessProfile:
    """
    Models ownership of a business (franchise, LLC, S-corp, sole prop, etc.).

    Revenue & costs
    ---------------
    net_profit = annual_revenue * (1 - expense_ratio)
    Revenue compounds at revenue_growth_rate each year starting from start_year.
    initial_investment is a one-time draw from brokerage in the start year.

    Tax treatment (applied on top of W-2 income)
    ---------------------------------------------
    SE tax: 15.3% on 92.35% of net profit; employer-half is deductible from AGI.
    QBI deduction: 20% of net profit if use_qbi_deduction=True (phased out above
      $191,950 single / $383,900 MFJ at 2024 thresholds, inflated each year).
    Self-employed health insurance: fully deductible from AGI.
    Solo 401k: up to $69,000/yr (IRS limit); tracked in retirement balance.
    SEP-IRA: up to 25% of net SE income; alternative or supplement to solo 401k.

    Asset value
    -----------
    Business equity = net_profit x equity_multiple, included in net worth.
    If sale_year is set, equity is liquidated into brokerage that year.
    Set equity_multiple=0 to exclude business equity from net worth.
    """
    annual_revenue: float = 0.0           # gross revenue in today's dollars
    expense_ratio: float = 0.60           # operating costs as fraction of revenue
    revenue_growth_rate: float = 0.05     # annual nominal revenue growth rate
    initial_investment: float = 0.0       # one-time startup/acquisition cost (in start_year)
    start_year: int = 1                   # projection year business starts earning

    use_qbi_deduction: bool = True        # 20% QBI pass-through deduction
    self_employed_health_insurance: float = 0.0   # annual premium, AGI-deductible
    solo_401k_contribution: float = 0.0   # owner solo 401k contribution (IRS limit: $69k)
    sep_ira_contribution: float = 0.0     # SEP-IRA contribution (<=25% net SE income)

    equity_multiple: float = 3.0          # business value = net_profit * this
    sale_year: Optional[int] = None       # sell business in this year; proceeds -> brokerage
    ownership_pct: float = 1.0            # your ownership share (e.g. 0.50 for 50/50 partnership)


@dataclass
class KidCarProfile:
    """
    Configuration for a first car given to each child.

    buy_at_age controls when each child receives a car:
      - 16  → at driving age (handed down from household or bought new)
      - 22  → at college graduation (start_age + years_per_child)
      - None → defaults to college graduation age if a CollegeProfile is
                configured, otherwise age 16

    All dollar amounts are today's dollars; the engine inflates them.
    Financed with a down payment from brokerage and an amortising loan.
    """
    car_price: float = 15_000
    down_payment_pct: float = 0.20
    loan_rate: float = 0.07
    loan_term_years: int = 5
    buy_at_age: Optional[int] = None  # None = graduation age if college configured, else 16


@dataclass
class CarProfile:
    """
    Car-purchase and financing parameters for household cars.

    Cars are purchased every `replace_every_years` years.  Each purchase is
    financed with a down payment (from brokerage) plus an amortising loan whose
    annual P&I reduces breathing room.  When a new car is bought, the old one is
    handed down to any child who has reached `hand_down_age`, or sold for
    `residual_value`.  All dollar amounts are today's dollars; the engine inflates
    them.

    `first_purchase_years`: list of projection years in which each car is first
    bought.  e.g. [3, 5] for a two-car household buying in yr 3 and yr 5.
    Before the first purchase year the car does not exist; no loan, no payment.
    If None, falls back to the legacy stagger (yr 1, yr 0, ...).

    `kids_car`: optional sub-profile for a first car given to each child.
    Set buy_at_age=16 for driving age, 22 (or None) for college graduation.
    """
    car_price: float = 25_000
    down_payment: float = 5_000
    loan_rate: float = 0.065
    loan_term_years: int = 5
    replace_every_years: int = 10
    residual_value: float = 5_000
    hand_down_age: int = 16
    num_cars: int = 1
    kids_car: Optional[KidCarProfile] = None
    # First purchase years for each car, in projection-year terms.
    # e.g. [3, 5] means Car 1 bought in yr 3, Car 2 in yr 5.
    # Length must equal num_cars.  None = use legacy stagger (yr 1, yr 0, ...).
    first_purchase_years: Optional[list[int]] = None


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
    car:      Optional[CarProfile]      = None
    business: Optional[BusinessProfile]  = None

    def events_for_year(self, year: int) -> list[TimelineEvent]:
        return [e for e in self.timeline_events if e.year == year]