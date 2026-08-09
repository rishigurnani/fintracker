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


def by_filing_status(status, single, mfj, hoh=None):
    """Pick a value by filing status.

    ``status`` may be a :class:`FilingStatus` or a plain bool where ``True`` means
    married-filing-jointly (convenient at call sites that only track ``is_married``).
    ``hoh`` falls back to the ``single`` value when omitted, matching the common
    case where head-of-household shares the single figure.
    """
    if status is True or status == FilingStatus.MARRIED_FILING_JOINTLY:
        return mfj
    if hoh is not None and status == FilingStatus.HEAD_OF_HOUSEHOLD:
        return hoh
    return single


class RangePhase:
    """Mixin for a value that applies over an inclusive ``[start, end]`` range.

    Subclasses expose ``start``, ``end``, and ``value`` (typically as properties
    over their own domain-specific field names) so a single lookup routine works
    for every phased schedule — childcare costs, Roth contributions, etc.
    """

    def covers(self, x) -> bool:
        return self.start <= x <= self.end


def value_over_phases(phases, x, default: float = 0.0) -> float:
    """Return the ``value`` of the first phase covering ``x``; ``default`` if none.

    Works with any iterable of :class:`RangePhase` (or anything exposing
    ``start``/``end``/``value``).  ``phases`` may be ``None``.
    """
    for phase in phases or ():
        if phase.start <= x <= phase.end:
            return phase.value
    return default


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
class ChildcarePhase(RangePhase):
    """
    Cost of childcare for a child in a given age range.

    age_start and age_end are inclusive (e.g. age_start=0, age_end=2 covers ages 0, 1, 2).
    monthly_cost is in today's dollars — the engine inflates it each year.
    Any age not covered by a phase costs $0 (child is self-sufficient or at college).
    """
    age_start: int          # first age this phase applies to (inclusive)
    age_end:   int          # last age this phase applies to (inclusive)
    monthly_cost: float     # today's dollars per month per child at this age

    # RangePhase interface — see value_over_phases()
    @property
    def start(self) -> int: return self.age_start
    @property
    def end(self) -> int: return self.age_end
    @property
    def value(self) -> float: return self.monthly_cost


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
        return value_over_phases(self.phases, age)


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

    # Insurance premiums (today's dollars, inflated yearly). These are the
    # recurring premiums a W-2 household actually pays out of take-home — the
    # projection previously ignored them, overstating take-home.
    #   * health: the employee's share of employer-sponsored (or marketplace)
    #     premiums. Applied while working and before Medicare eligibility; at
    #     Medicare age it is superseded by RetirementProfile's Medicare model.
    #   * disability: replaces earned income if you can't work — only relevant
    #     while working, so it stops at retirement.
    #   * life: applied every year (zero it out when a term policy lapses).
    annual_health_insurance_premium: float = 0.0
    annual_disability_insurance_premium: float = 0.0
    annual_life_insurance_premium: float = 0.0

    # Life-insurance death benefit (coverage amount, not a premium). Paid into the
    # estate/net worth in the year the primary dies — see RetirementProfile's
    # life_expectancy_age. A FIXED nominal amount (level term policies do not
    # inflation-adjust), so it is NOT scaled by inflation. Zero it if your term
    # policy will have lapsed by then. Only pays out when life_expectancy_age is
    # reached within the projection horizon.
    annual_life_insurance_death_benefit: float = 0.0

    # Your own long-term / end-of-life care (parallel to annual_parent_care_cost,
    # which covers your parents). Modeled as the final self_ltc_years_before_death
    # years of life (through the death year inclusive) — care clusters at the end of
    # life, not at a fixed age. Needs a modeled death year (RetirementProfile with
    # life_expectancy_age); without one it is never applied. Today's dollars,
    # inflated yearly.
    annual_self_ltc_cost: float = 0.0
    self_ltc_years_before_death: int = 3

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
            + self.annual_health_insurance_premium
            + self.annual_disability_insurance_premium
            + self.annual_life_insurance_premium
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
class RothContributionPhase(RangePhase):
    """
    A projection-year range during which a specific Roth IRA contribution is made.

    year_start and year_end are inclusive projection years (1-based).
    annual_amount is the contribution in that phase (today's dollars, fixed nominal).
    Years not covered by any phase default to $0.

    Example — contribute only in years 6-9::

        roth_contribution_schedule:
          - year_start: 6
            year_end:   9
            annual_amount: 7000
    """
    year_start: int
    year_end: int
    annual_amount: float

    # RangePhase interface — see value_over_phases()
    @property
    def start(self) -> int: return self.year_start
    @property
    def end(self) -> int: return self.year_end
    @property
    def value(self) -> float: return self.annual_amount


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
    # Healthcare inflates faster than the general basket (historically ~5% vs ~3%),
    # so medical costs get their own rate: out-of-pocket medical, the health-
    # insurance premium share, your own long-term care, and Medicare premiums +
    # IRMAA surcharges all compound at this rate instead of annual_inflation_rate.
    # (IRMAA *bracket thresholds* stay CPI-indexed at annual_inflation_rate, as the
    # SSA re-indexes them.) Disability/life premiums and every non-medical cost use
    # the general rate. Applied at a fixed rate even in Monte Carlo runs.
    annual_healthcare_inflation_rate: float = 0.05
    annual_salary_growth_rate: float = 0.04
    partner_salary_growth_rate: float = 0.04
    annual_home_appreciation_rate: float = 0.035

    # Salary real-growth plateau. After salary_growth_peak_age the projection
    # stops granting real raises: nominal growth is capped at inflation minus
    # salary_real_decline_rate (0 = flat plateau; >0 = late-career real decline).
    # Age requires a RetirementProfile; without one, salary grows unchanged.
    salary_growth_peak_age: int = 55
    salary_real_decline_rate: float = 0.0

    # Long-term capital-gains tax rate on taxable brokerage gains. Applied when
    # gains are realized (any brokerage drawdown) and to remaining unrealized gains
    # at retirement. US long-term rates are 0/15/20% by income (+3.8% NIIT for high
    # earners); 15% is typical. Defaults to 0% (off) for backward compatibility.
    # Falls back to RetirementProfile.capital_gains_tax_rate when left at 0.
    capital_gains_tax_rate: float = 0.0

    # Annual dividend/distribution yield on the taxable brokerage. Unlike a 401k,
    # a taxable account throws off dividends every year that are taxed as they are
    # paid (a real drag on compounding). The model reinvests the dividend — it is
    # part of annual_market_return — and only leaks the *tax* on it each year, at
    # the qualified-dividend (LTCG) rate + state. ~2% is a typical broad-market
    # yield; set to 0 to disable the drag.
    taxable_dividend_yield: float = 0.02

    # Effective cap-gains rate applied to the *remaining unrealized* gains in the
    # retirement-readiness haircut. SIMPLIFIED ASSUMPTION: a retiree draws the
    # brokerage down gradually and often lands in the 0%/15% LTCG bracket, so this
    # is typically lower than the working-years rate above (e.g. 0.0 or 0.075).
    # When None, the haircut falls back to capital_gains_tax_rate (no discount).
    retirement_capital_gains_tax_rate: Optional[float] = None

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

    # Compounding granularity for investment growth, in months (12 = annual, the
    # default and the historical behaviour; 1 = monthly; 3/6 = quarterly/semiannual;
    # 0.5 = twice-monthly; >12 = coarser than annual). Annual rates are converted
    # geometrically — rate_period = (1 + annual)^(period/12) − 1 — so the starting
    # balance always grows by exactly (1 + annual); the period only changes how much
    # a year's DEPOSITS (contributions and swept surplus) grow, since they are
    # dollar-cost-averaged across the sub-periods instead of added as a year-end lump.
    compounding_period_months: float = 12.0

    # Starting Roth IRA balance (if you already have one)
    current_roth_ira_balance: float = 0.0

    # Optional phase schedule for Roth contributions.
    # When set, overrides annual_roth_ira_contribution for each projection year.
    # Years not covered by any phase contribute $0.
    roth_contribution_schedule: Optional[list] = None  # list[RothContributionPhase]

    def roth_contribution_for_year(self, year: int) -> float:
        """Return the Roth IRA contribution amount for a given projection year.
        Uses the phase schedule if set, otherwise the flat annual amount.
        Returns 0.0 for years not covered by any phase."""
        if self.roth_contribution_schedule:
            return value_over_phases(self.roth_contribution_schedule, year)
        return self.annual_roth_ira_contribution

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

    # Backdoor Roth IRA strategy.
    # When True: contributes annual_roth_ira_contribution post-tax each year
    # (up to the IRS limit: $7,000/person, $14,000 if married).
    # The contribution basis (what you put in) is withdrawn tax-free before
    # touching brokerage when a deficit occurs.
    # Withdrawal waterfall: income → cash → Roth basis → brokerage.
    use_backdoor_roth: bool = False

    # Order in which accounts are drawn to cover a *retirement* cash-flow deficit
    # (a year at/after retirement_age where spending exceeds income). Before
    # retirement the legacy waterfall (cash → Roth basis → brokerage) is used and
    # the 401k/IRA is never touched. Valid keys (see projections.WITHDRAWAL_SOURCES):
    #   "cash_buffer", "uninvested_cash", "retirement_401k", "brokerage", "roth_basis"
    # 401k/IRA withdrawals are taxed as ordinary income at
    # RetirementProfile.retirement_withdrawal_tax_rate (grossed up so the net still
    # funds the deficit) and count toward MAGI for Medicare IRMAA.
    # Any valid source omitted from the list is appended (so nothing is left
    # unspendable); unknown keys are ignored.
    # When None (default), the engine picks the order from the starting balances:
    #   * a large traditional 401k/IRA relative to the brokerage → "bracket-fill"
    #     (draw 401k before brokerage) to shrink future RMDs and smooth taxes;
    #   * otherwise the conventional order (spend brokerage before the 401k).
    # Cash is always drawn first and Roth basis is preserved for last.
    retirement_withdrawal_order: Optional[list] = None  # list[str]


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
    work_start_age: int = 22   # first year of SS-covered work; sets years worked & credits
    retirement_age: int = 65
    desired_annual_income: float = 80_000     # today's dollars
    years_in_retirement: int = 30
    expected_post_retirement_return: float = 0.05
    # Social Security. The dollar benefits are AUTO-ESTIMATED at config load from
    # income + career + claim age + haircut (see config._auto_estimate_ss); the
    # YAML sets only the claim age(s) and the haircut, never a dollar amount.
    estimated_social_security_annual: float = 0.0        # today's $ (auto-filled)
    partner_social_security_annual: float = 0.0          # today's $ (auto-filled)
    social_security_claim_age: Optional[int] = None      # None → retirement_age
    partner_social_security_claim_age: Optional[int] = None
    partner_current_age: Optional[int] = None            # None → current_age
    social_security_haircut: float = 1.0                 # 1.0 = full scheduled benefit

    # Post-retirement withdrawal tax rates.
    # Set these to see Roth's genuine advantage over 401k/brokerage.
    # Defaults to 0% (no tax adjustment) for backward compatibility.
    # Typical values: 401k ~22-32% (ordinary income); brokerage ~15-20% (cap gains).
    retirement_withdrawal_tax_rate: float = 0.0   # applied to 401k/IRA balance
    capital_gains_tax_rate: float = 0.0           # applied to brokerage balance

    # Medicare — modelled as a recurring healthcare cost once the primary reaches
    # medicare_start_age. annual_medicare_premium is the base Part B + Part D
    # premium per enrolled person (today's dollars, inflated). On top of the base,
    # an income-tiered IRMAA surcharge is applied (see constants.irmaa_annual_surcharge).
    # A married couple is assumed to enrol together (both pay), a simplification
    # consistent with the engine modelling no separate partner age.
    medicare_start_age: int = 65
    annual_medicare_premium: float = 2_100.0   # ~Part B ($1,750) + Part D (~$350) base

    # Automatic retirement: when True (default), earned income stops at
    # retirement_age without needing a manual stop_working timeline event, which
    # previously left the projection paying a growing salary into retirement.
    # Only takes effect when a RetirementProfile is configured; set False to keep
    # earning past retirement_age (e.g. to model phased/partial retirement via
    # explicit timeline events instead).
    auto_retire: bool = True

    # Life expectancy / death age (the primary's). When set, it *defines the
    # projection endpoint*: the projection runs through the year the primary
    # reaches this age — extending past OR truncating projection_years — and then
    # all income, spending, and drawdowns stop. This makes the year-by-year tables
    # run right through death, naturally bounds end-of-life costs like self-LTC
    # (which otherwise charge to the horizon), and marks the year any life-insurance
    # death benefit pays into the estate. None (default) keeps the legacy
    # behaviour: run exactly projection_years.
    life_expectancy_age: Optional[int] = None

    # Monte-Carlo-only *realized* death-age range. When BOTH are set, each MC
    # simulation draws the primary's actual death age uniformly from
    # [death_age_min, death_age_max] and runs to that year (survivors-only
    # aggregation past it). This is distinct from life_expectancy_age, which stays
    # the *planning* horizon that forward-looking decisions forecast against (e.g.
    # the medical-burden failsafe): at 75 you don't know your realized death, so you
    # plan to life_expectancy_age even in a sim where you happen to die earlier.
    # None (either unset) → MC uses the deterministic death (life_expectancy_age),
    # exactly as before. Ignored by the deterministic projection.
    death_age_min: Optional[int] = None
    death_age_max: Optional[int] = None

    # Retirement "spending smile". Empirically, retirees' real discretionary
    # spending declines with age (the go-go / slow-go / no-go years). This scales
    # ONLY discretionary lifestyle (vacation, pets, monthly "other") by the
    # primary's age — never medical, Medicare, insurance, care, or LTC costs,
    # which keep inflating. Defaults model full spend through 74, then −10% for
    # 75–84 and −20% at 85+. Set both factors to 1.0 to disable the smile.
    spending_smile_slowgo_age: int = 75      # start of the −10% ("slow-go") band
    spending_smile_slowgo_factor: float = 0.90
    spending_smile_nogo_age: int = 85        # start of the −20% ("no-go") band
    spending_smile_nogo_factor: float = 0.80

    @property
    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.current_age)

    def discretionary_spending_factor(self, age: Optional[int]) -> float:
        """Retirement-smile multiplier on discretionary lifestyle for a given age.

        Returns 1.0 when age is unknown or below the slow-go threshold.
        """
        if age is None:
            return 1.0
        if age >= self.spending_smile_nogo_age:
            return self.spending_smile_nogo_factor
        if age >= self.spending_smile_slowgo_age:
            return self.spending_smile_slowgo_factor
        return 1.0


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

    # Operating costs — the recurring cost of *owning* a car, on top of the loan
    # payment (which is the only cost the engine previously modelled). Applied
    # per household car for every year the car exists. Today's dollars, inflated
    # yearly. Kids' cars are not charged operating costs (kept out of scope).
    annual_insurance_per_car: float = 1_500.0
    annual_maintenance_per_car: float = 1_000.0
    annual_fuel_per_car: float = 2_000.0
    annual_registration_per_car: float = 200.0

    @property
    def annual_operating_cost_per_car(self) -> float:
        """Total yearly cost to run one car (insurance + maintenance + fuel + reg)."""
        return (
            self.annual_insurance_per_car
            + self.annual_maintenance_per_car
            + self.annual_fuel_per_car
            + self.annual_registration_per_car
        )


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
# Failsafes — conditional events that fire when the running state crosses a
# threshold, rather than on a fixed year. Because the condition is checked
# against the live state each year, in Monte Carlo a failsafe triggers
# path-dependently: only in the simulations that actually hit the threshold.
# ---------------------------------------------------------------------------

@dataclass
class FailsafeCondition:
    """One trigger test evaluated on start-of-year state.

    ``metric`` is one of: ``brokerage_balance``, ``liquid_assets``,
    ``investable_assets``, ``retirement_balance``, ``home_equity``,
    ``net_worth``. ``comparator`` is ``below`` or ``above``. When
    ``present_value`` is true the metric is deflated to today's dollars before
    the comparison, so ``threshold`` is read in today's dollars. The condition
    is only armed within ``[start_year, end_year]`` (``end_year`` None **or 0** =
    to the horizon).
    """
    metric: str
    comparator: str                    # 'below' | 'above'
    threshold: float
    present_value: bool = True
    start_year: int = 1
    end_year: Optional[int] = None


@dataclass
class FailsafeAction:
    """What a failsafe does while active.

    Sustained levers (``partner_income`` / ``primary_income``) *replace* that
    person's earned income for the active window and revert when it ends; they
    are taxed as ordinary earned income. One-off levers fire once, at activation.
    When ``present_value`` is true all amounts are in today's dollars and are
    inflated to nominal for the year they apply.
    """
    partner_income: Optional[float] = None
    primary_income: Optional[float] = None
    one_time_income: float = 0.0
    one_time_expense: float = 0.0
    present_value: bool = True
    # Suspend 401k/IRA elective deferrals (and the contingent employer match)
    # for each year the action is active. HSA and 529 contributions are left
    # untouched. Naturally paired with a short duration + once=false so it
    # re-evaluates every year the trigger holds (see docs).
    suspend_retirement_contributions: bool = False
    # Override the annual vacation budget while active (honours ``present_value``,
    # so a value of 4000 means $4k in today's dollars that year). None leaves it.
    annual_vacation: Optional[float] = None
    # Multiplier applied to all healthcare costs (OOP, health premium, self-LTC,
    # Medicare) while active, e.g. 0.5 to halve them ("move abroad"). None leaves.
    medical_cost_multiplier: Optional[float] = None


@dataclass
class Failsafe:
    """A conditional event: fire ``action`` when ``conditions`` are met.

    ``match`` = ``any`` fires when any condition is true, ``all`` requires all.
    ``delay_years`` is the lag between the trigger firing and the action taking
    effect (e.g. a job hunt). ``duration_years`` is how long a sustained action
    lasts; ``None`` **or ``0``** (any non-positive value) means permanent — only a
    value >= 1 bounds the window. With ``once`` true the failsafe fires at most
    once per simulation path.
    """
    name: str
    conditions: list[FailsafeCondition]
    action: FailsafeAction
    match: str = "any"                 # 'any' | 'all'
    delay_years: int = 0
    duration_years: Optional[int] = None
    once: bool = True


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
    failsafes: list[Failsafe] = field(default_factory=list)
    projection_years: int = 30
    retirement: Optional[RetirementProfile] = None
    college: Optional[CollegeProfile] = None
    car:      Optional[CarProfile]      = None
    business: Optional[BusinessProfile]  = None

    def events_for_year(self, year: int) -> list[TimelineEvent]:
        return [e for e in self.timeline_events if e.year == year]