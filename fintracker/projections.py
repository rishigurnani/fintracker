"""
Long-term projection engine.

Architecture
------------
ProjectionEngine drives two public entry-points:

  run_deterministic()            → list[YearlySnapshot]
  run_monte_carlo()              → MonteCarloResult
  compute_retirement_readiness() → Optional[RetirementReadiness]

Internally, each year is processed as:

  1. _apply_timeline_events() — mutates EngineState for that year's events
  2. _compute_year()          — pure calculation; returns YearlySnapshot
       ├── _contributions()   — 401k / HSA / 529 amounts
       ├── _housing()         — cost, equity, amortisation
       ├── _lifestyle()       — medical / childcare / pets / parent care
       ├── _college()         — 529 drawdown, AOTC credit
       ├── _cars()            — loan payments, purchase/sale cash flows
       └── _asset_growth()    — new balances for all accounts
  3. _advance_state()         — rolls EngineState forward to next year
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fintracker.models import (
    BusinessProfile, CarProfile, KidCarProfile, CollegeProfile, FilingStatus, FinancialPlan,
    HousingProfile, IncomeProfile, InvestmentProfile,
    RetirementProfile, StrategyToggles,
)
from fintracker.tax_engine import TaxEngine
from fintracker.mortgage import MortgageCalculator


# ---------------------------------------------------------------------------
# IRS / tax constants
# ---------------------------------------------------------------------------

_HSA_LIMIT_SINGLE         = 4_150
_HSA_LIMIT_FAMILY         = 8_300
_401K_LIMIT               = 30_500
_SOLO_401K_LIMIT          = 69_000
_SE_TAX_RATE              = 0.1530
_SE_TAX_DEDUCTIBLE_SHARE  = 0.9235
_QBI_PHASEOUT_SINGLE      = 191_950
_QBI_PHASEOUT_MFJ         = 383_900
_AOTC_MAX_CREDIT          = 2_500
_AOTC_PHASEOUT_SINGLE_LOW  = 80_000
_AOTC_PHASEOUT_SINGLE_HIGH = 90_000
_AOTC_PHASEOUT_MFJ_LOW     = 160_000
_AOTC_PHASEOUT_MFJ_HIGH    = 180_000


# ---------------------------------------------------------------------------
# Historical S&P 500 annual total returns (1926–2025)
# Source: provided by user; used for bootstrap sampling in Monte Carlo.
# Bootstrap preserves the true empirical distribution — fat tails, skew,
# and crash years — rather than assuming normality.
# ---------------------------------------------------------------------------
_SP500_HISTORICAL_RETURNS: tuple[float, ...] = (
    # (year, return) — only returns stored, sorted newest-first for readability
    0.1788, 0.2502, 0.2629, -0.1811, 0.2871, 0.1840, 0.3149, -0.0438,
    0.2183, 0.1196, 0.0138, 0.1369, 0.3239, 0.1600, 0.0211, 0.1506,
    0.2646, -0.3700, 0.0549, 0.1579, 0.0491, 0.1088, 0.2868, -0.2210,
    -0.1189, -0.0910, 0.2104, 0.2858, 0.3336, 0.2296, 0.3758, 0.0132,
    0.1008, 0.0762, 0.3047, -0.0310, 0.3169, 0.1661, 0.0525, 0.1867,
    0.3173, 0.0627, 0.2256, 0.2155, -0.0491, 0.3242, 0.1844, 0.0656,
    -0.0718, 0.2384, 0.3720, -0.2647, -0.1466, 0.1898, 0.1431, 0.0401,
    -0.0850, 0.1106, 0.2398, -0.1006, 0.1245, 0.1648, 0.2280, -0.0873,
    0.2689, 0.0047, 0.1196, 0.4336, -0.1078, 0.0656, 0.3156, 0.5262,
    -0.0099, 0.1837, 0.2402, 0.3171, 0.1879, 0.0550, 0.0571, -0.0807,
    0.3644, 0.1975, 0.2590, 0.2034, -0.1159, -0.0978, -0.0041, 0.3112,
    -0.3503, 0.3392, 0.4767, -0.0144, 0.5399, -0.0819, -0.4334, -0.2490,
    -0.0842, 0.4361, 0.3749, 0.1162,
)


# ---------------------------------------------------------------------------
# Engine state — typed, explicit, no loose dicts
# ---------------------------------------------------------------------------

@dataclass
class EngineState:
    """
    Mutable state that rolls forward one year at a time.

    All income values are nominal dollars for the current projection year.
    `gross_income` is always derived as income_primary + income_partner and
    is kept in sync by every mutation that touches either component.
    """
    # Income
    income_primary: float
    income_partner: float
    filing_status: FilingStatus
    is_married: bool
    is_working: bool
    is_partner_working: bool

    # Family
    num_children: int
    num_pets: int
    child_birth_years: list[int]

    # Housing
    is_renting: bool
    monthly_rent: float
    mortgage_calc: Optional[MortgageCalculator]
    amort_lookup: dict[int, float]
    mortgage_year_offset: int
    mortgage_interest_rate: float
    home_price_ref: float
    home_value: float
    mortgage_balance: float

    # Balances
    retirement_balance: float
    brokerage_balance: float
    hsa_balance: float
    college_529_balance: float
    uninvested_cash: float
    cash_buffer: float
    business_equity: float
    business_revenue: float

    # Flags
    parent_care_active: bool

    # Car loan state — one dict per car
    cars: list[dict]
    # Kid car loans — one entry per child who has received a car
    kid_car_loans: list[dict]

    @property
    def gross_income(self) -> float:
        return self.income_primary + self.income_partner


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class YearlySnapshot:
    """Complete financial picture for a single projection year."""
    year: int

    # Income
    gross_income: float
    net_income: float
    annual_tax_total: float

    # Expenses
    annual_housing_cost: float
    annual_lifestyle_cost: float
    annual_medical_oop: float
    annual_college_cost: float
    annual_529_drawdown: float
    annual_parent_care_cost: float
    annual_retirement_contributions: float
    annual_hsa_contributions: float
    annual_brokerage_contribution: float
    annual_aotc_credit: float
    annual_car_payment: float
    annual_wedding_save: float

    # Cash flow
    annual_breathing_room: float

    # Assets
    retirement_balance: float
    brokerage_balance: float
    college_529_balance: float
    home_value: float
    home_equity: float
    hsa_balance: float
    uninvested_cash: float

    # Liabilities
    mortgage_balance: float

    # Net worth
    net_worth: float

    # Meta
    filing_status: FilingStatus
    num_children: int
    is_renting: bool
    is_married: bool
    is_working: bool
    is_partner_working: bool

    # Business income and equity
    annual_business_income: float = 0.0
    business_equity: float = 0.0
    # Car one-off costs (for display / debugging)
    car_purchase_cost: float = 0.0
    car_sale_proceeds: float = 0.0
    # Intentional cash buffer (earns 0%; maintained before sweeping to brokerage)
    cash_buffer: float = 0.0

    @property
    def total_assets(self) -> float:
        return (
            self.retirement_balance + self.brokerage_balance
            + self.college_529_balance + self.home_equity
            + self.hsa_balance + self.uninvested_cash + self.cash_buffer
            + self.business_equity
        )

    @property
    def liquid_assets(self) -> float:
        return self.brokerage_balance


@dataclass
class RetirementReadiness:
    """Result of the retirement readiness analysis."""
    years_to_retirement: int
    retirement_year: int
    projected_balance_at_retirement: float
    required_balance: float
    on_track: bool
    funded_pct: float
    annual_surplus_or_gap: float
    desired_income_nominal: float
    social_security_offset: float


@dataclass
class MonteCarloResult:
    """Result of N Monte Carlo simulation runs."""
    years: list[int]

    # Net worth percentile bands
    p10_net_worth: list[float]
    p25_net_worth: list[float]
    p50_net_worth: list[float]
    p75_net_worth: list[float]
    p90_net_worth: list[float]
    mean_net_worth: list[float]

    # Liquidity risk: per-year probability that liquid assets (brokerage) go
    # negative in that simulation year.  Values in [0, 1].
    prob_negative_liquid: list[float]

    # Brokerage balance percentiles (same pool as liquid assets chart)
    p10_liquid: list[float]
    p50_liquid: list[float]
    p90_liquid: list[float]

    # Summary statistics
    prob_millionaire_10yr: float = 0.0
    num_simulations: int = 1_000

    # Simulation parameters (stored for display)
    use_historical_returns: bool = True
    market_return_std: float = 0.15
    inflation_std: float = 0.015
    salary_growth_std: float = 0.02


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def _legacy_car_down(car) -> float:
    """Sum of down payments for legacy-mode cars (no first_purchase_years)."""
    if not car:
        return 0.0
    if car.first_purchase_years:
        # All cars have explicit purchase years — no upfront deduction
        return 0.0
    return car.down_payment * car.num_cars


class ProjectionEngine:
    """
    Runs deterministic and Monte Carlo projections for a FinancialPlan.

    Usage::

        engine = ProjectionEngine(plan)
        snapshots = engine.run_deterministic()
        mc = engine.run_monte_carlo(n_simulations=1_000)
        rr = engine.compute_retirement_readiness(snapshots)
    """

    def __init__(self, plan: FinancialPlan) -> None:
        self._plan = plan
        self._tax = TaxEngine()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_deterministic(self) -> list[YearlySnapshot]:
        state = self._initial_state()
        snapshots: list[YearlySnapshot] = []
        for year in range(1, self._plan.projection_years + 1):
            self._apply_timeline_events(state, year)
            snap = self._compute_year(state, year)
            snapshots.append(snap)
            self._advance_state(state, snap)
        return snapshots

    def run_monte_carlo(
        self,
        n_simulations: int = 1_000,
        seed: Optional[int] = None,
        market_return_std: float = 0.15,
        inflation_std: float = 0.015,
        salary_growth_std: float = 0.02,
        use_historical_returns: bool = True,
    ) -> MonteCarloResult:
        """
        Run N Monte Carlo simulations with randomized economic parameters.

        Parameters
        ----------
        n_simulations         : number of simulation runs
        seed                  : RNG seed for reproducibility (None = random)
        use_historical_returns: if True (default), sample market returns by
                                bootstrap from the historical S&P 500 dataset
                                (1926–2025).  This preserves the true empirical
                                distribution — fat tails, skewness, -43% crashes,
                                +54% booms — rather than assuming normality.
                                If False, draws from N(mean, market_return_std).
        market_return_std     : std dev used only when use_historical_returns=False
        inflation_std         : std dev of annual inflation (always normal)
        salary_growth_std     : std dev of annual salary growth (always normal)

        In both modes, inflation and salary growth remain normally distributed
        since those datasets are smaller and more symmetric.
        """
        rng = np.random.default_rng(seed)
        inv = self._plan.investments
        years = list(range(1, self._plan.projection_years + 1))
        all_nw:  list[list[float]] = []
        all_liq: list[list[float]] = []   # brokerage balance per sim per year

        hist = np.array(_SP500_HISTORICAL_RETURNS)

        for _ in range(n_simulations):
            if use_historical_returns:
                # Bootstrap: sample with replacement from the historical dataset.
                # Each draw is an independent annual return — no autocorrelation
                # assumed (consistent with weak-form market efficiency).
                mkt = rng.choice(hist, size=len(years), replace=True)
            else:
                mkt = rng.normal(inv.annual_market_return, market_return_std, len(years))
            inf = np.clip(rng.normal(inv.annual_inflation_rate, inflation_std, len(years)), 0, 0.15)
            sg  = np.clip(rng.normal(inv.annual_salary_growth_rate, salary_growth_std, len(years)), -0.10, 0.20)
            state = self._initial_state()
            sim_nw:  list[float] = []
            sim_liq: list[float] = []
            for i, year in enumerate(years):
                self._apply_timeline_events(state, year)
                snap = self._compute_year(
                    state, year,
                    market_return_override=float(mkt[i]),
                    inflation_override=float(inf[i]),
                    salary_growth_override=float(sg[i]),
                )
                sim_nw.append(snap.net_worth)
                # Liquid position = brokerage + cash_buffer: both are accessible.
                # Measuring brokerage alone understates liquidity when a buffer exists.
                sim_liq.append(snap.brokerage_balance + snap.cash_buffer)
                self._advance_state(state, snap,
                                    market_return=float(mkt[i]),
                                    inflation=float(inf[i]),
                                    salary_growth=float(sg[i]))
            all_nw.append(sim_nw)
            all_liq.append(sim_liq)

        by_year_nw  = list(zip(*all_nw))
        by_year_liq = list(zip(*all_liq))

        def pct(arr, p): return float(np.percentile(arr, p))

        yr10 = list(by_year_nw[9]) if len(by_year_nw) >= 10 else []

        # Probability of negative liquid assets in each year
        prob_neg = [
            sum(1 for v in yr if v < 0) / n_simulations
            for yr in by_year_liq
        ]

        return MonteCarloResult(
            years=years,
            p10_net_worth=[pct(yr, 10) for yr in by_year_nw],
            p25_net_worth=[pct(yr, 25) for yr in by_year_nw],
            p50_net_worth=[pct(yr, 50) for yr in by_year_nw],
            p75_net_worth=[pct(yr, 75) for yr in by_year_nw],
            p90_net_worth=[pct(yr, 90) for yr in by_year_nw],
            mean_net_worth=[float(np.mean(yr)) for yr in by_year_nw],
            prob_negative_liquid=prob_neg,
            p10_liquid=[pct(yr, 10) for yr in by_year_liq],
            p50_liquid=[pct(yr, 50) for yr in by_year_liq],
            p90_liquid=[pct(yr, 90) for yr in by_year_liq],
            prob_millionaire_10yr=(
                sum(1 for nw in yr10 if nw >= 1_000_000) / n_simulations if yr10 else 0.0
            ),
            num_simulations=n_simulations,
            use_historical_returns=use_historical_returns,
            market_return_std=market_return_std,
            inflation_std=inflation_std,
            salary_growth_std=salary_growth_std,
        )

    def compute_retirement_readiness(
        self,
        snapshots: Optional[list[YearlySnapshot]] = None,
    ) -> Optional[RetirementReadiness]:
        """
        Returns None when no RetirementProfile is configured.

        Required balance uses a growing annuity — spending inflates at
        annual_inflation_rate throughout retirement, not just to retirement day.
        This is the correct model: a retiree spending $290k/yr at age 65 will
        need roughly $299k at 66, $308k at 67, etc.

        Formula: PV of growing annuity
            required = PMT / (r - g) * (1 - ((1+g)/(1+r))^n)
        where PMT = nominal income needed at retirement start,
              r   = expected_post_retirement_return,
              g   = annual_inflation_rate (spending growth in retirement),
              n   = years_in_retirement.

        Edge cases:
          r == g : required = PMT * n / (1 + r)   (L'Hôpital limit)
          r == 0 : required = PMT * n              (no return, no growth help)
        """
        rp = self._plan.retirement
        if rp is None:
            return None

        if snapshots is None:
            snapshots = self.run_deterministic()

        inflation    = self._plan.investments.annual_inflation_rate
        years_to_ret = rp.years_to_retirement

        # Projected balance = all investable assets at retirement year
        # (same pool used by the Retirement Readiness panel — single source of truth)
        snap = next((s for s in snapshots if s.year == years_to_ret), snapshots[-1])
        projected = snap.retirement_balance + snap.hsa_balance + snap.brokerage_balance

        # Income need at retirement start (nominal dollars)
        nominal_income = rp.desired_annual_income * (1 + inflation) ** years_to_ret
        ss_nominal     = rp.estimated_social_security_annual * (1 + inflation) ** years_to_ret
        income_needed  = max(0.0, nominal_income - ss_nominal)

        r, g, n = rp.expected_post_retirement_return, inflation, rp.years_in_retirement

        if r == 0:
            # No investment return: simple sum, no discounting
            required = income_needed * n
        elif abs(r - g) < 1e-9:
            # r ≈ g: L'Hôpital limit of growing annuity formula
            required = income_needed * n / (1 + r)
        else:
            # Growing annuity: PV = PMT/(r-g) * (1 - ((1+g)/(1+r))^n)
            required = income_needed / (r - g) * (1 - ((1 + g) / (1 + r)) ** n)

        funded_pct  = projected / required if required > 0 else math.inf
        balance_gap = projected - required

        # Convert balance surplus/gap to annual income equivalent
        # Using the same growing annuity formula in reverse
        if r == 0:
            annual_gap = balance_gap / n if n > 0 else 0.0
        elif abs(r - g) < 1e-9:
            annual_gap = balance_gap * (1 + r) / n if n > 0 else 0.0
        else:
            annual_gap = balance_gap * (r - g) / (1 - ((1 + g) / (1 + r)) ** n)

        return RetirementReadiness(
            years_to_retirement=years_to_ret,
            retirement_year=snap.year,
            projected_balance_at_retirement=projected,
            required_balance=required,
            on_track=projected >= required,
            funded_pct=funded_pct,
            annual_surplus_or_gap=annual_gap,
            desired_income_nominal=nominal_income,
            social_security_offset=ss_nominal,
        )

    # ------------------------------------------------------------------ #
    # State initialisation                                                 #
    # ------------------------------------------------------------------ #

    def _initial_state(self) -> EngineState:
        p   = self._plan
        inv = p.investments
        is_married = p.income.filing_status == FilingStatus.MARRIED_FILING_JOINTLY

        mortgage_calc = None
        amort_lookup: dict[int, float] = {}
        if not p.housing.is_renting and p.housing.loan_amount > 0:
            mortgage_calc = MortgageCalculator(p.housing, inv.annual_home_appreciation_rate)
            amort_lookup  = self._amort_lookup(mortgage_calc)

        initial_brokerage = (
            inv.current_liquid_cash
            - inv.one_time_upcoming_expenses
            - (p.housing.down_payment if not p.housing.is_renting else 0.0)
            # Only deduct down payments for legacy cars (pre-purchased at projection start).
            # Cars with first_purchase_years will have their down payment deducted
            # in the year they are first bought via _cars().
            - (_legacy_car_down(p.car))
            + inv.current_brokerage_balance
        )

        return EngineState(
            income_primary=p.income.gross_annual_income,
            income_partner=p.income.spouse_gross_annual_income,
            filing_status=p.income.filing_status,
            is_married=is_married,
            is_working=True,
            is_partner_working=p.income.spouse_gross_annual_income > 0,
            num_children=p.lifestyle.num_children,
            num_pets=p.lifestyle.num_pets,
            child_birth_years=[0] * p.lifestyle.num_children,
            is_renting=p.housing.is_renting,
            monthly_rent=p.housing.monthly_rent,
            mortgage_calc=mortgage_calc,
            amort_lookup=amort_lookup,
            mortgage_year_offset=0,
            mortgage_interest_rate=p.housing.interest_rate,
            home_price_ref=p.housing.home_price,
            home_value=p.housing.home_price if not p.housing.is_renting else 0.0,
            mortgage_balance=p.housing.loan_amount if not p.housing.is_renting else 0.0,
            retirement_balance=inv.current_retirement_balance,
            brokerage_balance=initial_brokerage,
            hsa_balance=0.0,
            college_529_balance=0.0,
            uninvested_cash=0.0,
            cash_buffer=0.0,
            parent_care_active=p.lifestyle.annual_parent_care_cost > 0,
            cars=self._init_cars(p.car),
            kid_car_loans=[],
            business_equity=0.0,
            business_revenue=(p.business.annual_revenue if p.business else 0.0),
        )

    @staticmethod
    def _amort_lookup(mc: MortgageCalculator) -> dict[int, float]:
        return {row.year: row.balance for row in mc.full_schedule() if row.month % 12 == 0}

    @staticmethod
    def _init_cars(car: Optional[CarProfile]) -> list[dict]:
        """
        Initialise one state-dict per car.

        If first_purchase_years is configured, each car starts with no loan and
        no payments until its specified purchase year, at which point _cars()
        will buy it and start the loan.  Before that year the entry is inert.

        Legacy fallback: if first_purchase_years is None, uses the old stagger
        (car 0 bought at yr 1, car 1 at yr 0) so existing configs are unchanged.
        """
        if car is None:
            return []
        cars = []
        for i in range(car.num_cars):
            if car.first_purchase_years and i < len(car.first_purchase_years):
                # Explicit first purchase year — car hasn't been bought yet
                cars.append({
                    "loan_balance":    0.0,
                    "loan_year":       0,
                    "purchase_year":   None,          # None = not yet purchased
                    "first_buy_year":  car.first_purchase_years[i],
                    "monthly_payment": 0.0,
                })
            else:
                # Legacy: treat as already purchased and financed at projection start
                principal  = max(0.0, car.car_price - car.down_payment)
                monthly_pi = ProjectionEngine._car_monthly_pi(
                    principal, car.loan_rate, car.loan_term_years)
                cars.append({
                    "loan_balance":    principal,
                    "loan_year":       1,
                    "purchase_year":   1 - i,
                    "first_buy_year":  None,
                    "monthly_payment": monthly_pi,
                })
        return cars

    # ------------------------------------------------------------------ #
    # Timeline events                                                      #
    # ------------------------------------------------------------------ #

    def _apply_timeline_events(self, state: EngineState, year: int) -> None:
        p = self._plan
        for ev in p.events_for_year(year):

            if ev.marriage:
                state.filing_status = FilingStatus.MARRIED_FILING_JOINTLY
                state.is_married    = True

            if ev.new_child:
                birth = ev.child_birth_year_override if ev.child_birth_year_override is not None else year
                state.child_birth_years.append(birth)
                state.num_children += 1

            if ev.new_pet:
                state.num_pets += 1

            # Work continuity
            if ev.stop_working:
                state.is_working    = False
                state.income_primary = 0.0
            if ev.resume_working:
                state.is_working    = True
            if ev.partner_stop_working:
                state.is_partner_working = False
                state.income_partner     = 0.0
            if ev.partner_resume_working:
                state.is_partner_working = True

            # Income changes (must come after stop/resume so resume + income_change works)
            if ev.income_change is not None:
                state.income_primary = ev.income_change
                if ev.resume_working:
                    state.is_working = True
            if ev.partner_income_change is not None:
                state.income_partner = ev.partner_income_change
                if ev.partner_resume_working:
                    state.is_partner_working = True

            # Parent care
            if ev.start_parent_care:
                state.parent_care_active = True
            if ev.stop_parent_care:
                state.parent_care_active = False

            # One-off cash
            state.brokerage_balance += ev.extra_one_time_income
            state.brokerage_balance -= ev.extra_one_time_expense

            # Home purchase
            if ev.buy_home:
                self._apply_home_purchase(state, ev)

    def _apply_home_purchase(self, state: EngineState, ev) -> None:
        p          = self._plan
        new_price  = ev.new_home_price or ev.home_price_override or state.home_value
        new_down   = ev.new_home_down_payment or new_price * 0.20
        new_rate   = ev.new_home_interest_rate or state.mortgage_interest_rate

        if ev.sell_current_home and not state.is_renting:
            equity   = max(0.0, state.home_value - state.mortgage_balance)
            proceeds = equity - state.home_value * ev.seller_closing_cost_rate
            state.brokerage_balance += max(0.0, proceeds)

        state.brokerage_balance -= new_down + new_price * ev.buyer_closing_cost_rate

        new_hp = HousingProfile(
            home_price=new_price, down_payment=new_down, interest_rate=new_rate,
            loan_term_years=p.housing.loan_term_years,
            annual_property_tax_rate=p.housing.annual_property_tax_rate,
            annual_insurance=p.housing.annual_insurance,
            annual_maintenance_rate=p.housing.annual_maintenance_rate,
            pmi_annual_rate=p.housing.pmi_annual_rate,
        )
        new_calc = MortgageCalculator(new_hp, p.investments.annual_home_appreciation_rate)

        state.mortgage_calc        = new_calc
        state.amort_lookup         = self._amort_lookup(new_calc)
        state.mortgage_year_offset = ev.year - 1
        state.mortgage_interest_rate = new_rate
        state.home_price_ref       = new_price
        state.home_value           = new_price
        state.mortgage_balance     = new_hp.loan_amount
        state.is_renting           = False

    # ------------------------------------------------------------------ #
    # Year computation — orchestrator + focused helpers                   #
    # ------------------------------------------------------------------ #

    def _compute_year(
        self,
        state: EngineState,
        year: int,
        market_return_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
        salary_growth_override: Optional[float] = None,
    ) -> YearlySnapshot:
        p   = self._plan
        inv = p.investments
        mkt = market_return_override if market_return_override is not None else inv.annual_market_return
        inf = inflation_override if inflation_override is not None else inv.annual_inflation_rate
        inf_f = (1 + inf) ** (year - 1)

        # --- Contributions & tax ---
        hsa, k401, partner_k401, r529 = self._contributions(state, year)
        biz_net, biz_se_tax, biz_equity, biz_solo_401k = self._business(state, year)
        tax, aotc = self._tax_and_credits(state, year, hsa, k401, r529, inf_f)
        net_income = state.gross_income + biz_net - tax - biz_se_tax - hsa - k401 - partner_k401 - biz_solo_401k

        # --- Expenses ---
        housing_cost, home_equity, home_value, eoy_mortgage = self._housing(state, year, inf_f)
        lifestyle_cost, medical_oop, parent_care = self._lifestyle(state, inf_f)
        college_gross, drawdown_529, net_college, annual_529_save = self._college(state, year, inf_f, r529)
        wedding_save = self._wedding_save(state, year, inf_f)
        car_pmt, car_purchase, car_sale = self._cars(state, year, inf_f)
        brokerage_earmark = inv.annual_brokerage_contribution

        breathing_room = (
            net_income
            - housing_cost
            - lifestyle_cost
            - annual_529_save
            - net_college
            - brokerage_earmark
            - car_pmt
            - wedding_save
        )

        # --- Asset growth ---
        ret_bal, hsa_bal, col529_bal, brok_bal, uninvested, new_buffer = self._asset_growth(
            state, year, mkt, hsa, k401 + biz_solo_401k, partner_k401,
            annual_529_save, drawdown_529, brokerage_earmark, breathing_room,
            annual_expenses=lifestyle_cost + housing_cost,
        )

        nw = ret_bal + hsa_bal + col529_bal + brok_bal + home_equity + uninvested + new_buffer + biz_equity

        return YearlySnapshot(
            year=year,
            gross_income=state.gross_income,
            net_income=net_income,
            annual_tax_total=tax,
            annual_housing_cost=housing_cost,
            annual_lifestyle_cost=lifestyle_cost,
            annual_medical_oop=medical_oop,
            annual_college_cost=college_gross,
            annual_529_drawdown=drawdown_529,
            annual_parent_care_cost=parent_care,
            annual_retirement_contributions=k401 + partner_k401,
            annual_hsa_contributions=hsa,
            annual_brokerage_contribution=brokerage_earmark,
            annual_aotc_credit=aotc,
            annual_car_payment=car_pmt,
            annual_wedding_save=wedding_save,
            annual_breathing_room=breathing_room,
            retirement_balance=ret_bal,
            brokerage_balance=brok_bal,
            college_529_balance=col529_bal,
            home_value=home_value,
            home_equity=home_equity,
            hsa_balance=hsa_bal,
            uninvested_cash=uninvested,
            cash_buffer=new_buffer,
            mortgage_balance=eoy_mortgage,
            net_worth=nw,
            filing_status=state.filing_status,
            num_children=state.num_children,
            is_renting=state.is_renting,
            is_married=state.is_married,
            is_working=state.is_working,
            is_partner_working=state.is_partner_working,
            annual_business_income=biz_net,
            business_equity=biz_equity,
            car_purchase_cost=car_purchase,
            car_sale_proceeds=car_sale,
        )

    # ------------------------------------------------------------------ #
    # Computation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _contributions(
        self, state: EngineState, year: int
    ) -> tuple[float, float, float, float]:
        """Returns (hsa, k401, partner_k401, r529)."""
        inv   = self._plan.investments
        strat = self._plan.strategies

        is_family = state.is_married or state.num_children > 0
        hsa_limit = _HSA_LIMIT_FAMILY if is_family else _HSA_LIMIT_SINGLE
        hsa       = min(inv.annual_hsa_contribution, hsa_limit) if strat.maximize_hsa else 0.0

        k401         = min(inv.annual_401k_contribution, _401K_LIMIT)
        partner_k401 = (
            min(inv.partner_annual_401k_contribution, _401K_LIMIT)
            if state.income_partner > 0 else 0.0
        )

        # 529: stop contributing once all children have graduated (only enforced
        # when a CollegeProfile is configured)
        col = self._plan.college
        if col and state.child_birth_years:
            graduated = all(
                (year - by) >= col.start_age + col.years_per_child
                for by in state.child_birth_years
            )
            r529 = 0.0 if graduated else inv.annual_529_contribution
        else:
            r529 = inv.annual_529_contribution

        return hsa, k401, partner_k401, r529

    def _tax_and_credits(
        self,
        state: EngineState,
        year: int,
        hsa: float,
        k401: float,
        r529: float,
        inf_f: float,
    ) -> tuple[float, float]:
        """Returns (effective_tax, aotc_credit)."""
        p = self._plan
        tmp_inc = IncomeProfile(
            gross_annual_income=state.gross_income,
            filing_status=state.filing_status,
            state=p.income.state,
            other_state_flat_rate=p.income.other_state_flat_rate,
        )
        tmp_inv = InvestmentProfile(
            annual_hsa_contribution=hsa,
            annual_401k_contribution=k401,
            annual_529_contribution=r529,
        )
        raw_tax  = self._tax.calculate(tmp_inc, tmp_inv, p.strategies,
                                       num_children=state.num_children).total_annual_tax
        aotc     = self._aotc_credit(state, year, state.gross_income, state.is_married, inf_f)
        eff_tax  = max(0.0, raw_tax - aotc)
        return eff_tax, aotc

    def _housing(
        self, state: EngineState, year: int, inf_f: float
    ) -> tuple[float, float, float, float]:
        """Returns (annual_cost, home_equity, home_value, eoy_mortgage_balance)."""
        p = self._plan
        if state.is_renting:
            return state.monthly_rent * 12 * inf_f, 0.0, 0.0, 0.0

        mc = state.mortgage_calc
        if mc:
            monthly_pi    = mc.monthly_pi_payment()
            ref           = state.home_price_ref
            monthly_other = (
                ref * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
                + p.housing.annual_insurance
            ) / 12 * inf_f
            pmi = (
                mc._pmi_payment(state.mortgage_balance)
                if state.mortgage_balance / ref > 0.80 and mc._p.requires_pmi else 0.0
            )
            cost = (monthly_pi + monthly_other + pmi) * 12

            mortgage_yr   = year - state.mortgage_year_offset
            eoy_balance   = state.amort_lookup.get(mortgage_yr, state.mortgage_balance)
            equity        = max(0.0, state.home_value - eoy_balance)
            return cost, equity, state.home_value, eoy_balance

        # Owned outright
        ref  = state.home_price_ref
        cost = (
            ref * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
            + p.housing.annual_insurance
        ) * inf_f
        return cost, state.home_value, state.home_value, 0.0

    def _lifestyle(
        self, state: EngineState, inf_f: float
    ) -> tuple[float, float, float]:
        """Returns (annual_lifestyle, medical_oop, parent_care)."""
        lif = self._plan.lifestyle

        medical     = lif.scaled_medical_oop(state.is_married, state.num_children) * inf_f
        pets        = state.num_pets * lif.annual_pet_cost * inf_f
        childcare   = state.num_children * lif.monthly_childcare * 12 * inf_f
        vacation    = lif.annual_vacation * inf_f
        other       = lif.monthly_other_recurring * 12 * inf_f
        parent_care = lif.annual_parent_care_cost * inf_f if state.parent_care_active else 0.0

        return medical + pets + childcare + vacation + other + parent_care, medical, parent_care

    def _college(
        self,
        state: EngineState,
        year: int,
        inf_f: float,
        r529: float,
    ) -> tuple[float, float, float, float]:
        """Returns (gross_cost, drawdown_529, net_from_brokerage, annual_529_save)."""
        col = self._plan.college
        annual_529_save = r529 * state.num_children

        if not col or not state.child_birth_years:
            return 0.0, 0.0, 0.0, annual_529_save

        # 529 available = current balance + this year's contributions (same-year drawdown allowed)
        available = state.college_529_balance + annual_529_save
        gross, drawdown, remaining = 0.0, 0.0, available

        for by in state.child_birth_years:
            age = year - by
            if col.start_age <= age < col.start_age + col.years_per_child:
                cost    = col.annual_cost_per_child * inf_f
                gross  += cost
                drawn   = min(remaining, cost)
                drawdown += drawn
                remaining -= drawn

        net_brokerage = max(0.0, gross - drawdown)
        return gross, drawdown, net_brokerage, annual_529_save

    def _aotc_credit(
        self,
        state: EngineState,
        year: int,
        gross_income: float,
        is_married: bool,
        inf_f: float,
    ) -> float:
        col = self._plan.college
        if not col or not col.use_aotc_credit:
            return 0.0

        low  = _AOTC_PHASEOUT_MFJ_LOW  if is_married else _AOTC_PHASEOUT_SINGLE_LOW
        high = _AOTC_PHASEOUT_MFJ_HIGH if is_married else _AOTC_PHASEOUT_SINGLE_HIGH
        if gross_income >= high:
            return 0.0

        phase = max(0.0, 1.0 - (gross_income - low) / (high - low)) if gross_income > low else 1.0

        eligible = sum(
            1 for by in state.child_birth_years
            if col.start_age <= (year - by) < col.start_age + min(col.years_per_child, 4)
            and (year - by - col.start_age + 1) <= min(col.years_per_child, 4)
        )
        return eligible * _AOTC_MAX_CREDIT * phase

    def _wedding_save(self, state: EngineState, year: int, inf_f: float) -> float:
        """Annual wedding fund savings — stops when each child turns 25."""
        rate = self._plan.lifestyle.annual_wedding_fund_per_child
        if not rate:
            return 0.0
        return rate * sum(
            1 for by in state.child_birth_years if (year - by) <= 25
        )

    def _business(
        self,
        state: EngineState,
        year: int,
    ) -> tuple[float, float, float, float]:
        """
        Returns (net_income, se_tax, business_equity, solo_401k_contribution).

        net_income   — owner's draw after SE tax, QBI deduction, and health
                       insurance deduction; flows into breathing room
        se_tax       — self-employment tax owed (shown separately in cash flow)
        business_equity — current business asset value (net_profit × multiple)
        solo_401k_contribution — amount deposited to retirement this year

        Revenue is stored in state.business_revenue and grown in _advance_state.
        The initial_investment is deducted from brokerage in the start year.
        """
        biz = self._plan.business
        if biz is None or year < biz.start_year:
            return 0.0, 0.0, state.business_equity, 0.0

        # One-time initial investment in start year
        if year == biz.start_year and biz.initial_investment > 0:
            state.brokerage_balance -= biz.initial_investment

        # Business sale: liquidate equity into brokerage once, then silence permanently.
        if biz.sale_year is not None and year >= biz.sale_year:
            if year == biz.sale_year:
                proceeds = state.business_equity
                state.brokerage_balance += proceeds
            return 0.0, 0.0, 0.0, 0.0

        revenue    = state.business_revenue
        net_profit = revenue * (1.0 - biz.expense_ratio)

        # --- Self-employment tax ---
        # SE tax is 15.3% on 92.35% of net profit.
        # The employer half (7.65%) is deductible from AGI — reduces taxable income.
        se_base     = net_profit * _SE_TAX_DEDUCTIBLE_SHARE
        se_tax      = se_base * _SE_TAX_RATE
        employer_half_deduction = se_tax / 2.0

        # --- Health insurance deduction ---
        hi_deduction = min(biz.self_employed_health_insurance, net_profit)

        # --- QBI deduction ---
        # 20% of qualified business income, phased out above income thresholds.
        # Simplified: apply phase-out linearly over a $50k window above the limit.
        qbi_deduction = 0.0
        if biz.use_qbi_deduction:
            limit = (_QBI_PHASEOUT_MFJ if state.is_married else _QBI_PHASEOUT_SINGLE)
            total_income = state.gross_income + net_profit
            if total_income <= limit:
                phase = 1.0
            elif total_income >= limit + 50_000:
                phase = 0.0
            else:
                phase = 1.0 - (total_income - limit) / 50_000
            qbi_deduction = net_profit * 0.20 * phase

        # --- Solo 401k ---
        # Capped at IRS limit and net profit (can't contribute more than earned)
        solo_k = min(biz.solo_401k_contribution, _SOLO_401K_LIMIT, max(0.0, net_profit))

        # --- SEP-IRA ---
        # Up to 25% of net self-employment income (after SE tax deduction)
        sep_base = max(0.0, net_profit - employer_half_deduction)
        sep = min(biz.sep_ira_contribution, 0.25 * sep_base)
        # SEP flows into retirement alongside solo 401k
        solo_k_total = min(solo_k + sep, _SOLO_401K_LIMIT)

        # --- Net income to owner ---
        # Gross profit minus all deductions; the actual tax impact on W-2 income
        # is handled in _tax_and_credits via the normal tax engine (which will see
        # a lower AGI because of employer_half_deduction + hi_deduction + qbi_deduction).
        # Here we return the owner's take-home after SE tax and retirement contributions.
        net_income = net_profit - se_tax - solo_k_total

        # --- Business equity ---
        biz_equity = net_profit * biz.equity_multiple

        return net_income, se_tax, biz_equity, solo_k_total

    def _cars(
        self, state: EngineState, year: int, inf_f: float
    ) -> tuple[float, float, float]:
        """Returns (annual_payment, purchase_cost, sale_proceeds)."""
        car = self._plan.car
        if not car:
            return 0.0, 0.0, 0.0

        total_pmt, total_purchase, total_sale = 0.0, 0.0, 0.0

        for c in state.cars:
            # --- First purchase (explicit first_buy_year mode) ---
            if c["purchase_year"] is None:
                # Car hasn't been bought yet; wait for its first_buy_year
                if c.get("first_buy_year") == year:
                    nominal_price = car.car_price * inf_f
                    nominal_down  = car.down_payment * inf_f
                    state.brokerage_balance -= nominal_down
                    total_purchase += nominal_down
                    principal = max(0.0, nominal_price - nominal_down)
                    monthly   = self._car_monthly_pi(principal, car.loan_rate, car.loan_term_years)
                    c["loan_balance"]    = principal
                    c["loan_year"]       = 1
                    c["purchase_year"]   = year
                    c["monthly_payment"] = monthly
                # Nothing to do before first_buy_year — skip to next car
                if c["purchase_year"] is None:
                    continue

            # --- Replacement cycle ---
            years_owned = year - c["purchase_year"]
            if years_owned > 0 and years_owned % car.replace_every_years == 0:
                proceeds = self._car_old_proceeds(state, car, year)
                state.brokerage_balance += proceeds
                total_sale += proceeds

                nominal_price = car.car_price * inf_f
                nominal_down  = car.down_payment * inf_f
                state.brokerage_balance -= nominal_down
                total_purchase += nominal_down

                principal = max(0.0, nominal_price - nominal_down)
                monthly   = self._car_monthly_pi(principal, car.loan_rate, car.loan_term_years)
                c["loan_balance"]    = principal
                c["loan_year"]       = 1
                c["purchase_year"]   = year
                c["monthly_payment"] = monthly

            # --- Annual loan payment ---
            if c["loan_balance"] > 0 and c["loan_year"] <= car.loan_term_years:
                annual_pmt = c["monthly_payment"] * 12
                annual_pmt = min(annual_pmt, c["loan_balance"] * (1 + car.loan_rate / 12) * 12)
                total_pmt += annual_pmt

                r = car.loan_rate / 12
                n_paid  = (c["loan_year"] - 1) * 12
                n_total = car.loan_term_years * 12
                remaining = (
                    c["monthly_payment"] * (1 - (1 + r) ** -(n_total - n_paid)) / r
                    if r > 0
                    else c["loan_balance"] - c["monthly_payment"] * 12
                )
                c["loan_balance"] = max(0.0, remaining)
                c["loan_year"]   += 1

        # --- Kids' first cars ---
        if car and car.kids_car:
            col = self._plan.college  # may be None; buy_at_age defaulting handles it
            kc  = car.kids_car
            # buy_at_age default: graduation age if college configured, else 16
            if kc.buy_at_age is not None:
                buy_age = kc.buy_at_age
            elif col is not None:
                buy_age = col.start_age + col.years_per_child  # graduation age
            else:
                buy_age = 16

            for child_idx, birth_year in enumerate(state.child_birth_years):
                child_age = year - birth_year
                # Buy car in exactly the graduation year
                if child_age == buy_age:
                    # Check not already bought for this child
                    already = any(l["child_idx"] == child_idx for l in state.kid_car_loans)
                    if not already:
                        nominal_price = kc.car_price * inf_f
                        down          = nominal_price * kc.down_payment_pct
                        principal     = nominal_price - down
                        monthly_pmt   = self._car_monthly_pi(principal, kc.loan_rate, kc.loan_term_years)
                        state.brokerage_balance -= down
                        total_purchase += down
                        state.kid_car_loans.append({
                            "child_idx":     child_idx,
                            "loan_balance":  principal,
                            "loan_year":     1,
                            "monthly_payment": monthly_pmt,
                        })

            # Annual payments on active kid car loans
            for loan in state.kid_car_loans:
                if loan["loan_balance"] > 0 and loan["loan_year"] <= kc.loan_term_years:
                    annual_pmt = loan["monthly_payment"] * 12
                    annual_pmt = min(annual_pmt, loan["loan_balance"] * (1 + kc.loan_rate / 12) * 12)
                    total_pmt += annual_pmt
                    r = kc.loan_rate / 12
                    n_paid  = (loan["loan_year"] - 1) * 12
                    n_total = kc.loan_term_years * 12
                    remaining = (
                        loan["monthly_payment"] * (1 - (1 + r) ** -(n_total - n_paid)) / r
                        if r > 0
                        else loan["loan_balance"] - loan["monthly_payment"] * 12
                    )
                    loan["loan_balance"] = max(0.0, remaining)
                    loan["loan_year"] += 1

        return total_pmt, total_purchase, total_sale

    def _car_old_proceeds(
        self, state: EngineState, car: CarProfile, year: int
    ) -> float:
        """Sell old car for residual_value, or hand down to an age-eligible child."""
        if not state.child_birth_years:
            return car.residual_value
        if any((year - by) >= car.hand_down_age for by in state.child_birth_years):
            return 0.0
        return car.residual_value

    def _asset_growth(
        self,
        state: EngineState,
        year: int,
        mkt: float,
        hsa: float,
        k401: float,
        partner_k401: float,
        annual_529_save: float,
        drawdown_529: float,
        brokerage_earmark: float,
        breathing_room: float,
        annual_expenses: float = 0.0,
    ) -> tuple[float, float, float, float, float, float]:
        """Returns (retirement, hsa, col529, brokerage, uninvested_cash, cash_buffer)."""
        col = self._plan.college
        inv = self._plan.investments

        ret_bal  = state.retirement_balance * (1 + mkt) + k401 + partner_k401
        hsa_bal  = state.hsa_balance        * (1 + mkt) + hsa

        r529_growth = (
            col.early_529_return if year <= col.glide_path_years else col.late_529_return
        ) if col else mkt
        col529_bal = max(0.0,
            state.college_529_balance * (1 + r529_growth) + annual_529_save - drawdown_529
        )

        # --- Cash buffer ---
        # Target floor = N months of annual expenses held as liquid cash (0% return).
        # Buffer is topped up from breathing room BEFORE surplus is swept to brokerage.
        # Deficits drain the buffer first, then brokerage (or uninvested_cash).
        buffer_floor = annual_expenses * inv.cash_buffer_months / 12
        current_buf  = state.cash_buffer

        if breathing_room >= 0:
            topup      = min(breathing_room, max(0.0, buffer_floor - current_buf))
            new_buffer = current_buf + topup
            investable = breathing_room - topup   # what remains after topping buffer
        else:
            deficit    = -breathing_room          # positive amount
            buf_drawn  = min(current_buf, deficit)
            new_buffer = current_buf - buf_drawn
            investable = -(deficit - buf_drawn)   # remaining deficit (negative) or 0

        # --- Brokerage / uninvested ---
        if inv.auto_invest_surplus:
            brok_bal   = state.brokerage_balance * (1 + mkt) + brokerage_earmark + investable
            uninvested = 0.0
        else:
            brok_bal = state.brokerage_balance * (1 + mkt) + brokerage_earmark
            if investable >= 0:
                uninvested = state.uninvested_cash + investable
            else:
                deficit2   = -investable
                avail      = state.uninvested_cash
                drawn      = min(avail, deficit2)
                uninvested = avail - drawn
                brok_bal  += -(deficit2 - drawn)

        return ret_bal, hsa_bal, col529_bal, brok_bal, uninvested, new_buffer

    # ------------------------------------------------------------------ #
    # State advancement                                                    #
    # ------------------------------------------------------------------ #

    def _advance_state(
        self,
        state: EngineState,
        snap: YearlySnapshot,
        market_return: Optional[float] = None,
        inflation: Optional[float] = None,
        salary_growth: Optional[float] = None,
    ) -> None:
        p   = self._plan
        sg  = salary_growth if salary_growth is not None else p.investments.annual_salary_growth_rate
        psg = salary_growth if salary_growth is not None else p.investments.partner_salary_growth_rate

        if state.is_working:
            state.income_primary *= (1 + sg)
        if state.is_partner_working:
            state.income_partner *= (1 + psg)

        state.retirement_balance  = snap.retirement_balance
        state.brokerage_balance   = snap.brokerage_balance
        state.hsa_balance         = snap.hsa_balance
        state.college_529_balance = snap.college_529_balance
        state.uninvested_cash     = snap.uninvested_cash
        state.cash_buffer         = snap.cash_buffer
        state.business_equity     = snap.business_equity
        if self._plan.business and snap.year >= self._plan.business.start_year:
            state.business_revenue *= (1 + self._plan.business.revenue_growth_rate)

        if not state.is_renting:
            state.home_value = snap.home_value * (1 + p.investments.annual_home_appreciation_rate)
            self._advance_mortgage(state, snap)

    def _advance_mortgage(self, state: EngineState, snap: YearlySnapshot) -> None:
        mc = state.mortgage_calc
        if not mc or state.mortgage_balance <= 0:
            return
        amort  = state.amort_lookup
        offset = state.mortgage_year_offset
        myr    = snap.year - offset
        if myr in amort:
            state.mortgage_balance = amort[myr]
        elif amort and myr > max(amort):
            state.mortgage_balance = 0.0
        else:
            rate    = state.mortgage_interest_rate
            ann_int = state.mortgage_balance * rate
            ann_pi  = mc.monthly_pi_payment() * 12
            state.mortgage_balance = max(0.0, state.mortgage_balance - max(0.0, ann_pi - ann_int))

    # ------------------------------------------------------------------ #
    # Car helpers                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _car_monthly_pi(principal: float, annual_rate: float, term_years: int) -> float:
        """Standard amortising loan monthly P&I payment."""
        if principal <= 0:
            return 0.0
        if annual_rate == 0:
            return principal / (term_years * 12)
        r = annual_rate / 12
        n = term_years * 12
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)