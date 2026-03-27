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
    CarProfile, CollegeProfile, FilingStatus, FinancialPlan,
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
_AOTC_MAX_CREDIT          = 2_500
_AOTC_PHASEOUT_SINGLE_LOW  = 80_000
_AOTC_PHASEOUT_SINGLE_HIGH = 90_000
_AOTC_PHASEOUT_MFJ_LOW     = 160_000
_AOTC_PHASEOUT_MFJ_HIGH    = 180_000


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

    # Flags
    parent_care_active: bool

    # Car loan state — one dict per car
    cars: list[dict]

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

    # Car one-off costs (for display / debugging)
    car_purchase_cost: float = 0.0
    car_sale_proceeds: float = 0.0

    @property
    def total_assets(self) -> float:
        return (
            self.retirement_balance + self.brokerage_balance
            + self.college_529_balance + self.home_equity
            + self.hsa_balance + self.uninvested_cash
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
    p10_net_worth: list[float]
    p25_net_worth: list[float]
    p50_net_worth: list[float]
    p75_net_worth: list[float]
    p90_net_worth: list[float]
    mean_net_worth: list[float]
    prob_retire_at_65: float = 0.0
    prob_millionaire_10yr: float = 0.0
    num_simulations: int = 1_000


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

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
    ) -> MonteCarloResult:
        rng = np.random.default_rng(seed)
        inv = self._plan.investments
        years = list(range(1, self._plan.projection_years + 1))
        all_nw: list[list[float]] = []

        for _ in range(n_simulations):
            mkt   = rng.normal(inv.annual_market_return, 0.15, len(years))
            inf   = np.clip(rng.normal(inv.annual_inflation_rate, 0.015, len(years)), 0, 0.15)
            sg    = np.clip(rng.normal(inv.annual_salary_growth_rate, 0.02, len(years)), -0.10, 0.20)
            state = self._initial_state()
            sim: list[float] = []
            for i, year in enumerate(years):
                self._apply_timeline_events(state, year)
                snap = self._compute_year(
                    state, year,
                    market_return_override=float(mkt[i]),
                    inflation_override=float(inf[i]),
                    salary_growth_override=float(sg[i]),
                )
                sim.append(snap.net_worth)
                self._advance_state(state, snap,
                                    market_return=float(mkt[i]),
                                    inflation=float(inf[i]),
                                    salary_growth=float(sg[i]))
            all_nw.append(sim)

        by_year = list(zip(*all_nw))
        def pct(arr, p): return float(np.percentile(arr, p))
        yr10 = list(by_year[9]) if len(by_year) >= 10 else []

        return MonteCarloResult(
            years=years,
            p10_net_worth=[pct(yr, 10) for yr in by_year],
            p25_net_worth=[pct(yr, 25) for yr in by_year],
            p50_net_worth=[pct(yr, 50) for yr in by_year],
            p75_net_worth=[pct(yr, 75) for yr in by_year],
            p90_net_worth=[pct(yr, 90) for yr in by_year],
            mean_net_worth=[float(np.mean(yr)) for yr in by_year],
            prob_millionaire_10yr=(
                sum(1 for nw in yr10 if nw >= 1_000_000) / n_simulations if yr10 else 0.0
            ),
            num_simulations=n_simulations,
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
            - (p.car.down_payment * p.car.num_cars if p.car else 0.0)
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
            parent_care_active=p.lifestyle.annual_parent_care_cost > 0,
            cars=self._init_cars(p.car),
        )

    @staticmethod
    def _amort_lookup(mc: MortgageCalculator) -> dict[int, float]:
        return {row.year: row.balance for row in mc.full_schedule() if row.month % 12 == 0}

    @staticmethod
    def _init_cars(car: Optional[CarProfile]) -> list[dict]:
        """Initialise one state-dict per car, staggered by one year each."""
        if car is None:
            return []
        cars = []
        for i in range(car.num_cars):
            principal  = max(0.0, car.car_price - car.down_payment)
            monthly_pi = ProjectionEngine._car_monthly_pi(principal, car.loan_rate, car.loan_term_years)
            cars.append({
                "loan_balance":   principal,
                "loan_year":      1,
                "purchase_year":  1 - i,  # car 0 → yr 1, car 1 → yr 2, etc.
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
        tax, aotc = self._tax_and_credits(state, year, hsa, k401, r529, inf_f)
        net_income = state.gross_income - tax - hsa - k401 - partner_k401

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
        ret_bal, hsa_bal, col529_bal, brok_bal, uninvested = self._asset_growth(
            state, year, mkt, hsa, k401, partner_k401,
            annual_529_save, drawdown_529, brokerage_earmark, breathing_room,
        )

        nw = ret_bal + hsa_bal + col529_bal + brok_bal + home_equity + uninvested

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
            mortgage_balance=eoy_mortgage,
            net_worth=nw,
            filing_status=state.filing_status,
            num_children=state.num_children,
            is_renting=state.is_renting,
            is_married=state.is_married,
            is_working=state.is_working,
            is_partner_working=state.is_partner_working,
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

    def _cars(
        self, state: EngineState, year: int, inf_f: float
    ) -> tuple[float, float, float]:
        """Returns (annual_payment, purchase_cost, sale_proceeds)."""
        car = self._plan.car
        if not car:
            return 0.0, 0.0, 0.0

        total_pmt, total_purchase, total_sale = 0.0, 0.0, 0.0

        for c in state.cars:
            years_owned = year - c["purchase_year"]

            # Replacement cycle
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

            # Annual loan payment
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
    ) -> tuple[float, float, float, float, float]:
        """Returns (retirement, hsa, col529, brokerage, uninvested_cash)."""
        col = self._plan.college

        ret_bal  = state.retirement_balance * (1 + mkt) + k401 + partner_k401
        hsa_bal  = state.hsa_balance        * (1 + mkt) + hsa

        r529_growth = (
            col.early_529_return if year <= col.glide_path_years else col.late_529_return
        ) if col else mkt
        col529_bal = max(0.0,
            state.college_529_balance * (1 + r529_growth) + annual_529_save - drawdown_529
        )

        inv = self._plan.investments
        if inv.auto_invest_surplus:
            brok_bal    = state.brokerage_balance * (1 + mkt) + brokerage_earmark + breathing_room
            uninvested  = 0.0
        else:
            brok_bal = state.brokerage_balance * (1 + mkt) + brokerage_earmark
            if breathing_room >= 0:
                uninvested = state.uninvested_cash + breathing_room
            else:
                deficit    = breathing_room  # negative
                avail      = state.uninvested_cash
                drawn      = min(avail, -deficit)
                uninvested = avail - drawn
                brok_bal  += deficit + drawn  # remaining deficit hits brokerage

        return ret_bal, hsa_bal, col529_bal, brok_bal, uninvested

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