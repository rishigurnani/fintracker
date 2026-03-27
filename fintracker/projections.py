"""
Long-term projection engine.

Features:
  - Deterministic year-by-year projection + Monte Carlo simulation
  - Home purchase mid-projection with exact amortization
  - Healthcare auto-scaling by family size
  - HSA single/family tier upgrade on marriage
  - Dual income with independent salary growth rates
  - Start/stop work events (sabbatical, caregiving, early retirement)
  - Annual brokerage contribution earmark (separate from organic surplus)
  - College costs with 529 drawdown and AOTC tax credit
  - Parent care costs with start/stop timeline events
  - Retirement readiness analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fintracker.models import (
    CarProfile, CollegeProfile, FilingStatus, FinancialPlan, HousingProfile,
    IncomeProfile, InvestmentProfile, RetirementProfile,
    StrategyToggles, TimelineEvent,
)
from fintracker.tax_engine import TaxEngine
from fintracker.mortgage import MortgageCalculator

_HSA_LIMIT_SINGLE = 4_150
_HSA_LIMIT_FAMILY = 8_300

# AOTC: up to $2,500/student/year, first 4 years of college
_AOTC_MAX_CREDIT = 2_500
_AOTC_PHASEOUT_SINGLE_LOW  = 80_000
_AOTC_PHASEOUT_SINGLE_HIGH = 90_000
_AOTC_PHASEOUT_MFJ_LOW     = 160_000
_AOTC_PHASEOUT_MFJ_HIGH    = 180_000


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
    annual_college_cost: float          # gross college expense this year
    annual_529_drawdown: float          # portion paid from 529 (tax-free)
    annual_parent_care_cost: float      # parent care extracted from lifestyle for visibility
    annual_retirement_contributions: float
    annual_hsa_contributions: float
    annual_brokerage_contribution: float  # earmarked brokerage investment
    annual_aotc_credit: float           # American Opportunity Tax Credit applied

    # Cash flow
    annual_breathing_room: float        # after all expenses and earmarked investments

    # Assets
    retirement_balance: float
    brokerage_balance: float
    college_529_balance: float          # 529 balance (separate from brokerage)
    home_value: float
    home_equity: float
    hsa_balance: float

    # Liabilities
    mortgage_balance: float

    # Net worth
    net_worth: float

    # Meta
    filing_status: FilingStatus
    num_children: int
    is_renting: bool
    is_married: bool
    is_working: bool            # primary person working this year
    is_partner_working: bool

    # Default-valued fields must come last in a dataclass
    # Uninvested surplus (only non-zero when auto_invest_surplus=False)
    uninvested_cash: float = 0.0
    # Car costs
    annual_car_payment: float = 0.0   # total P&I paid on car loans this year
    car_purchase_cost: float = 0.0    # down payment(s) paid this year
    car_sale_proceeds: float = 0.0    # proceeds from selling old car(s)
    # Wedding fund
    annual_wedding_save: float = 0.0  # amount saved toward children's weddings

    @property
    def total_assets(self) -> float:
        return (
            self.retirement_balance
            + self.brokerage_balance
            + self.college_529_balance
            + self.home_equity
            + self.hsa_balance
            + self.uninvested_cash
        )

    @property
    def liquid_assets(self) -> float:
        return self.brokerage_balance


@dataclass
class RetirementReadiness:
    """Result of the retirement readiness analysis."""
    years_to_retirement: int
    retirement_year: int                  # projection year when retirement happens (0 if beyond horizon)
    projected_balance_at_retirement: float
    required_balance: float               # to fund desired income for N years
    on_track: bool
    funded_pct: float                     # projected / required (1.0 = exactly funded)
    annual_surplus_or_gap: float          # positive = surplus per yr, negative = shortfall
    desired_income_nominal: float         # desired income inflated to retirement year dollars
    social_security_offset: float         # annual SS benefit in nominal retirement dollars


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



def _car_monthly_payment(principal: float, annual_rate: float, term_years: int) -> float:
    """Standard amortising loan monthly payment."""
    if principal <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / (term_years * 12)
    r = annual_rate / 12
    n = term_years * 12
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _init_car_state(car: "CarProfile") -> list:
    """
    Initialise state for each car. Multiple cars are assumed to be purchased
    one year apart so their replacement cycles are staggered (reduces lumpy
    cash-flow spikes). Car 0 starts in yr 1, car 1 starts in yr 2, etc.
    """
    cars = []
    for i in range(car.num_cars):
        loan_principal = max(0.0, car.car_price - car.down_payment)
        monthly_pmt = _car_monthly_payment(loan_principal, car.loan_rate, car.loan_term_years)
        cars.append({
            "loan_balance": loan_principal,
            "loan_year": 1,           # which year of the current loan (1-based)
            "purchase_year": 1 - i,   # yr car was last purchased (negative = pre-projection)
            "monthly_payment": monthly_pmt,
        })
    return cars



def _compute_car_costs(
    state: dict,
    year: int,
    inf_factor: float,
    car: "CarProfile | None",
    college: "CollegeProfile | None",
) -> tuple[float, float, float]:
    """
    Returns (annual_car_payment, car_purchase_cost, car_sale_proceeds).

    annual_car_payment: total P&I for all active car loans this year
    car_purchase_cost:  total down payments made this year (already deducted
                        from state['brokerage_balance'] as a side effect)
    car_sale_proceeds:  cash received from selling old cars this year (already
                        added to state['brokerage_balance'] as a side effect)
    """
    if car is None:
        return 0.0, 0.0, 0.0

    annual_payment  = 0.0
    purchase_cost   = 0.0
    sale_proceeds   = 0.0

    for c in state["cars"]:
        # --- Purchase: buy a new car when replacement cycle fires ---
        # Car i was last purchased at c["purchase_year"]; replace every N years.
        years_owned = year - c["purchase_year"]
        if years_owned > 0 and years_owned % car.replace_every_years == 0:
            # Sell / hand down old car
            old_car_proceeds = _handle_old_car(state, car, college, year)
            state["brokerage_balance"] += old_car_proceeds
            sale_proceeds += old_car_proceeds

            # Buy new car: down payment + new loan
            nominal_price = car.car_price * inf_factor
            nominal_down  = car.down_payment * inf_factor
            state["brokerage_balance"] -= nominal_down
            purchase_cost += nominal_down

            loan_principal = max(0.0, nominal_price - nominal_down)
            monthly_pmt = _car_monthly_payment(loan_principal, car.loan_rate, car.loan_term_years)
            c["loan_balance"]   = loan_principal
            c["loan_year"]      = 1
            c["purchase_year"]  = year
            c["monthly_payment"] = monthly_pmt

        # --- Annual loan payment ---
        if c["loan_balance"] > 0 and c["loan_year"] <= car.loan_term_years:
            annual_pmt = c["monthly_payment"] * 12
            # Last year: cap at remaining balance to avoid over-paying
            annual_pmt = min(annual_pmt, c["loan_balance"] * (1 + car.loan_rate / 12) * 12)
            annual_payment += annual_pmt
            # Advance loan: approximate annual principal paydown
            r = car.loan_rate / 12
            n_paid = (c["loan_year"] - 1) * 12
            n_total = car.loan_term_years * 12
            if r > 0:
                remaining = c["monthly_payment"] * (1 - (1 + r) ** -(n_total - n_paid)) / r
            else:
                remaining = c["loan_balance"] - c["monthly_payment"] * 12
            c["loan_balance"] = max(0.0, remaining)
            c["loan_year"] += 1

    return annual_payment, purchase_cost, sale_proceeds


def _handle_old_car(state: dict, car: "CarProfile", college: "CollegeProfile | None", year: int) -> float:
    """
    When replacing a car, check if any child is old enough to receive it.
    If yes: hand down (0 proceeds). If no: sell for residual_value.
    """
    if not state.get("child_birth_years"):
        return car.residual_value  # no children → sell

    for birth_year in state["child_birth_years"]:
        child_age = year - birth_year
        if child_age >= car.hand_down_age:
            return 0.0  # hand down — no cash proceeds

    return car.residual_value  # all children too young → sell


class ProjectionEngine:
    def __init__(self, plan: FinancialPlan):
        self._plan = plan
        self._tax_engine = TaxEngine()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_deterministic(self) -> list[YearlySnapshot]:
        snapshots: list[YearlySnapshot] = []
        state = self._initial_state()
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
        np_rng = np.random.default_rng(seed)
        inv = self._plan.investments
        years = list(range(1, self._plan.projection_years + 1))
        all_net_worths: list[list[float]] = []

        for _ in range(n_simulations):
            market_returns = np_rng.normal(inv.annual_market_return, 0.15, len(years))
            inflations = np.clip(np_rng.normal(inv.annual_inflation_rate, 0.015, len(years)), 0, 0.15)
            salary_growths = np.clip(
                np_rng.normal(inv.annual_salary_growth_rate, 0.02, len(years)), -0.10, 0.20,
            )
            state = self._initial_state()
            sim_nw: list[float] = []
            for i, year in enumerate(years):
                self._apply_timeline_events(state, year)
                snap = self._compute_year(
                    state, year,
                    market_return_override=float(market_returns[i]),
                    inflation_override=float(inflations[i]),
                    salary_growth_override=float(salary_growths[i]),
                )
                sim_nw.append(snap.net_worth)
                self._advance_state(state, snap,
                                    market_return=float(market_returns[i]),
                                    inflation=float(inflations[i]),
                                    salary_growth=float(salary_growths[i]))
            all_net_worths.append(sim_nw)

        by_year = list(zip(*all_net_worths))

        def pct(arr, p): return float(np.percentile(arr, p))
        year10 = list(by_year[9]) if len(by_year) >= 10 else []

        return MonteCarloResult(
            years=years,
            p10_net_worth=[pct(yr, 10) for yr in by_year],
            p25_net_worth=[pct(yr, 25) for yr in by_year],
            p50_net_worth=[pct(yr, 50) for yr in by_year],
            p75_net_worth=[pct(yr, 75) for yr in by_year],
            p90_net_worth=[pct(yr, 90) for yr in by_year],
            mean_net_worth=[float(np.mean(yr)) for yr in by_year],
            prob_millionaire_10yr=(
                sum(1 for nw in year10 if nw >= 1_000_000) / n_simulations if year10 else 0.0
            ),
            num_simulations=n_simulations,
        )

    def compute_retirement_readiness(
        self, snapshots: Optional[list[YearlySnapshot]] = None
    ) -> Optional[RetirementReadiness]:
        """
        Compute retirement readiness given the deterministic projection.
        Returns None if no RetirementProfile is configured.
        """
        rp = self._plan.retirement
        if rp is None:
            return None

        if snapshots is None:
            snapshots = self.run_deterministic()

        inv = self._plan.investments
        years_to_ret = rp.years_to_retirement
        inflation = inv.annual_inflation_rate

        # Find the snapshot at retirement year (or use last if beyond horizon)
        retirement_snap = next(
            (s for s in snapshots if s.year == years_to_ret), None
        )
        if retirement_snap is None:
            retirement_snap = snapshots[-1]
        retirement_year = retirement_snap.year

        projected_balance = (
            retirement_snap.retirement_balance
            + retirement_snap.hsa_balance
            + retirement_snap.brokerage_balance
        )

        # Desired income inflated to nominal retirement-year dollars
        nominal_income = rp.desired_annual_income * (1 + inflation) ** years_to_ret

        # Social Security offset (also inflate to retirement dollars)
        ss_nominal = rp.estimated_social_security_annual * (1 + inflation) ** years_to_ret
        net_income_needed = max(0.0, nominal_income - ss_nominal)

        # Required balance: present value of an annuity at post-retirement return
        r = rp.expected_post_retirement_return
        n = rp.years_in_retirement
        if r == 0:
            required = net_income_needed * n
        else:
            # PV of annuity: PMT * (1 - (1+r)^-n) / r
            required = net_income_needed * (1 - (1 + r) ** -n) / r

        funded_pct = projected_balance / required if required > 0 else float("inf")
        on_track = projected_balance >= required

        # Annual surplus or gap: how much extra (or short) per year
        # Convert balance surplus/gap to annual equivalent over retirement horizon
        balance_gap = projected_balance - required
        if r == 0:
            annual_equiv = balance_gap / n if n > 0 else 0
        else:
            # Annuity payment that the gap/surplus would fund
            annual_equiv = balance_gap * r / (1 - (1 + r) ** -n)

        return RetirementReadiness(
            years_to_retirement=years_to_ret,
            retirement_year=retirement_year,
            projected_balance_at_retirement=projected_balance,
            required_balance=required,
            on_track=on_track,
            funded_pct=funded_pct,
            annual_surplus_or_gap=annual_equiv,
            desired_income_nominal=nominal_income,
            social_security_offset=ss_nominal,
        )

    # ------------------------------------------------------------------ #
    # State initialisation                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_amort_lookup(mc: MortgageCalculator) -> dict[int, float]:
        return {row.year: row.balance for row in mc.full_schedule() if row.month % 12 == 0}

    def _initial_state(self) -> dict:
        p = self._plan
        is_married = p.income.filing_status == FilingStatus.MARRIED_FILING_JOINTLY

        mortgage_calc = None
        amort_lookup: dict[int, float] = {}
        if not p.housing.is_renting and p.housing.loan_amount > 0:
            mortgage_calc = MortgageCalculator(p.housing, p.investments.annual_home_appreciation_rate)
            amort_lookup = self._build_amort_lookup(mortgage_calc)

        # Child birth years: pre-populate from lifestyle.num_children (all treated as born year 0)
        child_birth_years: list[int] = [0] * p.lifestyle.num_children

        return {
            "income_primary": p.income.gross_annual_income,
            "income_partner": p.income.spouse_gross_annual_income,
            "gross_income": p.income.total_gross_income,
            "filing_status": p.income.filing_status,
            "is_married": is_married,
            "num_children": p.lifestyle.num_children,
            "num_pets": p.lifestyle.num_pets,
            "is_working": True,
            "is_partner_working": p.income.spouse_gross_annual_income > 0,
            "is_renting": p.housing.is_renting,
            "monthly_rent": p.housing.monthly_rent,
            "mortgage_calc": mortgage_calc,
            "amort_lookup": amort_lookup,
            "mortgage_year_offset": 0,
            "mortgage_interest_rate": p.housing.interest_rate,
            "home_price_ref": p.housing.home_price,
            "retirement_balance": p.investments.current_retirement_balance,
            "brokerage_balance": (
                p.investments.current_liquid_cash
                - p.investments.one_time_upcoming_expenses
                - (p.housing.down_payment if not p.housing.is_renting else 0.0)
                # Initial car down payments: assumed paid from current liquid assets
                - (p.car.down_payment * p.car.num_cars if p.car else 0.0)
                + p.investments.current_brokerage_balance
            ),
            "hsa_balance": 0.0,
            "college_529_balance": 0.0,
            "home_value": p.housing.home_price if not p.housing.is_renting else 0.0,
            "mortgage_balance": p.housing.loan_amount if not p.housing.is_renting else 0.0,
            "parent_care_active": p.lifestyle.annual_parent_care_cost > 0,
            "child_birth_years": child_birth_years,   # list[int]: projection year each child was born
            "child_529_years_used": {},                # {child_idx: years_of_college_used}
            "uninvested_cash": 0.0,                    # cash not invested (auto_invest_surplus=False)
            # Car state: one entry per car, each a dict with loan_balance and loan_year
            "cars": _init_car_state(p.car) if p.car else [],
        }

    # ------------------------------------------------------------------ #
    # Timeline event application                                          #
    # ------------------------------------------------------------------ #

    def _apply_timeline_events(self, state: dict, year: int) -> None:
        p = self._plan
        for event in p.events_for_year(year):

            if event.marriage:
                state["filing_status"] = FilingStatus.MARRIED_FILING_JOINTLY
                state["is_married"] = True

            if event.new_child:
                birth_year = event.child_birth_year_override if event.child_birth_year_override is not None else year
                state["child_birth_years"].append(birth_year)
                state["num_children"] += 1

            if event.new_pet:
                state["num_pets"] += 1

            # --- Work start/stop ---
            if event.stop_working:
                state["is_working"] = False
                state["income_primary"] = 0.0
                state["gross_income"] = state["income_primary"] + state["income_partner"]
            if event.resume_working:
                state["is_working"] = True
                # income_change must accompany resume_working to set the new salary
            if event.partner_stop_working:
                state["is_partner_working"] = False
                state["income_partner"] = 0.0
                state["gross_income"] = state["income_primary"] + state["income_partner"]
            if event.partner_resume_working:
                state["is_partner_working"] = True

            # --- Income changes ---
            if event.income_change is not None:
                state["income_primary"] = event.income_change
                if event.resume_working:
                    state["is_working"] = True
                state["gross_income"] = state["income_primary"] + state["income_partner"]
            if event.partner_income_change is not None:
                state["income_partner"] = event.partner_income_change
                if event.partner_resume_working:
                    state["is_partner_working"] = True
                state["gross_income"] = state["income_primary"] + state["income_partner"]

            # --- Parent care ---
            if event.start_parent_care:
                state["parent_care_active"] = True
            if event.stop_parent_care:
                state["parent_care_active"] = False

            # --- One-time cash flows ---
            if event.extra_one_time_income:
                state["brokerage_balance"] += event.extra_one_time_income
            if event.extra_one_time_expense:
                state["brokerage_balance"] -= event.extra_one_time_expense

            # --- Home purchase ---
            if event.buy_home:
                new_price = event.new_home_price or event.home_price_override or state["home_value"]
                new_down = event.new_home_down_payment or new_price * 0.20
                new_rate = event.new_home_interest_rate or state["mortgage_interest_rate"]

                if event.sell_current_home and not state["is_renting"]:
                    equity = max(0.0, state["home_value"] - state["mortgage_balance"])
                    seller_closing = state["home_value"] * event.seller_closing_cost_rate
                    state["brokerage_balance"] += max(0.0, equity - seller_closing)

                state["brokerage_balance"] -= new_down + new_price * event.buyer_closing_cost_rate

                new_hp = HousingProfile(
                    home_price=new_price, down_payment=new_down, interest_rate=new_rate,
                    loan_term_years=p.housing.loan_term_years,
                    annual_property_tax_rate=p.housing.annual_property_tax_rate,
                    annual_insurance=p.housing.annual_insurance,
                    annual_maintenance_rate=p.housing.annual_maintenance_rate,
                    pmi_annual_rate=p.housing.pmi_annual_rate,
                )
                new_calc = MortgageCalculator(new_hp, p.investments.annual_home_appreciation_rate)
                state["mortgage_calc"] = new_calc
                state["amort_lookup"] = self._build_amort_lookup(new_calc)
                state["mortgage_year_offset"] = event.year - 1
                state["mortgage_interest_rate"] = new_rate
                state["home_price_ref"] = new_price
                state["home_value"] = new_price
                state["mortgage_balance"] = new_hp.loan_amount
                state["is_renting"] = False

    # ------------------------------------------------------------------ #
    # Compute one year                                                     #
    # ------------------------------------------------------------------ #

    def _compute_year(
        self,
        state: dict,
        year: int,
        market_return_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
        salary_growth_override: Optional[float] = None,
    ) -> YearlySnapshot:
        p = self._plan
        inv = p.investments
        strat = p.strategies

        market_return = market_return_override if market_return_override is not None else inv.annual_market_return
        inflation = inflation_override if inflation_override is not None else inv.annual_inflation_rate
        inf_factor = (1 + inflation) ** (year - 1)

        gross_income = state["gross_income"]
        is_married = state["is_married"]
        num_children = state["num_children"]

        # --- Contributions ---
        is_family_hsa = is_married or num_children > 0
        hsa_irs_limit = _HSA_LIMIT_FAMILY if is_family_hsa else _HSA_LIMIT_SINGLE
        hsa_cont   = min(inv.annual_hsa_contribution, hsa_irs_limit) if strat.maximize_hsa else 0.0
        k401_cont  = min(inv.annual_401k_contribution, 30_500)
        partner_k401 = (
            min(inv.partner_annual_401k_contribution, 30_500)
            if state["income_partner"] > 0 else 0.0
        )
        # 529 contributions: stop once all children have graduated college.
        # Only applies when a CollegeProfile is configured — without one,
        # contributions continue for as long as annual_529_contribution > 0.
        if p.college and state["child_birth_years"]:
            all_done = all(
                (year - by) >= p.college.start_age + p.college.years_per_child
                for by in state["child_birth_years"]
            )
            r529_cont = 0.0 if all_done else inv.annual_529_contribution
        else:
            # No college profile: always contribute the stated amount
            r529_cont = inv.annual_529_contribution

        # --- Tax ---
        tmp_income = IncomeProfile(
            gross_annual_income=gross_income,
            filing_status=state["filing_status"],
            state=p.income.state,
            other_state_flat_rate=p.income.other_state_flat_rate,
        )
        tmp_inv = InvestmentProfile(
            annual_hsa_contribution=hsa_cont,
            annual_401k_contribution=k401_cont,
            annual_529_contribution=r529_cont,
        )
        tax_result = self._tax_engine.calculate(
            tmp_income, tmp_inv, strat, num_children=num_children
        )

        # --- AOTC credit (reduces effective tax) ---
        aotc_credit = self._compute_aotc(state, year, gross_income, is_married, inf_factor, p.college)
        effective_tax = max(0.0, tax_result.total_annual_tax - aotc_credit)

        net_income = gross_income - effective_tax - hsa_cont - k401_cont - partner_k401

        # --- Housing ---
        mortgage_calc: Optional[MortgageCalculator] = state.get("mortgage_calc")
        end_of_year_balance = 0.0
        if state["is_renting"]:
            annual_housing = state["monthly_rent"] * 12 * inf_factor
            home_equity = 0.0
            home_value = 0.0
        elif mortgage_calc:
            monthly_pi = mortgage_calc.monthly_pi_payment()
            ref_price = state["home_price_ref"]
            monthly_other = (
                ref_price * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
                + p.housing.annual_insurance
            ) / 12 * inf_factor
            pmi = (
                mortgage_calc._pmi_payment(state["mortgage_balance"])
                if state["mortgage_balance"] / ref_price > 0.80 and mortgage_calc._p.requires_pmi
                else 0.0
            )
            annual_housing = (monthly_pi + monthly_other + pmi) * 12
            home_value = state["home_value"]
            amort_lookup = state.get("amort_lookup", {})
            offset = state.get("mortgage_year_offset", 0)
            mortgage_yr = year - offset
            end_of_year_balance = amort_lookup.get(mortgage_yr, state["mortgage_balance"])
            home_equity = max(0.0, home_value - end_of_year_balance)
        else:
            ref_price = state["home_price_ref"]
            annual_housing = (
                ref_price * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
                + p.housing.annual_insurance
            ) * inf_factor
            home_value = state["home_value"]
            home_equity = home_value

        # --- Lifestyle ---
        medical_oop   = p.lifestyle.scaled_medical_oop(is_married, num_children) * inf_factor
        pets_cost     = state["num_pets"] * p.lifestyle.annual_pet_cost * inf_factor
        kids_cost     = num_children * p.lifestyle.monthly_childcare * 12 * inf_factor
        vacation      = p.lifestyle.annual_vacation * inf_factor
        other         = p.lifestyle.monthly_other_recurring * 12 * inf_factor
        parent_care   = (
            p.lifestyle.annual_parent_care_cost * inf_factor
            if state["parent_care_active"] else 0.0
        )
        annual_lifestyle = medical_oop + pets_cost + kids_cost + vacation + other + parent_care

        # --- 529 savings contributions (after-tax) ---
        annual_529_save = r529_cont * num_children

        # --- Wedding fund: earmarked savings per child (goes to brokerage earmark) ---
        # Wedding fund: saved per child from LifestyleProfile — separate from college.
        # Deducted from breathing room and held in brokerage.
        # Stops when each child turns 25 (reasonable upper bound for wedding planning).
        annual_wedding_save = 0.0
        if p.lifestyle.annual_wedding_fund_per_child > 0:
            for by in state["child_birth_years"]:
                child_age = year - by
                if child_age <= 25:
                    annual_wedding_save += p.lifestyle.annual_wedding_fund_per_child

        # --- College costs: 529 drawdown + brokerage fallback ---
        # Pass available_529 = current balance + this year's contributions so that
        # contributions made this year are available for same-year college expenses.
        available_529_this_year = state["college_529_balance"] + annual_529_save
        college_gross, drawdown_529, net_college_brokerage = self._compute_college_costs(
            state, year, inf_factor, p.college, available_529=available_529_this_year
        )

        # --- Car purchases and loan payments ---
        annual_car_payment, car_purchase_cost, car_sale_proceeds = _compute_car_costs(
            state, year, inf_factor, p.car, p.college
        )

        # --- Earmarked brokerage investment (annual_brokerage_contribution) ---
        brokerage_earmark = inv.annual_brokerage_contribution

        # Breathing room: what's left after all expenses and earmarked investments
        breathing_room = (
            net_income
            - annual_housing
            - annual_lifestyle
            - annual_529_save
            - net_college_brokerage   # college costs not covered by 529
            - brokerage_earmark       # earmarked brokerage contribution
            - annual_car_payment      # car loan P&I
            - annual_wedding_save     # wedding fund savings (to brokerage)
        )

        # --- Car purchase/sale: direct brokerage hit (down payment already removed in _compute_car_costs) ---
        # car_purchase_cost and car_sale_proceeds are already applied to brokerage
        # inside _compute_car_costs via state mutation, so they flow through naturally.

        # --- Asset growth ---
        retirement_bal   = state["retirement_balance"] * (1 + market_return) + k401_cont + partner_k401
        hsa_bal          = state["hsa_balance"] * (1 + market_return) + hsa_cont
        # 529 grows at glide-path rate (aggressive early, conservative later),
        # NOT at the general market_return. This reflects standard age-based
        # 529 allocation: equity-heavy early, shift to bonds as college approaches.
        if p.college:
            r529_growth = (
                p.college.early_529_return if year <= p.college.glide_path_years
                else p.college.late_529_return
            )
        else:
            r529_growth = market_return
        col529_bal = state["college_529_balance"] * (1 + r529_growth) + annual_529_save - drawdown_529
        col529_bal = max(0.0, col529_bal)

        # Auto-invest surplus toggle:
        #   ON  (default): all surplus breathing_room is swept into brokerage → earns market return
        #   OFF: surplus sits in uninvested cash (0% return) — shows cost of not investing
        if strat.auto_invest_surplus:
            # Current (and default) behaviour: everything invested
            brokerage_bal    = (
                state["brokerage_balance"] * (1 + market_return)
                + brokerage_earmark
                + breathing_room
            )
            uninvested_cash  = 0.0
        else:
            # Earmark still goes to brokerage; organic surplus stays in cash (0% return)
            brokerage_bal    = (
                state["brokerage_balance"] * (1 + market_return)
                + brokerage_earmark
            )
            # Positive surplus accumulates as uninvested cash; deficits drain brokerage
            if breathing_room >= 0:
                uninvested_cash = state["uninvested_cash"] + breathing_room
            else:
                # Deficit: drain uninvested cash first, then brokerage
                remaining_deficit = breathing_room  # negative
                if state["uninvested_cash"] > 0:
                    drawn = min(state["uninvested_cash"], -remaining_deficit)
                    uninvested_cash  = state["uninvested_cash"] - drawn
                    remaining_deficit += drawn
                else:
                    uninvested_cash = 0.0
                brokerage_bal += remaining_deficit  # remaining deficit drains brokerage

        net_worth = retirement_bal + hsa_bal + col529_bal + brokerage_bal + home_equity + uninvested_cash

        return YearlySnapshot(
            year=year,
            gross_income=gross_income,
            net_income=net_income,
            annual_tax_total=effective_tax,
            annual_housing_cost=annual_housing,
            annual_lifestyle_cost=annual_lifestyle,
            annual_medical_oop=medical_oop,
            annual_college_cost=college_gross,
            annual_529_drawdown=drawdown_529,
            annual_parent_care_cost=parent_care,
            annual_retirement_contributions=k401_cont + partner_k401,
            annual_hsa_contributions=hsa_cont,
            annual_brokerage_contribution=brokerage_earmark,
            annual_aotc_credit=aotc_credit,
            annual_breathing_room=breathing_room,
            uninvested_cash=uninvested_cash,
            annual_car_payment=annual_car_payment,
            car_purchase_cost=car_purchase_cost,
            car_sale_proceeds=car_sale_proceeds,
            annual_wedding_save=annual_wedding_save,
            retirement_balance=retirement_bal,
            brokerage_balance=brokerage_bal,
            college_529_balance=col529_bal,
            home_value=home_value,
            home_equity=home_equity,
            hsa_balance=hsa_bal,
            mortgage_balance=end_of_year_balance if not state["is_renting"] else 0.0,
            net_worth=net_worth,
            filing_status=state["filing_status"],
            num_children=num_children,
            is_renting=state["is_renting"],
            is_married=is_married,
            is_working=state["is_working"],
            is_partner_working=state["is_partner_working"],
        )

    # ------------------------------------------------------------------ #
    # College cost helpers                                                 #
    # ------------------------------------------------------------------ #

    def _compute_college_costs(
        self,
        state: dict,
        year: int,
        inf_factor: float,
        college: Optional[CollegeProfile],
        available_529: float = 0.0,
    ) -> tuple[float, float, float]:
        """
        Returns (gross_college_cost, 529_drawdown, net_cost_from_brokerage).

        available_529: the 529 balance available for drawdown this year,
        including same-year contributions (state_balance + annual_529_save).
        Does NOT mutate state — caller applies drawdown in the balance formula.
        """
        if college is None or not state["child_birth_years"]:
            return 0.0, 0.0, 0.0

        gross_cost = 0.0
        drawdown_529 = 0.0
        remaining_529 = available_529  # local — do not mutate state

        for birth_year in state["child_birth_years"]:
            child_age = year - birth_year
            in_college = college.start_age <= child_age < college.start_age + college.years_per_child
            if not in_college:
                continue

            year_cost = college.annual_cost_per_child * inf_factor
            gross_cost += year_cost

            # Draw from 529 first (tax-free); remainder comes from brokerage via breathing_room
            this_drawdown = min(remaining_529, year_cost)
            drawdown_529 += this_drawdown
            remaining_529 -= this_drawdown  # reduce local counter only

        net_from_brokerage = max(0.0, gross_cost - drawdown_529)
        return gross_cost, drawdown_529, net_from_brokerage

    def _compute_aotc(
        self,
        state: dict,
        year: int,
        gross_income: float,
        is_married: bool,
        inf_factor: float,
        college: Optional[CollegeProfile],
    ) -> float:
        """
        Compute American Opportunity Tax Credit.
        Up to $2,500 per eligible student per year, first 4 years of college.
        Income phase-out applies.
        """
        if college is None or not college.use_aotc_credit:
            return 0.0

        # Phase-out thresholds (not inflated — IRS rarely adjusts AOTC thresholds)
        low  = _AOTC_PHASEOUT_MFJ_LOW  if is_married else _AOTC_PHASEOUT_SINGLE_LOW
        high = _AOTC_PHASEOUT_MFJ_HIGH if is_married else _AOTC_PHASEOUT_SINGLE_HIGH

        if gross_income >= high:
            return 0.0

        phase_out_pct = max(0.0, 1.0 - (gross_income - low) / (high - low)) if gross_income > low else 1.0

        eligible_students = 0
        for i, birth_year in enumerate(state["child_birth_years"]):
            child_age = year - birth_year
            in_college = college.start_age <= child_age < college.start_age + college.years_per_child
            # AOTC: only for first 4 years; years_per_child caps at 4 for credit purposes
            aotc_eligible_years = min(college.years_per_child, 4)
            aotc_year = child_age - college.start_age + 1  # 1-based year of college
            if in_college and aotc_year <= aotc_eligible_years:
                eligible_students += 1

        return eligible_students * _AOTC_MAX_CREDIT * phase_out_pct

    # ------------------------------------------------------------------ #
    # Advance state                                                        #
    # ------------------------------------------------------------------ #

    def _advance_state(
        self,
        state: dict,
        snap: YearlySnapshot,
        market_return: Optional[float] = None,
        inflation: Optional[float] = None,
        salary_growth: Optional[float] = None,
    ) -> None:
        p = self._plan
        sg   = salary_growth if salary_growth is not None else p.investments.annual_salary_growth_rate
        p_sg = salary_growth if salary_growth is not None else p.investments.partner_salary_growth_rate

        # Only grow income if currently working
        if state["is_working"]:
            state["income_primary"] *= (1 + sg)
        if state["is_partner_working"]:
            state["income_partner"] *= (1 + p_sg)
        state["gross_income"] = state["income_primary"] + state["income_partner"]

        state["retirement_balance"]  = snap.retirement_balance
        state["brokerage_balance"]   = snap.brokerage_balance
        state["hsa_balance"]         = snap.hsa_balance
        state["college_529_balance"] = snap.college_529_balance
        state["uninvested_cash"]     = snap.uninvested_cash

        if not state["is_renting"]:
            state["home_value"] = snap.home_value * (1 + p.investments.annual_home_appreciation_rate)
            mc: Optional[MortgageCalculator] = state.get("mortgage_calc")
            if mc and state["mortgage_balance"] > 0:
                amort = state.get("amort_lookup", {})
                offset = state.get("mortgage_year_offset", 0)
                myr = snap.year - offset
                if myr in amort:
                    state["mortgage_balance"] = amort[myr]
                elif amort and myr > max(amort):
                    state["mortgage_balance"] = 0.0
                else:
                    rate = state["mortgage_interest_rate"]
                    ann_int = state["mortgage_balance"] * rate
                    ann_pi = mc.monthly_pi_payment() * 12
                    state["mortgage_balance"] = max(0.0, state["mortgage_balance"] - max(0.0, ann_pi - ann_int))