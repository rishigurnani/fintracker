"""
Long-term projection engine.

Two modes:
  1. Deterministic: Fixed economic assumptions, year-by-year snapshot.
  2. Monte Carlo:   Randomized returns + inflation, N simulations,
                    producing percentile bands (10th/50th/90th).

Key behaviours:
  - Home purchase mid-projection: selling current home captures equity as cash;
    a fresh MortgageCalculator is spun up for the new home.
  - Healthcare auto-scaling: medical OOP grows when marriage or children occur.
  - HSA limit tier: single ($4,150) automatically upgrades to family ($8,300)
    in the year of marriage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from fintracker.models import (
    FilingStatus, FinancialPlan, HousingProfile, IncomeProfile,
    InvestmentProfile, StrategyToggles, TimelineEvent,
)
from fintracker.tax_engine import TaxEngine
from fintracker.mortgage import MortgageCalculator

# 2024 IRS HSA limits
_HSA_LIMIT_SINGLE = 4_150
_HSA_LIMIT_FAMILY = 8_300


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
    annual_medical_oop: float       # broken out so UI can show scaling clearly
    annual_retirement_contributions: float
    annual_hsa_contributions: float

    # Cash flow
    annual_breathing_room: float

    # Assets
    retirement_balance: float
    brokerage_balance: float
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

    @property
    def total_assets(self) -> float:
        return (
            self.retirement_balance
            + self.brokerage_balance
            + self.home_equity
            + self.hsa_balance
        )

    @property
    def liquid_assets(self) -> float:
        return self.brokerage_balance + self.hsa_balance


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


class ProjectionEngine:
    """
    Runs deterministic and Monte Carlo projections.

    Usage::

        engine = ProjectionEngine(plan)
        snapshots = engine.run_deterministic()
        mc_result = engine.run_monte_carlo(n_simulations=1000)
    """

    def __init__(self, plan: FinancialPlan):
        self._plan = plan
        self._tax_engine = TaxEngine()

    # ------------------------------------------------------------------
    # Deterministic projection
    # ------------------------------------------------------------------

    def run_deterministic(self) -> list[YearlySnapshot]:
        snapshots: list[YearlySnapshot] = []
        state = self._initial_state()

        for year in range(1, self._plan.projection_years + 1):
            self._apply_timeline_events(state, year)
            snap = self._compute_year(state, year)
            snapshots.append(snap)
            self._advance_state(state, snap)

        return snapshots

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        n_simulations: int = 1_000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        np_rng = np.random.default_rng(seed)

        market_mean = self._plan.investments.annual_market_return
        inflation_mean = self._plan.investments.annual_inflation_rate
        years = list(range(1, self._plan.projection_years + 1))
        all_net_worths: list[list[float]] = []

        for _ in range(n_simulations):
            market_returns = np_rng.normal(market_mean, 0.15, len(years))
            inflations = np.clip(np_rng.normal(inflation_mean, 0.015, len(years)), 0, 0.15)
            salary_growths = np.clip(
                np_rng.normal(self._plan.investments.annual_salary_growth_rate, 0.02, len(years)),
                -0.10, 0.20,
            )

            state = self._initial_state()
            sim_net_worths: list[float] = []

            for i, year in enumerate(years):
                self._apply_timeline_events(state, year)
                snap = self._compute_year(
                    state, year,
                    market_return_override=float(market_returns[i]),
                    inflation_override=float(inflations[i]),
                    salary_growth_override=float(salary_growths[i]),
                )
                sim_net_worths.append(snap.net_worth)
                self._advance_state(state, snap,
                                    market_return=float(market_returns[i]),
                                    inflation=float(inflations[i]),
                                    salary_growth=float(salary_growths[i]))

            all_net_worths.append(sim_net_worths)

        by_year = list(zip(*all_net_worths))

        def pct(arr, p):
            return float(np.percentile(arr, p))

        year10_worths = list(by_year[9]) if len(by_year) >= 10 else []

        return MonteCarloResult(
            years=years,
            p10_net_worth=[pct(yr, 10) for yr in by_year],
            p25_net_worth=[pct(yr, 25) for yr in by_year],
            p50_net_worth=[pct(yr, 50) for yr in by_year],
            p75_net_worth=[pct(yr, 75) for yr in by_year],
            p90_net_worth=[pct(yr, 90) for yr in by_year],
            mean_net_worth=[float(np.mean(yr)) for yr in by_year],
            prob_millionaire_10yr=(
                sum(1 for nw in year10_worths if nw >= 1_000_000) / n_simulations
                if year10_worths else 0.0
            ),
            num_simulations=n_simulations,
        )

    # ------------------------------------------------------------------
    # Internal: state initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_amort_lookup(mortgage_calc: MortgageCalculator) -> dict[int, float]:
        """Return {year: exact_end_of_year_balance} from the full amortization schedule."""
        lookup: dict[int, float] = {}
        for row in mortgage_calc.full_schedule():
            if row.month % 12 == 0:
                lookup[row.year] = row.balance
        return lookup

    def _initial_state(self) -> dict:
        p = self._plan
        is_married = p.income.filing_status == FilingStatus.MARRIED_FILING_JOINTLY

        # Build initial mortgage calculator if applicable
        mortgage_calc = None
        amort_lookup: dict[int, float] = {}
        mortgage_year_offset = 0  # which mortgage year == projection year 1
        if not p.housing.is_renting and p.housing.loan_amount > 0:
            mortgage_calc = MortgageCalculator(
                p.housing, p.investments.annual_home_appreciation_rate
            )
            amort_lookup = self._build_amort_lookup(mortgage_calc)

        return {
            "gross_income": p.income.total_gross_income,
            "filing_status": p.income.filing_status,
            "is_married": is_married,
            "num_children": p.lifestyle.num_children,
            "num_pets": p.lifestyle.num_pets,
            "is_renting": p.housing.is_renting,
            "monthly_rent": p.housing.monthly_rent,
            # live mortgage calculator — replaced when a buy_home event fires
            "mortgage_calc": mortgage_calc,
            "amort_lookup": amort_lookup,       # {mortgage_year: exact_balance}
            "mortgage_year_offset": mortgage_year_offset,  # projection_year - mortgage_year
            "mortgage_interest_rate": p.housing.interest_rate,
            "home_price_ref": p.housing.home_price,  # for tax/insurance calc
            "retirement_balance": p.investments.current_retirement_balance,
            "brokerage_balance": (
                p.investments.current_liquid_cash
                - p.investments.one_time_upcoming_expenses
                - (p.housing.down_payment if not p.housing.is_renting else 0.0)
                + p.investments.current_brokerage_balance
            ),
            "hsa_balance": 0.0,
            "home_value": p.housing.home_price if not p.housing.is_renting else 0.0,
            "mortgage_balance": p.housing.loan_amount if not p.housing.is_renting else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal: apply timeline events
    # ------------------------------------------------------------------

    def _apply_timeline_events(self, state: dict, year: int) -> None:
        p = self._plan
        for event in p.events_for_year(year):

            # --- Marriage ---
            if event.marriage:
                state["filing_status"] = FilingStatus.MARRIED_FILING_JOINTLY
                state["is_married"] = True

            # --- New child / pet ---
            if event.new_child:
                state["num_children"] += 1
            if event.new_pet:
                state["num_pets"] += 1

            # --- Income change ---
            if event.income_change is not None:
                state["gross_income"] = event.income_change

            # --- One-time cash flows ---
            if event.extra_one_time_income:
                state["brokerage_balance"] += event.extra_one_time_income
            if event.extra_one_time_expense:
                state["brokerage_balance"] -= event.extra_one_time_expense

            # --- Home purchase ---
            if event.buy_home:
                # Resolve new home details (fall back to home_price_override for
                # back-compat with old YAML that used that field name)
                new_price = (
                    event.new_home_price
                    or event.home_price_override
                    or state["home_value"]  # fallback: same price
                )
                new_down = event.new_home_down_payment
                new_rate = event.new_home_interest_rate or state["mortgage_interest_rate"]

                # If down payment not specified, default to 20% of new price
                if new_down is None:
                    new_down = new_price * 0.20

                # Sell current home: equity goes into brokerage
                if event.sell_current_home and not state["is_renting"]:
                    current_equity = max(0.0, state["home_value"] - state["mortgage_balance"])
                    # Typical closing costs on sale ≈ 6% of sale price
                    sale_proceeds = current_equity - (state["home_value"] * 0.06)
                    state["brokerage_balance"] += max(0.0, sale_proceeds)

                # Pay down payment from brokerage
                state["brokerage_balance"] -= new_down

                # Build fresh mortgage calculator for the new home
                new_housing_profile = HousingProfile(
                    home_price=new_price,
                    down_payment=new_down,
                    interest_rate=new_rate,
                    loan_term_years=p.housing.loan_term_years,
                    annual_property_tax_rate=p.housing.annual_property_tax_rate,
                    annual_insurance=p.housing.annual_insurance,
                    annual_maintenance_rate=p.housing.annual_maintenance_rate,
                    pmi_annual_rate=p.housing.pmi_annual_rate,
                )
                new_calc = MortgageCalculator(
                    new_housing_profile,
                    p.investments.annual_home_appreciation_rate,
                )
                state["mortgage_calc"] = new_calc
                state["amort_lookup"] = self._build_amort_lookup(new_calc)
                state["mortgage_year_offset"] = event.year - 1  # mortgage yr1 = projection yr event.year
                state["mortgage_interest_rate"] = new_rate
                state["home_price_ref"] = new_price
                state["home_value"] = new_price
                state["mortgage_balance"] = new_housing_profile.loan_amount
                state["is_renting"] = False

    # ------------------------------------------------------------------
    # Internal: compute one year
    # ------------------------------------------------------------------

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

        gross_income = state["gross_income"]
        is_married = state["is_married"]
        num_children = state["num_children"]

        # --- HSA limit: single vs family tier ---
        # Family tier applies once married OR once there's a dependent child
        is_family_hsa = is_married or num_children > 0
        hsa_irs_limit = _HSA_LIMIT_FAMILY if is_family_hsa else _HSA_LIMIT_SINGLE

        # Inflate contributions, capped at IRS limits
        inf_factor = (1 + inflation) ** (year - 1)
        # When maximize_hsa=True: contribute the full IRS limit (that's what 'maximize' means).
        # When False: use the user's stated contribution, capped at IRS limit as a safety check.
        if strat.maximize_hsa:
            hsa_cont = hsa_irs_limit
        else:
            hsa_cont = min(inv.annual_hsa_contribution * inf_factor, hsa_irs_limit)
        if strat.maximize_401k:
            k401_cont = 30_500  # full IRS limit (23k employee + catch-up if 50+)
        else:
            k401_cont = min(inv.annual_401k_contribution * inf_factor, 30_500)
        r529_cont = inv.annual_529_contribution * inf_factor

        tmp_income = IncomeProfile(
            gross_annual_income=gross_income,
            filing_status=state["filing_status"],
            state=p.income.state,
            other_state_flat_rate=p.income.other_state_flat_rate,
        )
        tmp_inv = InvestmentProfile(
            annual_hsa_contribution=hsa_cont if strat.maximize_hsa else 0.0,
            annual_401k_contribution=k401_cont if strat.maximize_401k else 0.0,
            annual_529_contribution=r529_cont,
        )

        tax_result = self._tax_engine.calculate(
            tmp_income, tmp_inv, strat, num_children=num_children
        )

        net_income = gross_income - tax_result.total_annual_tax - hsa_cont - k401_cont

        # --- Housing ---
        mortgage_calc: Optional[MortgageCalculator] = state.get("mortgage_calc")
        if state["is_renting"]:
            annual_housing = state["monthly_rent"] * 12 * inf_factor
            home_equity = 0.0
            home_value = 0.0
        elif mortgage_calc:
            monthly_pi = mortgage_calc.monthly_pi_payment()
            ref_price = state["home_price_ref"]
            monthly_taxes_ins = (
                ref_price * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
                + p.housing.annual_insurance
            ) / 12 * inf_factor
            pmi = (
                mortgage_calc._pmi_payment(state["mortgage_balance"])
                if (state["mortgage_balance"] / ref_price > 0.80
                    and mortgage_calc._p.requires_pmi)
                else 0.0
            )
            annual_housing = (monthly_pi + monthly_taxes_ins + pmi) * 12
            home_value = state["home_value"]
            # Use end-of-year balance for equity (more accurate than start-of-year)
            amort_lookup = state.get("amort_lookup", {})
            offset = state.get("mortgage_year_offset", 0)
            mortgage_year = year - offset
            end_of_year_balance = amort_lookup.get(mortgage_year, state["mortgage_balance"])
            home_equity = max(0.0, home_value - end_of_year_balance)
        else:
            # Owned outright — no mortgage, just taxes/maintenance
            ref_price = state["home_price_ref"]
            annual_housing = (
                ref_price * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
                + p.housing.annual_insurance
            ) * inf_factor
            home_value = state["home_value"]
            home_equity = home_value

        # --- Lifestyle with healthcare scaling ---
        medical_oop = p.lifestyle.scaled_medical_oop(is_married, num_children) * inf_factor
        pets_cost = state["num_pets"] * p.lifestyle.annual_pet_cost * inf_factor
        kids_cost = num_children * p.lifestyle.monthly_childcare * 12 * inf_factor
        vacation = p.lifestyle.annual_vacation * inf_factor
        other = p.lifestyle.monthly_other_recurring * 12 * inf_factor
        annual_lifestyle = medical_oop + pets_cost + kids_cost + vacation + other

        # 529 contributions (after-tax)
        annual_529 = r529_cont * num_children

        breathing_room = net_income - annual_housing - annual_lifestyle - annual_529

        # --- Asset growth ---
        retirement_bal = state["retirement_balance"] * (1 + market_return) + k401_cont
        hsa_bal = state["hsa_balance"] * (1 + market_return) + hsa_cont
        # Negative breathing_room correctly drains brokerage (deficit spending)
        brokerage_bal = state["brokerage_balance"] * (1 + market_return) + breathing_room

        net_worth = retirement_bal + hsa_bal + brokerage_bal + home_equity

        return YearlySnapshot(
            year=year,
            gross_income=gross_income,
            net_income=net_income,
            annual_tax_total=tax_result.total_annual_tax,
            annual_housing_cost=annual_housing,
            annual_lifestyle_cost=annual_lifestyle,
            annual_medical_oop=medical_oop,
            annual_retirement_contributions=k401_cont,
            annual_hsa_contributions=hsa_cont,
            annual_breathing_room=breathing_room,
            retirement_balance=retirement_bal,
            brokerage_balance=brokerage_bal,
            home_value=home_value,
            home_equity=home_equity,
            hsa_balance=hsa_bal,
            mortgage_balance=end_of_year_balance if not state["is_renting"] else 0.0,
            net_worth=net_worth,
            filing_status=state["filing_status"],
            num_children=num_children,
            is_renting=state["is_renting"],
            is_married=is_married,
        )

    # ------------------------------------------------------------------
    # Internal: advance state to next year
    # ------------------------------------------------------------------

    def _advance_state(
        self,
        state: dict,
        snap: YearlySnapshot,
        market_return: Optional[float] = None,
        inflation: Optional[float] = None,
        salary_growth: Optional[float] = None,
    ) -> None:
        p = self._plan
        sg = salary_growth if salary_growth is not None else p.investments.annual_salary_growth_rate

        state["gross_income"] *= (1 + sg)
        state["retirement_balance"] = snap.retirement_balance
        state["brokerage_balance"] = snap.brokerage_balance
        state["hsa_balance"] = snap.hsa_balance

        if not state["is_renting"]:
            # Appreciate home value
            state["home_value"] = snap.home_value * (1 + p.investments.annual_home_appreciation_rate)

            # Pay down mortgage using exact amortization schedule (avoids ~$20k drift over 30yr)
            mortgage_calc: Optional[MortgageCalculator] = state.get("mortgage_calc")
            if mortgage_calc and state["mortgage_balance"] > 0:
                amort_lookup = state.get("amort_lookup", {})
                offset = state.get("mortgage_year_offset", 0)
                mortgage_year = snap.year - offset  # which year of THIS mortgage we just finished
                if mortgage_year in amort_lookup:
                    state["mortgage_balance"] = amort_lookup[mortgage_year]
                elif amort_lookup and mortgage_year > max(amort_lookup):
                    state["mortgage_balance"] = 0.0  # paid off
                else:
                    # fallback: simplified approximation (e.g. mid-year purchase)
                    rate = state["mortgage_interest_rate"]
                    annual_interest = state["mortgage_balance"] * rate
                    annual_pi = mortgage_calc.monthly_pi_payment() * 12
                    annual_principal = max(0.0, annual_pi - annual_interest)
                    state["mortgage_balance"] = max(0.0, state["mortgage_balance"] - annual_principal)