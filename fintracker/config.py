"""
Configuration loader/saver.

Personal config lives in config/personal.yaml (gitignored).
A sample config is provided in config/sample.yaml (tracked in git).

Usage::

    plan = load_plan("config/personal.yaml")
    save_plan(plan, "config/personal.yaml")
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, TimelineEvent, FinancialPlan,
)

_PERSONAL_CONFIG_NAMES = {"personal.yaml", "personal.yml"}


def load_plan(path: str | Path) -> FinancialPlan:
    """Load a FinancialPlan from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return _dict_to_plan(data)


def save_plan(plan: FinancialPlan, path: str | Path) -> None:
    """Serialize a FinancialPlan to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_plan_to_dict(plan), f, default_flow_style=False, sort_keys=False)


def load_plan_or_sample(path: str | Path = "config/personal.yaml") -> FinancialPlan:
    """Try to load personal config; fall back to sample if not found."""
    try:
        return load_plan(path)
    except FileNotFoundError:
        sample = Path(__file__).parent.parent / "config" / "sample.yaml"
        if sample.exists():
            return load_plan(sample)
        return _default_plan()


def write_sample_config(output_path: str | Path = "config/sample.yaml") -> None:
    """Write a sample config file with documented defaults."""
    save_plan(_default_plan(), output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_plan(d: dict) -> FinancialPlan:
    inc_d = d.get("income", {})
    income = IncomeProfile(
        gross_annual_income=float(inc_d.get("gross_annual_income", 100_000)),
        filing_status=FilingStatus(inc_d.get("filing_status", "single")),
        state=State(inc_d.get("state", "GA")),
        other_state_flat_rate=float(inc_d.get("other_state_flat_rate", 0.05)),
        spouse_gross_annual_income=float(inc_d.get("spouse_gross_annual_income", 0)),
    )

    h_d = d.get("housing", {})
    housing = HousingProfile(
        home_price=float(h_d.get("home_price", 400_000)),
        down_payment=float(h_d.get("down_payment", 80_000)),
        interest_rate=float(h_d.get("interest_rate", 0.065)),
        loan_term_years=int(h_d.get("loan_term_years", 30)),
        annual_property_tax_rate=float(h_d.get("annual_property_tax_rate", 0.012)),
        annual_insurance=float(h_d.get("annual_insurance", 2_000)),
        annual_maintenance_rate=float(h_d.get("annual_maintenance_rate", 0.01)),
        pmi_annual_rate=float(h_d.get("pmi_annual_rate", 0.005)),
        is_renting=bool(h_d.get("is_renting", False)),
        monthly_rent=float(h_d.get("monthly_rent", 0)),
        annual_rent_increase_rate=float(h_d.get("annual_rent_increase_rate", 0.03)),
    )

    l_d = d.get("lifestyle", {})
    lifestyle = LifestyleProfile(
        monthly_childcare=float(l_d.get("monthly_childcare", 0)),
        num_children=int(l_d.get("num_children", 0)),
        num_pets=int(l_d.get("num_pets", 0)),
        annual_pet_cost=float(l_d.get("annual_pet_cost", 0)),
        annual_medical_oop=float(l_d.get("annual_medical_oop", 3_000)),
        medical_auto_scale=bool(l_d.get("medical_auto_scale", True)),
        medical_spouse_multiplier=float(l_d.get("medical_spouse_multiplier", 1.8)),
        medical_per_child_annual=float(l_d.get("medical_per_child_annual", 1_500)),
        annual_vacation=float(l_d.get("annual_vacation", 5_000)),
        monthly_other_recurring=float(l_d.get("monthly_other_recurring", 500)),
    )

    inv_d = d.get("investments", {})
    investments = InvestmentProfile(
        current_liquid_cash=float(inv_d.get("current_liquid_cash", 50_000)),
        current_retirement_balance=float(inv_d.get("current_retirement_balance", 0)),
        current_brokerage_balance=float(inv_d.get("current_brokerage_balance", 0)),
        one_time_upcoming_expenses=float(inv_d.get("one_time_upcoming_expenses", 0)),
        annual_401k_contribution=float(inv_d.get("annual_401k_contribution", 23_000)),
        annual_roth_ira_contribution=float(inv_d.get("annual_roth_ira_contribution", 0)),
        annual_hsa_contribution=float(inv_d.get("annual_hsa_contribution", 4_150)),
        annual_529_contribution=float(inv_d.get("annual_529_contribution", 0)),
        annual_brokerage_contribution=float(inv_d.get("annual_brokerage_contribution", 0)),
        annual_market_return=float(inv_d.get("annual_market_return", 0.08)),
        annual_inflation_rate=float(inv_d.get("annual_inflation_rate", 0.03)),
        annual_salary_growth_rate=float(inv_d.get("annual_salary_growth_rate", 0.04)),
        annual_home_appreciation_rate=float(inv_d.get("annual_home_appreciation_rate", 0.035)),
    )

    s_d = d.get("strategies", {})
    strategies = StrategyToggles(
        maximize_hsa=bool(s_d.get("maximize_hsa", True)),
        use_529_state_deduction=bool(s_d.get("use_529_state_deduction", False)),
        maximize_401k=bool(s_d.get("maximize_401k", True)),
        use_roth_ladder=bool(s_d.get("use_roth_ladder", False)),
        roth_conversion_annual_amount=float(s_d.get("roth_conversion_annual_amount", 0)),
    )

    events = [
        TimelineEvent(
            year=int(e["year"]),
            description=str(e.get("description", "")),
            income_change=e.get("income_change"),
            new_child=bool(e.get("new_child", False)),
            new_pet=bool(e.get("new_pet", False)),
            marriage=bool(e.get("marriage", False)),
            buy_home=bool(e.get("buy_home", False)),
            new_home_price=float(e["new_home_price"]) if e.get("new_home_price") else None,
            new_home_down_payment=float(e["new_home_down_payment"]) if e.get("new_home_down_payment") else None,
            new_home_interest_rate=float(e["new_home_interest_rate"]) if e.get("new_home_interest_rate") else None,
            sell_current_home=bool(e.get("sell_current_home", True)),
            home_price_override=e.get("home_price_override"),
            extra_one_time_expense=float(e.get("extra_one_time_expense", 0)),
            extra_one_time_income=float(e.get("extra_one_time_income", 0)),
        )
        for e in d.get("timeline_events", [])
    ]

    return FinancialPlan(
        income=income,
        housing=housing,
        lifestyle=lifestyle,
        investments=investments,
        strategies=strategies,
        timeline_events=events,
        projection_years=int(d.get("projection_years", 30)),
    )


def _plan_to_dict(plan: FinancialPlan) -> dict:
    return {
        "projection_years": plan.projection_years,
        "income": {
            "gross_annual_income": plan.income.gross_annual_income,
            "filing_status": plan.income.filing_status.value,
            "state": plan.income.state.value,
            "other_state_flat_rate": plan.income.other_state_flat_rate,
            "spouse_gross_annual_income": plan.income.spouse_gross_annual_income,
        },
        "housing": {
            "home_price": plan.housing.home_price,
            "down_payment": plan.housing.down_payment,
            "interest_rate": plan.housing.interest_rate,
            "loan_term_years": plan.housing.loan_term_years,
            "annual_property_tax_rate": plan.housing.annual_property_tax_rate,
            "annual_insurance": plan.housing.annual_insurance,
            "annual_maintenance_rate": plan.housing.annual_maintenance_rate,
            "pmi_annual_rate": plan.housing.pmi_annual_rate,
            "is_renting": plan.housing.is_renting,
            "monthly_rent": plan.housing.monthly_rent,
            "annual_rent_increase_rate": plan.housing.annual_rent_increase_rate,
        },
        "lifestyle": {
            "monthly_childcare": plan.lifestyle.monthly_childcare,
            "num_children": plan.lifestyle.num_children,
            "num_pets": plan.lifestyle.num_pets,
            "annual_pet_cost": plan.lifestyle.annual_pet_cost,
            "annual_medical_oop": plan.lifestyle.annual_medical_oop,
            "medical_auto_scale": plan.lifestyle.medical_auto_scale,
            "medical_spouse_multiplier": plan.lifestyle.medical_spouse_multiplier,
            "medical_per_child_annual": plan.lifestyle.medical_per_child_annual,
            "annual_vacation": plan.lifestyle.annual_vacation,
            "monthly_other_recurring": plan.lifestyle.monthly_other_recurring,
        },
        "investments": {
            "current_liquid_cash": plan.investments.current_liquid_cash,
            "current_retirement_balance": plan.investments.current_retirement_balance,
            "current_brokerage_balance": plan.investments.current_brokerage_balance,
            "one_time_upcoming_expenses": plan.investments.one_time_upcoming_expenses,
            "annual_401k_contribution": plan.investments.annual_401k_contribution,
            "annual_roth_ira_contribution": plan.investments.annual_roth_ira_contribution,
            "annual_hsa_contribution": plan.investments.annual_hsa_contribution,
            "annual_529_contribution": plan.investments.annual_529_contribution,
            "annual_brokerage_contribution": plan.investments.annual_brokerage_contribution,
            "annual_market_return": plan.investments.annual_market_return,
            "annual_inflation_rate": plan.investments.annual_inflation_rate,
            "annual_salary_growth_rate": plan.investments.annual_salary_growth_rate,
            "annual_home_appreciation_rate": plan.investments.annual_home_appreciation_rate,
        },
        "strategies": {
            "maximize_hsa": plan.strategies.maximize_hsa,
            "use_529_state_deduction": plan.strategies.use_529_state_deduction,
            "maximize_401k": plan.strategies.maximize_401k,
            "use_roth_ladder": plan.strategies.use_roth_ladder,
            "roth_conversion_annual_amount": plan.strategies.roth_conversion_annual_amount,
        },
        "timeline_events": [
            {
                "year": e.year,
                "description": e.description,
                "income_change": e.income_change,
                "new_child": e.new_child,
                "new_pet": e.new_pet,
                "marriage": e.marriage,
                "buy_home": e.buy_home,
                "new_home_price": e.new_home_price,
                "new_home_down_payment": e.new_home_down_payment,
                "new_home_interest_rate": e.new_home_interest_rate,
                "sell_current_home": e.sell_current_home,
                "extra_one_time_expense": e.extra_one_time_expense,
                "extra_one_time_income": e.extra_one_time_income,
            }
            for e in plan.timeline_events
        ],
    }


def _default_plan() -> FinancialPlan:
    """A sensible default plan used when no config is found."""
    return FinancialPlan(
        income=IncomeProfile(
            gross_annual_income=120_000,
            filing_status=FilingStatus.SINGLE,
            state=State.GEORGIA,
        ),
        housing=HousingProfile(
            home_price=400_000,
            down_payment=80_000,
            interest_rate=0.065,
        ),
        lifestyle=LifestyleProfile(
            annual_medical_oop=3_000,
            annual_vacation=5_000,
            monthly_other_recurring=500,
        ),
        investments=InvestmentProfile(
            current_liquid_cash=100_000,
            annual_401k_contribution=23_000,
            annual_hsa_contribution=4_150,
            annual_market_return=0.08,
            annual_inflation_rate=0.03,
            annual_salary_growth_rate=0.04,
        ),
        strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
        projection_years=30,
    )