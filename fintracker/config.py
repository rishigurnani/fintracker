"""
Configuration loader / saver.

Personal config lives in config/personal.yaml (gitignored).
A sample config is provided in config/sample.yaml (tracked in git).

Usage::

    plan = load_plan("config/personal.yaml")
    save_plan(plan, "config/personal.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fintracker.models import (
    CarProfile, CollegeProfile, FilingStatus, RetirementProfile, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, TimelineEvent, FinancialPlan,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_plan(path: str | Path) -> FinancialPlan:
    """Load a FinancialPlan from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return _dict_to_plan(yaml.safe_load(f))


def save_plan(plan: FinancialPlan, path: str | Path) -> None:
    """Serialize a FinancialPlan to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_plan_to_dict(plan), f, default_flow_style=False, sort_keys=False)


def load_plan_or_sample(path: str | Path = "config/personal.yaml") -> FinancialPlan:
    """Load personal config; fall back to sample.yaml then hard-coded defaults."""
    try:
        return load_plan(path)
    except FileNotFoundError:
        sample = Path(__file__).parent.parent / "config" / "sample.yaml"
        return load_plan(sample) if sample.exists() else _default_plan()


# ---------------------------------------------------------------------------
# Deserialization
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
        annual_parent_care_cost=float(l_d.get("annual_parent_care_cost", 0)),
        annual_wedding_fund_per_child=float(l_d.get("annual_wedding_fund_per_child", 0)),
    )

    inv_d = d.get("investments", {})
    s_d   = d.get("strategies", {})
    # auto_invest_surplus lives in investments:; read from strategies: for back-compat
    auto_invest = inv_d.get("auto_invest_surplus", s_d.get("auto_invest_surplus", True))
    investments = InvestmentProfile(
        current_liquid_cash=float(inv_d.get("current_liquid_cash", 50_000)),
        current_retirement_balance=float(inv_d.get("current_retirement_balance", 0)),
        current_brokerage_balance=float(inv_d.get("current_brokerage_balance", 0)),
        one_time_upcoming_expenses=float(inv_d.get("one_time_upcoming_expenses", 0)),
        annual_401k_contribution=float(inv_d.get("annual_401k_contribution", 23_000)),
        partner_annual_401k_contribution=float(inv_d.get("partner_annual_401k_contribution", 0)),
        annual_roth_ira_contribution=float(inv_d.get("annual_roth_ira_contribution", 0)),
        annual_hsa_contribution=float(inv_d.get("annual_hsa_contribution", 4_150)),
        annual_529_contribution=float(inv_d.get("annual_529_contribution", 0)),
        annual_brokerage_contribution=float(inv_d.get("annual_brokerage_contribution", 0)),
        annual_market_return=float(inv_d.get("annual_market_return", 0.08)),
        annual_inflation_rate=float(inv_d.get("annual_inflation_rate", 0.03)),
        annual_salary_growth_rate=float(inv_d.get("annual_salary_growth_rate", 0.04)),
        partner_salary_growth_rate=float(inv_d.get("partner_salary_growth_rate", 0.04)),
        annual_home_appreciation_rate=float(inv_d.get("annual_home_appreciation_rate", 0.035)),
        auto_invest_surplus=bool(auto_invest),
    )

    strategies = StrategyToggles(
        maximize_hsa=bool(s_d.get("maximize_hsa", True)),
        use_529_state_deduction=bool(s_d.get("use_529_state_deduction", False)),
        maximize_401k=bool(s_d.get("maximize_401k", True)),
        use_roth_ladder=bool(s_d.get("use_roth_ladder", False)),
        roth_conversion_annual_amount=float(s_d.get("roth_conversion_annual_amount", 0)),
    )

    events = [_dict_to_event(e) for e in d.get("timeline_events", [])]

    retirement = _dict_to_retirement(d["retirement"]) if "retirement" in d else None
    college    = _dict_to_college(d["college"])    if "college"    in d else None
    car        = _dict_to_car(d["car"])            if "car"        in d else None

    return FinancialPlan(
        income=income,
        housing=housing,
        lifestyle=lifestyle,
        investments=investments,
        strategies=strategies,
        timeline_events=events,
        projection_years=int(d.get("projection_years", 30)),
        retirement=retirement,
        college=college,
        car=car,
    )


def _dict_to_event(e: dict) -> TimelineEvent:
    return TimelineEvent(
        year=int(e["year"]),
        description=str(e.get("description", "")),
        income_change=e.get("income_change"),
        partner_income_change=e.get("partner_income_change"),
        stop_working=bool(e.get("stop_working", False)),
        resume_working=bool(e.get("resume_working", False)),
        partner_stop_working=bool(e.get("partner_stop_working", False)),
        partner_resume_working=bool(e.get("partner_resume_working", False)),
        start_parent_care=bool(e.get("start_parent_care", False)),
        stop_parent_care=bool(e.get("stop_parent_care", False)),
        child_birth_year_override=e.get("child_birth_year_override"),
        new_child=bool(e.get("new_child", False)),
        new_pet=bool(e.get("new_pet", False)),
        marriage=bool(e.get("marriage", False)),
        buy_home=bool(e.get("buy_home", False)),
        new_home_price=float(e["new_home_price"]) if e.get("new_home_price") else None,
        new_home_down_payment=float(e["new_home_down_payment"]) if e.get("new_home_down_payment") else None,
        new_home_interest_rate=float(e["new_home_interest_rate"]) if e.get("new_home_interest_rate") else None,
        sell_current_home=bool(e.get("sell_current_home", True)),
        buyer_closing_cost_rate=float(e.get("buyer_closing_cost_rate", 0.02)),
        seller_closing_cost_rate=float(e.get("seller_closing_cost_rate", 0.06)),
        home_price_override=e.get("home_price_override"),
        extra_one_time_expense=float(e.get("extra_one_time_expense", 0)),
        extra_one_time_income=float(e.get("extra_one_time_income", 0)),
    )


def _dict_to_retirement(r: dict) -> RetirementProfile:
    return RetirementProfile(
        current_age=int(r.get("current_age", 35)),
        retirement_age=int(r.get("retirement_age", 65)),
        desired_annual_income=float(r.get("desired_annual_income", 80_000)),
        years_in_retirement=int(r.get("years_in_retirement", 30)),
        expected_post_retirement_return=float(r.get("expected_post_retirement_return", 0.05)),
        estimated_social_security_annual=float(r.get("estimated_social_security_annual", 0)),
    )


def _dict_to_college(c: dict) -> CollegeProfile:
    return CollegeProfile(
        annual_cost_per_child=float(c.get("annual_cost_per_child", 35_000)),
        years_per_child=int(c.get("years_per_child", 4)),
        start_age=int(c.get("start_age", 18)),
        use_aotc_credit=bool(c.get("use_aotc_credit", True)),
        early_529_return=float(c.get("early_529_return", 0.08)),
        late_529_return=float(c.get("late_529_return", 0.04)),
        glide_path_years=int(c.get("glide_path_years", 10)),
    )


def _dict_to_car(c: dict) -> CarProfile:
    return CarProfile(
        car_price=float(c.get("car_price", 25_000)),
        down_payment=float(c.get("down_payment", 5_000)),
        loan_rate=float(c.get("loan_rate", 0.065)),
        loan_term_years=int(c.get("loan_term_years", 5)),
        replace_every_years=int(c.get("replace_every_years", 10)),
        residual_value=float(c.get("residual_value", 5_000)),
        hand_down_age=int(c.get("hand_down_age", 16)),
        num_cars=int(c.get("num_cars", 1)),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _plan_to_dict(plan: FinancialPlan) -> dict:
    d: dict = {
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
            "annual_parent_care_cost": plan.lifestyle.annual_parent_care_cost,
            "annual_wedding_fund_per_child": plan.lifestyle.annual_wedding_fund_per_child,
        },
        "investments": {
            "current_liquid_cash": plan.investments.current_liquid_cash,
            "current_retirement_balance": plan.investments.current_retirement_balance,
            "current_brokerage_balance": plan.investments.current_brokerage_balance,
            "one_time_upcoming_expenses": plan.investments.one_time_upcoming_expenses,
            "annual_401k_contribution": plan.investments.annual_401k_contribution,
            "partner_annual_401k_contribution": plan.investments.partner_annual_401k_contribution,
            "annual_roth_ira_contribution": plan.investments.annual_roth_ira_contribution,
            "annual_hsa_contribution": plan.investments.annual_hsa_contribution,
            "annual_529_contribution": plan.investments.annual_529_contribution,
            "annual_brokerage_contribution": plan.investments.annual_brokerage_contribution,
            "annual_market_return": plan.investments.annual_market_return,
            "annual_inflation_rate": plan.investments.annual_inflation_rate,
            "annual_salary_growth_rate": plan.investments.annual_salary_growth_rate,
            "partner_salary_growth_rate": plan.investments.partner_salary_growth_rate,
            "annual_home_appreciation_rate": plan.investments.annual_home_appreciation_rate,
            "auto_invest_surplus": plan.investments.auto_invest_surplus,
        },
        "strategies": {
            "maximize_hsa": plan.strategies.maximize_hsa,
            "use_529_state_deduction": plan.strategies.use_529_state_deduction,
            "maximize_401k": plan.strategies.maximize_401k,
            "use_roth_ladder": plan.strategies.use_roth_ladder,
            "roth_conversion_annual_amount": plan.strategies.roth_conversion_annual_amount,
        },
        "timeline_events": [_event_to_dict(e) for e in plan.timeline_events],
    }

    if plan.retirement:
        d["retirement"] = {
            "current_age": plan.retirement.current_age,
            "retirement_age": plan.retirement.retirement_age,
            "desired_annual_income": plan.retirement.desired_annual_income,
            "years_in_retirement": plan.retirement.years_in_retirement,
            "expected_post_retirement_return": plan.retirement.expected_post_retirement_return,
            "estimated_social_security_annual": plan.retirement.estimated_social_security_annual,
        }

    if plan.college:
        d["college"] = {
            "annual_cost_per_child": plan.college.annual_cost_per_child,
            "years_per_child": plan.college.years_per_child,
            "start_age": plan.college.start_age,
            "use_aotc_credit": plan.college.use_aotc_credit,
            "early_529_return": plan.college.early_529_return,
            "late_529_return": plan.college.late_529_return,
            "glide_path_years": plan.college.glide_path_years,
        }

    if plan.car:
        d["car"] = {
            "car_price": plan.car.car_price,
            "down_payment": plan.car.down_payment,
            "loan_rate": plan.car.loan_rate,
            "loan_term_years": plan.car.loan_term_years,
            "replace_every_years": plan.car.replace_every_years,
            "residual_value": plan.car.residual_value,
            "hand_down_age": plan.car.hand_down_age,
            "num_cars": plan.car.num_cars,
        }

    return d


def _event_to_dict(e: TimelineEvent) -> dict:
    return {
        "year": e.year,
        "description": e.description,
        "income_change": e.income_change,
        "partner_income_change": e.partner_income_change,
        "stop_working": e.stop_working,
        "resume_working": e.resume_working,
        "partner_stop_working": e.partner_stop_working,
        "partner_resume_working": e.partner_resume_working,
        "start_parent_care": e.start_parent_care,
        "stop_parent_care": e.stop_parent_care,
        "child_birth_year_override": e.child_birth_year_override,
        "new_child": e.new_child,
        "new_pet": e.new_pet,
        "marriage": e.marriage,
        "buy_home": e.buy_home,
        "new_home_price": e.new_home_price,
        "new_home_down_payment": e.new_home_down_payment,
        "new_home_interest_rate": e.new_home_interest_rate,
        "sell_current_home": e.sell_current_home,
        "buyer_closing_cost_rate": e.buyer_closing_cost_rate,
        "seller_closing_cost_rate": e.seller_closing_cost_rate,
        "extra_one_time_expense": e.extra_one_time_expense,
        "extra_one_time_income": e.extra_one_time_income,
    }


# ---------------------------------------------------------------------------
# Default plan
# ---------------------------------------------------------------------------

def _default_plan() -> FinancialPlan:
    return FinancialPlan(
        income=IncomeProfile(gross_annual_income=120_000,
                             filing_status=FilingStatus.SINGLE, state=State.GEORGIA),
        housing=HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065),
        lifestyle=LifestyleProfile(annual_medical_oop=3_000, annual_vacation=5_000,
                                   monthly_other_recurring=500),
        investments=InvestmentProfile(current_liquid_cash=100_000, annual_401k_contribution=23_000,
                                      annual_hsa_contribution=4_150, annual_market_return=0.08,
                                      annual_inflation_rate=0.03, annual_salary_growth_rate=0.04),
        strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
        projection_years=30,
    )