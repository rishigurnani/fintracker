"""Shared pytest fixtures for fintracker tests."""
import pytest
from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, FinancialPlan, TimelineEvent,
)


@pytest.fixture
def single_income_ga() -> IncomeProfile:
    return IncomeProfile(
        gross_annual_income=100_000,
        filing_status=FilingStatus.SINGLE,
        state=State.GEORGIA,
    )


@pytest.fixture
def mfj_income_ga() -> IncomeProfile:
    return IncomeProfile(
        gross_annual_income=180_000,
        filing_status=FilingStatus.MARRIED_FILING_JOINTLY,
        state=State.GEORGIA,
        spouse_gross_annual_income=0,
    )


@pytest.fixture
def standard_housing() -> HousingProfile:
    return HousingProfile(
        home_price=400_000,
        down_payment=80_000,   # 20% — no PMI
        interest_rate=0.065,
        loan_term_years=30,
    )


@pytest.fixture
def low_down_housing() -> HousingProfile:
    """Under 20% down — triggers PMI."""
    return HousingProfile(
        home_price=400_000,
        down_payment=40_000,   # 10%
        interest_rate=0.065,
        loan_term_years=30,
    )


@pytest.fixture
def standard_lifestyle() -> LifestyleProfile:
    return LifestyleProfile(
        monthly_childcare=0,
        num_children=0,
        annual_medical_oop=3_000,
        annual_vacation=5_000,
        monthly_other_recurring=500,
    )


@pytest.fixture
def standard_investments() -> InvestmentProfile:
    return InvestmentProfile(
        current_liquid_cash=100_000,
        current_retirement_balance=50_000,
        annual_401k_contribution=23_000,
        annual_hsa_contribution=4_150,
        annual_market_return=0.08,
        annual_inflation_rate=0.03,
        annual_salary_growth_rate=0.04,
        annual_home_appreciation_rate=0.035,
    )


@pytest.fixture
def all_strategies_on() -> StrategyToggles:
    return StrategyToggles(
        maximize_hsa=True,
        use_529_state_deduction=True,
        maximize_401k=True,
        use_roth_ladder=False,
    )


@pytest.fixture
def all_strategies_off() -> StrategyToggles:
    return StrategyToggles(
        maximize_hsa=False,
        use_529_state_deduction=False,
        maximize_401k=False,
        use_roth_ladder=False,
    )


@pytest.fixture
def simple_plan(
    single_income_ga, standard_housing, standard_lifestyle,
    standard_investments, all_strategies_on,
) -> FinancialPlan:
    return FinancialPlan(
        income=single_income_ga,
        housing=standard_housing,
        lifestyle=standard_lifestyle,
        investments=standard_investments,
        strategies=all_strategies_on,
        projection_years=10,
    )