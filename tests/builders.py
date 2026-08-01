"""Shared test data builders.

The test suite constructs many FinancialPlans that share the same
zero-growth / renting / no-active-strategy skeleton and differ only in a
few fields. These composable factories keep that skeleton in one place so
individual tests express only what actually matters to them.

Every factory takes keyword overrides and merges them over sensible test
defaults, so callers pass just the fields they care about, e.g.::

    make_plan(income=IncomeProfile(...), projection_years=5)
    investments(current_liquid_cash=200_000, annual_salary_growth_rate=0.05)
"""
from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, FinancialPlan,
)


def investments(**kw) -> InvestmentProfile:
    """InvestmentProfile with market/inflation/salary-growth pinned to 0 by default."""
    return InvestmentProfile(**{
        "annual_market_return": 0.0,
        "annual_inflation_rate": 0.0,
        "annual_salary_growth_rate": 0.0,
        **kw,
    })


def zero_lifestyle(**kw) -> LifestyleProfile:
    """LifestyleProfile with discretionary/medical costs zeroed and medical auto-scale off."""
    return LifestyleProfile(**{
        "annual_vacation": 0,
        "monthly_other_recurring": 0,
        "annual_medical_oop": 0,
        "medical_auto_scale": False,
        **kw,
    })


def renting_housing(monthly_rent=0, **kw) -> HousingProfile:
    """A renting HousingProfile (no home, no mortgage)."""
    return HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=monthly_rent, **kw)


def make_plan(**overrides) -> FinancialPlan:
    """Zero-growth baseline plan: renting, no active strategies, 10-year horizon.

    Override any FinancialPlan component (income, housing, lifestyle,
    investments, strategies, timeline_events, projection_years, ...).
    """
    return FinancialPlan(**{
        "income": IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
        "housing": renting_housing(),
        "lifestyle": LifestyleProfile(),
        "investments": investments(current_liquid_cash=500_000),
        "strategies": StrategyToggles(maximize_hsa=False, maximize_401k=False),
        "projection_years": 10,
        **overrides,
    })
