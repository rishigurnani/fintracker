"""
The HSA must pay qualified medical expenses (out-of-pocket + long-term care) in
retirement, tax-free, instead of only ever accumulating. While working it keeps
growing (medical is paid from income); in retirement it draws down for medical.
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, LifestyleProfile, RetirementProfile, State,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, renting_housing


def _plan(current_age, retirement_age, hsa_start, medical_oop, ltc=0.0,
          ltc_years_before_death=0, life_expectancy_age=None,
          market=0.0, post_ret=0.0, years=3):
    plan = make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=0),
        lifestyle=LifestyleProfile(annual_medical_oop=medical_oop, medical_auto_scale=False,
                                   annual_vacation=0, monthly_other_recurring=0,
                                   annual_self_ltc_cost=ltc,
                                   self_ltc_years_before_death=ltc_years_before_death),
        investments=investments(current_liquid_cash=0, current_retirement_balance=5_000_000,
                                annual_market_return=market, annual_hsa_contribution=0),
        retirement=RetirementProfile(current_age=current_age, retirement_age=retirement_age,
                                     medicare_start_age=200, expected_post_retirement_return=post_ret,
                                     retirement_withdrawal_tax_rate=0.2,
                                     life_expectancy_age=life_expectancy_age),
        projection_years=years,
    )
    # Seed a starting HSA balance directly (no contribution path needed).
    eng = ProjectionEngine(plan)
    return plan, eng, hsa_start


def _run_with_hsa(plan, eng, hsa_start):
    state = eng._initial_state()
    state.hsa_balance = hsa_start
    snaps = []
    for year in range(1, eng._horizon() + 1):
        eng._apply_timeline_events(state, year)
        snap = eng._compute_year(state, year)
        snaps.append(snap)
        eng._advance_state(state, snap)
    return snaps


class TestHsaMedicalSpend:

    def test_retired_hsa_pays_medical_and_declines(self):
        """Retired: the HSA covers medical OOP each year and its balance falls by
        that amount (zero growth to isolate the draw)."""
        plan, eng, hsa0 = _plan(current_age=66, retirement_age=65,
                                hsa_start=100_000, medical_oop=10_000)
        s = _run_with_hsa(plan, eng, hsa0)[0]
        assert s.annual_hsa_withdrawal == pytest.approx(10_000, rel=1e-6)
        assert s.hsa_balance == pytest.approx(90_000, rel=1e-6)

    def test_working_hsa_is_not_drawn(self):
        """Before retirement the HSA keeps growing — medical is paid from income,
        not the HSA."""
        plan, eng, hsa0 = _plan(current_age=40, retirement_age=65,
                                hsa_start=100_000, medical_oop=10_000)
        s = _run_with_hsa(plan, eng, hsa0)[0]
        assert s.annual_hsa_withdrawal == pytest.approx(0.0, abs=1e-6)
        assert s.hsa_balance == pytest.approx(100_000, rel=1e-6)

    def test_hsa_draw_capped_at_balance_never_negative(self):
        """A medical bill larger than the HSA drains it to zero, not below."""
        plan, eng, hsa0 = _plan(current_age=66, retirement_age=65,
                                hsa_start=5_000, medical_oop=40_000)
        s = _run_with_hsa(plan, eng, hsa0)[0]
        assert s.annual_hsa_withdrawal == pytest.approx(5_000, rel=1e-6)
        assert s.hsa_balance == pytest.approx(0.0, abs=1e-6)

    def test_hsa_covers_long_term_care(self):
        """Long-term care is HSA-qualified and is paid from the HSA in retirement."""
        # Death in projection year 1 (age 66), LTC window of 1 → LTC active that year.
        plan, eng, hsa0 = _plan(current_age=66, retirement_age=65, hsa_start=100_000,
                                medical_oop=0, ltc=30_000, ltc_years_before_death=1,
                                life_expectancy_age=66)
        s = _run_with_hsa(plan, eng, hsa0)[0]
        assert s.annual_hsa_withdrawal == pytest.approx(30_000, rel=1e-6)
        assert s.hsa_balance == pytest.approx(70_000, rel=1e-6)


def _medicare_plan(hsa_start, medicare_premium=6_000):
    """Retired, no income/expenses except Medicare, large HSA — isolates the HSA's
    Medicare payment."""
    plan = make_plan(
        income=IncomeProfile(0, FilingStatus.SINGLE, State.TEXAS),
        housing=renting_housing(monthly_rent=0),
        lifestyle=LifestyleProfile(annual_medical_oop=0, medical_auto_scale=False,
                                   annual_vacation=0, monthly_other_recurring=0),
        investments=investments(current_liquid_cash=0, current_retirement_balance=5_000_000,
                                annual_hsa_contribution=0),
        retirement=RetirementProfile(current_age=66, retirement_age=65, medicare_start_age=65,
                                     annual_medicare_premium=medicare_premium,
                                     expected_post_retirement_return=0.0,
                                     retirement_withdrawal_tax_rate=0.2),
        projection_years=2,
    )
    return plan, ProjectionEngine(plan), hsa_start


class TestHsaPaysMedicare:

    def test_hsa_pays_medicare_tax_free(self):
        """With a funded HSA, Medicare is paid from it — so the HSA drawdown equals
        the Medicare bill and no 401k withdrawal (hence no IRMAA) is triggered."""
        plan, eng, hsa0 = _medicare_plan(hsa_start=200_000, medicare_premium=6_000)
        s = _run_with_hsa(plan, eng, hsa0)[0]
        assert s.annual_medicare_cost == pytest.approx(6_000, rel=1e-6)   # base only, no IRMAA
        assert s.annual_hsa_withdrawal == pytest.approx(6_000, rel=1e-6)  # HSA paid it
        assert s.annual_retirement_withdrawal == pytest.approx(0.0, abs=1.0)
        assert s.hsa_balance == pytest.approx(194_000, rel=1e-6)

    def test_hsa_covered_medicare_does_not_exceed_no_hsa_case(self):
        """Paying Medicare from the HSA is tax-free, so it can only lower (never
        raise) MAGI-driven IRMAA versus funding Medicare from the 401k."""
        with_hsa = _run_with_hsa(*_medicare_plan(hsa_start=200_000))[0]
        no_hsa   = _run_with_hsa(*_medicare_plan(hsa_start=0))[0]
        assert with_hsa.annual_medicare_cost <= no_hsa.annual_medicare_cost + 1e-6
