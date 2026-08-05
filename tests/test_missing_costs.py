"""
Tests for the previously-missing personal-finance costs and automatic retirement:

  1. Insurance premiums (health / disability / life) on LifestyleProfile
  2. Your own long-term care (age-gated)
  3. Car operating costs (insurance / maintenance / fuel / registration)
  4. Medicare base premium + income-tiered IRMAA surcharge (65+)
  5. Automatic retirement at retirement_age

Each test pins market/inflation/salary growth to 0 (via the shared builders) so a
single cost shows up as an exact, isolated reduction in breathing room.
"""
import pytest

from fintracker.constants import irmaa_annual_surcharge
from fintracker.models import (
    CarProfile, FilingStatus, FinancialPlan, IncomeProfile, InvestmentProfile,
    LifestyleProfile, RetirementProfile, State, StrategyToggles, TimelineEvent,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, zero_lifestyle


def _project(plan):
    return ProjectionEngine(plan).run_deterministic()


def _retirement(current_age, retirement_age=65, **kw):
    return RetirementProfile(current_age=current_age, retirement_age=retirement_age, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Insurance premiums
# ══════════════════════════════════════════════════════════════════════════════

class TestInsurancePremiums:

    def _plan(self, **lifestyle_kw):
        return make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(**lifestyle_kw),
            investments=investments(current_liquid_cash=500_000),
            projection_years=3,
        )

    def test_premiums_reduce_breathing_room_while_working(self):
        base = _project(self._plan())[0]
        withp = _project(self._plan(
            annual_health_insurance_premium=6_000,
            annual_disability_insurance_premium=1_000,
            annual_life_insurance_premium=800,
        ))[0]
        assert withp.annual_insurance_premiums == pytest.approx(7_800)
        assert base.annual_breathing_room - withp.annual_breathing_room == pytest.approx(7_800)

    def test_premiums_folded_into_lifestyle_cost(self):
        s = _project(self._plan(annual_life_insurance_premium=1_200))[0]
        assert s.annual_lifestyle_cost == pytest.approx(1_200)
        assert s.annual_insurance_premiums == pytest.approx(1_200)

    def test_health_and_disability_stop_when_not_working(self):
        # Stop working in year 2 → health & disability lapse; life continues.
        plan = self._plan(
            annual_health_insurance_premium=6_000,
            annual_disability_insurance_premium=1_000,
            annual_life_insurance_premium=800,
        )
        plan = FinancialPlan(**{**plan.__dict__,
                                "timeline_events": [TimelineEvent(year=2, description="retire early",
                                                                  stop_working=True)]})
        snaps = _project(plan)
        assert snaps[0].annual_insurance_premiums == pytest.approx(7_800)   # working
        assert snaps[1].annual_insurance_premiums == pytest.approx(800)     # only life

    def test_health_premium_stops_at_medicare_age_but_life_continues(self):
        # Age 64 in yr1 → 65 in yr2. Still "working" (no auto-retire here), but
        # health premium should stop at Medicare age; life keeps going.
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(annual_health_insurance_premium=5_000,
                                     annual_life_insurance_premium=900),
            investments=investments(current_liquid_cash=500_000),
            retirement=_retirement(64, auto_retire=False),
            projection_years=3,
        )
        snaps = _project(plan)
        assert snaps[0].annual_insurance_premiums == pytest.approx(5_900)  # age 64: health+life
        assert snaps[1].annual_insurance_premiums == pytest.approx(900)    # age 65: life only


# ══════════════════════════════════════════════════════════════════════════════
# 2. Your own long-term care
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfLongTermCare:

    def _plan(self, current_age, years_before_death, cost, life_expectancy_age):
        return make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(annual_self_ltc_cost=cost,
                                     self_ltc_years_before_death=years_before_death),
            investments=investments(current_liquid_cash=2_000_000),
            retirement=_retirement(current_age, auto_retire=False,
                                   life_expectancy_age=life_expectancy_age),
            # horizon is governed by life_expectancy_age when set (death year = 4 here)
            projection_years=4,
        )

    def test_applies_only_in_final_years_of_life(self):
        # age 78 now, dies at 81 (projection year 4). years_before_death=3 → the
        # final three years through death (years 2,3,4 = ages 79,80,81); year 1 off.
        snaps = _project(self._plan(current_age=78, years_before_death=3,
                                    cost=90_000, life_expectancy_age=81))
        assert snaps[0].annual_self_ltc_cost == 0.0                    # age 78
        assert snaps[1].annual_self_ltc_cost == pytest.approx(90_000)  # age 79
        assert snaps[2].annual_self_ltc_cost == pytest.approx(90_000)  # age 80
        assert snaps[3].annual_self_ltc_cost == pytest.approx(90_000)  # age 81 (death)

    def test_window_of_one_charges_death_year_only(self):
        # years_before_death=1 → only the death year (age 81, year 4) is charged.
        snaps = _project(self._plan(current_age=78, years_before_death=1,
                                    cost=90_000, life_expectancy_age=81))
        costs = [s.annual_self_ltc_cost for s in snaps]
        assert costs == [0.0, 0.0, 0.0, pytest.approx(90_000)]

    def test_never_applies_without_death_year(self):
        # A RetirementProfile without life_expectancy_age has no modeled death, so
        # self-LTC (which anchors to the end of life) never triggers.
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(annual_self_ltc_cost=90_000,
                                     self_ltc_years_before_death=5),
            investments=investments(current_liquid_cash=2_000_000),
            retirement=_retirement(70, auto_retire=False),   # no life_expectancy_age
            projection_years=6,
        )
        assert all(s.annual_self_ltc_cost == 0.0 for s in _project(plan))

    def test_never_applies_without_retirement_profile(self):
        # No RetirementProfile → no death year → self-LTC never triggers.
        plan = make_plan(
            lifestyle=zero_lifestyle(annual_self_ltc_cost=50_000,
                                     self_ltc_years_before_death=5),
            investments=investments(current_liquid_cash=500_000),
            projection_years=3,
        )
        assert all(s.annual_self_ltc_cost == 0.0 for s in _project(plan))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Car operating costs
# ══════════════════════════════════════════════════════════════════════════════

class TestCarOperatingCosts:

    def _plan(self, num_cars=1, first_purchase_years=None, **op_kw):
        car_kw = dict(car_price=25_000, down_payment=5_000, loan_rate=0.05,
                      loan_term_years=5, replace_every_years=20, num_cars=num_cars,
                      annual_insurance_per_car=1_500, annual_maintenance_per_car=1_000,
                      annual_fuel_per_car=2_000, annual_registration_per_car=200)
        car_kw.update(op_kw)
        if first_purchase_years is not None:
            car_kw["first_purchase_years"] = first_purchase_years
        return make_plan(
            income=IncomeProfile(200_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=500_000),
            car=CarProfile(**car_kw),
            projection_years=5,
        )

    def test_single_car_operating_cost(self):
        s = _project(self._plan())[0]
        assert s.annual_car_operating_cost == pytest.approx(4_700)  # 1500+1000+2000+200

    def test_scales_with_number_of_cars(self):
        one = _project(self._plan(num_cars=1))[0]
        two = _project(self._plan(num_cars=2))[0]
        assert two.annual_car_operating_cost == pytest.approx(2 * one.annual_car_operating_cost)

    def test_not_charged_before_first_purchase(self):
        # Car first bought in year 3 → no operating cost in years 1-2.
        snaps = _project(self._plan(first_purchase_years=[3]))
        assert snaps[0].annual_car_operating_cost == 0.0
        assert snaps[1].annual_car_operating_cost == 0.0
        assert snaps[2].annual_car_operating_cost == pytest.approx(4_700)

    def test_operating_cost_inflates(self):
        plan = self._plan()
        plan = FinancialPlan(**{**plan.__dict__,
                                "investments": investments(current_liquid_cash=500_000,
                                                           annual_inflation_rate=0.10)})
        snaps = _project(plan)
        assert snaps[1].annual_car_operating_cost == pytest.approx(4_700 * 1.10)

    def test_no_car_means_no_operating_cost(self):
        plan = make_plan(lifestyle=zero_lifestyle(), projection_years=3)
        assert all(s.annual_car_operating_cost == 0.0 for s in _project(plan))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Medicare + IRMAA
# ══════════════════════════════════════════════════════════════════════════════

class TestMedicareAndIrmaa:

    def _plan(self, current_age, married=False, income=40_000, medicare_premium=2_100,
              desired_income=40_000, years=3):
        return make_plan(
            income=IncomeProfile(income,
                                 FilingStatus.MARRIED_FILING_JOINTLY if married else FilingStatus.SINGLE,
                                 State.TEXAS,
                                 spouse_gross_annual_income=income if married else 0.0),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=2_000_000),
            retirement=_retirement(current_age, desired_annual_income=desired_income,
                                   annual_medicare_premium=medicare_premium, auto_retire=False),
            projection_years=years,
        )

    def test_no_medicare_before_start_age(self):
        # Age 63 in yr1, 64 in yr2 → still no Medicare.
        snaps = _project(self._plan(current_age=63))
        assert snaps[0].annual_medicare_cost == 0.0
        assert snaps[1].annual_medicare_cost == 0.0

    def test_base_premium_at_65_low_income_no_irmaa(self):
        # Age 65 in yr1, low income → base premium only, no surcharge.
        s = _project(self._plan(current_age=65, income=30_000, desired_income=30_000))[0]
        assert s.annual_medicare_cost == pytest.approx(2_100)

    def test_married_couple_pays_double_base(self):
        single = _project(self._plan(current_age=65, income=30_000, desired_income=30_000))[0]
        couple = _project(self._plan(current_age=65, married=True, income=30_000,
                                     desired_income=30_000))[0]
        assert couple.annual_medicare_cost == pytest.approx(2 * single.annual_medicare_cost)

    def test_irmaa_surcharge_applied_for_high_earner_still_working(self):
        # Age 66, still working at high W-2 income → IRMAA surcharge on top of base.
        s = _project(self._plan(current_age=66, income=200_000))[0]
        expected = 2_100 + irmaa_annual_surcharge(200_000, is_married=False)
        assert s.annual_medicare_cost == pytest.approx(expected)
        assert s.annual_medicare_cost > 2_100

    def test_medicare_folded_into_lifestyle_and_breathing_room(self):
        s = _project(self._plan(current_age=65, income=30_000, desired_income=30_000))[0]
        assert s.annual_lifestyle_cost == pytest.approx(2_100)  # only cost is Medicare

    # --- IRMAA helper unit tests ---

    def test_irmaa_tiers_single(self):
        assert irmaa_annual_surcharge(100_000, is_married=False) == 0.0
        assert irmaa_annual_surcharge(120_000, is_married=False) == pytest.approx(994.80)
        assert irmaa_annual_surcharge(600_000, is_married=False) == pytest.approx(6_003.60)

    def test_irmaa_thresholds_double_for_married(self):
        # $180k: over the single first threshold ($103k) but under the MFJ one ($206k).
        assert irmaa_annual_surcharge(180_000, is_married=False) > 0.0
        assert irmaa_annual_surcharge(180_000, is_married=True) == 0.0

    def test_irmaa_thresholds_inflation_indexed(self):
        # A MAGI just over the nominal first bound falls back to $0 once the bound
        # is inflated above it.
        assert irmaa_annual_surcharge(110_000, is_married=False) > 0.0
        assert irmaa_annual_surcharge(110_000, is_married=False, inflation_factor=1.5) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Automatic retirement
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoRetirement:

    def _plan(self, current_age, retirement_age=65, auto_retire=True, years=None,
              spouse_income=0.0):
        yrs = years or (retirement_age - current_age + 3)
        return make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS,
                                 spouse_gross_annual_income=spouse_income),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=1_000_000),
            retirement=_retirement(current_age, retirement_age, auto_retire=auto_retire),
            projection_years=yrs,
        )

    def test_income_stops_at_retirement_age(self):
        # Age 63 in yr1 → age 65 in yr3 (retirement year).
        snaps = _project(self._plan(current_age=63, retirement_age=65))
        assert snaps[1].gross_income > 0          # yr2, age 64: still working
        assert snaps[1].is_working is True
        assert snaps[2].gross_income == 0.0       # yr3, age 65: retired
        assert snaps[2].is_working is False
        assert snaps[3].gross_income == 0.0       # stays retired

    def test_partner_also_retires(self):
        snaps = _project(self._plan(current_age=64, retirement_age=65, spouse_income=90_000))
        # yr1 age 64: both working; yr2 age 65: both retired.
        assert snaps[0].gross_income == pytest.approx(240_000)
        assert snaps[1].gross_income == 0.0
        assert snaps[1].is_partner_working is False

    def test_disabled_keeps_paying_salary(self):
        snaps = _project(self._plan(current_age=64, retirement_age=65, auto_retire=False))
        assert snaps[1].gross_income > 0          # age 65, still earning
        assert snaps[1].is_working is True

    def test_no_retirement_profile_never_auto_retires(self):
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=investments(current_liquid_cash=500_000),
            projection_years=40,
        )
        snaps = _project(plan)
        assert all(s.is_working for s in snaps)
        assert snaps[-1].gross_income > 0

    def test_resume_working_after_auto_retirement_overrides(self):
        # Auto-retire at 65, then a resume_working event brings income back.
        plan = self._plan(current_age=64, retirement_age=65, years=5)
        plan = FinancialPlan(**{**plan.__dict__,
                                "timeline_events": [TimelineEvent(year=4, description="consulting gig",
                                                                  resume_working=True,
                                                                  income_change=60_000)]})
        snaps = _project(plan)
        assert snaps[1].gross_income == 0.0        # yr2 age 65: auto-retired
        assert snaps[3].gross_income == pytest.approx(60_000)  # yr4: resumed
        assert snaps[3].is_working is True


# ══════════════════════════════════════════════════════════════════════════════
# 6. Healthcare inflation — medical costs compound at their own (higher) rate
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthcareInflation:
    """General inflation = 0, healthcare inflation = 10% so the medical costs that
    should track the healthcare rate creep upward while everything else stays flat."""

    def _inv(self, **kw):
        return investments(current_liquid_cash=3_000_000,
                           annual_inflation_rate=0.0,
                           annual_healthcare_inflation_rate=0.10, **kw)

    def test_medical_oop_compounds_at_healthcare_rate(self):
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(annual_medical_oop=4_000, medical_auto_scale=False),
            investments=self._inv(),
            projection_years=3,
        )
        snaps = _project(plan)
        assert snaps[0].annual_medical_oop == pytest.approx(4_000)          # yr1
        assert snaps[1].annual_medical_oop == pytest.approx(4_400)          # ×1.10
        assert snaps[2].annual_medical_oop == pytest.approx(4_840)          # ×1.10²

    def test_self_ltc_compounds_at_healthcare_rate(self):
        # Death at 82 (projection year 3); LTC over the final 3 years = ages 80-82.
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(annual_self_ltc_cost=100_000, self_ltc_years_before_death=3),
            investments=self._inv(),
            retirement=_retirement(80, auto_retire=False, life_expectancy_age=82),
            projection_years=3,
        )
        snaps = _project(plan)
        assert snaps[0].annual_self_ltc_cost == pytest.approx(100_000)      # yr1 age 80
        assert snaps[2].annual_self_ltc_cost == pytest.approx(121_000)      # ×1.10²

    def test_health_premium_tracks_healthcare_but_life_tracks_general(self):
        # general 0, healthcare 10%: health premium grows, life stays flat.
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(annual_health_insurance_premium=5_000,
                                     annual_life_insurance_premium=1_000),
            investments=self._inv(),
            projection_years=3,
        )
        snaps = _project(plan)
        assert snaps[0].annual_insurance_premiums == pytest.approx(6_000)   # 5000 + 1000
        assert snaps[2].annual_insurance_premiums == pytest.approx(5_000 * 1.21 + 1_000)

    def test_medicare_premium_compounds_at_healthcare_rate(self):
        # Retired at 65 (Medicare age), modest spending so no IRMAA; base premium only.
        plan = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(),
            investments=self._inv(),
            retirement=_retirement(65, retirement_age=65, auto_retire=True,
                                   annual_medicare_premium=2_000),
            projection_years=3,
        )
        snaps = _project(plan)
        assert snaps[0].annual_medicare_cost == pytest.approx(2_000)        # yr1 age 65
        assert snaps[2].annual_medicare_cost == pytest.approx(2_420)        # ×1.10²


# ══════════════════════════════════════════════════════════════════════════════
# 7. Life expectancy / death — truncates the projection, bounds LTC, pays benefit
# ══════════════════════════════════════════════════════════════════════════════

class TestLifeExpectancy:

    def _plan(self, current_age=80, life_expectancy_age=None, projection_years=20, **life_kw):
        return make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(**life_kw),
            investments=investments(current_liquid_cash=3_000_000),
            retirement=_retirement(current_age, retirement_age=65, auto_retire=True,
                                   life_expectancy_age=life_expectancy_age),
            projection_years=projection_years,
        )

    def test_projection_truncates_at_death_year(self):
        # current_age 80, dies at 85 → lives through yrs 1..6 (ages 80..85).
        snaps = _project(self._plan(current_age=80, life_expectancy_age=85, projection_years=20))
        assert len(snaps) == 6
        assert snaps[-1].year == 6

    def test_no_life_expectancy_runs_full_horizon(self):
        snaps = _project(self._plan(current_age=80, life_expectancy_age=None, projection_years=12))
        assert len(snaps) == 12

    def test_life_expectancy_extends_horizon_beyond_projection_years(self):
        # projection_years=5 but death at 95 (yr 16) → life_expectancy governs the
        # endpoint, extending the projection so the tables run through death.
        snaps = _project(self._plan(current_age=80, life_expectancy_age=95, projection_years=5))
        assert len(snaps) == 16
        assert snaps[-1].year == 16

    def test_self_ltc_bounded_by_death(self):
        # current_age 80, death at 84 → yr5 is the death year. LTC over the final 3
        # years of life = ages 82, 83, 84 → yrs 3, 4, 5. It is anchored to death, so
        # it never charges out to the 20-year projection horizon.
        snaps = _project(self._plan(current_age=80, life_expectancy_age=84, projection_years=20,
                                    annual_self_ltc_cost=100_000, self_ltc_years_before_death=3))
        assert len(snaps) == 5
        ltc_years = [s.year for s in snaps if s.annual_self_ltc_cost > 0]
        assert ltc_years == [3, 4, 5]        # the final 3 years, bounded by death

    def test_death_benefit_pays_into_estate_in_final_year(self):
        with_benefit = _project(self._plan(current_age=80, life_expectancy_age=83, projection_years=20,
                                           annual_life_insurance_death_benefit=500_000))
        without = _project(self._plan(current_age=80, life_expectancy_age=83, projection_years=20))
        # Only the final (death) year carries the payout.
        assert with_benefit[-1].annual_life_insurance_payout == pytest.approx(500_000)
        assert all(s.annual_life_insurance_payout == 0 for s in with_benefit[:-1])
        # And it lands in net worth (fixed nominal, not inflated).
        assert (with_benefit[-1].net_worth - without[-1].net_worth) == pytest.approx(500_000, abs=1)

    def test_death_benefit_requires_death_within_horizon(self):
        # No life_expectancy_age → nobody dies → benefit never pays.
        snaps = _project(self._plan(current_age=80, life_expectancy_age=None, projection_years=6,
                                    annual_life_insurance_death_benefit=500_000))
        assert all(s.annual_life_insurance_payout == 0 for s in snaps)
