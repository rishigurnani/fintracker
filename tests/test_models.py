"""Unit tests for the data models in fintracker/models.py.

These cover the behaviour-bearing methods and properties on the model
dataclasses (the module has 14 downstream dependents but no paired test file).
Pure model logic only — engine integration is covered elsewhere.
"""
import pytest

from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile,
    ChildcarePhase, ChildcareProfile, LifestyleProfile,
    RothContributionPhase, MatchTier, EmployerMatch, InvestmentProfile,
    StrategyToggles, RetirementProfile, CollegeProfile,
    TimelineEvent, FinancialPlan,
)


class TestIncomeProfile:
    def test_total_gross_income_sums_both_earners(self):
        inc = IncomeProfile(gross_annual_income=120_000, spouse_gross_annual_income=80_000)
        assert inc.total_gross_income == 200_000

    def test_total_gross_income_solo(self):
        assert IncomeProfile(gross_annual_income=100_000).total_gross_income == 100_000

    def test_defaults(self):
        inc = IncomeProfile(gross_annual_income=1)
        assert inc.filing_status is FilingStatus.SINGLE
        assert inc.state is State.GEORGIA
        assert inc.spouse_gross_annual_income == 0.0


class TestHousingProfile:
    def _house(self, **kw):
        base = dict(home_price=400_000, down_payment=80_000, interest_rate=0.06)
        base.update(kw)
        return HousingProfile(**base)

    def test_down_payment_pct(self):
        assert self._house(home_price=400_000, down_payment=80_000).down_payment_pct == pytest.approx(0.20)

    def test_down_payment_pct_zero_home_price_is_zero(self):
        # Guard against division by zero.
        assert self._house(home_price=0, down_payment=0).down_payment_pct == 0.0

    def test_loan_amount(self):
        assert self._house(home_price=400_000, down_payment=80_000).loan_amount == 320_000

    def test_loan_amount_never_negative(self):
        # Down payment exceeding price must clamp to zero, not go negative.
        assert self._house(home_price=100_000, down_payment=150_000).loan_amount == 0.0

    def test_requires_pmi_true_below_20pct(self):
        assert self._house(home_price=400_000, down_payment=40_000).requires_pmi is True

    def test_requires_pmi_false_at_or_above_20pct(self):
        assert self._house(home_price=400_000, down_payment=80_000).requires_pmi is False

    def test_requires_pmi_false_when_renting(self):
        assert self._house(home_price=400_000, down_payment=0, is_renting=True).requires_pmi is False


class TestChildcareProfile:
    def _profile(self):
        return ChildcareProfile(phases=[
            ChildcarePhase(age_start=0, age_end=2, monthly_cost=2_500),
            ChildcarePhase(age_start=3, age_end=4, monthly_cost=1_500),
        ])

    def test_cost_within_phase(self):
        assert self._profile().monthly_cost_at_age(1) == 2_500

    def test_boundaries_are_inclusive(self):
        p = self._profile()
        assert p.monthly_cost_at_age(0) == 2_500
        assert p.monthly_cost_at_age(2) == 2_500
        assert p.monthly_cost_at_age(3) == 1_500
        assert p.monthly_cost_at_age(4) == 1_500

    def test_uncovered_age_is_zero(self):
        assert self._profile().monthly_cost_at_age(10) == 0.0

    def test_empty_profile_is_zero(self):
        assert ChildcareProfile().monthly_cost_at_age(1) == 0.0


class TestLifestyleProfile:
    def test_annual_total_aggregates_all_recurring(self):
        lif = LifestyleProfile(
            monthly_childcare=1_000, annual_pet_cost=1_200, annual_medical_oop=3_000,
            annual_vacation=5_000, monthly_other_recurring=500, annual_parent_care_cost=6_000,
        )
        # 12*1000 + 1200 + 3000 + 5000 + 12*500 + 6000
        assert lif.annual_total == pytest.approx(12_000 + 1_200 + 3_000 + 5_000 + 6_000 + 6_000)

    def test_scaled_medical_oop_no_autoscale_returns_raw(self):
        lif = LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=False)
        assert lif.scaled_medical_oop(is_married=True, num_children=3) == 3_000

    def test_scaled_medical_oop_single_no_kids_is_base(self):
        lif = LifestyleProfile(annual_medical_oop=3_000)
        assert lif.scaled_medical_oop(is_married=False, num_children=0) == 3_000

    def test_scaled_medical_oop_married_applies_multiplier(self):
        lif = LifestyleProfile(annual_medical_oop=3_000, medical_spouse_multiplier=1.8)
        assert lif.scaled_medical_oop(is_married=True, num_children=0) == pytest.approx(5_400)

    def test_scaled_medical_oop_adds_per_child(self):
        lif = LifestyleProfile(annual_medical_oop=3_000, medical_per_child_annual=1_500)
        # single: 3000 + 2*1500
        assert lif.scaled_medical_oop(is_married=False, num_children=2) == pytest.approx(6_000)

    def test_scaled_medical_oop_married_with_children(self):
        lif = LifestyleProfile(
            annual_medical_oop=3_000, medical_spouse_multiplier=1.8, medical_per_child_annual=1_500,
        )
        # 3000*1.8 + 2*1500
        assert lif.scaled_medical_oop(is_married=True, num_children=2) == pytest.approx(5_400 + 3_000)


class TestEmployerMatch:
    def test_single_tier_full_contribution(self):
        m = EmployerMatch(tiers=[MatchTier(match_pct=0.50, up_to_pct_of_salary=0.06)])
        # employee contributes well past the 6% ceiling → matched on 6% of 100k = 6000 * 50%
        assert m.compute_match(20_000, 100_000, projection_year=1) == pytest.approx(3_000)

    def test_single_tier_partial_contribution(self):
        m = EmployerMatch(tiers=[MatchTier(match_pct=0.50, up_to_pct_of_salary=0.06)])
        # employee only contributes 3000 (< 6000 ceiling) → matched on 3000 * 50%
        assert m.compute_match(3_000, 100_000, projection_year=1) == pytest.approx(1_500)

    def test_tiered_match(self):
        m = EmployerMatch(tiers=[
            MatchTier(match_pct=1.00, up_to_pct_of_salary=0.03),
            MatchTier(match_pct=0.50, up_to_pct_of_salary=0.02),
        ])
        # 100% on first 3% (3000) + 50% on next 2% (2000) = 3000 + 1000
        assert m.compute_match(20_000, 100_000, projection_year=1) == pytest.approx(4_000)

    def test_annual_cap_limits_total(self):
        m = EmployerMatch(tiers=[MatchTier(1.00, 0.10)], annual_cap=5_000.0)
        # uncapped would be 10% of 100k = 10000, capped to 5000
        assert m.compute_match(20_000, 100_000, projection_year=1) == 5_000.0

    def test_cliff_vesting_forfeits_before_vesting_year(self):
        m = EmployerMatch(tiers=[MatchTier(1.00, 0.04)], vesting_years=3)
        assert m.compute_match(10_000, 100_000, projection_year=2) == 0.0

    def test_cliff_vesting_pays_from_vesting_year(self):
        m = EmployerMatch(tiers=[MatchTier(1.00, 0.04)], vesting_years=3)
        assert m.compute_match(10_000, 100_000, projection_year=3) == pytest.approx(4_000)

    def test_profit_sharing_added_regardless_of_contribution(self):
        m = EmployerMatch(tiers=[], profit_sharing_annual=3_000.0)
        assert m.compute_match(0, 100_000, projection_year=1) == 3_000.0


class TestInvestmentProfile:
    def test_roth_flat_amount_when_no_schedule(self):
        inv = InvestmentProfile(annual_roth_ira_contribution=7_000)
        assert inv.roth_contribution_for_year(1) == 7_000
        assert inv.roth_contribution_for_year(30) == 7_000

    def test_roth_schedule_in_range(self):
        inv = InvestmentProfile(
            annual_roth_ira_contribution=7_000,  # ignored when a schedule is set
            roth_contribution_schedule=[RothContributionPhase(year_start=6, year_end=9, annual_amount=5_000)],
        )
        assert inv.roth_contribution_for_year(7) == 5_000

    def test_roth_schedule_out_of_range_is_zero(self):
        inv = InvestmentProfile(
            roth_contribution_schedule=[RothContributionPhase(year_start=6, year_end=9, annual_amount=5_000)],
        )
        assert inv.roth_contribution_for_year(3) == 0.0
        assert inv.roth_contribution_for_year(10) == 0.0

    def test_investable_cash_subtracts_one_time_expenses(self):
        inv = InvestmentProfile(current_liquid_cash=100_000, one_time_upcoming_expenses=30_000)
        assert inv.investable_cash == 70_000

    def test_investable_cash_never_negative(self):
        inv = InvestmentProfile(current_liquid_cash=20_000, one_time_upcoming_expenses=50_000)
        assert inv.investable_cash == 0.0


class TestRetirementProfile:
    def test_years_to_retirement(self):
        assert RetirementProfile(current_age=35, retirement_age=65).years_to_retirement == 30

    def test_years_to_retirement_clamps_when_past(self):
        assert RetirementProfile(current_age=70, retirement_age=65).years_to_retirement == 0


class TestModelDefaults:
    """Guards default field values relied on across the app and config layer."""

    def test_strategy_toggles_backdoor_roth_defaults_off(self):
        assert StrategyToggles().use_backdoor_roth is False

    def test_college_glide_path_defaults(self):
        c = CollegeProfile()
        assert c.early_529_return == 0.08
        assert c.late_529_return == 0.04
        assert c.glide_path_years == 10


class TestFinancialPlan:
    def _plan(self, events):
        return FinancialPlan(
            income=IncomeProfile(gross_annual_income=100_000),
            housing=HousingProfile(home_price=0, down_payment=0, interest_rate=0.0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(),
            timeline_events=events,
        )

    def test_events_for_year_filters_by_year(self):
        e1 = TimelineEvent(year=3, description="a")
        e2 = TimelineEvent(year=5, description="b")
        e3 = TimelineEvent(year=3, description="c")
        plan = self._plan([e1, e2, e3])
        assert plan.events_for_year(3) == [e1, e3]

    def test_events_for_year_empty_when_none_match(self):
        plan = self._plan([TimelineEvent(year=2, description="x")])
        assert plan.events_for_year(9) == []
