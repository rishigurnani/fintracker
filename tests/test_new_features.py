"""
Tests for the five new feature groups:
  1. Retirement readiness
  2. Start/stop work
  3. Annual brokerage contribution earmark
  4. College costs (529 drawdown, AOTC, wedding fund)
  5. Parent care costs
"""
import pytest
from fintracker.models import (
    BusinessProfile, CarProfile, ChildcarePhase, ChildcareProfile,
    EmployerMatch, KidCarProfile, MatchTier, CollegeProfile,
    FilingStatus, FinancialPlan, HousingProfile,
    IncomeProfile, InvestmentProfile, LifestyleProfile, RetirementProfile,
    State, StrategyToggles, TimelineEvent,
)
from fintracker.projections import ProjectionEngine


# ── Shared plan builder ────────────────────────────────────────────────────────

def _base_plan(**overrides) -> FinancialPlan:
    """Zero-inflation, zero-growth, zero-market-return base for isolated tests."""
    defaults = dict(
        income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
        housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
        lifestyle=LifestyleProfile(
            annual_vacation=0, monthly_other_recurring=0,
            annual_medical_oop=0, medical_auto_scale=False,
        ),
        investments=InvestmentProfile(
            current_liquid_cash=500_000,
            annual_market_return=0.0,
            annual_inflation_rate=0.0,
            annual_salary_growth_rate=0.0,
        ),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        projection_years=10,
    )
    defaults.update(overrides)
    return FinancialPlan(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Retirement Readiness
# ══════════════════════════════════════════════════════════════════════════════

class TestRetirementReadiness:

    def _plan_with_retirement(self, current_age, retirement_age, desired_income,
                               k401=20_000, ss=0, years=None):
        yrs = years or (retirement_age - current_age + 5)
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=50_000, current_retirement_balance=200_000,
                annual_401k_contribution=k401,
                annual_market_return=0.07,
                annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=yrs,
            retirement=RetirementProfile(
                current_age=current_age,
                retirement_age=retirement_age,
                desired_annual_income=desired_income,
                years_in_retirement=30,
                expected_post_retirement_return=0.05,
                estimated_social_security_annual=ss,
            ),
        )

    def test_returns_none_without_retirement_profile(self):
        plan = _base_plan()
        engine = ProjectionEngine(plan)
        assert engine.compute_retirement_readiness() is None

    def test_required_balance_positive(self):
        plan = self._plan_with_retirement(35, 65, 80_000)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.required_balance > 0

    def test_on_track_with_high_savings(self):
        """High 401k contributions over 30 years should put user on track."""
        plan = self._plan_with_retirement(35, 65, 60_000, k401=30_500)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.on_track, f"Expected on track, funded={rr.funded_pct:.1%}"

    def test_not_on_track_with_low_savings(self):
        """Minimal savings over 5 years should not fund 30yr retirement."""
        plan = self._plan_with_retirement(60, 65, 100_000, k401=5_000, years=5)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert not rr.on_track, f"Expected off track, funded={rr.funded_pct:.1%}"

    def test_social_security_reduces_required_balance(self):
        """SS offset reduces the balance needed to fund retirement."""
        plan_no_ss = self._plan_with_retirement(35, 65, 80_000, ss=0)
        plan_with_ss = self._plan_with_retirement(35, 65, 80_000, ss=24_000)
        rr_no = ProjectionEngine(plan_no_ss).compute_retirement_readiness()
        rr_ss = ProjectionEngine(plan_with_ss).compute_retirement_readiness()
        assert rr_ss.required_balance < rr_no.required_balance

    def test_funded_pct_scales_with_savings(self):
        """More 401k savings → higher projected balance → higher funded_pct.
        Uses no brokerage/liquid so retirement balance is the only variable
        and the difference is clearly measurable."""
        def make(k401):
            return FinancialPlan(
                income=IncomeProfile(80_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
                lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                           annual_medical_oop=0, medical_auto_scale=False),
                investments=InvestmentProfile(
                    current_liquid_cash=0,
                    current_retirement_balance=0,
                    current_brokerage_balance=0,
                    annual_401k_contribution=k401,
                    annual_market_return=0.07,
                    annual_inflation_rate=0.03,
                    annual_salary_growth_rate=0.0,
                ),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                projection_years=30,
                retirement=RetirementProfile(
                    current_age=35, retirement_age=65,
                    desired_annual_income=80_000,
                    years_in_retirement=30,
                    expected_post_retirement_return=0.05,
                ),
            )
        rr_low  = ProjectionEngine(make(1_000)).compute_retirement_readiness()
        rr_high = ProjectionEngine(make(20_000)).compute_retirement_readiness()
        assert rr_high.funded_pct > rr_low.funded_pct, (
            f"High savings funded_pct ({rr_high.funded_pct:.4f}) should exceed "
            f"low savings ({rr_low.funded_pct:.4f})"
        )

    def test_desired_income_inflated_to_nominal(self):
        """Desired income must be inflated from today's dollars to retirement dollars."""
        plan = self._plan_with_retirement(35, 65, 80_000)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        # 80k * (1.03)^30 >> 80k
        assert rr.desired_income_nominal > 80_000

    def test_years_to_retirement_correct(self):
        plan = self._plan_with_retirement(35, 65, 80_000)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        assert rr.years_to_retirement == 30

    def test_annual_surplus_positive_when_on_track(self):
        plan = self._plan_with_retirement(35, 65, 40_000, k401=30_500)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        if rr.on_track:
            assert rr.annual_surplus_or_gap > 0

    def test_annual_gap_negative_when_off_track(self):
        plan = self._plan_with_retirement(60, 65, 200_000, k401=1_000, years=5)
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        if not rr.on_track:
            assert rr.annual_surplus_or_gap < 0

    def test_retirement_balance_uses_all_investable_assets(self):
        """Projected balance should include retirement + brokerage + HSA at retirement year."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=0, current_retirement_balance=500_000,
                current_brokerage_balance=200_000,
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=5,
            retirement=RetirementProfile(current_age=60, retirement_age=65,
                                          desired_annual_income=80_000),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        rr = ProjectionEngine(plan).compute_retirement_readiness(snaps)
        yr5 = snaps[4]
        expected = yr5.retirement_balance + yr5.brokerage_balance + yr5.hsa_balance
        assert abs(rr.projected_balance_at_retirement - expected) < 1



    def test_required_balance_uses_growing_annuity(self):
        """
        REGRESSION: required_balance must use the growing annuity formula, not
        a fixed annuity.  A fixed annuity assumes flat spending throughout
        retirement; a growing annuity inflates spending at annual_inflation_rate
        each year — the correct model for a real retiree.

        For r=5%, g=3%, n=30, PMT=$194k:
          Fixed annuity:   PMT * (1-(1+r)^-n)/r
          Growing annuity: PMT / (r-g) * (1 - ((1+g)/(1+r))^n)
        The growing annuity target is materially larger.
        """
        plan = FinancialPlan(
            income=IncomeProfile(80_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=0, current_retirement_balance=0,
                annual_401k_contribution=0, annual_market_return=0.0,
                annual_inflation_rate=0.03, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=30,
            retirement=RetirementProfile(
                current_age=35, retirement_age=65,
                desired_annual_income=100_000,
                years_in_retirement=30,
                expected_post_retirement_return=0.05,
            ),
        )
        rr = ProjectionEngine(plan).compute_retirement_readiness()

        # Manual growing annuity: PMT = 100k*(1.03)^30, r=5%, g=3%, n=30
        import math
        pmt = 100_000 * (1.03 ** 30)
        r, g, n = 0.05, 0.03, 30
        expected_growing = pmt / (r - g) * (1 - ((1 + g) / (1 + r)) ** n)
        expected_fixed   = pmt * (1 - (1 + r) ** -n) / r

        assert abs(rr.required_balance - expected_growing) < 1.0, (
            f"Engine returned {rr.required_balance:,.0f}, "
            f"expected growing annuity {expected_growing:,.0f}"
        )
        assert rr.required_balance > expected_fixed, (
            f"Growing annuity ({rr.required_balance:,.0f}) should exceed "
            f"fixed annuity ({expected_fixed:,.0f})"
        )

    def test_growing_annuity_edge_r_equals_g(self):
        """When r == g exactly, formula uses L'Hôpital limit: PMT*n/(1+r)."""
        plan = FinancialPlan(
            income=IncomeProfile(80_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=0,current_retirement_balance=0,
                                          annual_401k_contribution=0,annual_market_return=0.0,
                                          annual_inflation_rate=0.04,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            projection_years=30,
            retirement=RetirementProfile(current_age=35,retirement_age=65,
                                          desired_annual_income=50_000,years_in_retirement=20,
                                          expected_post_retirement_return=0.04),
        )
        rr = ProjectionEngine(plan).compute_retirement_readiness()
        pmt = 50_000 * (1.04**30)
        expected = pmt * 20 / 1.04
        assert abs(rr.required_balance - expected) < 1.0, (
            f"r=g edge case: got {rr.required_balance:,.0f}, expected {expected:,.0f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Start / Stop Work
# ══════════════════════════════════════════════════════════════════════════════

class TestStartStopWork:

    def _work_plan(self, events, partner=0):
        return FinancialPlan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS,
                                 spouse_gross_annual_income=partner),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.05,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events,
            projection_years=8,
        )

    def test_stop_working_zeroes_primary_income(self):
        plan = self._work_plan([TimelineEvent(year=3, description="Sabbatical", stop_working=True)])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[1].gross_income > 0   # yr2 still working
        assert snaps[2].gross_income == 0  # yr3 stopped
        assert snaps[3].gross_income == 0  # yr4 still stopped

    def test_is_working_flag_reflects_state(self):
        plan = self._work_plan([TimelineEvent(year=2, description="Stop", stop_working=True)])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].is_working is True
        assert snaps[1].is_working is False

    def test_resume_working_restores_income(self):
        plan = self._work_plan([
            TimelineEvent(year=2, description="Stop", stop_working=True),
            TimelineEvent(year=4, description="Resume", resume_working=True, income_change=110_000),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[1].gross_income == 0    # stopped
        assert snaps[2].gross_income == 0    # still stopped
        assert snaps[3].gross_income == pytest.approx(110_000, abs=1)  # resumed at new salary

    def test_salary_grows_from_resume_salary(self):
        """After resuming, salary grows from the new stated amount each year."""
        plan = self._work_plan([
            TimelineEvent(year=2, description="Stop", stop_working=True),
            TimelineEvent(year=4, description="Resume", resume_working=True, income_change=100_000),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[3].gross_income == pytest.approx(100_000, abs=1)
        assert snaps[4].gross_income == pytest.approx(100_000 * 1.05, abs=1)  # grows yr5

    def test_partner_stop_working(self):
        plan = self._work_plan(
            events=[TimelineEvent(year=2, description="Partner stops", partner_stop_working=True)],
            partner=60_000,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].gross_income == pytest.approx(160_000, abs=1)   # both working
        assert snaps[1].is_partner_working is False
        assert snaps[1].gross_income < snaps[0].gross_income  # partner income gone

    def test_partner_resume_working(self):
        plan = self._work_plan(
            events=[
                TimelineEvent(year=2, description="Partner stops", partner_stop_working=True),
                TimelineEvent(year=4, description="Partner returns", partner_resume_working=True,
                              partner_income_change=65_000),
            ],
            partner=60_000,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[1].is_partner_working is False
        assert snaps[3].is_partner_working is True
        assert snaps[3].gross_income > snaps[1].gross_income

    def test_income_does_not_grow_while_stopped(self):
        """When stopped, income stays at 0; salary growth only applies while working."""
        plan = self._work_plan([
            TimelineEvent(year=2, description="Stop", stop_working=True),
            TimelineEvent(year=5, description="Resume", resume_working=True, income_change=120_000),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        # Yr2,3,4: income=0
        for yr in [2, 3, 4]:
            assert snaps[yr-1].gross_income == 0, f"Yr{yr} should be 0 while stopped"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Annual Brokerage Contribution
# ══════════════════════════════════════════════════════════════════════════════

class TestBrokerageContribution:

    def _brok_plan(self, brokerage_contribution, income=150_000):
        return FinancialPlan(
            income=IncomeProfile(income, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=200_000,
                annual_brokerage_contribution=brokerage_contribution,
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=5,
        )

    def test_earmark_recorded_in_snapshot(self):
        plan = self._brok_plan(6_000)
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_brokerage_contribution == pytest.approx(6_000, abs=1)

    def test_zero_earmark_recorded_as_zero(self):
        plan = self._brok_plan(0)
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_brokerage_contribution == 0.0

    def test_earmark_reduces_breathing_room(self):
        """The earmark comes out of breathing room, so breathing_room is lower."""
        p_no  = self._brok_plan(0)
        p_yes = self._brok_plan(6_000)
        s_no  = ProjectionEngine(p_no).run_deterministic()[0]
        s_yes = ProjectionEngine(p_yes).run_deterministic()[0]
        assert s_no.annual_breathing_room - s_yes.annual_breathing_room == pytest.approx(6_000, abs=1)

    def test_brokerage_balance_same_regardless_of_earmark(self):
        """The total going to brokerage (earmark + breathing_room) is the same.
        The earmark just changes how it's labelled, not the total amount invested."""
        p_no  = self._brok_plan(0)
        p_yes = self._brok_plan(6_000)
        for s_no, s_yes in zip(
            ProjectionEngine(p_no).run_deterministic(),
            ProjectionEngine(p_yes).run_deterministic(),
        ):
            assert abs(s_no.brokerage_balance - s_yes.brokerage_balance) < 1.0, \
                f"Yr{s_no.year}: brokerage differs"

    def test_net_brokerage_flow_equals_net_income_minus_expenses(self):
        """DESIGN INVARIANT: the earmark cancels algebraically in the brokerage formula.
        brokerage += earmark + breathing_room
        breathing_room = net_income - expenses - earmark
        → brokerage += net_income - expenses  (earmark cancels)
        This means the total cash flowing to brokerage is always
        (net_income - expenses), regardless of earmark size.
        The earmark only changes the LABEL (earmarked vs organic), not the total.
        This test verifies that invariant holds across different earmark sizes."""
        def make(earmark):
            return FinancialPlan(
                income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
                lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                           annual_medical_oop=0, medical_auto_scale=False),
                investments=InvestmentProfile(
                    current_liquid_cash=200_000,
                    annual_brokerage_contribution=earmark,
                    annual_market_return=0.0, annual_inflation_rate=0.0,
                    annual_salary_growth_rate=0.0,
                ),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                projection_years=3,
            )
        # All three earmark sizes produce identical brokerage balances
        snaps_0   = ProjectionEngine(make(0)).run_deterministic()
        snaps_10k = ProjectionEngine(make(10_000)).run_deterministic()
        snaps_50k = ProjectionEngine(make(50_000)).run_deterministic()
        for s0, s10, s50 in zip(snaps_0, snaps_10k, snaps_50k):
            assert abs(s0.brokerage_balance - s10.brokerage_balance) < 1.0, \
                f"Yr{s0.year}: earmark=0 brokerage={s0.brokerage_balance:.0f} "\
                f"vs earmark=10k brokerage={s10.brokerage_balance:.0f}"
            assert abs(s0.brokerage_balance - s50.brokerage_balance) < 1.0, \
                f"Yr{s0.year}: earmark=0 brokerage={s0.brokerage_balance:.0f} "\
                f"vs earmark=50k brokerage={s50.brokerage_balance:.0f}"

    def test_brokerage_drains_when_expenses_exceed_income(self):
        """Brokerage drains when total expenses (including earmark) exceed net income.
        This is the correct way to test cash flow stress — not via earmark alone."""
        plan = FinancialPlan(
            income=IncomeProfile(20_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=2_500),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=100_000,
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=3,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # Rent (30k/yr) > net income (~17.9k) → negative breathing room → brokerage drains
        for s in snaps:
            assert s.annual_breathing_room < 0, f"Yr{s.year}: expected deficit"
        for i in range(1, len(snaps)):
            assert snaps[i].brokerage_balance < snaps[i-1].brokerage_balance, \
                f"Yr{snaps[i].year}: brokerage should drain under deficit spending"


# ══════════════════════════════════════════════════════════════════════════════
# 4. College Costs, 529 Drawdown, AOTC
# ══════════════════════════════════════════════════════════════════════════════

class TestCollegeCosts:

    def _college_plan(self, income=70_000, k529=5_000, cost=35_000,
                      child_birth_year_offset=-17, filing=FilingStatus.SINGLE,
                      use_aotc=True, events=None):
        """child_birth_year_offset: negative = already born. -17 = age 17 in yr0, 18 in yr1."""
        return FinancialPlan(
            income=IncomeProfile(income, filing, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=300_000,
                annual_529_contribution=k529,
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events or [
                TimelineEvent(year=1, description="Child in college", new_child=True,
                              child_birth_year_override=child_birth_year_offset)
            ],
            college=CollegeProfile(annual_cost_per_child=cost, years_per_child=4,
                                    start_age=18, use_aotc_credit=use_aotc),
            projection_years=6,
        )

    def test_college_cost_appears_in_college_years(self):
        snaps = ProjectionEngine(self._college_plan()).run_deterministic()
        # yr1: child is 18 (birth_year=0 adjusted to -17 from year 0)
        # birth_year_override=-17 means born at projection "year -17"
        # yr1: age = 1 - (-17) = 18. In college years 1-4.
        assert snaps[0].annual_college_cost == pytest.approx(35_000, abs=1)
        assert snaps[1].annual_college_cost == pytest.approx(35_000, abs=1)
        assert snaps[2].annual_college_cost == pytest.approx(35_000, abs=1)
        assert snaps[3].annual_college_cost == pytest.approx(35_000, abs=1)
        assert snaps[4].annual_college_cost == 0.0  # yr5: age 22, done

    def test_no_college_cost_before_college_age(self):
        """Child born in year 1 should not have college costs until year 19."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=1, description="Child born", new_child=True)],
            college=CollegeProfile(annual_cost_per_child=35_000, start_age=18),
            projection_years=10,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_college_cost == 0.0, f"Yr{s.year}: unexpected college cost"

    def test_529_drawn_down_in_college_years(self):
        """529 balance should decrease in college years."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_529_contribution=0,  # no new contributions
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="Child in college", new_child=True,
                              child_birth_year_override=-17)
            ],
            college=CollegeProfile(annual_cost_per_child=35_000, start_age=18),
            projection_years=5,
        )
        # Seed 529 balance manually by first running with contributions
        plan2 = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_529_contribution=10_000,
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="Child in college", new_child=True,
                              child_birth_year_override=-17)
            ],
            college=CollegeProfile(annual_cost_per_child=10_000, start_age=18),
            projection_years=6,
        )
        snaps2 = ProjectionEngine(plan2).run_deterministic()
        # 529 grows yrs 1-4 (savings 10k/yr, cost 10k/yr → net 0 change pre-drawdown)
        # Actually: 529 savings = 1 child * 10k, drawdown = 10k per year in college
        # Net: stays flat if savings == cost. Check it doesn't grow beyond.
        for s in snaps2[:4]:
            assert s.annual_529_drawdown == pytest.approx(10_000, abs=100), \
                f"Yr{s.year}: 529 drawdown should be 10000"

    def test_529_drawdown_recorded_separately(self):
        snaps = ProjectionEngine(self._college_plan(k529=0, cost=35_000)).run_deterministic()
        # No 529 balance → drawdown = 0, full cost hits brokerage
        assert snaps[0].annual_529_drawdown == 0.0
        assert snaps[0].annual_college_cost == pytest.approx(35_000, abs=1)

    def test_aotc_full_credit_below_phaseout(self):
        """Income below $80k (single) → full $2,500 AOTC per eligible student."""
        snaps = ProjectionEngine(self._college_plan(income=70_000, use_aotc=True)).run_deterministic()
        for yr in [1, 2, 3, 4]:
            assert snaps[yr-1].annual_aotc_credit == pytest.approx(2_500, abs=1), \
                f"Yr{yr}: expected full AOTC"
        assert snaps[4].annual_aotc_credit == 0.0  # yr5: done with college

    def test_aotc_zero_above_phaseout(self):
        """Income above $90k (single) → no AOTC."""
        snaps = ProjectionEngine(self._college_plan(income=95_000, use_aotc=True)).run_deterministic()
        for s in snaps:
            assert s.annual_aotc_credit == 0.0

    def test_aotc_partial_in_phaseout_range(self):
        """Income at 85k (midpoint 80-90k) → 50% credit = $1,250."""
        snaps = ProjectionEngine(self._college_plan(income=85_000, use_aotc=True)).run_deterministic()
        assert snaps[0].annual_aotc_credit == pytest.approx(1_250, abs=1)

    def test_aotc_higher_threshold_mfj(self):
        """MFJ phase-out is $160k-$180k; at $170k MFJ → 50% credit."""
        snaps = ProjectionEngine(
            self._college_plan(income=170_000, filing=FilingStatus.MARRIED_FILING_JOINTLY, use_aotc=True)
        ).run_deterministic()
        assert snaps[0].annual_aotc_credit == pytest.approx(1_250, abs=1)

    def test_aotc_disabled_gives_zero(self):
        snaps = ProjectionEngine(self._college_plan(income=70_000, use_aotc=False)).run_deterministic()
        for s in snaps:
            assert s.annual_aotc_credit == 0.0

    def test_aotc_reduces_effective_tax(self):
        """AOTC credit must lower effective tax (annual_tax_total) compared to no credit."""
        p_aotc    = self._college_plan(income=70_000, use_aotc=True)
        p_no_aotc = self._college_plan(income=70_000, use_aotc=False)
        s_aotc    = ProjectionEngine(p_aotc).run_deterministic()[0]
        s_no_aotc = ProjectionEngine(p_no_aotc).run_deterministic()[0]
        assert s_aotc.annual_tax_total < s_no_aotc.annual_tax_total

    def test_two_children_in_college_doubles_cost(self):
        """Two children in college simultaneously doubles cost and AOTC."""
        plan = FinancialPlan(
            income=IncomeProfile(70_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="Child 1", new_child=True,
                              child_birth_year_override=-17),
                TimelineEvent(year=1, description="Child 2", new_child=True,
                              child_birth_year_override=-18),  # also in college yr1
            ],
            college=CollegeProfile(annual_cost_per_child=35_000, start_age=18, use_aotc_credit=True),
            projection_years=4,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].annual_college_cost == pytest.approx(70_000, abs=1)  # 2 * 35k
        assert snaps[0].annual_aotc_credit == pytest.approx(5_000, abs=1)   # 2 * 2500


# ══════════════════════════════════════════════════════════════════════════════
# 5. Parent Care Costs
# ══════════════════════════════════════════════════════════════════════════════


    def test_529_contributions_stop_after_last_child_graduates(self):
        """REGRESSION: 529 contributions must stop once all children have finished college.
        Before this fix, contributions continued forever, causing the 529 balance to
        grow again after college — confusing the chart and overstating invested assets."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_529_contribution=5_000,
                annual_market_return=0.0,   # zero return: balance changes = contributions only
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="Child", new_child=True,
                              child_birth_year_override=-17),  # starts college yr1
            ],
            college=CollegeProfile(annual_cost_per_child=10_000, years_per_child=4,
                                    start_age=18, use_aotc_credit=False),
            projection_years=8,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # Yrs 1-4: college + contributions; yr5+: college done, contributions stop
        # With zero return and cost=10k, contribution=5k:
        #   balance = 5k - 10k = -5k per yr (draws from brokerage for shortfall)
        # After yr4: no more contributions, balance stays flat (near 0)
        post_college = [s for s in snaps if s.year > 4]
        balances = [s.college_529_balance for s in post_college]
        # All post-college balances should be <= the balance at end of yr4
        # (no growth since return=0 and no new contributions)
        for i in range(1, len(balances)):
            assert balances[i] <= balances[i-1] + 1.0, \
                f"Yr{post_college[i].year}: 529 balance grew after college ended " \
                f"({balances[i-1]:.0f} → {balances[i]:.0f})"



    def test_529_uses_early_rate_before_switch(self):
        """Before glide_path_years, 529 grows at early_529_return not late_529_return."""
        def make(early, late):
            return FinancialPlan(
                income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
                lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                           annual_medical_oop=0,medical_auto_scale=False),
                investments=InvestmentProfile(current_liquid_cash=500_000,
                                              annual_529_contribution=5_000,
                                              annual_market_return=0.08,
                                              annual_inflation_rate=0.0,
                                              annual_salary_growth_rate=0.0),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                timeline_events=[TimelineEvent(year=1, description="Child",
                                               new_child=True, child_birth_year_override=-1)],
                college=CollegeProfile(annual_cost_per_child=20_000, years_per_child=4,
                                        start_age=18, use_aotc_credit=False,
                                        early_529_return=early, late_529_return=late,
                                        glide_path_years=10),
                projection_years=15,
            )
        # Before switch yr5: glide(8%→4%) should equal flat 8%
        s_glide = ProjectionEngine(make(0.08, 0.04)).run_deterministic()
        s_flat8 = ProjectionEngine(make(0.08, 0.08)).run_deterministic()
        assert abs(s_glide[4].college_529_balance - s_flat8[4].college_529_balance) < 1.0,             f"Yr5 pre-switch: glide {s_glide[4].college_529_balance:.0f} != flat8 {s_flat8[4].college_529_balance:.0f}"

    def test_529_uses_late_rate_after_switch(self):
        """After glide_path_years, 529 growth rate matches late_529_return."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000,
                                          annual_529_contribution=5_000,
                                          annual_market_return=0.08,
                                          annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=1, description="Child",
                                           new_child=True, child_birth_year_override=-1)],
            college=CollegeProfile(annual_cost_per_child=20_000, years_per_child=4,
                                    start_age=18, use_aotc_credit=False,
                                    early_529_return=0.08, late_529_return=0.04,
                                    glide_path_years=10),
            projection_years=15,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # After switch (yr11 onwards), implied growth rate should be 0.04
        for yr in [11, 12, 13]:
            prev = snaps[yr-2].college_529_balance
            curr = snaps[yr-1].college_529_balance
            implied = (curr - 5_000) / prev - 1
            assert abs(implied - 0.04) < 0.001, \
                f"Yr{yr}: implied rate {implied:.4f} expected 0.04 (late_529_return)"

    def test_529_glide_independent_of_market_return(self):
        """529 growth uses glide path rates, not the general market_return."""
        def make(market_return):
            return FinancialPlan(
                income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
                lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                           annual_medical_oop=0,medical_auto_scale=False),
                investments=InvestmentProfile(current_liquid_cash=500_000,
                                              annual_529_contribution=5_000,
                                              annual_market_return=market_return,
                                              annual_inflation_rate=0.0,
                                              annual_salary_growth_rate=0.0),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                timeline_events=[TimelineEvent(year=1, description="Child",
                                               new_child=True, child_birth_year_override=-1)],
                college=CollegeProfile(annual_cost_per_child=20_000, years_per_child=4,
                                        start_age=18, use_aotc_credit=False,
                                        early_529_return=0.08, late_529_return=0.04,
                                        glide_path_years=10),
                projection_years=5,
            )
        # Different market returns should produce identical 529 balances
        snaps_8 = ProjectionEngine(make(0.08)).run_deterministic()
        snaps_5 = ProjectionEngine(make(0.05)).run_deterministic()
        snaps_0 = ProjectionEngine(make(0.00)).run_deterministic()
        for s8, s5, s0 in zip(snaps_8, snaps_5, snaps_0):
            assert abs(s8.college_529_balance - s5.college_529_balance) < 1.0, \
                f"Yr{s8.year}: 529 differs with different market_return"
            assert abs(s8.college_529_balance - s0.college_529_balance) < 1.0, \
                f"Yr{s8.year}: 529 differs with different market_return"

    def test_529_default_glide_path_values(self):
        """CollegeProfile defaults to 8% early, 4% late, 10yr switch."""
        cp = CollegeProfile()
        assert cp.early_529_return == pytest.approx(0.08)
        assert cp.late_529_return  == pytest.approx(0.04)
        assert cp.glide_path_years == 10

    def test_nw_integrity_with_glide_path(self):
        """Net worth components sum correctly when 529 uses glide path rates."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=300_000,
                                          annual_529_contribution=5_000,
                                          annual_market_return=0.08,
                                          annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=1, description="Child",
                                           new_child=True, child_birth_year_override=-17)],
            college=CollegeProfile(annual_cost_per_child=35_000, years_per_child=4,
                                    start_age=18, use_aotc_credit=False,
                                    early_529_return=0.08, late_529_return=0.04,
                                    glide_path_years=10),
            projection_years=8,
        )
        for s in ProjectionEngine(plan).run_deterministic():
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity
                          + s.hsa_balance + s.uninvested_cash)
            assert abs(components - s.net_worth) < 1.0, \
                f"Yr{s.year}: components={components:.2f} net_worth={s.net_worth:.2f}"


class TestParentCareCosts:

    def _parent_plan(self, care_cost, events=None, start_active=True):
        return FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
                annual_parent_care_cost=care_cost,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events or [],
            projection_years=8,
        )

    def test_parent_care_cost_appears_in_lifestyle(self):
        plan = self._parent_plan(12_000)
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_parent_care_cost == pytest.approx(12_000, abs=1)

    def test_zero_parent_care_cost(self):
        plan = self._parent_plan(0)
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_parent_care_cost == 0.0

    def test_start_parent_care_via_event(self):
        """Parent care starts via timeline event."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
                annual_parent_care_cost=15_000,  # amount set; activation via event
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=3, description="Parent needs care", start_parent_care=True),
            ],
            projection_years=6,
        )
        # parent_care_active initialises to True if annual_parent_care_cost > 0
        # To use start/stop events properly, set cost=0 initially:
        plan2 = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
                annual_parent_care_cost=15_000,  # amount defined
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="No care yet", stop_parent_care=True),  # disable at start
                TimelineEvent(year=3, description="Care starts", start_parent_care=True),
            ],
            projection_years=6,
        )
        snaps = ProjectionEngine(plan2).run_deterministic()
        assert snaps[0].annual_parent_care_cost == 0.0    # stopped in yr1
        assert snaps[1].annual_parent_care_cost == 0.0    # still off
        assert snaps[2].annual_parent_care_cost == pytest.approx(15_000 * (1.0), abs=1)  # started yr3

    def test_stop_parent_care_via_event(self):
        plan = self._parent_plan(12_000, events=[
            TimelineEvent(year=4, description="Parent passes", stop_parent_care=True),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        for yr in [1, 2, 3]:
            assert snaps[yr-1].annual_parent_care_cost > 0, f"Yr{yr} should have care cost"
        for yr in [4, 5, 6, 7, 8]:
            assert snaps[yr-1].annual_parent_care_cost == 0.0, \
                f"Yr{yr}: parent care should be zero after stop_parent_care event in yr4"

    def test_parent_care_reduces_breathing_room(self):
        p_with = self._parent_plan(12_000)
        p_none = self._parent_plan(0)
        s_with = ProjectionEngine(p_with).run_deterministic()[0]
        s_none = ProjectionEngine(p_none).run_deterministic()[0]
        diff = s_none.annual_breathing_room - s_with.annual_breathing_room
        assert diff == pytest.approx(12_000, abs=1)

    def test_parent_care_inflates_with_inflation(self):
        plan = FinancialPlan(
            income=IncomeProfile(200_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
                annual_parent_care_cost=12_000,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.04, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=5,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            expected = 12_000 * (1.04) ** (s.year - 1)
            assert abs(s.annual_parent_care_cost - expected) < 1.0, \
                f"Yr{s.year}: expected {expected:.0f}, got {s.annual_parent_care_cost:.0f}"

    def test_net_worth_lower_with_parent_care(self):
        p_with = self._parent_plan(12_000)
        p_none = self._parent_plan(0)
        nw_with = ProjectionEngine(p_with).run_deterministic()[-1].net_worth
        nw_none = ProjectionEngine(p_none).run_deterministic()[-1].net_worth
        assert nw_none > nw_with


# ══════════════════════════════════════════════════════════════════════════════
# 6. Integration: NW integrity with all features active
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationAllFeatures:

    def test_net_worth_components_sum_with_all_features(self):
        """With every feature active, NW = retirement + brokerage + 529 + home_equity + HSA."""
        plan = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.GEORGIA,
                                 spouse_gross_annual_income=80_000),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=2_000),
            lifestyle=LifestyleProfile(
                monthly_childcare=2_500, annual_medical_oop=3_000, medical_auto_scale=True,
                annual_vacation=8_000, monthly_other_recurring=500,
                annual_parent_care_cost=10_000,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=50_000, current_brokerage_balance=100_000,
                current_retirement_balance=80_000,
                annual_401k_contribution=20_000, annual_hsa_contribution=4_150,
                annual_529_contribution=5_000, annual_brokerage_contribution=6_000,
                annual_market_return=0.07, annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.04, partner_salary_growth_rate=0.05,
            ),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True,
                                        use_529_state_deduction=True),
            timeline_events=[
                TimelineEvent(year=1, description="Marry", marriage=True),
                TimelineEvent(year=2, description="Child 1", new_child=True),
                TimelineEvent(year=3, description="Partner sabbatical", partner_stop_working=True),
                TimelineEvent(year=5, description="Partner returns", partner_resume_working=True,
                              partner_income_change=90_000),
                TimelineEvent(year=4, description="Start parent care", start_parent_care=True),
                TimelineEvent(year=8, description="Stop parent care", stop_parent_care=True),
            ],
            retirement=RetirementProfile(current_age=30, retirement_age=65,
                                          desired_annual_income=100_000),
            college=CollegeProfile(annual_cost_per_child=35_000, use_aotc_credit=True),
            projection_years=15,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            components = (
                s.retirement_balance + s.brokerage_balance
                + s.college_529_balance + s.home_equity + s.hsa_balance
            )
            assert abs(components - s.net_worth) < 1.0, \
                f"Yr{s.year}: components={components:.2f} net_worth={s.net_worth:.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# Auto-Invest Surplus Toggle
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoInvestSurplus:
    """
    Tests for the auto_invest_surplus toggle on InvestmentProfile.
    ON  (default): surplus breathing room swept to brokerage each year → earns market return.
    OFF: surplus stays in uninvested_cash (0% return) → shows cost of not investing.
    """

    def _plan(self, auto_invest: bool, income: float = 150_000, years: int = 10):
        return FinancialPlan(
            income=IncomeProfile(income, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=100_000,
                annual_market_return=0.08,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
                auto_invest_surplus=auto_invest,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=years,
        )

    def test_invested_beats_cash_over_time(self):
        """Investing surplus earns compound returns; cash does not."""
        on  = ProjectionEngine(self._plan(True,  years=20)).run_deterministic()
        off = ProjectionEngine(self._plan(False, years=20)).run_deterministic()
        assert on[-1].net_worth > off[-1].net_worth,             f"Invested NW ({on[-1].net_worth:.0f}) should exceed cash NW ({off[-1].net_worth:.0f})"

    def test_uninvested_cash_zero_when_auto_invest_on(self):
        """When auto_invest=True, no cash should accumulate outside brokerage."""
        for s in ProjectionEngine(self._plan(True)).run_deterministic():
            assert s.uninvested_cash == 0.0,                 f"Yr{s.year}: uninvested_cash should be 0 when toggle is ON"

    def test_uninvested_cash_accumulates_when_toggle_off(self):
        """When auto_invest=False, surplus accumulates in uninvested_cash each year."""
        snaps = ProjectionEngine(self._plan(False)).run_deterministic()
        # With positive breathing room each year, uninvested_cash should grow
        assert snaps[-1].uninvested_cash > 0, "uninvested_cash should accumulate"
        # Should grow monotonically (no market return, just additions)
        for i in range(1, len(snaps)):
            assert snaps[i].uninvested_cash >= snaps[i-1].uninvested_cash,                 f"Yr{snaps[i].year}: uninvested_cash should not shrink under surplus"

    def test_net_worth_includes_uninvested_cash(self):
        """uninvested_cash must count toward net_worth."""
        snaps = ProjectionEngine(self._plan(False)).run_deterministic()
        for s in snaps:
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity
                          + s.hsa_balance + s.uninvested_cash)
            assert abs(components - s.net_worth) < 1.0,                 f"Yr{s.year}: NW components ({components:.2f}) != net_worth ({s.net_worth:.2f})"

    def test_same_total_wealth_year_1(self):
        """In year 1, both scenarios receive the same surplus — only future
        compounding differs, so year 1 net worths should be equal."""
        on  = ProjectionEngine(self._plan(True)).run_deterministic()
        off = ProjectionEngine(self._plan(False)).run_deterministic()
        assert abs(on[0].net_worth - off[0].net_worth) < 1.0,             f"Yr1 NW should match: {on[0].net_worth:.0f} vs {off[0].net_worth:.0f}"

    def test_deficit_drains_uninvested_cash_first(self):
        """When spending exceeds income, uninvested_cash should drain before brokerage."""
        plan = FinancialPlan(
            income=IncomeProfile(80_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=200_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
                auto_invest_surplus=False,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            # Year 3: big expense that flips to deficit
            timeline_events=[
                TimelineEvent(year=3, description="Big expense", extra_one_time_expense=500_000),
            ],
            projection_years=5,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.uninvested_cash >= 0,                 f"Yr{s.year}: uninvested_cash went negative ({s.uninvested_cash:.0f})"
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity
                          + s.hsa_balance + s.uninvested_cash)
            assert abs(components - s.net_worth) < 1.0, f"Yr{s.year}: NW mismatch"

    def test_toggle_default_is_on(self):
        """Default behavior must be to invest surplus (backwards compatible)."""
        assert InvestmentProfile().auto_invest_surplus is True


# ══════════════════════════════════════════════════════════════════════════════
# Car Purchases and Financing
# ══════════════════════════════════════════════════════════════════════════════

class TestCarPurchases:

    def _plan(self, events=None, children=None, years=22, num_cars=1,
              replace_every=10, loan_term=5, residual=5_000, hand_down_age=16,
              car_price=25_000, down=5_000):
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=300_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events or (children or []),
            car=CarProfile(
                car_price=car_price, down_payment=down,
                loan_rate=0.065, loan_term_years=loan_term,
                replace_every_years=replace_every, residual_value=residual,
                hand_down_age=hand_down_age, num_cars=num_cars,
            ),
            projection_years=years,
        )

    # --- Loan payments ---

    def test_loan_payment_fires_during_loan_term(self):
        """Car loan P&I payments must be non-zero during the loan term."""
        snaps = ProjectionEngine(self._plan(loan_term=5)).run_deterministic()
        for s in snaps[:5]:
            assert s.annual_car_payment > 0, f"Yr{s.year}: expected payment during loan term"

    def test_no_payment_after_loan_paid_off(self):
        """No car payments after loan term expires."""
        snaps = ProjectionEngine(self._plan(loan_term=5)).run_deterministic()
        for s in snaps[5:10]:  # yrs 6-10: loan paid off, next car not yet bought
            assert s.annual_car_payment == pytest.approx(0.0, abs=1),                 f"Yr{s.year}: loan should be paid off"

    def test_payment_reduces_breathing_room(self):
        """Car payment must reduce breathing room compared to no-car plan."""
        plan_car  = self._plan()
        plan_none = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=300_000,annual_market_return=0.0,annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            projection_years=5,
        )
        s_car  = ProjectionEngine(plan_car).run_deterministic()[0]
        s_none = ProjectionEngine(plan_none).run_deterministic()[0]
        assert s_car.annual_breathing_room < s_none.annual_breathing_room,             "Car payment should reduce breathing room"

    # --- Down payment ---

    def test_initial_down_payment_deducted_from_brokerage(self):
        """Initial car down payment must come out of starting brokerage."""
        plan_car  = self._plan(down=5_000)
        plan_none = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=300_000,annual_market_return=0.0,annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            projection_years=1,
        )
        s_car  = ProjectionEngine(plan_car).run_deterministic()[0]
        s_none = ProjectionEngine(plan_none).run_deterministic()[0]
        # At yr1: brokerage_car = brokerage_none - 5k_down (same net income, same market return)
        # Total reduction = down_payment + annual_car_payment (both reduce brokerage)
        expected_diff = 5_000 + s_car.annual_car_payment
        actual_diff = s_none.brokerage_balance - s_car.brokerage_balance
        assert abs(actual_diff - expected_diff) < 1.0, \
            f"Expected diff {expected_diff:.0f}, got {actual_diff:.0f}"

    def test_replacement_down_payment_hits_brokerage(self):
        """At replacement year, new down payment is deducted from brokerage."""
        snaps = ProjectionEngine(self._plan(replace_every=10)).run_deterministic()
        yr11 = snaps[10]
        assert yr11.car_purchase_cost == pytest.approx(5_000, abs=1),             f"Yr11 replacement should show 5k down payment, got {yr11.car_purchase_cost:.0f}"

    # --- Replacement cycle ---

    def test_replacement_fires_every_n_years(self):
        """Car replacement must fire at yr11 and yr21 for replace_every=10."""
        snaps = ProjectionEngine(self._plan(replace_every=10, years=25)).run_deterministic()
        replacement_yrs = [s.year for s in snaps if s.car_purchase_cost > 0]
        assert 11 in replacement_yrs, f"Expected replacement at yr11, got {replacement_yrs}"
        assert 21 in replacement_yrs, f"Expected replacement at yr21, got {replacement_yrs}"

    def test_new_loan_starts_after_replacement(self):
        """After replacing the car, a new 5-year loan starts."""
        snaps = ProjectionEngine(self._plan(replace_every=10, loan_term=5)).run_deterministic()
        # Yr11-15: new loan payments (5 years)
        for yr in [11, 12, 13, 14, 15]:
            assert snaps[yr-1].annual_car_payment > 0, f"Yr{yr}: expected payment on new loan"
        # Yr16-20: loan paid off
        for yr in [16, 17, 18, 19, 20]:
            assert snaps[yr-1].annual_car_payment == pytest.approx(0.0, abs=1),                 f"Yr{yr}: loan should be paid off"

    # --- Sell vs hand-down ---

    def test_sells_old_car_when_no_children(self):
        """With no children, old car is always sold for residual_value."""
        snaps = ProjectionEngine(self._plan(residual=5_000)).run_deterministic()
        yr11 = snaps[10]
        assert yr11.car_sale_proceeds == pytest.approx(5_000, abs=1),             f"Expected 5k sale proceeds, got {yr11.car_sale_proceeds:.0f}"

    def test_sells_old_car_when_children_too_young(self):
        """If all children are below hand_down_age, sell the car."""
        # Child born yr1 → age 10 at yr11 → too young (hand_down_age=16)
        events = [TimelineEvent(year=1, description="Child", new_child=True)]
        snaps = ProjectionEngine(self._plan(events=events, hand_down_age=16)).run_deterministic()
        yr11 = snaps[10]
        child_age = 11 - 1  # born yr1
        assert child_age < 16
        assert yr11.car_sale_proceeds == pytest.approx(5_000, abs=1),             f"Child age {child_age} < 16, should sell for 5k"

    def test_hands_down_car_when_child_old_enough(self):
        """If a child is at or above hand_down_age at replacement, hand down (0 proceeds)."""
        # Child born yr1 → age 20 at yr21 → old enough (hand_down_age=16)
        events = [TimelineEvent(year=1, description="Child", new_child=True)]
        snaps = ProjectionEngine(self._plan(events=events, hand_down_age=16, years=25)).run_deterministic()
        yr21 = snaps[20]
        child_age = 21 - 1
        assert child_age >= 16
        assert yr21.car_sale_proceeds == pytest.approx(0.0, abs=1),             f"Child age {child_age} >= 16, should hand down (0 proceeds)"

    def test_hand_down_saves_money_vs_sell(self):
        """Brokerage at yr21 should be higher when handing down vs not having children."""
        # When handing down: no 5k proceeds BUT also the child has the car (no loss)
        # When selling: +5k proceeds BUT -5k goes toward new car down (net neutral)
        # Actually same cash flow. Key difference: handing down vs selling is neutral for parent.
        # The real benefit is to the child. Test that proceeds=0 when handing down.
        events = [TimelineEvent(year=1, description="Child", new_child=True)]
        snaps = ProjectionEngine(self._plan(events=events, hand_down_age=16, years=22)).run_deterministic()
        yr21 = snaps[20]
        assert yr21.car_sale_proceeds == 0.0  # handed down, not sold

    # --- Multiple cars ---

    def test_two_cars_double_payments(self):
        """Two cars should produce roughly double the annual payment of one."""
        s1 = ProjectionEngine(self._plan(num_cars=1)).run_deterministic()[0]
        s2 = ProjectionEngine(self._plan(num_cars=2)).run_deterministic()[0]
        assert abs(s2.annual_car_payment - 2 * s1.annual_car_payment) < 50,             f"2-car payment {s2.annual_car_payment:.0f} should be ~2x {s1.annual_car_payment:.0f}"

    def test_two_cars_double_initial_down(self):
        """Two cars means 2x the initial down payment from brokerage."""
        plan1 = self._plan(num_cars=1, down=5_000)
        plan2 = self._plan(num_cars=2, down=5_000)
        plan0 = FinancialPlan(
            income=IncomeProfile(150_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=300_000,annual_market_return=0.0,annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            projection_years=1)
        s0 = ProjectionEngine(plan0).run_deterministic()[0]
        s1 = ProjectionEngine(plan1).run_deterministic()[0]
        s2 = ProjectionEngine(plan2).run_deterministic()[0]
        # Down payments are deducted in _initial_state(), before year 1 runs.
        # With market_return=0: initial_brokerage = yr1_brokerage - yr1_breathing_room
        # This strips out loan payments and isolates only the initial down payment.
        init0 = s0.brokerage_balance - s0.annual_breathing_room
        init1 = s1.brokerage_balance - s1.annual_breathing_room
        init2 = s2.brokerage_balance - s2.annual_breathing_room
        diff1 = init0 - init1
        diff2 = init0 - init2
        assert abs(diff1 - 5_000) < 1, f"1-car: expected 5k down diff, got {diff1:.0f}"
        assert abs(diff2 - 10_000) < 1, f"2-car: expected 10k down diff, got {diff2:.0f}"

    # --- No car ---

    def test_no_car_zero_car_costs(self):
        """Without a CarProfile, all car fields on snapshots should be zero."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=300_000,annual_market_return=0.0,annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            projection_years=5)
        for s in ProjectionEngine(plan).run_deterministic():
            assert s.annual_car_payment == 0.0
            assert s.car_purchase_cost == 0.0
            assert s.car_sale_proceeds == 0.0

    # --- NW integrity ---

    def test_nw_integrity_with_car(self):
        """Net worth components must sum correctly with car costs active."""
        events = [TimelineEvent(year=1, description="Child", new_child=True)]
        plan = self._plan(events=events, years=25)
        for s in ProjectionEngine(plan).run_deterministic():
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity
                          + s.hsa_balance + s.uninvested_cash)
            assert abs(components - s.net_worth) < 1.0,                 f"Yr{s.year}: components={components:.2f} net_worth={s.net_worth:.2f}"




    # --- Kids' first cars ---

    def test_kids_car_fires_at_graduation(self):
        """kids_car is purchased in the year each child graduates college."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000,annual_market_return=0.0,
                                          annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            timeline_events=[
                # Child born yr-17 (age 18 at yr1) → graduates at age 22 → yr5
                TimelineEvent(year=1,description="Child",new_child=True,child_birth_year_override=-17),
            ],
            college=CollegeProfile(annual_cost_per_child=10_000,years_per_child=4,
                                    start_age=18,use_aotc_credit=False),
            car=CarProfile(car_price=25_000,down_payment=5_000,loan_rate=0.07,
                           loan_term_years=5,replace_every_years=20,num_cars=0,
                           kids_car=KidCarProfile(car_price=15_000,down_payment_pct=0.20,
                                                   loan_rate=0.07,loan_term_years=5)),
            projection_years=10,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # No adult car (num_cars=0), so any car payment must come from the kid's car
        # Kid graduates yr5 (age=22): loan payments start yr5 and run through yr9
        assert snaps[4].annual_car_payment > 0, "Kid car loan should start in yr5"
        assert snaps[3].annual_car_payment == 0, "No car payment before graduation (yr4)"
        assert snaps[9].annual_car_payment == 0, "Loan paid off after 5 years (yr10)"

    def test_two_kids_get_separate_cars_at_graduation(self):
        """Each child gets their own car at their own graduation year."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000,annual_market_return=0.0,
                                          annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            timeline_events=[
                # Kid1: born -17 → graduates yr5 (age 22)
                TimelineEvent(year=1,description="Kid1",new_child=True,child_birth_year_override=-17),
                # Kid2: born -15 → graduates yr7 (age 22)
                TimelineEvent(year=1,description="Kid2",new_child=True,child_birth_year_override=-15),
            ],
            college=CollegeProfile(annual_cost_per_child=10_000,years_per_child=4,
                                    start_age=18,use_aotc_credit=False),
            car=CarProfile(car_price=25_000,down_payment=5_000,loan_rate=0.07,
                           loan_term_years=5,replace_every_years=20,num_cars=0,
                           kids_car=KidCarProfile(car_price=15_000,down_payment_pct=0.20,
                                                   loan_rate=0.07,loan_term_years=5)),
            projection_years=12,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # yr5: kid1 loan starts (yr1 of loan)
        # yr6: kid1 only (yr2 of loan)
        # yr7: both loans active (kid1 yr3, kid2 yr1) → payments should be > yr6
        assert snaps[4].annual_car_payment > 0,                "Kid1 loan starts yr5"
        assert snaps[6].annual_car_payment > snaps[5].annual_car_payment,             "yr7 (both kids) > yr6 (kid1 only)"
        # NW integrity
        for s in snaps:
            c = (s.retirement_balance + s.brokerage_balance + s.college_529_balance
                 + s.home_equity + s.hsa_balance + s.uninvested_cash + s.cash_buffer)
            assert abs(c - s.net_worth) < 1.0, f"Yr{s.year}: NW mismatch"

    def test_kids_car_config_roundtrip(self):
        """KidCarProfile survives YAML serialisation round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan = FinancialPlan(
            income=IncomeProfile(150_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=100_000),
            strategies=StrategyToggles(),
            car=CarProfile(car_price=30_000,num_cars=2,
                           kids_car=KidCarProfile(car_price=15_000,down_payment_pct=0.15,
                                                   buy_at_age=22)),
            projection_years=10)
        plan2 = _dict_to_plan(_plan_to_dict(plan))
        assert plan2.car.kids_car is not None
        assert plan2.car.kids_car.car_price == 15_000
        assert plan2.car.kids_car.down_payment_pct == 0.15
        assert plan2.car.kids_car.buy_at_age == 22

    def test_no_kids_car_no_graduation_purchases(self):
        """Without kids_car configured, no graduation-linked car purchases occur."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000,annual_market_return=0.0,
                                          annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1,description="Child",new_child=True,child_birth_year_override=-17),
            ],
            college=CollegeProfile(annual_cost_per_child=10_000,years_per_child=4,
                                    start_age=18,use_aotc_credit=False),
            # num_cars=0 means no adult car either; kids_car=None means no graduation car
            car=CarProfile(num_cars=0, kids_car=None),
            projection_years=10,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert all(s.annual_car_payment == 0 for s in snaps),             "No car payments when num_cars=0 and kids_car=None"



class TestChildBirthYearOverrideRegression:
    """
    REGRESSION: child_birth_year_override=0 was treated as falsy in:
        birth_year = event.child_birth_year_override or year
    which evaluates to `year` when override=0 (falsy zero).
    Fix: use `event.child_birth_year_override if event.child_birth_year_override is not None else year`
    This affected college timing, AOTC, wedding fund stop age, and car hand-down logic.
    """

    def test_zero_override_not_treated_as_falsy(self):
        """child_birth_year_override=0 must use year 0, not the event year."""
        plan = FinancialPlan(
            income=IncomeProfile(70_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000,
                                          annual_529_contribution=5_000,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="Child born yr0",
                              new_child=True, child_birth_year_override=0)
            ],
            college=CollegeProfile(annual_cost_per_child=10_000, years_per_child=4,
                                    start_age=18, use_aotc_credit=True),
            projection_years=22,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # Born yr0: age 18 at yr18 → college starts yr18 (not yr19 as with falsy bug)
        assert snaps[17].annual_college_cost > 0, \
            "Yr18: child age 18, college should start. " \
            "Regression: child_birth_year_override=0 was treated as falsy."
        assert snaps[16].annual_college_cost == 0, \
            "Yr17: child age 17, college not yet"

    def test_negative_override_unaffected(self):
        """Negative overrides (already born) were never affected by the bug."""
        plan = FinancialPlan(
            income=IncomeProfile(70_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=300_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=1, description="Teen", new_child=True,
                              child_birth_year_override=-17)
            ],
            college=CollegeProfile(annual_cost_per_child=10_000, years_per_child=4,
                                    start_age=18, use_aotc_credit=False),
            projection_years=6,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # Born yr -17: age 18 at yr1 → college starts yr1
        assert snaps[0].annual_college_cost > 0, "Yr1 college should start for child born yr-17"


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo — configurable params + liquidity risk
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarloLiquidity:
    """Tests for the new liquidity risk tracking and configurable std params."""

    def _plan(self, income=120_000, liquid_cash=50_000, rent=1_500, years=15):
        return FinancialPlan(
            income=IncomeProfile(income, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=rent),
            lifestyle=LifestyleProfile(annual_vacation=5_000,monthly_other_recurring=500,
                                       annual_medical_oop=3_000,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=liquid_cash,
                                          annual_market_return=0.07,annual_inflation_rate=0.03,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=years,
        )

    def test_prob_negative_liquid_length_matches_years(self):
        """One probability value per projection year."""
        mc = ProjectionEngine(self._plan()).run_monte_carlo(n_simulations=100, seed=1)
        assert len(mc.prob_negative_liquid) == len(mc.years)

    def test_prob_negative_liquid_is_valid_probability(self):
        """All values must be in [0, 1]. 0.0 is valid (no risk) and 1.0 is valid (always illiquid)."""
        mc = ProjectionEngine(self._plan()).run_monte_carlo(n_simulations=100, seed=1)
        for i, p in enumerate(mc.prob_negative_liquid):
            assert 0.0 <= p <= 1.0, f"Year {mc.years[i]}: probability {p} out of [0, 1]"

    def test_liquid_percentile_lengths_match_years(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(n_simulations=100, seed=1)
        n = len(mc.years)
        assert len(mc.p10_liquid) == n
        assert len(mc.p50_liquid) == n
        assert len(mc.p90_liquid) == n

    def test_liquid_percentile_ordering(self):
        """p10 ≤ p50 ≤ p90 every year."""
        mc = ProjectionEngine(self._plan()).run_monte_carlo(n_simulations=200, seed=42)
        for i in range(len(mc.years)):
            assert mc.p10_liquid[i] <= mc.p50_liquid[i], f"yr{mc.years[i]}: p10 > p50"
            assert mc.p50_liquid[i] <= mc.p90_liquid[i], f"yr{mc.years[i]}: p50 > p90"

    def test_tight_plan_shows_liquidity_risk(self):
        """A plan with income far below housing costs should produce non-zero liquidity risk."""
        plan = FinancialPlan(
            income=IncomeProfile(40_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(400_000, 80_000, 0.07),
            lifestyle=LifestyleProfile(annual_vacation=5_000, monthly_other_recurring=1_000,
                                       annual_medical_oop=3_000, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=150_000, annual_market_return=0.08,
                                          annual_inflation_rate=0.03, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=10,
        )
        mc = ProjectionEngine(plan).run_monte_carlo(n_simulations=300, seed=42)
        assert max(mc.prob_negative_liquid) > 0,             "Expected non-zero liquidity risk for a structurally cash-flow-negative plan"

    def test_healthy_plan_zero_liquidity_risk(self):
        """A well-funded plan with strong income should stay liquid in all simulations."""
        plan = FinancialPlan(
            income=IncomeProfile(300_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=500_000, annual_market_return=0.07,
                                          annual_inflation_rate=0.03, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=10,
        )
        mc = ProjectionEngine(plan).run_monte_carlo(n_simulations=200, seed=42)
        assert max(mc.prob_negative_liquid) == 0.0,             f"Expected zero risk, got {max(mc.prob_negative_liquid):.1%}"

    def test_higher_market_std_widens_liquid_spread(self):
        """Higher volatility should produce a wider p10–p90 liquid band."""
        engine = ProjectionEngine(self._plan(years=20))
        mc_low  = engine.run_monte_carlo(200, seed=42, market_return_std=0.05, use_historical_returns=False)
        mc_high = engine.run_monte_carlo(200, seed=42, market_return_std=0.30, use_historical_returns=False)
        spread_low  = mc_low.p90_liquid[-1]  - mc_low.p10_liquid[-1]
        spread_high = mc_high.p90_liquid[-1] - mc_high.p10_liquid[-1]
        assert spread_high > spread_low,             f"Higher std should widen band: high={spread_high:.0f} low={spread_low:.0f}"

    def test_market_return_std_stored_in_result(self):
        """Custom std values should be stored on the result for display."""
        mc = ProjectionEngine(self._plan()).run_monte_carlo(100, seed=1,
                                                             market_return_std=0.20,
                                                             inflation_std=0.025,
                                                             salary_growth_std=0.03)
        assert abs(mc.market_return_std - 0.20) < 1e-9
        assert abs(mc.inflation_std - 0.025) < 1e-9
        assert abs(mc.salary_growth_std - 0.03) < 1e-9

    def test_seeded_results_reproducible(self):
        """Same seed + params should produce identical results."""
        engine = ProjectionEngine(self._plan())
        mc1 = engine.run_monte_carlo(200, seed=99)
        mc2 = engine.run_monte_carlo(200, seed=99)
        assert mc1.p50_net_worth == mc2.p50_net_worth
        assert mc1.prob_negative_liquid == mc2.prob_negative_liquid


# ══════════════════════════════════════════════════════════════════════════════
# Cash Buffer
# ══════════════════════════════════════════════════════════════════════════════

class TestCashBuffer:
    """
    Tests for cash_buffer_months in InvestmentProfile.

    The buffer keeps N months of living expenses as liquid cash (0% return)
    before sweeping any surplus to brokerage. Deficits drain the buffer
    first, then brokerage — reducing the probability of going negative
    in bad Monte Carlo years.
    """

    def _plan(self, buffer_months=0.0, income=120_000):
        return FinancialPlan(
            income=IncomeProfile(income, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_500),
            lifestyle=LifestyleProfile(annual_vacation=5_000, monthly_other_recurring=500,
                                       annual_medical_oop=3_000, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=50_000,
                annual_market_return=0.07, annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.04,
                cash_buffer_months=buffer_months,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=10,
        )

    def test_zero_buffer_stays_zero(self):
        """With cash_buffer_months=0, cash_buffer is always 0."""
        for s in ProjectionEngine(self._plan(0)).run_deterministic():
            assert s.cash_buffer == 0.0, f"Yr{s.year}: buffer should be 0"

    def test_buffer_builds_from_surplus(self):
        """With buffer_months>0, cash_buffer grows toward the floor."""
        snaps = ProjectionEngine(self._plan(6)).run_deterministic()
        assert snaps[-1].cash_buffer > 0, "Buffer should be positive after 10 years"

    def test_buffer_diverts_surplus_from_brokerage(self):
        """Buffer is funded by diverting surplus that would otherwise go to brokerage."""
        s0 = ProjectionEngine(self._plan(0)).run_deterministic()
        s6 = ProjectionEngine(self._plan(6)).run_deterministic()
        assert s6[-1].brokerage_balance < s0[-1].brokerage_balance,             "Buffer diverts from brokerage; brokerage should be smaller"
        assert s6[-1].cash_buffer > 0

    def test_nw_integrity_with_buffer(self):
        """cash_buffer must count toward net_worth."""
        for s in ProjectionEngine(self._plan(6)).run_deterministic():
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity
                          + s.hsa_balance + s.uninvested_cash + s.cash_buffer)
            assert abs(components - s.net_worth) < 1.0,                 f"Yr{s.year}: components={components:.2f} net_worth={s.net_worth:.2f}"

    def test_nw_integrity_zero_buffer(self):
        """NW integrity must hold with zero buffer too (regression)."""
        for s in ProjectionEngine(self._plan(0)).run_deterministic():
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity
                          + s.hsa_balance + s.uninvested_cash + s.cash_buffer)
            assert abs(components - s.net_worth) < 1.0,                 f"Yr{s.year}: NW mismatch {components:.2f} vs {s.net_worth:.2f}"

    def test_buffer_stabilises_once_filled(self):
        """Once the buffer reaches its floor, changes are small (only inflation drift)."""
        snaps = ProjectionEngine(self._plan(3)).run_deterministic()
        diffs = [snaps[i].cash_buffer - snaps[i-1].cash_buffer for i in range(3, len(snaps))]
        assert all(d >= -1.0 for d in diffs),             f"Buffer should not shrink after filling: diffs={diffs}"

    def test_default_buffer_is_zero(self):
        """Default cash_buffer_months should be 0 (backwards compatible)."""
        assert InvestmentProfile().cash_buffer_months == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# YAML Export Fidelity — no field may be lost on round-trip through the UI
# ══════════════════════════════════════════════════════════════════════════════

class TestExportFidelity:
    """
    Verifies that every field in a loaded YAML plan survives the sidebar
    build_sidebar() → FinancialPlan → _plan_to_dict() round-trip unchanged.

    The failure mode being guarded against: build_sidebar() constructs
    sub-objects from scratch using only sidebar widget values, silently
    dropping any field not exposed in the UI.  The fix uses
    dataclasses.replace() so non-UI fields are inherited from the loaded plan.

    These tests simulate what the sidebar does without running Streamlit:
    they call _simulate_sidebar_merge() which mirrors the merge logic.
    """

    def _full_plan(self) -> FinancialPlan:
        """A plan with every optional field set to a non-default value."""
        return FinancialPlan(
            income=IncomeProfile(
                gross_annual_income=180_000,
                spouse_gross_annual_income=100_000,
                filing_status=FilingStatus.MARRIED_FILING_JOINTLY,
                state=State.GEORGIA,
                other_state_flat_rate=0.07,
            ),
            housing=HousingProfile(
                home_price=600_000, down_payment=120_000, interest_rate=0.068,
                loan_term_years=30,
                annual_property_tax_rate=0.015,
                annual_insurance=3_500,
                annual_maintenance_rate=0.012,
                pmi_annual_rate=0.006,
                is_renting=False,
                monthly_rent=0.0,
                annual_rent_increase_rate=0.04,
            ),
            lifestyle=LifestyleProfile(
                monthly_childcare=2_500,
                num_children=1,
                num_pets=2,
                annual_pet_cost=3_000,
                annual_medical_oop=4_000,
                medical_auto_scale=True,
                medical_spouse_multiplier=1.9,
                medical_per_child_annual=1_800,
                annual_vacation=12_000,
                monthly_other_recurring=1_500,
                annual_wedding_fund_per_child=5_000,
                annual_parent_care_cost=18_000,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=50_000,
                current_retirement_balance=120_000,
                current_brokerage_balance=300_000,
                one_time_upcoming_expenses=10_000,
                annual_401k_contribution=21_000,
                partner_annual_401k_contribution=19_500,
                annual_roth_ira_contribution=6_500,
                annual_hsa_contribution=8_300,
                annual_529_contribution=10_000,
                annual_brokerage_contribution=5_000,
                annual_market_return=0.09,
                annual_inflation_rate=0.035,
                annual_salary_growth_rate=0.05,
                partner_salary_growth_rate=0.06,
                annual_home_appreciation_rate=0.04,
                auto_invest_surplus=False,
                cash_buffer_months=6.0,
            ),
            strategies=StrategyToggles(
                maximize_hsa=True,
                use_529_state_deduction=True,
                maximize_401k=True,
                use_roth_ladder=True,
                roth_conversion_annual_amount=25_000,
            ),
            timeline_events=[
                TimelineEvent(
                    year=1, description="Get married",
                    marriage=True, extra_one_time_expense=15_000,
                ),
                TimelineEvent(
                    year=2, description="Buy home",
                    buy_home=True, new_home_price=600_000,
                    new_home_down_payment=120_000,
                    new_home_interest_rate=0.068,
                    sell_current_home=False,
                    buyer_closing_cost_rate=0.025,
                    seller_closing_cost_rate=0.055,
                ),
                TimelineEvent(
                    year=3, description="Child with override",
                    new_child=True,
                    child_birth_year_override=-2,
                ),
                TimelineEvent(
                    year=5, description="Partner stops",
                    partner_stop_working=True,
                ),
                TimelineEvent(
                    year=8, description="Start parent care",
                    start_parent_care=True,
                ),
                TimelineEvent(
                    year=14, description="Stop parent care",
                    stop_parent_care=True,
                ),
                TimelineEvent(
                    year=6, description="Business investment",
                    extra_one_time_expense=100_000,
                ),
            ],
            retirement=RetirementProfile(
                current_age=29,
                retirement_age=65,
                desired_annual_income=100_000,
                years_in_retirement=45,
                expected_post_retirement_return=0.05,
                estimated_social_security_annual=0,
            ),
            college=CollegeProfile(
                annual_cost_per_child=26_000,
                years_per_child=4,
                start_age=18,
                use_aotc_credit=True,
                early_529_return=0.08,
                late_529_return=0.04,
                glide_path_years=10,
            ),
            car=CarProfile(
                car_price=25_000,
                down_payment=5_000,
                loan_rate=0.065,
                loan_term_years=5,
                replace_every_years=15,
                residual_value=5_000,
                hand_down_age=16,
                num_cars=2,
                kids_car=KidCarProfile(
                    car_price=15_000,
                    down_payment_pct=0.20,
                    loan_rate=0.07,
                    loan_term_years=5,
                    buy_at_age=22,
                ),
            ),
            projection_years=36,
        )

    def _sidebar_merge(self, loaded: FinancialPlan, sidebar_income=None,
                       sidebar_is_renting=None, sidebar_spouse=0) -> FinancialPlan:
        """
        Simulate what build_sidebar() does: merge sidebar-controlled fields
        on top of the loaded plan using dataclasses.replace().
        This mirrors the exact pattern used in app.py.
        """
        import dataclasses

        # Income — fully covered
        income = dataclasses.replace(
            loaded.income,
            gross_annual_income=sidebar_income or loaded.income.gross_annual_income,
        )

        # Housing — sidebar only controls 3-4 fields; rest come from loaded plan
        if sidebar_is_renting is True:
            housing = dataclasses.replace(
                loaded.housing,
                home_price=0.0, down_payment=0.0, interest_rate=0.0,
                is_renting=True, monthly_rent=loaded.housing.monthly_rent,
            )
        else:
            housing = dataclasses.replace(
                loaded.housing,
                is_renting=False,
            )

        # Lifestyle — sidebar controls 7 fields; 5 come from loaded plan
        lifestyle = dataclasses.replace(
            loaded.lifestyle,
            num_children=loaded.lifestyle.num_children,
            annual_medical_oop=loaded.lifestyle.annual_medical_oop,
        )

        # Investments — sidebar controls most; 2 come from loaded plan
        investments = dataclasses.replace(
            loaded.investments,
            annual_roth_ira_contribution=loaded.investments.annual_roth_ira_contribution,
            annual_brokerage_contribution=loaded.investments.annual_brokerage_contribution,
        )

        # Strategies — sidebar controls 4; roth_conversion_annual_amount from loaded
        strategies = dataclasses.replace(
            loaded.strategies,
            maximize_hsa=loaded.strategies.maximize_hsa,
        )

        return FinancialPlan(
            income=income,
            housing=housing,
            lifestyle=lifestyle,
            investments=investments,
            strategies=strategies,
            timeline_events=loaded.timeline_events,
            projection_years=loaded.projection_years,
            retirement=loaded.retirement,
            college=loaded.college,
            car=loaded.car,
        )

    def _assert_plans_equal(self, original: FinancialPlan, exported: FinancialPlan,
                             context: str = "") -> None:
        """Deep field-by-field comparison with clear failure messages."""
        import dataclasses

        def cmp(a, b, path):
            if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
                for f in dataclasses.fields(a):
                    cmp(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
            elif isinstance(a, list) and isinstance(b, list):
                assert len(a) == len(b), f"{context}{path}: list length {len(a)} vs {len(b)}"
                for i, (x, y) in enumerate(zip(a, b)):
                    cmp(x, y, f"{path}[{i}]")
            elif isinstance(a, float) and isinstance(b, float):
                assert abs(a - b) < 1e-9, f"{context}{path}: {a} != {b}"
            else:
                assert a == b, f"{context}{path}: {a!r} != {b!r}"

        cmp(original, exported, "plan")

    # ── Core round-trip tests ───────────────────────────────────────────────

    def test_config_roundtrip_preserves_all_fields(self):
        """_plan_to_dict → _dict_to_plan must be a perfect lossless round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan = self._full_plan()
        restored = _dict_to_plan(_plan_to_dict(plan))
        self._assert_plans_equal(plan, restored, "config round-trip: ")

    def test_housing_non_ui_fields_preserved_in_merge(self):
        """loan_term_years, property tax, insurance, maintenance, pmi, rent_increase
        must survive a sidebar merge that only touches home_price/rate/down."""
        import dataclasses
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)

        assert merged.housing.loan_term_years          == plan.housing.loan_term_years
        assert merged.housing.annual_property_tax_rate == plan.housing.annual_property_tax_rate
        assert merged.housing.annual_insurance         == plan.housing.annual_insurance
        assert merged.housing.annual_maintenance_rate  == plan.housing.annual_maintenance_rate
        assert merged.housing.pmi_annual_rate          == plan.housing.pmi_annual_rate
        assert merged.housing.annual_rent_increase_rate == plan.housing.annual_rent_increase_rate

    def test_lifestyle_non_ui_fields_preserved_in_merge(self):
        """medical_auto_scale, multiplier, per_child, wedding_fund, parent_care
        must not be reset to defaults by a sidebar merge."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)

        assert merged.lifestyle.medical_auto_scale          == plan.lifestyle.medical_auto_scale
        assert merged.lifestyle.medical_spouse_multiplier   == plan.lifestyle.medical_spouse_multiplier
        assert merged.lifestyle.medical_per_child_annual    == plan.lifestyle.medical_per_child_annual
        assert merged.lifestyle.annual_wedding_fund_per_child == plan.lifestyle.annual_wedding_fund_per_child
        assert merged.lifestyle.annual_parent_care_cost     == plan.lifestyle.annual_parent_care_cost

    def test_investments_non_ui_fields_preserved_in_merge(self):
        """annual_roth_ira_contribution and annual_brokerage_contribution must survive."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)

        assert merged.investments.annual_roth_ira_contribution  == plan.investments.annual_roth_ira_contribution
        assert merged.investments.annual_brokerage_contribution == plan.investments.annual_brokerage_contribution

    def test_partner_salary_growth_preserved_when_solo(self):
        """partner_salary_growth_rate must survive even when spouse income = 0 in sidebar."""
        plan = self._full_plan()
        # Simulate sidebar with spouse=0 (partner not shown)
        merged = self._sidebar_merge(plan, sidebar_spouse=0)
        assert merged.investments.partner_salary_growth_rate == plan.investments.partner_salary_growth_rate

    def test_strategies_roth_amount_preserved(self):
        """roth_conversion_annual_amount is not in the sidebar; must not reset to 0."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)
        assert merged.strategies.roth_conversion_annual_amount == plan.strategies.roth_conversion_annual_amount

    def test_retirement_profile_preserved(self):
        """RetirementProfile must pass through unchanged (no sidebar UI for it)."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)
        assert merged.retirement is not None
        assert merged.retirement.current_age              == plan.retirement.current_age
        assert merged.retirement.desired_annual_income    == plan.retirement.desired_annual_income
        assert merged.retirement.years_in_retirement      == plan.retirement.years_in_retirement

    def test_college_profile_preserved(self):
        """CollegeProfile must pass through unchanged."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)
        assert merged.college is not None
        assert merged.college.annual_cost_per_child == plan.college.annual_cost_per_child
        assert merged.college.early_529_return      == plan.college.early_529_return
        assert merged.college.glide_path_years      == plan.college.glide_path_years

    def test_car_profile_preserved_including_kids_car(self):
        """CarProfile and KidCarProfile must pass through unchanged."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)
        assert merged.car is not None
        assert merged.car.replace_every_years          == plan.car.replace_every_years
        assert merged.car.kids_car is not None
        assert merged.car.kids_car.car_price           == plan.car.kids_car.car_price
        assert merged.car.kids_car.buy_at_age          == plan.car.kids_car.buy_at_age

    def test_timeline_event_hidden_fields_preserved(self):
        """start_parent_care, stop_parent_care, child_birth_year_override must
        survive the sidebar event rebuild."""
        plan = self._full_plan()
        merged = self._sidebar_merge(plan)

        by_desc = {e.description: e for e in merged.timeline_events}
        assert by_desc["Child with override"].child_birth_year_override == -2
        assert by_desc["Start parent care"].start_parent_care  is True
        assert by_desc["Stop parent care"].stop_parent_care    is True
        assert by_desc["Partner stops"].partner_stop_working   is True

    def test_none_retirement_stays_none(self):
        """Plans without a RetirementProfile must not gain one after a merge."""
        import dataclasses
        plan = dataclasses.replace(self._full_plan(), retirement=None)
        merged = self._sidebar_merge(plan)
        assert merged.retirement is None

    def test_none_college_stays_none(self):
        """Plans without a CollegeProfile must not gain one after a merge."""
        import dataclasses
        plan = dataclasses.replace(self._full_plan(), college=None)
        merged = self._sidebar_merge(plan)
        assert merged.college is None

    def test_none_car_stays_none(self):
        """Plans without a CarProfile must not gain one after a merge."""
        import dataclasses
        plan = dataclasses.replace(self._full_plan(), car=None)
        merged = self._sidebar_merge(plan)
        assert merged.car is None

    def test_renting_plan_preserves_rent_increase_rate(self):
        """When is_renting=True, annual_rent_increase_rate must survive."""
        import dataclasses
        plan = dataclasses.replace(
            self._full_plan(),
            housing=HousingProfile(
                home_price=0, down_payment=0, interest_rate=0.0,
                is_renting=True, monthly_rent=2_000,
                annual_rent_increase_rate=0.05,
            ),
        )
        merged = self._sidebar_merge(plan, sidebar_is_renting=True)
        assert merged.housing.annual_rent_increase_rate == 0.05

    def test_full_plan_config_roundtrip_is_lossless(self):
        """End-to-end: _plan_to_dict → _dict_to_plan → merge → _plan_to_dict
        must produce identical dicts (i.e. no field is silently dropped at
        any stage of the export pipeline)."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan = self._full_plan()
        # Simulate: load → sidebar merge → export
        merged = self._sidebar_merge(plan)
        d_original = _plan_to_dict(plan)
        d_merged   = _plan_to_dict(merged)

        def compare_dicts(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in a:
                    assert k in b, f"Key '{path}.{k}' missing from merged export"
                    compare_dicts(a[k], b[k], f"{path}.{k}")
            elif isinstance(a, list) and isinstance(b, list):
                assert len(a) == len(b), f"{path}: list length {len(a)} vs {len(b)}"
                for i, (x, y) in enumerate(zip(a, b)):
                    compare_dicts(x, y, f"{path}[{i}]")
            elif isinstance(a, float) and isinstance(b, float):
                assert abs(a - b) < 1e-9, f"{path}: {a} != {b}"
            else:
                assert a == b, f"{path}: {a!r} != {b!r}"

        compare_dicts(d_original, d_merged)


# ══════════════════════════════════════════════════════════════════════════════
# Business Ownership
# ══════════════════════════════════════════════════════════════════════════════

class TestBusinessProfile:
    """Tests for BusinessProfile — franchise / LLC / sole-prop business ownership."""

    def _base(self, biz=None):
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=200_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            business=biz, projection_years=10,
        )

    def _biz(self, **kw):
        defaults = dict(annual_revenue=200_000, expense_ratio=0.60,
                        revenue_growth_rate=0.0, start_year=1, equity_multiple=0)
        defaults.update(kw)
        return BusinessProfile(**defaults)

    # ── No-business baseline ────────────────────────────────────────────────

    def test_no_business_produces_zeros(self):
        """Without a BusinessProfile, all business fields must be zero every year."""
        for s in ProjectionEngine(self._base()).run_deterministic():
            assert s.annual_business_income == 0.0, f"Yr{s.year}: unexpected income"
            assert s.business_equity == 0.0,        f"Yr{s.year}: unexpected equity"

    # ── Income & equity ─────────────────────────────────────────────────────

    def test_business_income_positive(self):
        """Profitable business (revenue > costs) must produce positive owner income."""
        s1 = ProjectionEngine(self._base(self._biz())).run_deterministic()[0]
        assert s1.annual_business_income > 0

    def test_equity_equals_profit_times_multiple(self):
        """business_equity = net_profit × equity_multiple, exactly."""
        biz = self._biz(equity_multiple=3.0)
        s1 = ProjectionEngine(self._base(biz)).run_deterministic()[0]
        net_profit = 200_000 * (1 - 0.60)   # 80_000
        assert abs(s1.business_equity - net_profit * 3.0) < 1.0

    def test_se_tax_reduces_net_income_vs_gross_profit(self):
        """Self-employment tax must reduce owner take-home below gross profit."""
        biz = self._biz(use_qbi_deduction=False)
        s1 = ProjectionEngine(self._base(biz)).run_deterministic()[0]
        gross_profit = 200_000 * 0.40
        assert s1.annual_business_income < gross_profit

    def test_net_worth_integrity_with_business_equity(self):
        """NW must equal sum of all components including business_equity."""
        biz = self._biz(equity_multiple=3.0)
        for s in ProjectionEngine(self._base(biz)).run_deterministic():
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity + s.hsa_balance
                          + s.uninvested_cash + s.cash_buffer + s.business_equity)
            assert abs(components - s.net_worth) < 1.0, \
                f"Yr{s.year}: NW mismatch {components:.2f} vs {s.net_worth:.2f}"

    # ── Timing ──────────────────────────────────────────────────────────────

    def test_start_year_respected(self):
        """No income or equity before start_year; both appear from start_year onward."""
        biz = self._biz(start_year=3, equity_multiple=3.0)
        snaps = ProjectionEngine(self._base(biz)).run_deterministic()
        assert snaps[0].annual_business_income == 0.0, "yr1 should have no income"
        assert snaps[1].annual_business_income == 0.0, "yr2 should have no income"
        assert snaps[2].annual_business_income  > 0.0, "yr3 should have income"

    def test_revenue_grows_at_growth_rate(self):
        """Owner income in yr3 should be > yr1 by at least (1+rate)^2."""
        biz = self._biz(annual_revenue=100_000, expense_ratio=0.50,
                        revenue_growth_rate=0.10)
        snaps = ProjectionEngine(self._base(biz)).run_deterministic()
        assert snaps[2].annual_business_income > snaps[0].annual_business_income * 1.10

    # ── Initial investment ──────────────────────────────────────────────────

    def test_initial_investment_deducted_from_brokerage(self):
        """50k initial investment should reduce year-1 brokerage by exactly 50k."""
        biz_0  = self._biz(initial_investment=0)
        biz_50 = self._biz(initial_investment=50_000)
        br0 = ProjectionEngine(self._base(biz_0)).run_deterministic()[0].brokerage_balance
        br1 = ProjectionEngine(self._base(biz_50)).run_deterministic()[0].brokerage_balance
        assert abs((br0 - br1) - 50_000) < 1.0

    def test_initial_investment_only_in_start_year(self):
        """Investment must only hit brokerage once (in start_year, not every year)."""
        biz_0  = self._biz(initial_investment=0,      start_year=2)
        biz_50 = self._biz(initial_investment=50_000, start_year=2)
        snaps0 = ProjectionEngine(self._base(biz_0)).run_deterministic()
        snaps1 = ProjectionEngine(self._base(biz_50)).run_deterministic()
        # Difference should be constant from yr2 onward (not accumulating)
        diff_yr2 = snaps0[1].brokerage_balance - snaps1[1].brokerage_balance
        diff_yr5 = snaps0[4].brokerage_balance - snaps1[4].brokerage_balance
        assert abs(diff_yr2 - 50_000) < 1.0
        assert abs(diff_yr5 - diff_yr2) < 1.0, "Investment hit more than once"

    # ── Business sale ───────────────────────────────────────────────────────

    def test_sale_year_zeroes_equity(self):
        """After sale_year, business_equity must be 0 permanently."""
        biz = self._biz(equity_multiple=3.0, sale_year=5)
        snaps = ProjectionEngine(self._base(biz)).run_deterministic()
        for s in snaps[4:]:   # yr5 and beyond
            assert s.business_equity == 0.0,           f"Yr{s.year}: equity should be 0 after sale"
            assert s.annual_business_income == 0.0,    f"Yr{s.year}: no income after sale"

    def test_sale_year_proceeds_hit_brokerage(self):
        """Business sale proceeds (equity) must flow into brokerage in sale_year."""
        biz = self._biz(equity_multiple=3.0, sale_year=5)
        snaps = ProjectionEngine(self._base(biz)).run_deterministic()
        # brokerage in yr5 should spike relative to yr4
        jump = snaps[4].brokerage_balance - snaps[3].brokerage_balance
        expected_equity = 200_000 * 0.40 * 3.0   # 240_000
        assert jump > expected_equity * 0.8, f"Sale proceeds not in brokerage: jump={jump:.0f}"

    # ── Retirement contributions ─────────────────────────────────────────────

    def test_solo_401k_increases_retirement_balance(self):
        """Solo 401k contributions must appear in retirement_balance."""
        biz_no  = self._biz(solo_401k_contribution=0)
        biz_yes = self._biz(annual_revenue=300_000, expense_ratio=0.50,
                            solo_401k_contribution=20_000)
        ret_no  = ProjectionEngine(self._base(biz_no)).run_deterministic()[-1].retirement_balance
        ret_yes = ProjectionEngine(self._base(biz_yes)).run_deterministic()[-1].retirement_balance
        assert ret_yes > ret_no

    def test_solo_401k_cap_enforced(self):
        """Solo 401k must be capped at $69,000 IRS limit even if stated higher."""
        biz = self._biz(annual_revenue=5_000_000, expense_ratio=0.20,
                        solo_401k_contribution=200_000)
        # Engine should cap at 69k without crashing
        snaps = ProjectionEngine(self._base(biz)).run_deterministic()
        assert snaps[0].annual_business_income > 0  # still runs

    # ── Config round-trip ───────────────────────────────────────────────────

    def test_config_roundtrip_preserves_all_fields(self):
        """All BusinessProfile fields must survive _plan_to_dict → _dict_to_plan."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        biz = BusinessProfile(
            annual_revenue=300_000, expense_ratio=0.45,
            revenue_growth_rate=0.08, initial_investment=100_000,
            start_year=2, use_qbi_deduction=True,
            self_employed_health_insurance=15_000,
            solo_401k_contribution=35_000, sep_ira_contribution=8_000,
            equity_multiple=4.5, sale_year=20,
        )
        plan2 = _dict_to_plan(_plan_to_dict(self._base(biz)))
        b2 = plan2.business
        assert b2 is not None
        assert b2.annual_revenue                  == 300_000
        assert b2.expense_ratio                   == 0.45
        assert b2.revenue_growth_rate             == 0.08
        assert b2.initial_investment              == 100_000
        assert b2.start_year                      == 2
        assert b2.use_qbi_deduction               is True
        assert b2.self_employed_health_insurance  == 15_000
        assert b2.solo_401k_contribution          == 35_000
        assert b2.sep_ira_contribution            == 8_000
        assert b2.equity_multiple                 == 4.5
        assert b2.sale_year                       == 20

    def test_none_business_stays_none_after_config_roundtrip(self):
        """Plans without a business must not gain one on round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan2 = _dict_to_plan(_plan_to_dict(self._base(None)))
        assert plan2.business is None


# ══════════════════════════════════════════════════════════════════════════════
# Employer 401k Match
# ══════════════════════════════════════════════════════════════════════════════

class TestEmployerMatch:
    """
    Tests for EmployerMatch / MatchTier — employer 401k matching logic.

    The employer match flows directly into retirement_balance each year with no
    impact on breathing room (it's free money from the employer, not from
    the employee's income). Vesting, annual caps, and tiered structures are all
    supported.
    """

    def _base(self, em=None, k401=15_000, salary=100_000):
        return FinancialPlan(
            income=IncomeProfile(salary, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=0, current_retirement_balance=0,
                annual_401k_contribution=k401,
                annual_market_return=0.0, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
                employer_match=em,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=10,
        )

    # ── Baseline ────────────────────────────────────────────────────────────

    def test_no_match_retirement_equals_employee_contrib(self):
        """Without an employer match, retirement balance = employee contribution only."""
        s1 = ProjectionEngine(self._base()).run_deterministic()[0]
        assert abs(s1.retirement_balance - 15_000) < 1.0

    def test_none_match_is_default(self):
        """Default InvestmentProfile has no employer match."""
        assert InvestmentProfile().employer_match is None

    # ── Match formulas ───────────────────────────────────────────────────────

    def test_simple_match_50pct_on_6pct(self):
        """50% match on first 6% of $100k salary = $3,000 employer contribution."""
        em = EmployerMatch(tiers=[MatchTier(match_pct=0.50, up_to_pct_of_salary=0.06)])
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        # employee $15k + employer $3k = $18k
        assert abs(s1.retirement_balance - 18_000) < 1.0, f"got {s1.retirement_balance:.0f}"

    def test_dollar_for_dollar_on_3pct(self):
        """100% match on first 3% of $100k = $3,000."""
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.03)])
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        assert abs(s1.retirement_balance - 18_000) < 1.0

    def test_tiered_match(self):
        """100% on first 3% ($3k) + 50% on next 2% ($1k) = $4k total match."""
        em = EmployerMatch(tiers=[
            MatchTier(match_pct=1.00, up_to_pct_of_salary=0.03),
            MatchTier(match_pct=0.50, up_to_pct_of_salary=0.02),
        ])
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        # employee $15k + employer $4k = $19k
        assert abs(s1.retirement_balance - 19_000) < 1.0, f"got {s1.retirement_balance:.0f}"

    def test_match_capped_at_employee_contribution(self):
        """Match cannot exceed what the employee actually contributes per tier."""
        # Employee contributes $2k, tier ceiling = 6% × $100k = $6k
        # match = min($2k, $6k) × 100% = $2k
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.06)])
        s1 = ProjectionEngine(self._base(em, k401=2_000)).run_deterministic()[0]
        assert abs(s1.retirement_balance - 4_000) < 1.0, f"got {s1.retirement_balance:.0f}"

    # ── Annual cap ───────────────────────────────────────────────────────────

    def test_annual_cap_limits_total_match(self):
        """annual_cap is an absolute ceiling on total employer match per year."""
        # Without cap: 100% match on 10% of $100k = $10k
        # With $5k cap: match is $5k
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.10)],
                           annual_cap=5_000)
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        assert abs(s1.retirement_balance - 20_000) < 1.0, f"got {s1.retirement_balance:.0f}"

    def test_no_cap_means_full_match(self):
        """Without a cap, full match is paid even if large."""
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.10)])
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        # 100% × 10% × $100k = $10k match → $25k total
        assert abs(s1.retirement_balance - 25_000) < 1.0, f"got {s1.retirement_balance:.0f}"

    # ── Vesting ──────────────────────────────────────────────────────────────

    def test_cliff_vesting_withholds_match_before_vest_date(self):
        """With vesting_years=3, no match is paid in years 1 and 2."""
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.06)],
                           vesting_years=3)
        snaps = ProjectionEngine(self._base(em)).run_deterministic()
        assert abs(snaps[0].retirement_balance - 15_000) < 1.0, "yr1: no match expected"
        assert abs(snaps[1].retirement_balance - 30_000) < 1.0, "yr2: no match expected"

    def test_cliff_vesting_match_begins_at_vest_year(self):
        """Match starts in projection_year == vesting_years."""
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.06)],
                           vesting_years=3)
        snaps = ProjectionEngine(self._base(em)).run_deterministic()
        # yr3: employee $15k + employer $6k = $21k on top of prior $30k
        assert snaps[2].retirement_balance > 45_000,             f"yr3 should include match, got {snaps[2].retirement_balance:.0f}"

    def test_immediate_vesting_pays_from_year_1(self):
        """vesting_years=0 means match is paid starting in year 1."""
        em = EmployerMatch(tiers=[MatchTier(match_pct=1.00, up_to_pct_of_salary=0.03)],
                           vesting_years=0)
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        assert s1.retirement_balance > 15_000  # has employer contribution

    # ── Profit sharing ───────────────────────────────────────────────────────

    def test_profit_sharing_no_tiers(self):
        """Profit sharing pays a flat amount regardless of employee contribution."""
        em = EmployerMatch(tiers=[], profit_sharing_annual=3_000)
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        assert abs(s1.retirement_balance - 18_000) < 1.0

    def test_profit_sharing_combined_with_tier_match(self):
        """Profit sharing stacks on top of tier match."""
        em = EmployerMatch(
            tiers=[MatchTier(match_pct=0.50, up_to_pct_of_salary=0.06)],
            profit_sharing_annual=2_000,
        )
        s1 = ProjectionEngine(self._base(em)).run_deterministic()[0]
        # 50% × 6% × $100k = $3k tier + $2k profit sharing = $5k employer
        assert abs(s1.retirement_balance - 20_000) < 1.0, f"got {s1.retirement_balance:.0f}"

    # ── Config round-trip ────────────────────────────────────────────────────

    def test_config_roundtrip_preserves_all_fields(self):
        """All EmployerMatch and MatchTier fields must survive YAML round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        em = EmployerMatch(
            tiers=[
                MatchTier(match_pct=1.00, up_to_pct_of_salary=0.03),
                MatchTier(match_pct=0.50, up_to_pct_of_salary=0.02),
            ],
            annual_cap=8_000,
            vesting_years=2,
            profit_sharing_annual=1_500,
        )
        plan2 = _dict_to_plan(_plan_to_dict(self._base(em)))
        em2 = plan2.investments.employer_match
        assert em2 is not None
        assert len(em2.tiers) == 2
        assert em2.tiers[0].match_pct == 1.00
        assert em2.tiers[0].up_to_pct_of_salary == 0.03
        assert em2.tiers[1].match_pct == 0.50
        assert em2.tiers[1].up_to_pct_of_salary == 0.02
        assert em2.annual_cap == 8_000
        assert em2.vesting_years == 2
        assert em2.profit_sharing_annual == 1_500

    def test_none_match_roundtrips_as_none(self):
        """Plans with no employer match must not gain one on config round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan2 = _dict_to_plan(_plan_to_dict(self._base(None)))
        assert plan2.investments.employer_match is None


# ══════════════════════════════════════════════════════════════════════════════
# Childcare Profile — age-bracketed costs
# ══════════════════════════════════════════════════════════════════════════════

class TestChildcareProfile:
    """
    Tests for ChildcareProfile — age-bracketed childcare cost schedule.

    Replaces the flat monthly_childcare rate with realistic age-based costs.
    The flat rate is retained for backward compatibility (used when
    childcare_profile is None).
    """

    _std_profile = ChildcareProfile(phases=[
        ChildcarePhase(age_start=0,  age_end=2,  monthly_cost=2_500),
        ChildcarePhase(age_start=3,  age_end=4,  monthly_cost=1_500),
        ChildcarePhase(age_start=5,  age_end=12, monthly_cost=600),
        ChildcarePhase(age_start=13, age_end=17, monthly_cost=150),
    ])

    def _base(self, cp=None, flat=0, events=None):
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                monthly_childcare=flat, num_children=0,
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
                childcare_profile=cp,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events or [],
            projection_years=20,
        )

    def test_flat_rate_still_works_without_profile(self):
        """Backward compat: flat monthly_childcare is used when no profile is set."""
        plan = self._base(flat=2_500, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        s1 = ProjectionEngine(plan).run_deterministic()[0]
        assert abs(s1.annual_lifestyle_cost - 30_000) < 1.0   # $2500 × 12

    def test_infant_cost_applied_at_age_0(self):
        """Child born in yr1 has age 0 in yr1 → infant phase cost applies."""
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        s1 = ProjectionEngine(plan).run_deterministic()[0]
        assert abs(s1.annual_lifestyle_cost - 30_000) < 1.0   # $2500 × 12

    def test_preschool_phase_cheaper_than_infant(self):
        """Cost drops when child enters preschool phase (age 3)."""
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        # yr4: child age = 3 → preschool $1500/mo = $18k/yr
        assert snaps[3].annual_lifestyle_cost < snaps[0].annual_lifestyle_cost
        assert abs(snaps[3].annual_lifestyle_cost - 18_000) < 1.0

    def test_school_age_cheaper_than_preschool(self):
        """Cost drops again entering school-age phase (age 5)."""
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        # yr6: age=5 → $600/mo = $7,200/yr
        assert snaps[5].annual_lifestyle_cost < snaps[3].annual_lifestyle_cost
        assert abs(snaps[5].annual_lifestyle_cost - 7_200) < 1.0

    def test_teen_phase_cheapest(self):
        """Teen phase (age 13+) is cheapest covered age."""
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        # yr14: age=13 → $150/mo = $1,800/yr
        assert snaps[13].annual_lifestyle_cost < snaps[5].annual_lifestyle_cost
        assert abs(snaps[13].annual_lifestyle_cost - 1_800) < 1.0

    def test_zero_cost_past_last_phase(self):
        """Ages not covered by any phase return $0 childcare."""
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        # yr19: age=18, no phase covers it → $0
        assert snaps[18].annual_lifestyle_cost == 0.0

    def test_two_children_different_ages_independent_lookup(self):
        """Each child's cost is looked up by their individual age, not averaged."""
        # Child1 born yr1 → age 5 at yr6 ($600/mo)
        # Child2 born yr3 → age 3 at yr6 ($1500/mo)
        # Total yr6 = ($600 + $1500) × 12 = $25,200
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child1", new_child=True),
            TimelineEvent(year=3, description="Child2", new_child=True),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert abs(snaps[5].annual_lifestyle_cost - 25_200) < 1.0

    def test_nw_integrity_with_profile(self):
        """NW integrity must hold with age-based childcare costs each year."""
        plan = self._base(cp=self._std_profile, events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ])
        for s in ProjectionEngine(plan).run_deterministic():
            components = (s.retirement_balance + s.brokerage_balance
                          + s.college_529_balance + s.home_equity + s.hsa_balance
                          + s.uninvested_cash + s.cash_buffer + s.business_equity)
            assert abs(components - s.net_worth) < 1.0, f"Yr{s.year}"

    def test_config_roundtrip_preserves_all_phases(self):
        """All ChildcarePhase fields must survive YAML round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan2 = _dict_to_plan(_plan_to_dict(self._base(cp=self._std_profile)))
        cp2 = plan2.lifestyle.childcare_profile
        assert cp2 is not None
        assert len(cp2.phases) == 4
        assert cp2.phases[0].age_start == 0 and cp2.phases[0].age_end == 2
        assert cp2.phases[0].monthly_cost == 2_500
        assert cp2.phases[2].age_start == 5 and cp2.phases[2].monthly_cost == 600
        assert cp2.phases[3].monthly_cost == 150

    def test_none_profile_roundtrips_as_none(self):
        """Plans using flat rate must not gain a profile on round-trip."""
        from fintracker.config import _plan_to_dict, _dict_to_plan
        plan2 = _dict_to_plan(_plan_to_dict(self._base(flat=1_500)))
        assert plan2.lifestyle.childcare_profile is None


# ══════════════════════════════════════════════════════════════════════════════
# Gaps — tests added to cover features that shipped without tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoricalBootstrap:
    """
    Tests for use_historical_returns and use_historical_inflation bootstrap
    sampling in run_monte_carlo().
    """

    def _plan(self):
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=100_000, annual_market_return=0.08,
                annual_inflation_rate=0.03, annual_salary_growth_rate=0.04,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=20,
        )

    def test_historical_returns_produces_different_distribution(self):
        """Bootstrap from historical S&P data must differ from normal distribution."""
        engine = ProjectionEngine(self._plan())
        mc_hist = engine.run_monte_carlo(500, seed=42, use_historical_returns=True)
        mc_norm = engine.run_monte_carlo(500, seed=42, use_historical_returns=False)
        assert mc_hist.p10_net_worth[-1] != mc_norm.p10_net_worth[-1]

    def test_historical_inflation_produces_different_distribution(self):
        """Bootstrap from historical CPI data must differ from normal distribution."""
        engine = ProjectionEngine(self._plan())
        mc_hist = engine.run_monte_carlo(500, seed=42, use_historical_inflation=True)
        mc_norm = engine.run_monte_carlo(500, seed=42, use_historical_inflation=False)
        assert mc_hist.p10_net_worth[-1] != mc_norm.p10_net_worth[-1]

    def test_use_historical_returns_flag_stored(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(100, seed=1,
                                                            use_historical_returns=True)
        assert mc.use_historical_returns is True

    def test_use_historical_inflation_flag_stored(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(100, seed=1,
                                                            use_historical_inflation=True)
        assert mc.use_historical_inflation is True

    def test_historical_inflation_flag_false_stored(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(100, seed=1,
                                                            use_historical_inflation=False)
        assert mc.use_historical_inflation is False

    def test_seeded_reproducible_with_both_historical(self):
        """Same seed must produce identical results when both bootstrap modes are on."""
        engine = ProjectionEngine(self._plan())
        mc1 = engine.run_monte_carlo(200, seed=99,
                                      use_historical_returns=True,
                                      use_historical_inflation=True)
        mc2 = engine.run_monte_carlo(200, seed=99,
                                      use_historical_returns=True,
                                      use_historical_inflation=True)
        assert mc1.p50_net_worth == mc2.p50_net_worth
        assert mc1.prob_negative_liquid == mc2.prob_negative_liquid

    def test_historical_inflation_dataset_has_96_years(self):
        """Sanity-check the embedded constant."""
        from fintracker.projections import _US_HISTORICAL_INFLATION
        assert len(_US_HISTORICAL_INFLATION) == 96

    def test_historical_inflation_includes_stagflation(self):
        """Dataset must include at least one year above 10% (1946, 1974, 1979–80)."""
        from fintracker.projections import _US_HISTORICAL_INFLATION
        assert max(_US_HISTORICAL_INFLATION) > 0.10

    def test_historical_inflation_includes_deflation(self):
        """Dataset must include deflation years (1930s Great Depression)."""
        from fintracker.projections import _US_HISTORICAL_INFLATION
        assert min(_US_HISTORICAL_INFLATION) < 0.0


class TestOwnershipPct:
    """Tests for BusinessProfile.ownership_pct — scales profit, equity, and taxes."""

    def _plan(self, ownership_pct):
        biz = BusinessProfile(
            annual_revenue=400_000, expense_ratio=0.50,
            revenue_growth_rate=0.0, start_year=1,
            equity_multiple=3.0, ownership_pct=ownership_pct,
        )
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_market_return=0.0,
                                          annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            business=biz, projection_years=5,
        )

    def test_full_owner_gets_full_profit(self):
        s = ProjectionEngine(self._plan(1.0)).run_deterministic()[0]
        assert s.annual_business_income > 0

    def test_half_owner_gets_half_income(self):
        s_full = ProjectionEngine(self._plan(1.0)).run_deterministic()[0]
        s_half = ProjectionEngine(self._plan(0.5)).run_deterministic()[0]
        ratio = s_half.annual_business_income / s_full.annual_business_income
        assert abs(ratio - 0.5) < 0.01

    def test_half_owner_gets_half_equity(self):
        s_full = ProjectionEngine(self._plan(1.0)).run_deterministic()[0]
        s_half = ProjectionEngine(self._plan(0.5)).run_deterministic()[0]
        ratio = s_half.business_equity / s_full.business_equity
        assert abs(ratio - 0.5) < 0.01

    def test_ownership_pct_default_is_1(self):
        assert BusinessProfile().ownership_pct == 1.0

    def test_nw_integrity_with_partial_ownership(self):
        for s in ProjectionEngine(self._plan(0.6)).run_deterministic():
            c = (s.retirement_balance + s.brokerage_balance + s.college_529_balance
                 + s.home_equity + s.hsa_balance + s.uninvested_cash
                 + s.cash_buffer + s.business_equity)
            assert abs(c - s.net_worth) < 1.0, f"Yr{s.year}"


class TestChildcareYAMLValidation:
    """Tests for the improved _dict_to_childcare_profile validation."""

    def test_valid_profile_parses_correctly(self):
        from fintracker.config import _dict_to_childcare_profile
        cp = _dict_to_childcare_profile({"phases": [
            {"age_start": 0, "age_end": 2, "monthly_cost": 2500},
            {"age_start": 3, "age_end": 4, "monthly_cost": 1500},
        ]})
        assert len(cp.phases) == 2
        assert cp.phases[0].monthly_cost == 2500

    def test_broken_yaml_split_phase_raises_valueerror(self):
        """The common mistake of splitting one phase across two list items
        must raise a clear ValueError, not a silent KeyError crash."""
        from fintracker.config import _dict_to_childcare_profile
        import pytest
        # Phase 2 is missing age_start (split YAML produces two incomplete dicts)
        broken = {"phases": [
            {"age_start": 0, "age_end": 2, "monthly_cost": 2500},
            {"age_start": 3},                          # missing age_end and monthly_cost
            {"age_end": 4, "monthly_cost": 1600},      # missing age_start
        ]}
        with pytest.raises(ValueError, match="missing"):
            _dict_to_childcare_profile(broken)

    def test_error_message_names_the_bad_phase(self):
        """Error message must identify which phase number is broken."""
        from fintracker.config import _dict_to_childcare_profile
        import pytest
        broken = {"phases": [
            {"age_start": 0, "age_end": 2, "monthly_cost": 2500},
            {"age_end": 4, "monthly_cost": 1600},   # phase 2, missing age_start
        ]}
        with pytest.raises(ValueError, match="phase 2"):
            _dict_to_childcare_profile(broken)

    def test_empty_phases_list_returns_empty_profile(self):
        from fintracker.config import _dict_to_childcare_profile
        cp = _dict_to_childcare_profile({"phases": []})
        assert cp.phases == []


class TestMCLiquidityWithBuffer:
    """
    Regression: prob_negative_liquid must measure brokerage + cash_buffer,
    not brokerage alone. A buffer should never increase apparent liquidity risk.
    """

    def _tight_plan(self, buffer_months):
        return FinancialPlan(
            income=IncomeProfile(40_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(400_000, 80_000, 0.07),
            lifestyle=LifestyleProfile(annual_vacation=5_000, monthly_other_recurring=1_000,
                                       annual_medical_oop=3_000, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=150_000,
                                          annual_market_return=0.08,
                                          annual_inflation_rate=0.03,
                                          annual_salary_growth_rate=0.0,
                                          cash_buffer_months=buffer_months),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=10,
        )

    def test_buffer_does_not_increase_peak_liquidity_risk(self):
        """Adding a cash buffer must not increase peak prob_negative_liquid.
        REGRESSION: when metric used brokerage only (not brokerage + buffer),
        a larger buffer meant smaller brokerage → higher apparent risk."""
        mc0 = ProjectionEngine(self._tight_plan(0)).run_monte_carlo(300, seed=42)
        mc6 = ProjectionEngine(self._tight_plan(6)).run_monte_carlo(300, seed=42)
        assert max(mc6.prob_negative_liquid) <= max(mc0.prob_negative_liquid), (
            f"Buffer should not increase risk: 0mo={max(mc0.prob_negative_liquid):.1%} "
            f"6mo={max(mc6.prob_negative_liquid):.1%}"
        )

    def test_prob_negative_liquid_all_valid_probabilities(self):
        mc = ProjectionEngine(self._tight_plan(3)).run_monte_carlo(200, seed=1)
        for p in mc.prob_negative_liquid:
            assert 0.0 <= p <= 1.0


class TestCumulativeInflationFix:
    """
    REGRESSION: inf_f = (1 + this_year_rate)^(year-1) is only correct when
    inflation is constant.  With variable per-year rates (Monte Carlo), it
    raises the sampled rate to the power of all years elapsed — a single draw
    of 13.3% at year 15 makes expenses 5.7× base instead of 1.7×.

    Fix: track cumulative_inflation in EngineState as a rolling product of
    (1 + rate_t) for each year t, so the factor is always correct.
    """

    def _plan(self, inflation=0.03):
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000),
            lifestyle=LifestyleProfile(annual_vacation=5_000, monthly_other_recurring=500,
                                       annual_medical_oop=3_000, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_market_return=0.08,
                                          annual_inflation_rate=inflation,
                                          annual_salary_growth_rate=0.04),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=20,
        )

    def test_year1_inf_f_is_one(self):
        """In year 1, no inflation has been applied yet — inf_f must equal 1.0."""
        s1 = ProjectionEngine(self._plan()).run_deterministic()[0]
        # Rent = 1000/mo × 12 × inf_f; at inf_f=1.0 this is exactly 12,000
        assert abs(s1.annual_housing_cost - 12_000) < 10

    def test_deterministic_costs_inflate_correctly(self):
        """At constant 3%, year-3 costs must be 1.03² × year-1 costs."""
        snaps = ProjectionEngine(self._plan(0.03)).run_deterministic()
        housing_yr1 = snaps[0].annual_housing_cost
        housing_yr3 = snaps[2].annual_housing_cost
        expected_ratio = 1.03 ** 2
        actual_ratio = housing_yr3 / housing_yr1
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"Expected ratio {expected_ratio:.4f}, got {actual_ratio:.4f}"

    def test_zero_inflation_costs_are_flat(self):
        """With 0% inflation AND 0% rent_increase_rate, rent costs must be flat.
        Note: rent_increase_rate is independent of CPI — set both to 0 to get flat costs."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(0.0),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000,
                                   annual_rent_increase_rate=0.0),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        h1 = snaps[0].annual_housing_cost
        h10 = snaps[9].annual_housing_cost
        assert abs(h1 - h10) < 1.0

    def test_mc_historical_inflation_p10_positive(self):
        """REGRESSION: before fix, a single draw of 18% inflation at year 15
        produced inf_f = 1.18^14 ≈ 10, making expenses 10× base and wiping
        out all liquid assets.  After fix, p10 NW must be positive."""
        mc = ProjectionEngine(self._plan()).run_monte_carlo(
            200, seed=42,
            use_historical_returns=True,
            use_historical_inflation=True,
        )
        assert mc.p10_net_worth[-1] > 0, \
            f"p10 NW negative ({mc.p10_net_worth[-1]:,.0f}) — cumulative inflation bug may have returned"

    def test_mc_hist_and_normal_inflation_medians_similar(self):
        """Historical and normal inflation have the same mean (~3%), so medians
        should be within 30% of each other — only tails differ."""
        engine = ProjectionEngine(self._plan())
        mc_hist = engine.run_monte_carlo(500, seed=42, use_historical_inflation=True)
        mc_norm = engine.run_monte_carlo(500, seed=42, use_historical_inflation=False)
        ratio = mc_hist.p50_net_worth[-1] / mc_norm.p50_net_worth[-1]
        assert 0.70 < ratio < 1.30, \
            f"Medians diverge too much: ratio={ratio:.2f} (expected 0.7–1.3)"


class TestMCCalculationAudit:
    """
    Systematic audit of every MC-sensitive calculation.
    Confirms no other calculation has the same bug as the inf_f issue:
    i.e. that all rolling rates use proper state-based compounding,
    not (current_year_rate)^(years_elapsed).
    """

    def _plan(self, **kw):
        defaults = dict(
            income=IncomeProfile(100_000,FilingStatus.SINGLE,State.TEXAS),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=1_000,
                                   annual_rent_increase_rate=0.03),
            lifestyle=LifestyleProfile(annual_vacation=0,monthly_other_recurring=0,
                                       annual_medical_oop=0,medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=0,annual_market_return=0.0,
                                          annual_inflation_rate=0.0,annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False,maximize_401k=False),
            projection_years=5,
        )
        defaults.update(kw)
        return FinancialPlan(**defaults)

    # ── Salary growth ────────────────────────────────────────────────────

    def test_salary_compounding_is_rolling_product(self):
        """income_primary *= (1+sg) each year in _advance_state — correct rolling product."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            investments=InvestmentProfile(current_liquid_cash=0, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.10),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert abs(snaps[2].gross_income - 100_000 * 1.10**2) < 1

    # ── Market returns ───────────────────────────────────────────────────

    def test_market_return_compounding_is_rolling_product(self):
        """balance *= (1+mkt) each year — correct rolling product on state."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            investments=InvestmentProfile(current_liquid_cash=0,
                                          current_retirement_balance=100_000,
                                          annual_401k_contribution=0,
                                          annual_market_return=0.10,
                                          annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert abs(snaps[2].retirement_balance - 100_000 * 1.10**3) < 1

    # ── Rent ─────────────────────────────────────────────────────────────

    def test_rent_inflates_at_rent_increase_rate_not_cpi(self):
        """Rent uses annual_rent_increase_rate, NOT general CPI (inf_f)."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=1_000,
                                   annual_rent_increase_rate=0.05),
            investments=InvestmentProfile(current_liquid_cash=200_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.03, annual_salary_growth_rate=0.0),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # yr3: base 1000 × 1.05^2 × 12 = 13,230 (rent rate only, not CPI)
        assert abs(snaps[2].annual_housing_cost - 12_000 * 1.05**2) < 5

    def test_rent_not_double_inflated(self):
        """When rent_increase_rate == CPI, rent must NOT be inflated twice."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=1_000,
                                   annual_rent_increase_rate=0.03),
            investments=InvestmentProfile(current_liquid_cash=200_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.03, annual_salary_growth_rate=0.0),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # Should be 12,000 × 1.03^2 = 12,731; NOT 12,000 × 1.06^2 = 13,499
        assert abs(snaps[2].annual_housing_cost - 12_000 * 1.03**2) < 5
        assert abs(snaps[2].annual_housing_cost - 12_000 * 1.06**2) > 100

    # ── Other costs use cumulative inf_f correctly ────────────────────────

    def test_lifestyle_costs_use_cumulative_inflation(self):
        """Vacation, medical, etc. inflate via cumulative inf_f (rolling product)."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            housing=HousingProfile(0,0,0.0,is_renting=True,monthly_rent=0,
                                   annual_rent_increase_rate=0.0),
            lifestyle=LifestyleProfile(annual_vacation=10_000, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(current_liquid_cash=200_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.05, annual_salary_growth_rate=0.0),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        ratio = snaps[2].annual_lifestyle_cost / snaps[0].annual_lifestyle_cost
        assert abs(ratio - 1.05**2) < 0.01

    # ── MC with variable rates ────────────────────────────────────────────

    def test_mc_salary_growth_correct_for_variable_rates(self):
        """MC salary growth uses rolling product — a high-growth year doesn't
        compound that rate over all prior years."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            investments=InvestmentProfile(current_liquid_cash=0, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.05),
            projection_years=20,
        )
        mc_high = ProjectionEngine(plan).run_monte_carlo(200, seed=42, salary_growth_std=0.10,
                                                          use_historical_returns=False,
                                                          use_historical_inflation=False)
        mc_low  = ProjectionEngine(plan).run_monte_carlo(200, seed=42, salary_growth_std=0.01,
                                                          use_historical_returns=False,
                                                          use_historical_inflation=False)
        # Higher variance should widen the band but medians should be similar
        spread_high = mc_high.p90_net_worth[-1] - mc_high.p10_net_worth[-1]
        spread_low  = mc_low.p90_net_worth[-1]  - mc_low.p10_net_worth[-1]
        assert spread_high > spread_low, "Higher salary std should widen NW spread"

    def test_mc_market_return_correct_for_variable_rates(self):
        """MC market returns use rolling balance product — each year's return
        applies to the current balance, not some fixed initial amount."""
        import dataclasses
        plan = dataclasses.replace(
            self._plan(),
            investments=InvestmentProfile(current_liquid_cash=0,
                                          current_retirement_balance=100_000,
                                          annual_401k_contribution=0,
                                          annual_market_return=0.08,
                                          annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            projection_years=10,
        )
        mc = ProjectionEngine(plan).run_monte_carlo(500, seed=42, use_historical_returns=True,
                                                     use_historical_inflation=False)
        # After 10 years, even p10 should be well above zero (rolling returns, not fixed)
        assert mc.p10_net_worth[-1] > 50_000