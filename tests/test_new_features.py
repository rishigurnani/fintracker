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
    CollegeProfile, FilingStatus, FinancialPlan, HousingProfile,
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
            assert snaps[yr-1].annual_parent_care_cost == 0.0, f"Yr{yr} care should have stopped"

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
    Tests for the auto_invest_surplus toggle on StrategyToggles.
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
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False,
                                        auto_invest_surplus=auto_invest),
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
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False,
                                        auto_invest_surplus=False),
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
        assert StrategyToggles().auto_invest_surplus is True