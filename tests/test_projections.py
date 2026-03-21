"""
Tests for ProjectionEngine (deterministic and Monte Carlo).
"""
import pytest
from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, FinancialPlan, TimelineEvent,
)
from fintracker.projections import ProjectionEngine


@pytest.fixture
def basic_plan():
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
            current_retirement_balance=50_000,
            annual_401k_contribution=23_000,
            annual_hsa_contribution=4_150,
            annual_market_return=0.08,
            annual_inflation_rate=0.03,
            annual_salary_growth_rate=0.04,
            annual_home_appreciation_rate=0.035,
        ),
        strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
        projection_years=10,
    )


@pytest.fixture
def renting_plan():
    return FinancialPlan(
        income=IncomeProfile(gross_annual_income=80_000, filing_status=FilingStatus.SINGLE, state=State.TEXAS),
        housing=HousingProfile(
            home_price=0,
            down_payment=0,
            interest_rate=0.0,
            is_renting=True,
            monthly_rent=2_000,
            annual_rent_increase_rate=0.03,
        ),
        lifestyle=LifestyleProfile(annual_vacation=3_000),
        investments=InvestmentProfile(
            current_liquid_cash=30_000,
            annual_market_return=0.08,
            annual_inflation_rate=0.03,
            annual_salary_growth_rate=0.03,
        ),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        projection_years=10,
    )


class TestDeterministicProjection:

    def test_returns_correct_number_of_snapshots(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        assert len(snaps) == basic_plan.projection_years

    def test_year_numbers_are_sequential(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        for i, snap in enumerate(snaps):
            assert snap.year == i + 1

    def test_net_worth_generally_increases_over_time(self, basic_plan):
        """With positive savings rate, net worth should trend up."""
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        # Year 10 net worth should be higher than year 1
        assert snaps[-1].net_worth > snaps[0].net_worth

    def test_retirement_balance_grows(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        assert snaps[-1].retirement_balance > snaps[0].retirement_balance

    def test_home_equity_grows(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        assert snaps[-1].home_equity > snaps[0].home_equity

    def test_mortgage_balance_decreases(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        assert snaps[-1].mortgage_balance < snaps[0].mortgage_balance

    def test_gross_income_grows_with_salary_growth(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        # Income should have grown due to salary_growth_rate
        assert snaps[-1].gross_income > snaps[0].gross_income

    def test_renting_plan_no_home_equity(self, renting_plan):
        engine = ProjectionEngine(renting_plan)
        snaps = engine.run_deterministic()
        for snap in snaps:
            assert snap.home_equity == 0.0
            assert snap.is_renting is True

    def test_net_worth_components_sum_correctly(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        for snap in snaps:
            calculated_nw = (
                snap.retirement_balance
                + snap.brokerage_balance
                + snap.home_equity
                + snap.hsa_balance
            )
            assert calculated_nw == pytest.approx(snap.net_worth, abs=1.0)

    def test_higher_income_leads_to_higher_net_worth(self):
        """Sanity check: more income → more wealth."""
        def make_plan(income):
            return FinancialPlan(
                income=IncomeProfile(gross_annual_income=income, filing_status=FilingStatus.SINGLE, state=State.TEXAS),
                housing=HousingProfile(home_price=300_000, down_payment=60_000, interest_rate=0.065),
                lifestyle=LifestyleProfile(),
                investments=InvestmentProfile(
                    current_liquid_cash=50_000,
                    annual_market_return=0.08,
                    annual_inflation_rate=0.03,
                    annual_salary_growth_rate=0.04,
                ),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                projection_years=5,
            )

        low_plan = make_plan(60_000)
        high_plan = make_plan(200_000)
        low_nw = ProjectionEngine(low_plan).run_deterministic()[-1].net_worth
        high_nw = ProjectionEngine(high_plan).run_deterministic()[-1].net_worth
        assert high_nw > low_nw


class TestTimelineEvents:

    def test_marriage_changes_filing_status(self, basic_plan):
        basic_plan.timeline_events = [
            TimelineEvent(year=3, description="Get married", marriage=True)
        ]
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        assert snaps[0].filing_status == FilingStatus.SINGLE
        assert snaps[2].filing_status == FilingStatus.MARRIED_FILING_JOINTLY

    def test_new_child_increments_count(self, basic_plan):
        basic_plan.timeline_events = [
            TimelineEvent(year=2, description="First child", new_child=True),
            TimelineEvent(year=4, description="Second child", new_child=True),
        ]
        engine = ProjectionEngine(basic_plan)
        snaps = engine.run_deterministic()
        assert snaps[0].num_children == 0
        assert snaps[1].num_children == 1
        assert snaps[3].num_children == 2

    def test_one_time_income_increases_brokerage(self, basic_plan):
        basic_plan.timeline_events = [
            TimelineEvent(year=2, description="Inheritance", extra_one_time_income=50_000)
        ]
        engine = ProjectionEngine(basic_plan)
        snaps_with = engine.run_deterministic()

        basic_plan.timeline_events = []
        engine_without = ProjectionEngine(basic_plan)
        snaps_without = engine_without.run_deterministic()

        assert snaps_with[1].brokerage_balance > snaps_without[1].brokerage_balance

    def test_one_time_expense_reduces_brokerage(self, basic_plan):
        basic_plan.timeline_events = [
            TimelineEvent(year=1, description="Big purchase", extra_one_time_expense=20_000)
        ]
        engine = ProjectionEngine(basic_plan)
        snaps_with = engine.run_deterministic()

        basic_plan.timeline_events = []
        engine_without = ProjectionEngine(basic_plan)
        snaps_without = engine_without.run_deterministic()

        assert snaps_with[0].brokerage_balance < snaps_without[0].brokerage_balance


class TestMonteCarlo:

    def test_monte_carlo_returns_correct_years(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        mc = engine.run_monte_carlo(n_simulations=100, seed=42)
        assert len(mc.years) == basic_plan.projection_years

    def test_percentile_ordering(self, basic_plan):
        """p10 <= p25 <= p50 <= p75 <= p90 for all years."""
        engine = ProjectionEngine(basic_plan)
        mc = engine.run_monte_carlo(n_simulations=200, seed=42)
        for i in range(len(mc.years)):
            assert mc.p10_net_worth[i] <= mc.p25_net_worth[i]
            assert mc.p25_net_worth[i] <= mc.p50_net_worth[i]
            assert mc.p50_net_worth[i] <= mc.p75_net_worth[i]
            assert mc.p75_net_worth[i] <= mc.p90_net_worth[i]

    def test_prob_millionaire_is_probability(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        mc = engine.run_monte_carlo(n_simulations=100, seed=42)
        assert 0.0 <= mc.prob_millionaire_10yr <= 1.0

    def test_seeded_simulations_are_reproducible(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        mc1 = engine.run_monte_carlo(n_simulations=100, seed=99)
        mc2 = engine.run_monte_carlo(n_simulations=100, seed=99)
        assert mc1.p50_net_worth == mc2.p50_net_worth

    def test_median_positive_for_reasonable_income(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        mc = engine.run_monte_carlo(n_simulations=200, seed=42)
        # Median net worth should be positive after 10 years for $120k earner
        assert mc.p50_net_worth[-1] > 0

    def test_num_simulations_recorded(self, basic_plan):
        engine = ProjectionEngine(basic_plan)
        mc = engine.run_monte_carlo(n_simulations=50, seed=1)
        assert mc.num_simulations == 50


# ============================================================
# New tests: home purchase event + healthcare scaling
# ============================================================

class TestHomePurchaseEvent:

    def _renting_plan(self, **kwargs):
        return FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_450),
            lifestyle=LifestyleProfile(annual_medical_oop=3_000, annual_vacation=5_000),
            investments=InvestmentProfile(
                current_liquid_cash=200_000,
                annual_401k_contribution=23_000, annual_hsa_contribution=4_150,
                annual_market_return=0.08, annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.04, annual_home_appreciation_rate=0.035,
            ),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
            timeline_events=kwargs.get("events", []),
            projection_years=kwargs.get("years", 5),
        )

    def test_year1_is_renting_before_event(self):
        """Without a buy_home event the plan stays renting."""
        plan = self._renting_plan()
        snaps = ProjectionEngine(plan).run_deterministic()
        assert all(s.is_renting for s in snaps)

    def test_buy_home_event_switches_to_owning(self):
        plan = self._renting_plan(events=[
            TimelineEvent(year=2, description="Buy home", buy_home=True,
                          new_home_price=500_000, new_home_down_payment=100_000,
                          new_home_interest_rate=0.065, sell_current_home=False),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].is_renting is True,  "Year 1 still renting"
        assert snaps[1].is_renting is False, "Year 2 owns home"
        assert snaps[1].home_value == pytest.approx(500_000, rel=0.01)
        assert snaps[1].mortgage_balance == pytest.approx(400_000, rel=0.02)

    def test_home_equity_positive_after_purchase(self):
        plan = self._renting_plan(events=[
            TimelineEvent(year=1, description="Buy home", buy_home=True,
                          new_home_price=400_000, new_home_down_payment=80_000,
                          new_home_interest_rate=0.065, sell_current_home=False),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].home_equity > 0

    def test_sell_current_home_adds_equity_to_brokerage(self):
        """Selling a home should add net proceeds to the brokerage account."""
        # Start with an owned home
        plan_own = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(300_000, 100_000, 0.065),
            lifestyle=LifestyleProfile(annual_medical_oop=3_000),
            investments=InvestmentProfile(
                current_liquid_cash=100_000, annual_market_return=0.08,
                annual_inflation_rate=0.03, annual_salary_growth_rate=0.04,
                annual_home_appreciation_rate=0.035,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=2, description="Upsize", buy_home=True,
                              new_home_price=500_000, new_home_down_payment=100_000,
                              new_home_interest_rate=0.065, sell_current_home=True),
            ],
            projection_years=5,
        )
        plan_no_sell = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(300_000, 100_000, 0.065),
            lifestyle=LifestyleProfile(annual_medical_oop=3_000),
            investments=InvestmentProfile(
                current_liquid_cash=100_000, annual_market_return=0.08,
                annual_inflation_rate=0.03, annual_salary_growth_rate=0.04,
                annual_home_appreciation_rate=0.035,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=2, description="Upsize no sell", buy_home=True,
                              new_home_price=500_000, new_home_down_payment=100_000,
                              new_home_interest_rate=0.065, sell_current_home=False),
            ],
            projection_years=5,
        )
        snaps_sell = ProjectionEngine(plan_own).run_deterministic()
        snaps_no_sell = ProjectionEngine(plan_no_sell).run_deterministic()
        # Selling should yield a higher brokerage balance
        assert snaps_sell[1].brokerage_balance > snaps_no_sell[1].brokerage_balance

    def test_home_appreciates_after_purchase(self):
        plan = self._renting_plan(events=[
            TimelineEvent(year=1, description="Buy", buy_home=True,
                          new_home_price=400_000, new_home_down_payment=80_000,
                          new_home_interest_rate=0.065, sell_current_home=False),
        ], years=5)
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[4].home_value > snaps[0].home_value

    def test_mortgage_balance_decreases_after_purchase(self):
        plan = self._renting_plan(events=[
            TimelineEvent(year=1, description="Buy", buy_home=True,
                          new_home_price=400_000, new_home_down_payment=80_000,
                          new_home_interest_rate=0.065, sell_current_home=False),
        ], years=5)
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[4].mortgage_balance < snaps[0].mortgage_balance


class TestHealthcareScaling:

    def _plan_with_events(self, events):
        return FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_500),
            lifestyle=LifestyleProfile(
                annual_medical_oop=3_000,
                medical_auto_scale=True,
                medical_spouse_multiplier=1.8,
                medical_per_child_annual=1_500,
                annual_vacation=3_000,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=100_000, annual_market_return=0.08,
                annual_inflation_rate=0.0,   # zero inflation so scaling is isolated
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events,
            projection_years=6,
        )

    def test_single_no_kids_baseline(self):
        plan = self._plan_with_events([])
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_medical_oop == pytest.approx(3_000, abs=1)

    def test_marriage_increases_medical(self):
        plan = self._plan_with_events([
            TimelineEvent(year=2, description="Marry", marriage=True)
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].annual_medical_oop == pytest.approx(3_000, abs=1)   # single
        assert snaps[1].annual_medical_oop == pytest.approx(5_400, abs=1)   # 3000*1.8

    def test_first_child_adds_cost(self):
        plan = self._plan_with_events([
            TimelineEvent(year=1, description="Marry", marriage=True),
            TimelineEvent(year=2, description="Child 1", new_child=True),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].annual_medical_oop == pytest.approx(5_400, abs=1)   # married
        assert snaps[1].annual_medical_oop == pytest.approx(6_900, abs=1)   # +1500

    def test_second_child_adds_more_cost(self):
        plan = self._plan_with_events([
            TimelineEvent(year=1, description="Marry", marriage=True),
            TimelineEvent(year=2, description="Child 1", new_child=True),
            TimelineEvent(year=4, description="Child 2", new_child=True),
        ])
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[3].annual_medical_oop == pytest.approx(8_400, abs=1)   # married + 2 kids

    def test_auto_scale_false_keeps_fixed_cost(self):
        plan = FinancialPlan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000),
            lifestyle=LifestyleProfile(
                annual_medical_oop=5_000,
                medical_auto_scale=False,   # pinned
            ),
            investments=InvestmentProfile(
                current_liquid_cash=50_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=2, description="Marry", marriage=True),
                TimelineEvent(year=3, description="Child", new_child=True),
            ],
            projection_years=4,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_medical_oop == pytest.approx(5_000, abs=1), \
                f"Year {s.year} medical OOP should stay fixed at 5000"


class TestHSAFamilyTier:

    def test_hsa_stays_single_limit_before_marriage(self):
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=100_000,
                annual_hsa_contribution=8_300,  # wants max family limit
                annual_market_return=0.08, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=False),
            timeline_events=[],
            projection_years=3,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        # Before marriage: capped at single limit ($4,150)
        for s in snaps:
            assert s.annual_hsa_contributions <= 4_150 + 1  # +1 for float tolerance

    def test_hsa_upgrades_to_family_limit_on_marriage(self):
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=100_000,
                annual_hsa_contribution=8_300,  # wants full family limit
                annual_market_return=0.08, annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=2, description="Marry", marriage=True),
            ],
            projection_years=3,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].annual_hsa_contributions <= 4_150 + 1   # single cap
        assert snaps[1].annual_hsa_contributions == pytest.approx(8_300, abs=1)  # family cap