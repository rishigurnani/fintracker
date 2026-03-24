"""
Accounting correctness tests — locks in exact economic/financial behaviour.

These tests verify:
  1. Deficit spending drains brokerage (not silently ignored)
  2. Mortgage paydown matches exact amortization schedule (not simplified approx)
  3. Home equity uses end-of-year balance
  4. Childcare correctly inflates and scales with num_children
  5. Medical OOP correctly scales at marriage and each child birth
  6. HSA tier upgrade on marriage
  7. Salary growth compounds correctly year-over-year
  8. All lifestyle costs inflate at the configured rate
  9. 529 contributions are after-tax (subtracted from breathing room)
 10. Net worth = retirement + brokerage + home_equity + hsa (no double-counting)
"""
import pytest
from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, FinancialPlan, TimelineEvent,
)
from fintracker.projections import ProjectionEngine
from fintracker.mortgage import MortgageCalculator


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _simple_plan(**overrides) -> FinancialPlan:
    """Zero-inflation, zero-growth baseline for isolating one variable at a time."""
    defaults = dict(
        income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
        housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000),
        lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0),
        investments=InvestmentProfile(
            current_liquid_cash=500_000,
            annual_market_return=0.0,
            annual_inflation_rate=0.0,
            annual_salary_growth_rate=0.0,
        ),
        strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        projection_years=5,
    )
    defaults.update(overrides)
    return FinancialPlan(**defaults)


# ── 1. Deficit spending ────────────────────────────────────────────────────────

class TestDeficitSpending:

    def test_negative_breathing_room_drains_brokerage(self):
        """When expenses exceed income, the shortfall must come from brokerage."""
        plan = FinancialPlan(
            income=IncomeProfile(40_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(400_000, 80_000, 0.07),
            lifestyle=LifestyleProfile(annual_vacation=5_000, monthly_other_recurring=1_000),
            investments=InvestmentProfile(
                current_liquid_cash=300_000,
                annual_market_return=0.0,   # no growth so change is purely cash flow
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=3,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.annual_breathing_room < 0, f"Year {s.year} should be a deficit"
        # Brokerage must strictly decrease each year
        for i in range(1, len(snaps)):
            assert snaps[i].brokerage_balance < snaps[i-1].brokerage_balance, \
                f"Brokerage should decrease yr{i+1} vs yr{i}"

    def test_exact_deficit_drain_amount(self):
        """Brokerage decrease should equal exactly the deficit (with no market return)."""
        plan = _simple_plan(
            income=IncomeProfile(60_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(500_000, 100_000, 0.065),
            lifestyle=LifestyleProfile(annual_vacation=6_000),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        s = snaps[0]
        if s.annual_breathing_room < 0:
            expected_brokerage = 500_000 - 100_000 + s.annual_breathing_room
            # 400k starting cash (500k - 100k down), then deficit reduces it
            assert abs(s.brokerage_balance - expected_brokerage) < 10, \
                f"Brokerage should be exactly start + breathing_room"

    def test_surplus_grows_brokerage(self):
        """Positive breathing room with no market return grows brokerage exactly."""
        plan = _simple_plan(
            income=IncomeProfile(200_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=2_000),
            lifestyle=LifestyleProfile(annual_vacation=5_000),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for i in range(1, len(snaps)):
            assert snaps[i].brokerage_balance > snaps[i-1].brokerage_balance


# ── 2. Mortgage paydown accuracy ──────────────────────────────────────────────

class TestMortgagePaydown:

    @pytest.fixture
    def mortgage_plan(self):
        housing = HousingProfile(500_000, 100_000, 0.065)
        return FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=housing,
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=200_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=30,
        )

    def test_matches_exact_schedule_all_years(self, mortgage_plan):
        """Engine mortgage balance must match exact amortization within $2 for every year."""
        housing = mortgage_plan.housing
        calc = MortgageCalculator(housing)
        sched = calc.full_schedule()
        exact = {row.year: row.balance for row in sched if row.month % 12 == 0}

        snaps = ProjectionEngine(mortgage_plan).run_deterministic()
        for s in snaps:
            expected = exact.get(s.year, 0.0)
            assert abs(s.mortgage_balance - expected) < 2.0, \
                f"Year {s.year}: engine={s.mortgage_balance:.2f} exact={expected:.2f}"

    def test_mortgage_reaches_zero_at_payoff(self, mortgage_plan):
        snaps = ProjectionEngine(mortgage_plan).run_deterministic()
        assert snaps[-1].mortgage_balance == pytest.approx(0.0, abs=2.0)

    def test_home_equity_uses_end_of_year_balance(self, mortgage_plan):
        """Equity = home_value - end-of-year balance (not start-of-year)."""
        housing = mortgage_plan.housing
        calc = MortgageCalculator(housing)
        sched = calc.full_schedule()
        exact = {row.year: row.balance for row in sched if row.month % 12 == 0}

        snaps = ProjectionEngine(mortgage_plan).run_deterministic()
        for s in snaps:
            expected_equity = s.home_value - exact.get(s.year, 0.0)
            assert abs(s.home_equity - expected_equity) < 2.0, \
                f"Year {s.year} equity mismatch"

    def test_mid_projection_home_purchase_uses_exact_schedule(self):
        """Buying a home via timeline event also uses exact schedule."""
        plan = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=2_000),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=300_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[
                TimelineEvent(year=2, description="Buy", buy_home=True,
                              new_home_price=500_000, new_home_down_payment=100_000,
                              new_home_interest_rate=0.065, sell_current_home=False),
            ],
            projection_years=10,
        )
        new_housing = HousingProfile(500_000, 100_000, 0.065)
        calc = MortgageCalculator(new_housing)
        sched = calc.full_schedule()
        exact = {row.year: row.balance for row in sched if row.month % 12 == 0}

        snaps = ProjectionEngine(plan).run_deterministic()
        # Years 2+ are mortgage years 1, 2, 3, ...
        for s in snaps[1:]:  # from year 2 onwards
            mortgage_year = s.year - 1   # projection year 2 = mortgage year 1
            expected = exact.get(mortgage_year, 0.0)
            assert abs(s.mortgage_balance - expected) < 2.0, \
                f"Projection year {s.year} (mortgage year {mortgage_year}): " \
                f"engine={s.mortgage_balance:.2f} exact={expected:.2f}"


# ── 3. Childcare inflation and scaling ────────────────────────────────────────

class TestChildcare:

    def _childcare_plan(self, events=None, inflation=0.03):
        return FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_500),
            lifestyle=LifestyleProfile(
                monthly_childcare=2_500,
                num_children=0,
                annual_vacation=0,
                monthly_other_recurring=0,
                annual_medical_oop=0,
                medical_auto_scale=False,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_market_return=0.0,
                annual_inflation_rate=inflation,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events or [],
            projection_years=6,
        )

    def test_no_children_zero_childcare(self):
        snaps = ProjectionEngine(self._childcare_plan()).run_deterministic()
        for s in snaps:
            # lifestyle = rent*12*inf + childcare(0)
            assert s.annual_breathing_room == pytest.approx(
                snaps[0].annual_breathing_room, rel=0.1
            )

    def test_one_child_childcare_correct_amount(self):
        plan = self._childcare_plan(events=[
            TimelineEvent(year=2, description="Child", new_child=True)
        ], inflation=0.0)
        snaps = ProjectionEngine(plan).run_deterministic()
        assert snaps[0].num_children == 0
        assert snaps[1].num_children == 1
        # Year 2: 1 child * 2500 * 12 = 30,000 exactly (inflation=0)
        yr1_lifestyle = snaps[0].annual_lifestyle_cost
        yr2_lifestyle = snaps[1].annual_lifestyle_cost
        childcare_added = yr2_lifestyle - yr1_lifestyle
        assert childcare_added == pytest.approx(30_000, abs=1)

    def test_two_children_doubles_childcare(self):
        plan = self._childcare_plan(events=[
            TimelineEvent(year=1, description="Child 1", new_child=True),
            TimelineEvent(year=2, description="Child 2", new_child=True),
        ], inflation=0.0)
        snaps = ProjectionEngine(plan).run_deterministic()
        yr1_childcare = snaps[0].annual_lifestyle_cost - 0  # only childcare (no vacation/medical/other)
        yr2_childcare = snaps[1].annual_lifestyle_cost - 0
        assert yr1_childcare == pytest.approx(30_000, abs=10)   # 1 child
        assert yr2_childcare == pytest.approx(60_000, abs=10)   # 2 children

    def test_childcare_inflates_each_year(self):
        # annual_lifestyle_cost does NOT include housing — rent is in annual_housing_cost.
        # Use monthly_rent=0 so lifestyle == childcare with nothing to subtract.
        plan = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                monthly_childcare=2_500, num_children=0,
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.03, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=1, description="Child", new_child=True)],
            projection_years=4,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            inf = (1.03) ** (s.year - 1)
            expected = s.num_children * 2_500 * 12 * inf
            assert s.annual_lifestyle_cost == pytest.approx(expected, abs=5), \
                f"Year {s.year}: expected childcare {expected:.0f}, got {s.annual_lifestyle_cost:.0f}"


# ── 4. Salary growth ──────────────────────────────────────────────────────────

class TestSalaryGrowth:

    def test_salary_compounds_correctly(self):
        plan = _simple_plan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.05,
            ),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            expected = 100_000 * (1.05) ** (s.year - 1)
            assert s.gross_income == pytest.approx(expected, rel=1e-6), \
                f"Year {s.year}: expected {expected:.2f} got {s.gross_income:.2f}"

    def test_zero_growth_constant_income(self):
        plan = _simple_plan(
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.gross_income == pytest.approx(100_000, abs=1)


# ── 5. Expense inflation ──────────────────────────────────────────────────────

class TestExpenseInflation:

    def test_vacation_inflates_at_correct_rate(self):
        plan = _simple_plan(
            lifestyle=LifestyleProfile(
                annual_vacation=10_000,
                annual_medical_oop=0,
                medical_auto_scale=False,
                monthly_other_recurring=0,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.04,
                annual_salary_growth_rate=0.0,
            ),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            inf = (1.04) ** (s.year - 1)
            expected = 10_000 * inf
            assert s.annual_lifestyle_cost == pytest.approx(expected, rel=0.001), \
                f"Year {s.year}: expected vacation {expected:.2f} got {s.annual_lifestyle_cost:.2f}"

    def test_rent_inflates_each_year(self):
        plan = _simple_plan(
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=2_000),
            lifestyle=LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                                       annual_medical_oop=0, medical_auto_scale=False),
            investments=InvestmentProfile(
                current_liquid_cash=500_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.0,
            ),
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            inf = (1.03) ** (s.year - 1)
            expected_rent = 2_000 * 12 * inf
            assert s.annual_housing_cost == pytest.approx(expected_rent, rel=0.001)


# ── 6. Net worth accounting integrity ─────────────────────────────────────────

class TestNetWorthIntegrity:

    def test_net_worth_equals_sum_of_components(self):
        """NW must always equal retirement + brokerage + home_equity + hsa."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(400_000, 80_000, 0.065),
            lifestyle=LifestyleProfile(annual_vacation=5_000, annual_medical_oop=3_000),
            investments=InvestmentProfile(
                current_liquid_cash=100_000, current_retirement_balance=50_000,
                annual_401k_contribution=20_000, annual_hsa_contribution=4_150,
                annual_market_return=0.08, annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.04,
            ),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
            timeline_events=[
                TimelineEvent(year=3, description="Marry", marriage=True),
                TimelineEvent(year=4, description="Child", new_child=True),
            ],
            projection_years=15,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            components = s.retirement_balance + s.brokerage_balance + s.home_equity + s.hsa_balance
            assert abs(components - s.net_worth) < 1.0, \
                f"Year {s.year}: components={components:.2f} net_worth={s.net_worth:.2f}"

    def test_home_equity_never_negative(self):
        plan = FinancialPlan(
            income=IncomeProfile(80_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(400_000, 40_000, 0.07),  # 10% down, high rate
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=100_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
                annual_home_appreciation_rate=0.0,  # no appreciation
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=30,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            assert s.home_equity >= 0, f"Year {s.year}: negative equity {s.home_equity:.2f}"


# ── 7. 529 contributions ──────────────────────────────────────────────────────

class TestFivetwentynine:

    def test_529_reduces_breathing_room(self):
        """529 is after-tax; it must reduce breathing room not net income."""
        base = _simple_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.GEORGIA),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
                annual_529_contribution=5_000,
            ),
            lifestyle=LifestyleProfile(num_children=1),
            strategies=StrategyToggles(
                maximize_hsa=False, maximize_401k=False, use_529_state_deduction=True
            ),
        )
        no_529 = _simple_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.GEORGIA),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
                annual_529_contribution=0,
            ),
            lifestyle=LifestyleProfile(num_children=1),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
        )
        s_with = ProjectionEngine(base).run_deterministic()[0]
        s_without = ProjectionEngine(no_529).run_deterministic()[0]
        # Breathing room should be lower with 529
        assert s_with.annual_breathing_room < s_without.annual_breathing_room
        # Net diff = $5,000 outflow minus any state tax saved by the deduction.
        # e.g. GA at 5.39% saves ~$270 on a $5,000 deduction -> net impact ~$4,730.
        diff = s_without.annual_breathing_room - s_with.annual_breathing_room
        assert 4_500 < diff <= 5_000, f"Expected ~4730 (5000 outflow minus state tax saving), got {diff:.0f}"


# ── 8. Medical OOP scaling ────────────────────────────────────────────────────

class TestMedicalScaling:
    """These are unit-level checks on the scaled_medical_oop() method itself."""

    def test_single_no_kids_returns_base(self):
        lf = LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=True,
                               medical_spouse_multiplier=1.8, medical_per_child_annual=1_500)
        assert lf.scaled_medical_oop(False, 0) == pytest.approx(3_000)

    def test_married_multiplies_base(self):
        lf = LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=True,
                               medical_spouse_multiplier=2.0, medical_per_child_annual=1_500)
        assert lf.scaled_medical_oop(True, 0) == pytest.approx(6_000)

    def test_each_child_adds_per_child_cost(self):
        lf = LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=True,
                               medical_spouse_multiplier=1.0, medical_per_child_annual=1_500)
        assert lf.scaled_medical_oop(False, 1) == pytest.approx(4_500)
        assert lf.scaled_medical_oop(False, 2) == pytest.approx(6_000)
        assert lf.scaled_medical_oop(False, 3) == pytest.approx(7_500)

    def test_married_plus_children(self):
        lf = LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=True,
                               medical_spouse_multiplier=2.0, medical_per_child_annual=1_500)
        # 3000 * 2.0 + 2 * 1500 = 6000 + 3000 = 9000
        assert lf.scaled_medical_oop(True, 2) == pytest.approx(9_000)

    def test_auto_scale_false_returns_raw_value(self):
        lf = LifestyleProfile(annual_medical_oop=5_000, medical_auto_scale=False,
                               medical_spouse_multiplier=2.0, medical_per_child_annual=1_500)
        assert lf.scaled_medical_oop(True, 3) == pytest.approx(5_000)

    def test_medical_scales_in_projection_at_marriage(self):
        """Projection engine must call scaled_medical_oop correctly at each year."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                annual_medical_oop=3_000, medical_auto_scale=True,
                medical_spouse_multiplier=2.0, medical_per_child_annual=1_500,
                annual_vacation=0, monthly_other_recurring=0,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=500_000, annual_market_return=0.0,
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
        assert snaps[0].annual_medical_oop == pytest.approx(3_000, abs=1)   # single
        assert snaps[1].annual_medical_oop == pytest.approx(6_000, abs=1)   # married
        assert snaps[2].annual_medical_oop == pytest.approx(7_500, abs=1)   # married + 1 child
        assert snaps[3].annual_medical_oop == pytest.approx(7_500, abs=1)   # same (no new events)


# ============================================================
# Event correctness tests — every timeline event type
# ============================================================

class TestMarriageEvent:

    def _plan(self, events):
        return FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=True,
                                       medical_spouse_multiplier=2.0, medical_per_child_annual=1_500,
                                       annual_vacation=0, monthly_other_recurring=0),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_hsa_contribution=4_150,
                                          annual_401k_contribution=23_000,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
            timeline_events=events,
            projection_years=3,
        )

    def test_filing_status_switches(self):
        s = ProjectionEngine(self._plan([TimelineEvent(year=2, description='Marry', marriage=True)])).run_deterministic()
        assert s[0].filing_status == FilingStatus.SINGLE
        assert s[1].filing_status == FilingStatus.MARRIED_FILING_JOINTLY

    def test_is_married_flag_set(self):
        s = ProjectionEngine(self._plan([TimelineEvent(year=2, description='Marry', marriage=True)])).run_deterministic()
        assert not s[0].is_married
        assert s[1].is_married

    def test_medical_scales_on_marriage(self):
        s = ProjectionEngine(self._plan([TimelineEvent(year=2, description='Marry', marriage=True)])).run_deterministic()
        assert s[0].annual_medical_oop == pytest.approx(3_000, abs=1)
        assert s[1].annual_medical_oop == pytest.approx(6_000, abs=1)  # 3000 * 2.0

    def test_hsa_stated_amount_honored_after_marriage(self):
        """Stated annual_hsa_contribution=4150 is honored even after marriage.
        Marriage upgrades the IRS cap from 4150 to 8300, but the engine
        still contributes exactly what was stated. To receive 8300 the user
        must set annual_hsa_contribution=8300 in their YAML."""
        s = ProjectionEngine(self._plan([TimelineEvent(year=2, description='Marry', marriage=True)])).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(4_150, abs=1)   # single, stated 4150
        assert s[1].annual_hsa_contributions == pytest.approx(4_150, abs=1)   # married, stated 4150 still honored

    def test_hsa_family_cap_allows_8300_when_stated(self):
        """If user sets annual_hsa_contribution=8300, they get 4150 (single cap) before
        marriage and 8300 (family cap) after marriage."""
        plan = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(annual_medical_oop=0, medical_auto_scale=False,
                                       annual_vacation=0, monthly_other_recurring=0),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_hsa_contribution=8_300,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=False),
            timeline_events=[TimelineEvent(year=2, description='Marry', marriage=True)],
            projection_years=3,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(4_150, abs=1)   # capped at single limit
        assert s[1].annual_hsa_contributions == pytest.approx(8_300, abs=1)   # family cap unlocked

    def test_mfj_tax_lower_than_single(self):
        """MFJ filing status must reduce tax relative to single at same income."""
        from fintracker.tax_engine import TaxEngine
        te = TaxEngine()
        inv = InvestmentProfile(annual_hsa_contribution=8_300, annual_401k_contribution=23_000)
        strat = StrategyToggles(maximize_hsa=True, maximize_401k=True)
        t_s = te.calculate(IncomeProfile(180_000, FilingStatus.SINGLE, State.GEORGIA), inv, strat)
        t_m = te.calculate(IncomeProfile(180_000, FilingStatus.MARRIED_FILING_JOINTLY, State.GEORGIA), inv, strat)
        assert t_m.total_annual_tax < t_s.total_annual_tax


class TestMaximizeContributions:
    """
    REGRESSION CLASS — these tests document bugs that were introduced and fixed.

    The core rule: annual_401k_contribution and annual_hsa_contribution in
    InvestmentProfile are the user's stated nominal amounts. The projection
    engine ALWAYS uses exactly what is stated, capped at IRS limits.
    maximize_401k / maximize_hsa flags control tax treatment only,
    NOT the contribution amount.

    Bug history:
      - "maximize means maximize" fix incorrectly overrode stated 23k to 30.5k,
        silently diverting $7,500/yr from brokerage to retirement and causing
        liquid assets to go negative years earlier than the user expected.
    """

    def test_stated_401k_is_honored_when_maximize_true(self):
        """REGRESSION: maximize_401k=True must NOT override stated contribution.
        User set 23k → engine must contribute exactly 23k, not 30.5k.
        This was the bug that caused liquid assets to go negative unexpectedly."""
        plan = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_401k_contribution=23_000,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            projection_years=3,
        )
        s = ProjectionEngine(plan).run_deterministic()
        for snap in s:
            assert snap.annual_retirement_contributions == pytest.approx(23_000, abs=1),                 f"Year {snap.year}: expected 23000, got {snap.annual_retirement_contributions:.0f}. "                 f"maximize_401k=True must not override the stated contribution amount."

    def test_stated_hsa_is_honored_when_maximize_true(self):
        """REGRESSION: maximize_hsa=True must NOT override stated contribution."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_hsa_contribution=3_000,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=False),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(3_000, abs=1),             f"maximize_hsa=True overrode stated 3000 to {s[0].annual_hsa_contributions:.0f}"

    def test_stated_401k_at_irs_limit_works(self):
        """User who explicitly sets 30500 gets 30500 — IRS cap does not cut below stated."""
        plan = FinancialPlan(
            income=IncomeProfile(200_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_401k_contribution=30_500,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_retirement_contributions == pytest.approx(30_500, abs=1)

    def test_contribution_capped_at_irs_limit(self):
        """Engine must not allow contributions above IRS limits even if user states more."""
        plan = FinancialPlan(
            income=IncomeProfile(200_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_401k_contribution=99_000,  # above IRS limit
                                          annual_hsa_contribution=99_000,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_retirement_contributions <= 30_500 + 1
        assert s[0].annual_hsa_contributions <= 8_300 + 1

    def test_maximize_false_means_zero_hsa_contribution(self):
        """maximize_hsa=False means HSA not used at all, even if amount is set."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_hsa_contribution=4_150,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(0.0, abs=1)

    def test_contribution_amount_unchanged_by_maximize_flag(self):
        """REGRESSION: maximize_401k=True with stated 23k must contribute exactly 23k,
        NOT 30.5k. The flag controls tax deduction treatment only — contribution amount
        and retirement balance growth must be identical regardless of the flag."""
        def make(maximize):
            return FinancialPlan(
                income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
                lifestyle=LifestyleProfile(),
                investments=InvestmentProfile(current_liquid_cash=100_000,
                                              annual_401k_contribution=23_000,
                                              annual_market_return=0.0, annual_inflation_rate=0.0,
                                              annual_salary_growth_rate=0.0),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=maximize),
                projection_years=3,
            )
        s_on  = ProjectionEngine(make(True)).run_deterministic()
        s_off = ProjectionEngine(make(False)).run_deterministic()
        for on, off in zip(s_on, s_off):
            assert on.annual_retirement_contributions == pytest.approx(23_000, abs=1), \
                f"Year {on.year}: maximize=True contributed {on.annual_retirement_contributions:.0f}, expected 23000"
            assert off.annual_retirement_contributions == pytest.approx(23_000, abs=1), \
                f"Year {on.year}: maximize=False contributed {off.annual_retirement_contributions:.0f}, expected 23000"
            assert abs(on.retirement_balance - off.retirement_balance) < 1.0, \
                f"Year {on.year}: retirement balance differs — contribution amount must not change"

class TestIncomeChangeEvent:

    def _plan(self, events, salary_growth=0.0):
        return FinancialPlan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=salary_growth),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events,
            projection_years=4,
        )

    def test_income_snaps_to_new_value(self):
        s = ProjectionEngine(self._plan([TimelineEvent(year=2, description='Raise', income_change=150_000)])).run_deterministic()
        assert s[0].gross_income == pytest.approx(100_000, abs=1)
        assert s[1].gross_income == pytest.approx(150_000, abs=1)

    def test_salary_growth_applies_from_new_base(self):
        """After income_change, salary_growth_rate compounds from the new base."""
        s = ProjectionEngine(self._plan(
            [TimelineEvent(year=2, description='Raise', income_change=150_000)],
            salary_growth=0.05,
        )).run_deterministic()
        assert s[1].gross_income == pytest.approx(150_000, abs=1)
        assert s[2].gross_income == pytest.approx(150_000 * 1.05, abs=1)


class TestOneTimeEvents:

    def _plan(self, liquid, events):
        return FinancialPlan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=liquid, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events,
            projection_years=3,
        )

    def test_expense_drains_brokerage_in_event_year(self):
        s = ProjectionEngine(self._plan(200_000, [TimelineEvent(year=2, description='Wedding', extra_one_time_expense=30_000)])).run_deterministic()
        expected = s[0].brokerage_balance - 30_000 + s[1].annual_breathing_room
        assert abs(s[1].brokerage_balance - expected) < 1

    def test_expense_only_hits_once(self):
        s = ProjectionEngine(self._plan(200_000, [TimelineEvent(year=2, description='Wedding', extra_one_time_expense=30_000)])).run_deterministic()
        expected_yr3 = s[1].brokerage_balance + s[2].annual_breathing_room
        assert abs(s[2].brokerage_balance - expected_yr3) < 1

    def test_windfall_adds_to_brokerage_in_event_year(self):
        s = ProjectionEngine(self._plan(50_000, [TimelineEvent(year=2, description='Bonus', extra_one_time_income=100_000)])).run_deterministic()
        expected = s[0].brokerage_balance + 100_000 + s[1].annual_breathing_room
        assert abs(s[1].brokerage_balance - expected) < 1


class TestMultipleEventsSameYear:

    def test_marriage_and_home_purchase_same_year(self):
        plan = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.GEORGIA),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_450),
            lifestyle=LifestyleProfile(annual_medical_oop=3_000, medical_auto_scale=True,
                                       medical_spouse_multiplier=2.0, medical_per_child_annual=1_500,
                                       annual_vacation=5_000, monthly_other_recurring=500),
            investments=InvestmentProfile(current_liquid_cash=200_000, annual_hsa_contribution=4_150,
                                          annual_401k_contribution=23_000, annual_market_return=0.0,
                                          annual_inflation_rate=0.0, annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
            timeline_events=[
                TimelineEvent(year=1, description='Buy', buy_home=True, new_home_price=650_000,
                              new_home_down_payment=130_000, new_home_interest_rate=0.068,
                              sell_current_home=False),
                TimelineEvent(year=1, description='Marry', marriage=True),
                TimelineEvent(year=1, description='Wedding', extra_one_time_expense=25_000),
            ],
            projection_years=3,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].is_married
        assert not s[0].is_renting
        assert s[0].annual_medical_oop == pytest.approx(6_000, abs=50)
        assert s[0].annual_hsa_contributions == pytest.approx(4_150, abs=1)  # stated 4150 honored, not overridden to family 8300
        # Brokerage: started 200k, minus 130k down, minus 13k buyer closing (2% of 650k),
        # minus 25k wedding, plus breathing room
        buyer_closing = 650_000 * 0.02  # default buyer_closing_cost_rate=0.02
        expected_brok = (200_000 - 130_000 - buyer_closing - 25_000) * 1.0 + s[0].annual_breathing_room
        assert abs(s[0].brokerage_balance - expected_brok) < 1


# ============================================================
# Closing cost tests
# ============================================================

class TestClosingCosts:

    def _plan(self, events, liquid=200_000, brokerage=300_000):
        return FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_000),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=liquid,
                current_brokerage_balance=brokerage,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events,
            projection_years=3,
        )

    def test_buyer_closing_costs_deducted(self):
        """Buyer closing costs reduce brokerage by exactly price * rate."""
        rate = 0.02
        price = 650_000
        plan_with = self._plan([TimelineEvent(
            year=1, description="Buy", buy_home=True,
            new_home_price=price, new_home_down_payment=130_000,
            new_home_interest_rate=0.065, sell_current_home=False,
            buyer_closing_cost_rate=rate,
        )])
        plan_without = self._plan([TimelineEvent(
            year=1, description="Buy", buy_home=True,
            new_home_price=price, new_home_down_payment=130_000,
            new_home_interest_rate=0.065, sell_current_home=False,
            buyer_closing_cost_rate=0.0,
        )])
        s_with = ProjectionEngine(plan_with).run_deterministic()
        s_without = ProjectionEngine(plan_without).run_deterministic()
        diff = s_without[0].brokerage_balance - s_with[0].brokerage_balance
        assert abs(diff - price * rate) < 1.0, f"Expected {price*rate:.0f}, got {diff:.0f}"

    def test_buyer_closing_zero_no_extra_deduction(self):
        """Setting buyer_closing_cost_rate=0 charges nothing beyond the down payment."""
        plan_zero = self._plan([TimelineEvent(
            year=1, description="Buy", buy_home=True,
            new_home_price=500_000, new_home_down_payment=100_000,
            new_home_interest_rate=0.065, sell_current_home=False,
            buyer_closing_cost_rate=0.0,
        )])
        plan_two = self._plan([TimelineEvent(
            year=1, description="Buy", buy_home=True,
            new_home_price=500_000, new_home_down_payment=100_000,
            new_home_interest_rate=0.065, sell_current_home=False,
            buyer_closing_cost_rate=0.02,
        )])
        s0 = ProjectionEngine(plan_zero).run_deterministic()
        s2 = ProjectionEngine(plan_two).run_deterministic()
        assert s0[0].brokerage_balance > s2[0].brokerage_balance

    def test_default_buyer_closing_rate_is_two_pct(self):
        ev = TimelineEvent(year=1, description="Buy", buy_home=True,
                           new_home_price=500_000, new_home_down_payment=100_000,
                           new_home_interest_rate=0.065)
        assert ev.buyer_closing_cost_rate == pytest.approx(0.02)

    def test_seller_closing_rate_configurable(self):
        """Lowering seller closing rate from 6% to 4% increases sale proceeds."""
        def owned_plan(seller_rate):
            return FinancialPlan(
                income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(400_000, 100_000, 0.065),
                lifestyle=LifestyleProfile(),
                investments=InvestmentProfile(
                    current_liquid_cash=50_000,
                    annual_market_return=0.0,
                    annual_inflation_rate=0.0,
                    annual_salary_growth_rate=0.0,
                ),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                timeline_events=[TimelineEvent(
                    year=2, description="Sell+Buy", buy_home=True,
                    new_home_price=500_000, new_home_down_payment=100_000,
                    new_home_interest_rate=0.065, sell_current_home=True,
                    seller_closing_cost_rate=seller_rate,
                    buyer_closing_cost_rate=0.0,
                )],
                projection_years=3,
            )
        s6 = ProjectionEngine(owned_plan(0.06)).run_deterministic()
        s4 = ProjectionEngine(owned_plan(0.04)).run_deterministic()
        # 4% seller closing leaves more money than 6%
        assert s4[1].brokerage_balance > s6[1].brokerage_balance
        # Difference ≈ 2% of home value at sale
        diff = s4[1].brokerage_balance - s6[1].brokerage_balance
        assert 5_000 < diff < 15_000, f"Expected ~8k diff, got {diff:.0f}"

    def test_default_seller_closing_rate_is_six_pct(self):
        ev = TimelineEvent(year=1, description="Sell", buy_home=True,
                           new_home_price=500_000, new_home_down_payment=100_000,
                           new_home_interest_rate=0.065, sell_current_home=True)
        assert ev.seller_closing_cost_rate == pytest.approx(0.06)

    def test_both_closing_costs_applied_on_upsize(self):
        """Upsizing: buyer pays closing on new home AND seller pays closing on old home."""
        plan_both = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(400_000, 100_000, 0.065),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=50_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(
                year=2, description="Upsize",
                buy_home=True, sell_current_home=True,
                new_home_price=600_000, new_home_down_payment=120_000,
                new_home_interest_rate=0.065,
                buyer_closing_cost_rate=0.02,
                seller_closing_cost_rate=0.06,
            )],
            projection_years=3,
        )
        plan_no_costs = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(400_000, 100_000, 0.065),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=50_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(
                year=2, description="Upsize",
                buy_home=True, sell_current_home=True,
                new_home_price=600_000, new_home_down_payment=120_000,
                new_home_interest_rate=0.065,
                buyer_closing_cost_rate=0.0,
                seller_closing_cost_rate=0.0,
            )],
            projection_years=3,
        )
        s_both = ProjectionEngine(plan_both).run_deterministic()
        s_none = ProjectionEngine(plan_no_costs).run_deterministic()
        # With costs: lower brokerage
        assert s_both[1].brokerage_balance < s_none[1].brokerage_balance
        # Total cost = buyer (600k*2%) + seller (~400k*6%) ≈ 12k + 24k = 36k
        total_cost = s_none[1].brokerage_balance - s_both[1].brokerage_balance
        assert 30_000 < total_cost < 45_000, f"Expected ~36k, got {total_cost:.0f}"


# ============================================================
# Closing cost tests
# ============================================================

class TestBuyerClosingCosts:

    def _buy_plan(self, buyer_rate, liquid=300_000):
        return FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=liquid, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(
                year=1, description="Buy", buy_home=True,
                new_home_price=650_000, new_home_down_payment=130_000,
                new_home_interest_rate=0.068, sell_current_home=False,
                buyer_closing_cost_rate=buyer_rate,
            )],
            projection_years=2,
        )

    def test_buyer_closing_deducted_from_brokerage(self):
        s_2pct = ProjectionEngine(self._buy_plan(0.02)).run_deterministic()
        s_zero = ProjectionEngine(self._buy_plan(0.00)).run_deterministic()
        diff = s_zero[0].brokerage_balance - s_2pct[0].brokerage_balance
        assert abs(diff - 650_000 * 0.02) < 1, f"Expected {650_000*0.02:.0f}, diff={diff:.0f}"

    def test_buyer_closing_zero_means_no_extra_deduction(self):
        s = ProjectionEngine(self._buy_plan(0.00)).run_deterministic()
        expected = 300_000 - 130_000 + s[0].annual_breathing_room
        assert abs(s[0].brokerage_balance - expected) < 1

    def test_higher_buyer_rate_costs_more(self):
        s_2 = ProjectionEngine(self._buy_plan(0.02)).run_deterministic()
        s_3 = ProjectionEngine(self._buy_plan(0.03)).run_deterministic()
        diff = s_2[0].brokerage_balance - s_3[0].brokerage_balance
        assert abs(diff - 650_000 * 0.01) < 1, f"1% difference should be {650_000*0.01:.0f}, got {diff:.0f}"

    def test_default_rate_is_2pct(self):
        """TimelineEvent default buyer_closing_cost_rate must be 0.02."""
        ev = TimelineEvent(year=1, description="Buy", buy_home=True,
                           new_home_price=500_000, new_home_down_payment=100_000,
                           new_home_interest_rate=0.065)
        assert ev.buyer_closing_cost_rate == pytest.approx(0.02)


class TestSellerClosingCosts:

    def _sell_plan(self, seller_rate):
        return FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(300_000, 100_000, 0.065),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=200_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(
                year=2, description="Sell+Buy", buy_home=True,
                new_home_price=500_000, new_home_down_payment=100_000,
                new_home_interest_rate=0.065, sell_current_home=True,
                seller_closing_cost_rate=seller_rate, buyer_closing_cost_rate=0.0,
            )],
            projection_years=3,
        )

    def test_higher_seller_rate_reduces_proceeds(self):
        s_6 = ProjectionEngine(self._sell_plan(0.06)).run_deterministic()
        s_8 = ProjectionEngine(self._sell_plan(0.08)).run_deterministic()
        diff = s_6[1].brokerage_balance - s_8[1].brokerage_balance
        # Seller closing is applied to the home value at START of yr2 (post-appreciation)
        appreciated = 300_000 * 1.035
        expected_diff = appreciated * (0.08 - 0.06)
        assert abs(diff - expected_diff) < 2, f"Expected diff {expected_diff:.0f}, got {diff:.0f}"

    def test_zero_seller_rate_returns_full_equity(self):
        s = ProjectionEngine(self._sell_plan(0.00)).run_deterministic()
        # With 0% seller rate, all equity should come back
        appreciated = 300_000 * 1.035
        calc = MortgageCalculator(HousingProfile(300_000, 100_000, 0.065))
        exact = {r.year: r.balance for r in calc.full_schedule() if r.month % 12 == 0}
        full_equity = appreciated - exact[1]
        expected_yr2 = s[0].brokerage_balance + full_equity - 100_000 + s[1].annual_breathing_room
        assert abs(s[1].brokerage_balance - expected_yr2) < 2

    def test_default_seller_rate_is_6pct(self):
        ev = TimelineEvent(year=1, description="Sell", buy_home=True,
                           new_home_price=500_000, new_home_down_payment=100_000,
                           new_home_interest_rate=0.065, sell_current_home=True)
        assert ev.seller_closing_cost_rate == pytest.approx(0.06)


# ============================================================
# Regression: current_brokerage_balance included in initial pool
# ============================================================

class TestBrokerageBalance:
    """
    REGRESSION: current_brokerage_balance was missing from the sidebar
    InvestmentProfile constructor, silently defaulting to 0.
    A user with $232k brokerage + $20k liquid saw $0 brokerage in projections.
    """

    def test_brokerage_balance_included_in_initial_pool(self):
        """current_brokerage_balance must be added to the starting brokerage pool."""
        plan = FinancialPlan(
            income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=20_000,
                current_brokerage_balance=100_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=1,
        )
        s = ProjectionEngine(plan).run_deterministic()
        # Initial pool = 20k + 100k = 120k, then + breathing room
        # With zero market return the brokerage starts at 120k + breathing
        assert s[0].brokerage_balance > 100_000,             "current_brokerage_balance not reflected in starting pool"

    def test_zero_brokerage_balance_gives_lower_liquid(self):
        """Omitting current_brokerage_balance must produce lower liquid assets."""
        def make(brokerage):
            return FinancialPlan(
                income=IncomeProfile(100_000, FilingStatus.SINGLE, State.TEXAS),
                housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
                lifestyle=LifestyleProfile(),
                investments=InvestmentProfile(
                    current_liquid_cash=20_000,
                    current_brokerage_balance=brokerage,
                    annual_market_return=0.0, annual_inflation_rate=0.0,
                    annual_salary_growth_rate=0.0,
                ),
                strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
                projection_years=3,
            )
        s_with    = ProjectionEngine(make(100_000)).run_deterministic()
        s_without = ProjectionEngine(make(0)).run_deterministic()
        for w, wo in zip(s_with, s_without):
            assert w.brokerage_balance > wo.brokerage_balance,                 f"Year {w.year}: brokerage with 100k start ({w.brokerage_balance:.0f}) "                 f"should exceed zero-start ({wo.brokerage_balance:.0f})"


# ============================================================
# Regression: childcare hidden when num_children=0
# ============================================================

class TestChildcareWithFutureChildren:
    """
    REGRESSION: the sidebar only showed the monthly_childcare input when
    num_children > 0. Users with 0 current children planning future children
    via timeline events had childcare silently set to $0.
    The engine itself is fine — this tests the data flow so it stays correct.
    """

    def test_childcare_applies_when_child_arrives_via_event(self):
        """monthly_childcare must apply in the year a child arrives via timeline event,
        even though num_children starts at 0."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                monthly_childcare=2_500,
                num_children=0,          # no children today
                annual_vacation=0,
                monthly_other_recurring=0,
                annual_medical_oop=0,
                medical_auto_scale=False,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=200_000,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=2, description="Child", new_child=True)],
            projection_years=4,
        )
        s = ProjectionEngine(plan).run_deterministic()
        # Year 1: no child → no childcare cost
        assert s[0].annual_lifestyle_cost == pytest.approx(0.0, abs=1),             f"Year 1 has no child, childcare should be 0, got {s[0].annual_lifestyle_cost:.0f}"
        # Year 2: child arrives → 2500*12 = 30000
        assert s[1].annual_lifestyle_cost == pytest.approx(30_000, abs=1),             f"Year 2 child arrived, expected 30000 childcare, got {s[1].annual_lifestyle_cost:.0f}"

    def test_childcare_zero_when_monthly_childcare_not_set(self):
        """If monthly_childcare=0 (old default when field was hidden), children
        cost $0 in childcare — this must be visible, not silently wrong."""
        plan = FinancialPlan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(
                monthly_childcare=0,     # not set — the bug scenario
                num_children=0,
                annual_vacation=0, monthly_other_recurring=0,
                annual_medical_oop=0, medical_auto_scale=False,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=200_000, annual_market_return=0.0,
                annual_inflation_rate=0.0, annual_salary_growth_rate=0.0,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=[TimelineEvent(year=1, description="Child", new_child=True)],
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        # Child arrives but childcare=0, so lifestyle cost = 0
        assert s[0].annual_lifestyle_cost == pytest.approx(0.0, abs=1)
        # This is "correct" but wrong for the user — the test documents the danger


# ============================================================
# Dual income: independent growth and partner events
# ============================================================

class TestDualIncome:
    """
    Tests for dual-income household support added in the partner salary session.
    Each income must grow at its own rate independently.
    """

    def _dual_plan(self, primary, partner, primary_growth, partner_growth,
                   events=None, years=5, k401_partner=0):
        return FinancialPlan(
            income=IncomeProfile(
                gross_annual_income=primary,
                spouse_gross_annual_income=partner,
                filing_status=FilingStatus.MARRIED_FILING_JOINTLY,
                state=State.TEXAS,
            ),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(
                current_liquid_cash=100_000,
                annual_401k_contribution=0,
                partner_annual_401k_contribution=k401_partner,
                annual_market_return=0.0,
                annual_inflation_rate=0.0,
                annual_salary_growth_rate=primary_growth,
                partner_salary_growth_rate=partner_growth,
            ),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            timeline_events=events or [],
            projection_years=years,
        )

    def test_combined_income_is_sum(self):
        s = ProjectionEngine(
            self._dual_plan(180_000, 120_000, 0.0, 0.0)
        ).run_deterministic()
        assert s[0].gross_income == pytest.approx(300_000, abs=1)

    def test_primary_grows_independently(self):
        s = ProjectionEngine(
            self._dual_plan(100_000, 100_000, primary_growth=0.10, partner_growth=0.0)
        ).run_deterministic()
        # Year 2: primary = 110k, partner = 100k, total = 210k
        assert s[1].gross_income == pytest.approx(210_000, abs=1),             f"Expected 210000, got {s[1].gross_income:.0f}"

    def test_partner_grows_independently(self):
        s = ProjectionEngine(
            self._dual_plan(100_000, 100_000, primary_growth=0.0, partner_growth=0.10)
        ).run_deterministic()
        # Year 2: primary = 100k, partner = 110k, total = 210k
        assert s[1].gross_income == pytest.approx(210_000, abs=1)

    def test_different_growth_rates_diverge_correctly(self):
        s = ProjectionEngine(
            self._dual_plan(180_000, 120_000, primary_growth=0.05, partner_growth=0.08)
        ).run_deterministic()
        for snap in s:
            expected = 180_000 * (1.05)**(snap.year-1) + 120_000 * (1.08)**(snap.year-1)
            assert abs(snap.gross_income - expected) < 1.0,                 f"Year {snap.year}: expected {expected:.0f}, got {snap.gross_income:.0f}"

    def test_zero_partner_income_single_income_behavior(self):
        """partner=0 must behave identically to a single-income household."""
        dual = ProjectionEngine(
            self._dual_plan(180_000, 0, 0.04, 0.0)
        ).run_deterministic()
        single = ProjectionEngine(FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.MARRIED_FILING_JOINTLY, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=100_000, annual_401k_contribution=0,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.04),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=5,
        )).run_deterministic()
        for d, s in zip(dual, single):
            assert abs(d.gross_income - s.gross_income) < 1.0, f"Year {d.year}"
            assert abs(d.brokerage_balance - s.brokerage_balance) < 1.0, f"Year {d.year}"

    def test_partner_401k_adds_to_retirement(self):
        """Partner's 401k must be tracked separately and added to retirement balance."""
        s = ProjectionEngine(
            self._dual_plan(180_000, 120_000, 0.0, 0.0, k401_partner=20_000)
        ).run_deterministic()
        # With no primary 401k and partner 401k=20k
        assert s[0].annual_retirement_contributions == pytest.approx(20_000, abs=1)

    def test_partner_income_change_event(self):
        """partner_income_change event must update partner salary independently."""
        plan = self._dual_plan(
            180_000, 120_000, 0.0, 0.0,
            events=[TimelineEvent(year=2, description="Partner promotion",
                                  partner_income_change=160_000)],
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].gross_income == pytest.approx(300_000, abs=1)   # yr1: 180k + 120k
        assert s[1].gross_income == pytest.approx(340_000, abs=1)   # yr2: 180k + 160k

    def test_primary_income_change_does_not_affect_partner(self):
        """income_change event for primary must not clobber partner income."""
        plan = self._dual_plan(
            180_000, 120_000, 0.0, 0.0,
            events=[TimelineEvent(year=2, description="Raise", income_change=220_000)],
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[1].gross_income == pytest.approx(340_000, abs=1)   # 220k + 120k


# ============================================================
# Regression: full realistic scenario must not go negative
# ============================================================

class TestRealisticScenario:
    """
    REGRESSION: The full user scenario (180k income, 600k home, adoption expense,
    two children) must not produce negative liquid assets when using stated
    contribution amounts of 23k 401k and 4150 HSA.

    This test encodes the specific combination of bugs that caused the regression:
      - maximize_401k overriding 23k to 30.5k silently
      - Missing current_brokerage_balance in initial pool
    Either bug alone could cause the liquid asset dip.
    """

    def test_600k_home_adoption_never_negative(self):
        plan = FinancialPlan(
            income=IncomeProfile(180_000, FilingStatus.SINGLE, State.GEORGIA,
                                 spouse_gross_annual_income=0),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_450),
            lifestyle=LifestyleProfile(
                monthly_childcare=2_500, num_children=0, num_pets=1,
                annual_pet_cost=2_500, annual_medical_oop=3_000,
                medical_auto_scale=True, medical_spouse_multiplier=2.0,
                medical_per_child_annual=1_500, annual_vacation=10_000,
                monthly_other_recurring=500,
            ),
            investments=InvestmentProfile(
                current_liquid_cash=20_000,
                current_retirement_balance=80_000,
                current_brokerage_balance=232_000,
                annual_401k_contribution=23_000,
                annual_hsa_contribution=4_150,
                annual_market_return=0.08,
                annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.04,
                annual_home_appreciation_rate=0.035,
            ),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
            timeline_events=[
                TimelineEvent(year=1, description="Buy home", buy_home=True,
                              new_home_price=600_000, new_home_down_payment=130_000,
                              new_home_interest_rate=0.068, sell_current_home=False,
                              buyer_closing_cost_rate=0.02),
                TimelineEvent(year=1, description="Get married", marriage=True,
                              extra_one_time_expense=13_000),
                TimelineEvent(year=2, description="First child", new_child=True,
                              extra_one_time_expense=50_000),
                TimelineEvent(year=4, description="Second child", new_child=True),
                TimelineEvent(year=6, description="Second dog", new_pet=True),
            ],
            projection_years=30,
        )
        snaps = ProjectionEngine(plan).run_deterministic()
        negative = [s for s in snaps if s.brokerage_balance < 0]
        assert len(negative) == 0, (
            f"Liquid assets went negative in years: {[s.year for s in negative]}. "
            f"Worst: {min(s.brokerage_balance for s in negative):,.0f} in year "
            f"{min(negative, key=lambda s: s.brokerage_balance).year}. "
            f"Check maximize_401k semantics and current_brokerage_balance inclusion."
        )