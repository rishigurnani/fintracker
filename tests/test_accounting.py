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
        plan = self._childcare_plan(events=[
            TimelineEvent(year=1, description="Child", new_child=True)
        ], inflation=0.03)
        snaps = ProjectionEngine(plan).run_deterministic()
        for s in snaps:
            inf = (1.03) ** (s.year - 1)
            expected = s.num_children * 2_500 * 12 * inf
            # Isolate childcare: lifestyle = childcare + rent*12*inf (no vacation/other/medical)
            rent_cost = 1_500 * 12 * inf
            implied_childcare = s.annual_lifestyle_cost - rent_cost
            assert implied_childcare == pytest.approx(expected, abs=5), \
                f"Year {s.year}: expected childcare {expected:.0f}, got {implied_childcare:.0f}"


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

    def test_hsa_upgrades_to_family_limit_on_marriage(self):
        """maximize_hsa=True must contribute the full family limit after marriage."""
        s = ProjectionEngine(self._plan([TimelineEvent(year=2, description='Marry', marriage=True)])).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(4_150, abs=1)   # single
        assert s[1].annual_hsa_contributions == pytest.approx(8_300, abs=1)   # family

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

    def test_maximize_hsa_hits_single_irs_limit(self):
        """maximize_hsa=True should contribute exactly the IRS single limit."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_hsa_contribution=1_000,  # user set low — should be overridden
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=True, maximize_401k=False),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(4_150, abs=1)

    def test_maximize_401k_hits_irs_limit(self):
        """maximize_401k=True should contribute exactly the IRS limit regardless of user setting."""
        plan = FinancialPlan(
            income=IncomeProfile(200_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_401k_contribution=5_000,  # user set low
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=True),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_retirement_contributions == pytest.approx(30_500, abs=1)

    def test_maximize_false_uses_user_amount(self):
        """When maximize flags are off, use the user's stated contribution."""
        plan = FinancialPlan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=0),
            lifestyle=LifestyleProfile(),
            investments=InvestmentProfile(current_liquid_cash=200_000,
                                          annual_hsa_contribution=2_000,
                                          annual_401k_contribution=10_000,
                                          annual_market_return=0.0, annual_inflation_rate=0.0,
                                          annual_salary_growth_rate=0.0),
            strategies=StrategyToggles(maximize_hsa=False, maximize_401k=False),
            projection_years=2,
        )
        s = ProjectionEngine(plan).run_deterministic()
        assert s[0].annual_hsa_contributions == pytest.approx(2_000, abs=1)
        assert s[0].annual_retirement_contributions == pytest.approx(10_000, abs=1)


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
        assert s[0].annual_hsa_contributions == pytest.approx(8_300, abs=1)
        # Brokerage: started 200k, minus 130k down, minus 25k wedding, plus breathing room
        expected_brok = (200_000 - 130_000 - 25_000) * 1.0 + s[0].annual_breathing_room
        assert abs(s[0].brokerage_balance - expected_brok) < 1