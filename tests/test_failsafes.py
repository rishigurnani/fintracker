"""Tests for failsafes — conditional events that fire on the running state.

The fixture plan bleeds its brokerage every year (spending > income, partner not
working), so a "brokerage below $X" trigger is guaranteed to fire at a
predictable point, letting us pin down timing, delay, duration, present-value
indexing, the one-shot latch, and the Monte Carlo liquidity effect.
"""
import copy

import pytest

from fintracker.models import (
    IncomeProfile, FilingStatus, State, LifestyleProfile,
    Failsafe, FailsafeCondition, FailsafeAction,
)
from fintracker.projections import ProjectionEngine
from fintracker.config import _dict_to_plan, _plan_to_dict
from .builders import make_plan, investments


def declining_plan(years=12, **overrides):
    """A plan whose brokerage falls ~$15k/yr, crossing $100k PV near year 6-7."""
    return make_plan(
        income=IncomeProfile(90_000, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=LifestyleProfile(monthly_other_recurring=6_500,
                                   annual_medical_oop=0, medical_auto_scale=False),
        investments=investments(current_liquid_cash=160_000,
                                annual_inflation_rate=0.03, annual_market_return=0.02),
        projection_years=years,
        **overrides,
    )


def partner_income_by_year(plan):
    """Run the deterministic loop, returning [(year, income_partner, working)]."""
    eng = ProjectionEngine(plan)
    st = eng._initial_state()
    out = []
    for y in range(1, plan.projection_years + 1):
        eng._apply_timeline_events(st, y)
        eng._evaluate_failsafes(st, y)
        out.append((y, st.income_partner, st.is_partner_working, st.cumulative_inflation))
        snap = eng._compute_year(st, y)
        eng._advance_state(st, snap)
    return out


def make_failsafe(**overrides):
    kw = dict(
        name="partner returns to work",
        match="any",
        delay_years=1,
        duration_years=5,
        once=True,
        conditions=[FailsafeCondition(metric="brokerage_balance", comparator="below",
                                      threshold=100_000, present_value=True)],
        action=FailsafeAction(partner_income=100_000, present_value=True),
    )
    kw.update(overrides)
    return Failsafe(**kw)


class TestFiringAndTiming:
    def test_fires_when_threshold_crossed(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe()]
        active = [(y, inc) for y, inc, working, _ in partner_income_by_year(plan) if working and inc > 0]
        assert active, "failsafe never fired"

    def test_baseline_partner_stays_unemployed_without_failsafe(self):
        plan = declining_plan()
        assert all(not working for _, _, working, _ in partner_income_by_year(plan))

    def test_one_year_delay(self):
        # The condition crosses at some trigger year T; income must start at T+1,
        # so there is exactly one bleeding year with the partner still unemployed.
        plan = declining_plan()
        plan.failsafes = [make_failsafe(delay_years=1)]
        rows = partner_income_by_year(plan)
        first_active = next(y for y, inc, working, _ in rows if working)

        plan0 = declining_plan()
        plan0.failsafes = [make_failsafe(delay_years=0)]
        first_active_0 = next(y for y, inc, working, _ in partner_income_by_year(plan0) if working)
        assert first_active == first_active_0 + 1

    def test_duration_then_revert(self):
        plan = declining_plan(years=20)
        plan.failsafes = [make_failsafe(duration_years=5)]
        active_years = [y for y, inc, working, _ in partner_income_by_year(plan) if working]
        assert len(active_years) == 5
        # Contiguous block, then unemployed again afterwards.
        assert active_years == list(range(active_years[0], active_years[0] + 5))

    def test_permanent_when_duration_none(self):
        plan = declining_plan(years=20)
        plan.failsafes = [make_failsafe(duration_years=None)]
        rows = partner_income_by_year(plan)
        first = next(y for y, inc, working, _ in rows if working)
        # Once on, stays on through the horizon.
        assert all(working for y, inc, working, _ in rows if y >= first)


class TestPresentValueIndexing:
    def test_income_is_present_value_indexed(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(action=FailsafeAction(partner_income=100_000, present_value=True))]
        for y, inc, working, cum_infl in partner_income_by_year(plan):
            if working:
                assert inc == pytest.approx(100_000 * cum_infl, rel=1e-9)

    def test_nominal_action_not_indexed(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(action=FailsafeAction(partner_income=100_000, present_value=False))]
        for y, inc, working, cum_infl in partner_income_by_year(plan):
            if working:
                assert inc == pytest.approx(100_000, rel=1e-9)


class TestConditionsWindowAndLatch:
    def test_window_prevents_firing(self):
        # Crossing happens ~year 6-7; a window that closes at year 3 never arms.
        plan = declining_plan()
        plan.failsafes = [make_failsafe(conditions=[
            FailsafeCondition(metric="brokerage_balance", comparator="below",
                              threshold=100_000, present_value=True,
                              start_year=1, end_year=3)])]
        assert all(not working for _, _, working, _ in partner_income_by_year(plan))

    def test_fires_once_only(self):
        plan = declining_plan(years=20)
        plan.failsafes = [make_failsafe(duration_years=3, once=True)]
        # With once=True the block appears exactly once even though the brokerage
        # stays below threshold for the rest of the projection.
        active_years = [y for y, inc, working, _ in partner_income_by_year(plan) if working]
        assert active_years == list(range(active_years[0], active_years[0] + 3))

    def test_present_value_vs_nominal_threshold_differ(self):
        # Nominal brokerage stays above $100k a year longer than PV does, so a
        # nominal-threshold failsafe fires later than a present-value one.
        pv = declining_plan(); pv.failsafes = [make_failsafe(conditions=[
            FailsafeCondition(metric="brokerage_balance", comparator="below",
                              threshold=100_000, present_value=True)])]
        nom = declining_plan(); nom.failsafes = [make_failsafe(conditions=[
            FailsafeCondition(metric="brokerage_balance", comparator="below",
                              threshold=100_000, present_value=False)])]
        first_pv = next(y for y, i, w, _ in partner_income_by_year(pv) if w)
        first_nom = next(y for y, i, w, _ in partner_income_by_year(nom) if w)
        assert first_nom > first_pv


class TestMatchModes:
    def test_match_any_fires_on_single_condition(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(match="any", conditions=[
            FailsafeCondition("brokerage_balance", "below", 100_000, True),
            FailsafeCondition("net_worth", "below", 0, True),   # never true here
        ])]
        assert any(w for _, _, w, _ in partner_income_by_year(plan))

    def test_match_all_requires_both(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(match="all", conditions=[
            FailsafeCondition("brokerage_balance", "below", 100_000, True),
            FailsafeCondition("net_worth", "below", 0, True),   # never true here
        ])]
        assert all(not w for _, _, w, _ in partner_income_by_year(plan))


class TestMetrics:
    def test_unknown_metric_raises(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(conditions=[
            FailsafeCondition("not_a_metric", "below", 1, True)])]
        with pytest.raises(ValueError, match="Unknown failsafe metric"):
            partner_income_by_year(plan)

    def test_unknown_comparator_raises(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(conditions=[
            FailsafeCondition("brokerage_balance", "sideways", 1, True)])]
        with pytest.raises(ValueError, match="Unknown failsafe comparator"):
            partner_income_by_year(plan)


class TestMonteCarlo:
    def test_reduces_peak_liquidity_risk(self):
        # The whole point: rescue income shows up in the bad-luck paths, so the
        # worst-year probability of negative liquid assets falls.
        base = declining_plan(years=15)
        withfs = copy.deepcopy(base)
        withfs.failsafes = [make_failsafe(duration_years=6)]

        mc0 = ProjectionEngine(base).run_monte_carlo(n_simulations=800, seed=7)
        mc1 = ProjectionEngine(withfs).run_monte_carlo(n_simulations=800, seed=7)
        assert max(mc1.prob_negative_liquid) < max(mc0.prob_negative_liquid)

    def test_fire_rate_recorded_and_path_dependent(self):
        # MC samples ~10%-mean historical returns, so even the bleeding plan only
        # crosses the threshold on some paths: the recorded rate is strictly
        # between 0 and 1, keyed by failsafe name. That path-dependence is the
        # whole point of surfacing the metric.
        plan = declining_plan(years=15)
        plan.failsafes = [make_failsafe(name="rescue", duration_years=6)]
        mc = ProjectionEngine(plan).run_monte_carlo(n_simulations=400, seed=5)
        assert set(mc.failsafe_fire_rates) == {"rescue"}
        assert 0.0 < mc.failsafe_fire_rates["rescue"] < 1.0

    def test_never_triggered_reports_zero(self):
        # A threshold no path can reach records a 0% fire rate (not a missing key).
        plan = declining_plan(years=15)
        plan.failsafes = [make_failsafe(name="rescue", conditions=[
            FailsafeCondition("net_worth", "below", -1e12, True)])]
        mc = ProjectionEngine(plan).run_monte_carlo(n_simulations=200, seed=5)
        assert mc.failsafe_fire_rates == {"rescue": 0.0}

    def test_no_failsafes_means_empty_rates(self):
        mc = ProjectionEngine(declining_plan()).run_monte_carlo(n_simulations=100, seed=1)
        assert mc.failsafe_fire_rates == {}

    def test_fires_path_dependently(self):
        # In a healthier plan the failsafe fires in some sims but not all — its
        # presence still cannot raise liquidity risk.
        base = make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            investments=investments(current_liquid_cash=120_000,
                                    current_brokerage_balance=180_000),
            projection_years=20,
        )
        withfs = copy.deepcopy(base)
        withfs.failsafes = [make_failsafe(duration_years=5)]
        mc0 = ProjectionEngine(base).run_monte_carlo(n_simulations=600, seed=3)
        mc1 = ProjectionEngine(withfs).run_monte_carlo(n_simulations=600, seed=3)
        assert max(mc1.prob_negative_liquid) <= max(mc0.prob_negative_liquid) + 1e-9


class TestRecurringActions:
    """Suspend-contributions and vacation-cut are per-year (once=false, dur=1):
    they apply in every year the trigger holds and lift the moment it doesn't."""

    def _recurring(self, action):
        # Fires in any year brokerage < $100k PV, for that year only.
        return make_failsafe(name="belt-tighten", once=False, delay_years=0,
                             duration_years=1, action=action)

    def _yearly(self, plan):
        eng = ProjectionEngine(plan)
        st = eng._initial_state()
        rows = []
        for y in range(1, plan.projection_years + 1):
            eng._apply_timeline_events(st, y)
            eng._evaluate_failsafes(st, y)
            snap = eng._compute_year(st, y)
            rows.append((y, snap))
            eng._advance_state(st, snap)
        return rows

    def test_suspends_retirement_contributions_only_while_below(self):
        from fintracker.models import StrategyToggles, InvestmentProfile
        plan = declining_plan(years=12)
        plan.strategies = StrategyToggles(maximize_401k=True, maximize_hsa=False)
        plan.investments.annual_401k_contribution = 20_000
        plan.failsafes = [self._recurring(FailsafeAction(suspend_retirement_contributions=True))]
        rows = self._yearly(plan)
        # Early years (brokerage healthy) contribute; once brokerage is below the
        # threshold, contributions are zero.
        early = [snap.annual_retirement_contributions for y, snap in rows[:2]]
        late = [snap.annual_retirement_contributions for y, snap in rows[-3:]]
        assert all(c > 0 for c in early)
        assert all(c == 0 for c in late)

    def test_baseline_keeps_contributing(self):
        from fintracker.models import StrategyToggles
        plan = declining_plan(years=12)
        plan.strategies = StrategyToggles(maximize_401k=True, maximize_hsa=False)
        plan.investments.annual_401k_contribution = 20_000
        rows = self._yearly(plan)
        assert all(snap.annual_retirement_contributions > 0 for y, snap in rows)

    def test_vacation_cut_to_present_value_while_below(self):
        from fintracker.models import LifestyleProfile
        # All non-vacation lifestyle lines are inflation-flat in present value, so
        # the only today's-dollars change is the $20k -> $4k vacation cut = $16k.
        base_lifestyle = dict(annual_vacation=20_000, monthly_other_recurring=6_500,
                              annual_medical_oop=0, medical_auto_scale=False)
        plan = declining_plan(years=12)
        plan.lifestyle = LifestyleProfile(**base_lifestyle)
        plan.failsafes = [self._recurring(FailsafeAction(annual_vacation=4_000, present_value=True))]
        rows = self._yearly(plan)

        # Year 1 the brokerage is healthy (no cut); a late year is below $100k.
        first, last = rows[0][1], rows[-1][1]
        first_pv = first.to_todays_dollars(first.annual_lifestyle_cost)
        last_pv = last.to_todays_dollars(last.annual_lifestyle_cost)
        assert first_pv - last_pv == pytest.approx(16_000, abs=1_500)


class TestDurationSemantics:
    def test_duration_zero_is_permanent_like_none(self):
        # Regression: duration_years=0 ("permanent" per the UI/YAML convention)
        # must NOT produce an empty [start, start-1] window. 0 behaves like None.
        from fintracker.models import LifestyleProfile
        base = make_plan(
            lifestyle=LifestyleProfile(annual_medical_oop=10_000, medical_auto_scale=False),
            projection_years=10)

        def medical(dur):
            p = copy.deepcopy(base)
            p.failsafes = [Failsafe(
                name="cut", match="any", delay_years=0, duration_years=dur, once=True,
                conditions=[FailsafeCondition("net_worth", "above", 0, present_value=False)],
                action=FailsafeAction(medical_cost_multiplier=0.5))]
            return [s.annual_medical_oop for s in ProjectionEngine(p).run_deterministic()]

        base_run = [s.annual_medical_oop for s in ProjectionEngine(base).run_deterministic()]
        none_run, zero_run = medical(None), medical(0)
        assert zero_run == none_run            # 0 and None are the same thing
        for m0, mz in zip(base_run, zero_run):  # and both actually halve every year
            assert mz == pytest.approx(m0 * 0.5)
        assert base_run[-1] > 0                 # guard: there was something to cut

    def test_end_year_zero_means_to_horizon(self):
        # Same sentinel class as duration: end_year=0 must mean "to horizon",
        # not literal year 0 (which would make the window empty).
        from fintracker.models import LifestyleProfile
        base = make_plan(
            lifestyle=LifestyleProfile(annual_medical_oop=10_000, medical_auto_scale=False),
            projection_years=10)

        def medical(end_year):
            p = copy.deepcopy(base)
            p.failsafes = [Failsafe(
                name="cut", delay_years=0, duration_years=None, once=True,
                conditions=[FailsafeCondition("net_worth", "above", 0, present_value=False,
                                              start_year=1, end_year=end_year)],
                action=FailsafeAction(medical_cost_multiplier=0.5))]
            return [s.annual_medical_oop for s in ProjectionEngine(p).run_deterministic()]

        assert medical(0) == medical(None)      # 0 and None both mean "to horizon"
        base_run = [s.annual_medical_oop for s in ProjectionEngine(base).run_deterministic()]
        for m0, mz in zip(base_run, medical(0)):
            assert mz == pytest.approx(m0 * 0.5)  # actually fires, not an empty window


class TestMedicalMultiplier:
    def test_halves_medical_costs_while_active(self):
        from fintracker.models import LifestyleProfile
        # A retiree with real medical OOP so the cut is visible; permanent action.
        from fintracker.models import RetirementProfile
        lif = LifestyleProfile(annual_vacation=0, monthly_other_recurring=0,
                               annual_medical_oop=12_000, medical_auto_scale=False,
                               annual_self_ltc_cost=0)
        base = make_plan(lifestyle=lif, projection_years=8,
                         retirement=RetirementProfile(current_age=60, retirement_age=62))
        withfs = copy.deepcopy(base)
        withfs.failsafes = [make_failsafe(name="move abroad", delay_years=0, duration_years=None,
            conditions=[FailsafeCondition("net_worth", "above", 0, True)],  # always true
            action=FailsafeAction(medical_cost_multiplier=0.5))]

        b = ProjectionEngine(base).run_deterministic()
        f = ProjectionEngine(withfs).run_deterministic()
        # Medical OOP is halved every year the failsafe is active.
        for sb, sf in zip(b, f):
            assert sf.annual_medical_oop == pytest.approx(sb.annual_medical_oop * 0.5)
        assert b[0].annual_medical_oop > 0  # guard: there was something to halve

    def test_multiplier_absent_leaves_costs_unchanged(self):
        from fintracker.models import LifestyleProfile
        plan = make_plan(lifestyle=LifestyleProfile(annual_medical_oop=8_000, medical_auto_scale=False),
                         projection_years=5)
        plan.failsafes = [make_failsafe(action=FailsafeAction(partner_income=50_000))]
        # No medical multiplier on this action -> medical costs untouched.
        snaps = ProjectionEngine(plan).run_deterministic()
        assert all(s.annual_medical_oop > 0 for s in snaps)


class TestMedicalBurdenRatio:
    def _burden_plan(self, **overrides):
        from fintracker.models import LifestyleProfile, RetirementProfile
        return make_plan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=LifestyleProfile(annual_medical_oop=18_000, medical_auto_scale=False,
                                       annual_self_ltc_cost=90_000, self_ltc_start_age=75,
                                       annual_health_insurance_premium=9_000),
            investments=investments(current_liquid_cash=120_000, current_retirement_balance=150_000,
                                    annual_market_return=0.05, annual_inflation_rate=0.03,
                                    annual_healthcare_inflation_rate=0.05),
            retirement=RetirementProfile(current_age=30, retirement_age=60,
                                         expected_post_retirement_return=0.04),
            projection_years=55, **overrides,
        )

    def test_pv_future_medical_positive_and_deterministic(self):
        plan = self._burden_plan()
        eng = ProjectionEngine(plan)
        st = eng._initial_state()
        a = eng._pv_future_medical(st, 35)
        b = eng._pv_future_medical(st, 35)
        assert a == b and a > 0  # deterministic, positive

    def test_ratio_metric_is_pv_over_net_worth(self):
        plan = self._burden_plan()
        eng = ProjectionEngine(plan)
        st = eng._initial_state()
        for y in (35, 45):
            pv = eng._pv_future_medical(st, y)
            nw = eng._failsafe_metric(st, "net_worth", y)
            assert eng._failsafe_metric(st, "medical_burden_ratio", y) == pytest.approx(pv / nw)

    def test_move_abroad_triggers_and_halves_medical(self):
        base = self._burden_plan()
        withfs = copy.deepcopy(base)
        withfs.failsafes = [Failsafe(
            name="move abroad", match="any", delay_years=0, duration_years=None, once=True,
            conditions=[FailsafeCondition("medical_burden_ratio", "above", 0.5,
                                          present_value=False, start_year=35, end_year=45)],
            action=FailsafeAction(medical_cost_multiplier=0.5))]
        b = ProjectionEngine(base).run_deterministic()
        f = ProjectionEngine(withfs).run_deterministic()
        # Before the window it can't fire; medical is identical.
        assert f[33].annual_medical_oop == pytest.approx(b[33].annual_medical_oop)
        # From year 35 on (ratio ~1.7 > 0.5), medical OOP is halved.
        for yr in (35, 40, 50):
            assert f[yr - 1].annual_medical_oop == pytest.approx(b[yr - 1].annual_medical_oop * 0.5)

    def test_high_threshold_never_triggers(self):
        base = self._burden_plan()
        withfs = copy.deepcopy(base)
        withfs.failsafes = [Failsafe(
            name="move abroad", delay_years=0, duration_years=None, once=True,
            conditions=[FailsafeCondition("medical_burden_ratio", "above", 100.0,
                                          present_value=False, start_year=35, end_year=45)],
            action=FailsafeAction(medical_cost_multiplier=0.5))]
        b = ProjectionEngine(base).run_deterministic()
        f = ProjectionEngine(withfs).run_deterministic()
        for sb, sf in zip(b, f):
            assert sf.annual_medical_oop == pytest.approx(sb.annual_medical_oop)

    def test_present_value_flag_ignored_for_ratio(self):
        # The ratio is unit-free, so present_value must not deflate it: True/False
        # produce the same trigger behaviour.
        def run(pv_flag):
            plan = self._burden_plan()
            plan.failsafes = [Failsafe(
                name="x", delay_years=0, duration_years=None, once=True,
                conditions=[FailsafeCondition("medical_burden_ratio", "above", 0.5,
                                              present_value=pv_flag, start_year=35, end_year=45)],
                action=FailsafeAction(medical_cost_multiplier=0.5))]
            return [s.annual_medical_oop for s in ProjectionEngine(plan).run_deterministic()]
        assert run(True) == run(False)


class TestConfigRoundTrip:
    def test_yaml_roundtrip_preserves_failsafe(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(conditions=[
            FailsafeCondition("net_worth", "below", 250_000, True, start_year=15, end_year=30),
            FailsafeCondition("brokerage_balance", "below", 100_000, True),
        ], action=FailsafeAction(partner_income=100_000, one_time_expense=5_000, present_value=True))]

        restored = _dict_to_plan(_plan_to_dict(plan))
        assert len(restored.failsafes) == 1
        fs = restored.failsafes[0]
        assert fs.name == "partner returns to work"
        assert fs.match == "any" and fs.delay_years == 1 and fs.duration_years == 5
        assert fs.once is True
        assert len(fs.conditions) == 2
        assert fs.conditions[0].metric == "net_worth"
        assert fs.conditions[0].start_year == 15 and fs.conditions[0].end_year == 30
        assert fs.conditions[1].metric == "brokerage_balance"
        assert fs.action.partner_income == 100_000
        assert fs.action.one_time_expense == 5_000

    def test_roundtrip_preserves_recurring_action_fields(self):
        plan = declining_plan()
        plan.failsafes = [make_failsafe(name="belt-tighten", once=False, duration_years=1,
            action=FailsafeAction(suspend_retirement_contributions=True,
                                  annual_vacation=4_000, present_value=True))]
        fs = _dict_to_plan(_plan_to_dict(plan)).failsafes[0]
        assert fs.once is False and fs.duration_years == 1
        assert fs.action.suspend_retirement_contributions is True
        assert fs.action.annual_vacation == 4_000

    def test_roundtrip_preserves_medical_burden_ratio_failsafe(self):
        plan = declining_plan()
        plan.failsafes = [Failsafe(
            name="move abroad", delay_years=0, duration_years=None, once=True,
            conditions=[FailsafeCondition("medical_burden_ratio", "above", 0.5,
                                          present_value=False, start_year=35, end_year=45)],
            action=FailsafeAction(medical_cost_multiplier=0.5))]
        fs = _dict_to_plan(_plan_to_dict(plan)).failsafes[0]
        assert fs.conditions[0].metric == "medical_burden_ratio"
        assert fs.conditions[0].threshold == 0.5
        assert fs.conditions[0].start_year == 35 and fs.conditions[0].end_year == 45
        assert fs.action.medical_cost_multiplier == 0.5

    def test_absent_failsafes_key_when_none(self):
        plan = declining_plan()
        assert "failsafes" not in _plan_to_dict(plan)
