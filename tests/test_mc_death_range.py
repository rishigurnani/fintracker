"""Monte Carlo realized death-age range (RetirementProfile.death_age_min/max).

The feature draws each simulation's *realized* death year uniformly from the
configured age range and runs that path to its own death (survivors-only past
it). The subtlety these tests pin down: forward-looking forecasts must plan
against the *planning* death (life_expectancy_age), NOT the realized draw — at
75 you can't foresee dying at 85, so you budget medical costs out to your
planning horizon regardless.
"""
import math

import numpy as np
import pytest

from fintracker.models import (
    IncomeProfile, FilingStatus, State, LifestyleProfile, RetirementProfile,
)
from fintracker.projections import ProjectionEngine
from .builders import make_plan, investments, zero_lifestyle


def _plan(**retirement_kw):
    rp = dict(current_age=40, retirement_age=65, life_expectancy_age=90,
              expected_post_retirement_return=0.0)
    rp.update(retirement_kw)
    return make_plan(
        income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
        lifestyle=zero_lifestyle(annual_medical_oop=10_000, annual_self_ltc_cost=100_000,
                                 self_ltc_years_before_death=5),
        investments=investments(current_liquid_cash=2_000_000),
        retirement=RetirementProfile(**rp),
        projection_years=60,
    )


# ---------------------------------------------------------------------------
# Death-year bounds + horizon
# ---------------------------------------------------------------------------

class TestDeathBounds:
    def test_bounds_none_without_both_fields(self):
        assert ProjectionEngine(_plan())._mc_death_bounds() is None
        assert ProjectionEngine(_plan(death_age_min=80))._mc_death_bounds() is None
        assert ProjectionEngine(_plan(death_age_max=100))._mc_death_bounds() is None

    def test_bounds_convert_ages_to_projection_years(self):
        # current_age 40 → age A is projection year A-40+1.
        eng = ProjectionEngine(_plan(death_age_min=80, death_age_max=100))
        assert eng._mc_death_bounds() == (41, 61)

    def test_bounds_ordered_defensively(self):
        # min/max supplied backwards still yields an ordered (lo, hi) range.
        lo, hi = ProjectionEngine(_plan(death_age_min=100, death_age_max=80))._mc_death_bounds()
        assert lo <= hi

    def test_mc_horizon_runs_to_top_of_range(self):
        # death_age_max 100 → sims must be able to run to projection year 61.
        mc = ProjectionEngine(_plan(death_age_min=80, death_age_max=100)).run_monte_carlo(
            n_simulations=64, seed=1)
        assert mc.years[-1] == 61


# ---------------------------------------------------------------------------
# LTC gate is relative to whichever death year the caller supplies
# ---------------------------------------------------------------------------

class TestSelfLtcGate:
    def test_final_n_years_inclusive(self):
        # _plan configures self_ltc_years_before_death = 5.
        eng = ProjectionEngine(_plan())
        assert eng._self_ltc_active(30, 30) is True     # death year itself
        assert eng._self_ltc_active(26, 30) is True     # 4 before → inside N=5
        assert eng._self_ltc_active(25, 30) is False    # 5 before → outside N=5
        assert eng._self_ltc_active(30, None) is False  # no death year → never


# ---------------------------------------------------------------------------
# Per-sim realized death + survivors-only padding
# ---------------------------------------------------------------------------

class TestRealizedDeathPadding:
    def test_run_sim_rows_pads_after_realized_death(self):
        eng = ProjectionEngine(_plan(death_age_min=80, death_age_max=100))
        years = list(range(1, 61))
        n = 3
        z = np.zeros((n, len(years)))
        all_death = np.array([5, 20, 60])          # die in projection years 5, 20, 60
        nw, liq, _ = eng._run_sim_rows(z, z, z, years, {}, all_death)
        for row, death in zip(nw, all_death):
            alive = row[:death]                     # years 1..death
            dead = row[death:]                      # strictly after death
            assert not np.isnan(alive).any()        # lived years are real values
            assert np.isnan(dead).all()             # post-death years are NaN

    def test_no_range_leaves_full_rows(self):
        # Without a range the deterministic death governs; no NaN padding.
        eng = ProjectionEngine(_plan())              # life_expectancy 90 → year 51
        years = list(range(1, eng._horizon() + 1))
        z = np.zeros((2, len(years)))
        nw, _, _ = eng._run_sim_rows(z, z, z, years, {})
        assert not np.isnan(nw).any()


# ---------------------------------------------------------------------------
# THE crux: planning vs realized death
# ---------------------------------------------------------------------------

class TestPlanningVsRealizedDeath:
    def test_medical_forecast_ignores_realized_death(self):
        # Two fresh engines (independent forecast caches). One state dies far early
        # (realized), the other at the planning horizon. The medical-burden forecast
        # must be identical: it plans to life_expectancy_age either way.
        eng_a = ProjectionEngine(_plan(death_age_min=65, death_age_max=90))
        st_a = eng_a._initial_state()
        st_a.realized_death_year = 26                 # realized death at age 65

        eng_b = ProjectionEngine(_plan(death_age_min=65, death_age_max=90))
        st_b = eng_b._initial_state()                 # realized = planning death

        pv_early = eng_a._pv_future_medical(st_a, 10)
        pv_plan = eng_b._pv_future_medical(st_b, 10)
        assert pv_early == pytest.approx(pv_plan)
        assert pv_early > 0                           # forecast is non-trivial

    def test_forecast_includes_ltc_at_planning_horizon(self):
        # The forecast's LTC piece is gated on the PLANNING death, so PV medical is
        # strictly larger with LTC configured than without — even though a realized
        # early death might never actually reach the LTC years.
        with_ltc = ProjectionEngine(_plan())
        pv_with = with_ltc._pv_future_medical(with_ltc._initial_state(), 10)

        no_ltc_plan = _plan()
        no_ltc_plan.lifestyle.annual_self_ltc_cost = 0.0
        eng2 = ProjectionEngine(no_ltc_plan)
        pv_without = eng2._pv_future_medical(eng2._initial_state(), 10)
        assert pv_with > pv_without

    def test_actual_ltc_follows_realized_death(self):
        # In the actual projection, LTC occurs in the realized final years. A sim
        # dying at projection year 30 incurs LTC in its last 5 years (26..30), not
        # near the planning death (year 51).
        eng = ProjectionEngine(_plan(death_age_min=80, death_age_max=100))
        years = list(range(1, 61))
        state = eng._initial_state()
        state.realized_death_year = 30
        # Drive one path manually to read per-year LTC.
        ltc_years = []
        for i, year in enumerate(years):
            if year > state.realized_death_year:
                break
            eng._apply_timeline_events(state, year, {})
            eng._evaluate_failsafes(state, year)
            snap = eng._compute_year(state, year, market_return_override=0.0,
                                     inflation_override=0.0, salary_growth_override=0.0)
            if snap.annual_self_ltc_cost > 0:
                ltc_years.append(year)
            eng._advance_state(state, snap, market_return=0.0, inflation=0.0, salary_growth=0.0)
        assert ltc_years == [26, 27, 28, 29, 30]      # final 5 years before realized death


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_seeded_death_range_reproducible(self):
        p = _plan(death_age_min=75, death_age_max=100)
        a = ProjectionEngine(p).run_monte_carlo(n_simulations=200, seed=7)
        b = ProjectionEngine(p).run_monte_carlo(n_simulations=200, seed=7)
        assert a.p50_net_worth == b.p50_net_worth
        assert a.prob_negative_liquid == b.prob_negative_liquid

    def test_prob_negative_liquid_valid_probabilities(self):
        mc = ProjectionEngine(_plan(death_age_min=75, death_age_max=100)).run_monte_carlo(
            n_simulations=200, seed=3)
        assert all(0.0 <= p <= 1.0 for p in mc.prob_negative_liquid)
        assert all(math.isfinite(x) for x in mc.p50_net_worth)
