"""Tests for the joint block-bootstrap Monte Carlo sampler.

These pin down the two properties the sampler exists to restore, which the
previous independent per-year draws destroyed:

  #13  cross-series co-movement -- equity and inflation are sampled as
       calendar-year-aligned pairs, so stagflation (high inflation + poor real
       equity) is drawn as a bundle rather than as the product of two marginals.
  #14  temporal structure -- contiguous blocks preserve serial correlation
       (inflation is strongly persistent) and mean reversion.

Plus the salary-growth coupling that ties nominal wages to the sampled inflation.
"""
import numpy as np
import pytest

from fintracker import projections as P
from fintracker.projections import (
    ProjectionEngine,
    _stationary_block_indices,
    _coupled_salary_growth,
    _ALIGNED_EQUITY,
    _ALIGNED_INFLATION,
    _N_ALIGNED,
)
from fintracker.models import IncomeProfile, FilingStatus, State
from .builders import make_plan, investments


# --------------------------------------------------------------------------
# Aligned history
# --------------------------------------------------------------------------
class TestAlignedHistory:
    def test_common_window_length(self):
        # 1929-2024 inclusive.
        assert _N_ALIGNED == 96
        assert len(_ALIGNED_EQUITY) == 96
        assert len(_ALIGNED_INFLATION) == 96

    @pytest.mark.parametrize("year, equity, inflation", [
        (1974, -0.2647, 0.1230),   # the archetypal stagflation year
        (2008, -0.3700, 0.0010),   # GFC crash, near-zero inflation
        (2021,  0.2871, 0.0700),   # post-COVID boom + inflation surge
    ])
    def test_known_year_pairs_aligned(self, year, equity, inflation):
        i = year - P._ALIGNED_START_YEAR
        assert _ALIGNED_EQUITY[i] == pytest.approx(equity)
        assert _ALIGNED_INFLATION[i] == pytest.approx(inflation)


# --------------------------------------------------------------------------
# Stationary block bootstrap
# --------------------------------------------------------------------------
class TestStationaryBlockIndices:
    def test_indices_in_range(self):
        rng = np.random.default_rng(0)
        idx = _stationary_block_indices(rng, 500, 40, _N_ALIGNED, 5.0)
        assert idx.shape == (500, 40)
        assert idx.min() >= 0 and idx.max() < _N_ALIGNED

    def test_mean_block_length_matches_target(self):
        # A "continuation" is a step where the index advanced by exactly one
        # (mod N). The fraction of continuations is 1 - p, so the realized mean
        # block length is ~1/p = mean_block_years.
        rng = np.random.default_rng(1)
        target = 5.0
        idx = _stationary_block_indices(rng, 4000, 60, _N_ALIGNED, target)
        step = idx[:, 1:]
        prev = (idx[:, :-1] + 1) % _N_ALIGNED
        continuation_rate = (step == prev).mean()
        realized_mean = 1.0 / (1.0 - continuation_rate)
        assert realized_mean == pytest.approx(target, abs=0.5)

    def test_pairs_are_actual_historical_pairs(self):
        # Every sampled (equity, inflation) point must be a real calendar year's
        # pair -- the joint distribution is exactly the empirical one, never a
        # cross-product that never occurred.
        rng = np.random.default_rng(2)
        idx = _stationary_block_indices(rng, 300, 30, _N_ALIGNED, 5.0)
        hist_pairs = set(zip(_ALIGNED_EQUITY.tolist(), _ALIGNED_INFLATION.tolist()))
        sampled = set(zip(_ALIGNED_EQUITY[idx].ravel().tolist(),
                          _ALIGNED_INFLATION[idx].ravel().tolist()))
        assert sampled <= hist_pairs

    def test_inflation_serial_correlation_preserved(self):
        # Inflation is strongly persistent historically (lag-1 autocorr ~0.6).
        # Block sampling must reproduce clear positive persistence within a path,
        # whereas an IID draw destroys it (autocorr ~0).
        rng = np.random.default_rng(3)
        idx = _stationary_block_indices(rng, 2000, 40, _N_ALIGNED, 5.0)
        inf = _ALIGNED_INFLATION[idx]
        # Pooled lag-1 autocorrelation across all paths.
        a, b = inf[:, :-1].ravel(), inf[:, 1:].ravel()
        block_ac1 = np.corrcoef(a, b)[0, 1]

        iid = rng.choice(_ALIGNED_INFLATION, size=(2000, 40), replace=True)
        c, d = iid[:, :-1].ravel(), iid[:, 1:].ravel()
        iid_ac1 = np.corrcoef(c, d)[0, 1]

        assert block_ac1 > 0.35            # clearly persistent
        assert block_ac1 > iid_ac1 + 0.3   # and far more so than IID
        assert abs(iid_ac1) < 0.05         # IID has essentially none


# --------------------------------------------------------------------------
# Stagflation co-occurrence (#13)
# --------------------------------------------------------------------------
class TestStagflationCoOccurrence:
    @staticmethod
    def _stagflation_rate(equity, inflation):
        real = (1 + equity) / (1 + inflation) - 1
        return ((inflation > 0.08) & (real < 0)).mean()

    def test_historical_baseline(self):
        # 7 of 96 years: high inflation AND negative real equity return.
        rate = self._stagflation_rate(_ALIGNED_EQUITY, _ALIGNED_INFLATION)
        assert rate == pytest.approx(7 / 96, abs=1e-9)

    def test_block_bundles_stagflation_independent_does_not(self):
        rng = np.random.default_rng(4)
        idx = _stationary_block_indices(rng, 4000, 30, _N_ALIGNED, 5.0)
        block_rate = self._stagflation_rate(_ALIGNED_EQUITY[idx], _ALIGNED_INFLATION[idx])

        # Independent draws pair a random equity year with a random inflation
        # year -> co-occurrence collapses toward the product of the marginals.
        ind_eq = rng.choice(_ALIGNED_EQUITY, size=(4000, 30), replace=True)
        ind_inf = rng.choice(_ALIGNED_INFLATION, size=(4000, 30), replace=True)
        ind_rate = self._stagflation_rate(ind_eq, ind_inf)

        hist_rate = 7 / 96   # ~7.3%
        # Block sampling reproduces the true historical frequency...
        assert block_rate == pytest.approx(hist_rate, abs=0.012)
        # ...while independent draws materially under-count it (the flagged bug):
        # pairing a random equity year with a random inflation year pushes the
        # co-occurrence toward the product of the marginals.
        assert ind_rate < 0.75 * hist_rate
        assert block_rate > 1.4 * ind_rate


# --------------------------------------------------------------------------
# Salary growth coupled to inflation
# --------------------------------------------------------------------------
class TestCoupledSalaryGrowth:
    def test_salary_tracks_inflation(self):
        rng = np.random.default_rng(5)
        inflation = rng.choice(_ALIGNED_INFLATION, size=(3000, 20), replace=True)
        sg = _coupled_salary_growth(rng, inflation, real_premium=0.01, std=0.02)
        # Nominal salary growth moves with inflation (strong positive corr)...
        corr = np.corrcoef(sg.ravel(), inflation.ravel())[0, 1]
        assert corr > 0.6
        # ...and the real premium (sg - inflation) centers on its mean.
        real = sg - inflation
        assert real.mean() == pytest.approx(0.01, abs=0.01)

    def test_independent_salary_has_no_inflation_link(self):
        # Baseline: the legacy independent draw is uncorrelated with inflation.
        rng = np.random.default_rng(6)
        inflation = rng.choice(_ALIGNED_INFLATION, size=(3000, 20), replace=True)
        sg = rng.normal(0.04, 0.02, inflation.shape)
        corr = np.corrcoef(sg.ravel(), inflation.ravel())[0, 1]
        assert abs(corr) < 0.05


# --------------------------------------------------------------------------
# End-to-end wiring
# --------------------------------------------------------------------------
class TestRunMonteCarloWiring:
    def _plan(self):
        return make_plan(
            income=IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS),
            investments=investments(
                current_liquid_cash=100_000,
                current_brokerage_balance=400_000,
                annual_market_return=0.07,
                annual_inflation_rate=0.03,
                annual_salary_growth_rate=0.04,
            ),
            projection_years=20,
        )

    def test_block_bootstrap_is_default_and_recorded(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(n_simulations=100, seed=1)
        assert mc.block_bootstrap is True

    def test_disabled_when_inflation_not_historical(self):
        # Joint alignment needs both historical series; drop one and it must
        # fall back to independent draws (flag reports what actually ran).
        mc = ProjectionEngine(self._plan()).run_monte_carlo(
            n_simulations=100, seed=1, use_historical_inflation=False)
        assert mc.block_bootstrap is False

    def test_flag_off_disables_joint_path(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(
            n_simulations=100, seed=1, block_bootstrap=False)
        assert mc.block_bootstrap is False

    def test_percentiles_ordered(self):
        mc = ProjectionEngine(self._plan()).run_monte_carlo(n_simulations=300, seed=2)
        for lo, hi in zip(mc.p10_net_worth, mc.p50_net_worth):
            assert lo <= hi
        for lo, hi in zip(mc.p50_net_worth, mc.p90_net_worth):
            assert lo <= hi

    def test_seeded_reproducible(self):
        eng = ProjectionEngine(self._plan())
        a = eng.run_monte_carlo(n_simulations=150, seed=99)
        b = eng.run_monte_carlo(n_simulations=150, seed=99)
        assert a.p50_net_worth == b.p50_net_worth
