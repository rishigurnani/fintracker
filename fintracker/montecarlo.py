"""
Monte Carlo simulation harness for the projection engine.

Separated from :mod:`fintracker.projections` so the *statistical* concerns —
sampling economic paths (historical bootstrap or normal draws), the joint
block bootstrap, fanning the independent per-simulation loop across worker
processes, and aggregating trajectories into percentile bands — live apart from
the *domain* concern of advancing one financial plan year by year.

``MonteCarloSimulator`` owns the sampling + parallelism + aggregation; it drives
a :class:`~fintracker.projections.ProjectionEngine` (one per worker process) to
run each sampled path. ``ProjectionEngine.run_monte_carlo`` is a thin delegator
to this module.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fintracker.models import HousingProfile
from fintracker.mortgage import MortgageCalculator

# Monte Carlo runs the independent per-simulation loop across processes once the
# batch is large enough to amortise process-startup overhead; below this it stays
# serial (small runs and the test suite skip the pool). Output is identical
# either way — the sims are independent and the RNG is pre-drawn.
_MC_PARALLEL_MIN_SIMS = 1000


# ---------------------------------------------------------------------------
# Historical S&P 500 total annual returns (1926–2025), 100 years.
# Source: provided by user; used for bootstrap sampling in Monte Carlo.
# Bootstrap preserves the true empirical distribution — fat tails, skew,
# and crash years — rather than assuming normality.
# ---------------------------------------------------------------------------
_SP500_HISTORICAL_RETURNS: tuple[float, ...] = (
    # (year, return) — only returns stored, sorted newest-first for readability
    0.1788, 0.2502, 0.2629, -0.1811, 0.2871, 0.1840, 0.3149, -0.0438,
    0.2183, 0.1196, 0.0138, 0.1369, 0.3239, 0.1600, 0.0211, 0.1506,
    0.2646, -0.3700, 0.0549, 0.1579, 0.0491, 0.1088, 0.2868, -0.2210,
    -0.1189, -0.0910, 0.2104, 0.2858, 0.3336, 0.2296, 0.3758, 0.0132,
    0.1008, 0.0762, 0.3047, -0.0310, 0.3169, 0.1661, 0.0525, 0.1867,
    0.3173, 0.0627, 0.2256, 0.2155, -0.0491, 0.3242, 0.1844, 0.0656,
    -0.0718, 0.2384, 0.3720, -0.2647, -0.1466, 0.1898, 0.1431, 0.0401,
    -0.0850, 0.1106, 0.2398, -0.1006, 0.1245, 0.1648, 0.2280, -0.0873,
    0.2689, 0.0047, 0.1196, 0.4336, -0.1078, 0.0656, 0.3156, 0.5262,
    -0.0099, 0.1837, 0.2402, 0.3171, 0.1879, 0.0550, 0.0571, -0.0807,
    0.3644, 0.1975, 0.2590, 0.2034, -0.1159, -0.0978, -0.0041, 0.3112,
    -0.3503, 0.3392, 0.4767, -0.0144, 0.5399, -0.0819, -0.4334, -0.2490,
    -0.0842, 0.4361, 0.3749, 0.1162,
)


# ---------------------------------------------------------------------------
# Historical US CPI annual inflation rates (1929–2024), 96 years.
# Source: US Bureau of Labor Statistics. Used for bootstrap Monte Carlo sampling.
# Fat tails: deflation (Great Depression), 1970s stagflation (13.3%),
# post-WWII spike (18.1%), 2021-22 surge (7%). Normal(3%, 1.5%) misses all of these.
_US_HISTORICAL_INFLATION: tuple[float, ...] = (
     0.0060, -0.0640, -0.0930, -0.1030,  0.0080,  0.0150,  0.0300,  0.0140,
     0.0290, -0.0280,  0.0000,  0.0070,  0.0990,  0.0900,  0.0300,  0.0230,
     0.0220,  0.1810,  0.0880,  0.0300, -0.0210,  0.0590,  0.0600,  0.0080,
     0.0070, -0.0070,  0.0040,  0.0300,  0.0290,  0.0180,  0.0170,  0.0140,
     0.0070,  0.0130,  0.0160,  0.0100,  0.0190,  0.0350,  0.0300,  0.0470,
     0.0620,  0.0560,  0.0330,  0.0340,  0.0870,  0.1230,  0.0690,  0.0490,
     0.0670,  0.0900,  0.1330,  0.1250,  0.0890,  0.0380,  0.0380,  0.0390,
     0.0380,  0.0110,  0.0440,  0.0440,  0.0460,  0.0610,  0.0310,  0.0290,
     0.0270,  0.0270,  0.0250,  0.0330,  0.0170,  0.0160,  0.0270,  0.0340,
     0.0160,  0.0240,  0.0190,  0.0330,  0.0340,  0.0250,  0.0410,  0.0010,
     0.0270,  0.0150,  0.0300,  0.0170,  0.0150,  0.0080,  0.0070,  0.0210,
     0.0210,  0.0190,  0.0230,  0.0140,  0.0700,  0.0650,  0.0340,  0.0290,
)

# ---------------------------------------------------------------------------
# Calendar-year-aligned equity + inflation history (for the joint block
# bootstrap). The two series above are stored in OPPOSITE orders over DIFFERENT
# windows: S&P returns are newest-first over 1926-2025 (index 0 = 2025), CPI
# inflation is oldest-first over 1929-2024 (index 0 = 1929). To sample the two
# jointly we realign them onto their common window, 1929-2024 (96 years), so
# that row i carries the equity return AND the inflation rate from the SAME
# calendar year. This preserves the contemporaneous co-movement that
# independent per-series draws destroy -- e.g. 1974 stays bundled as
# (-26.5% equity, +12.3% inflation), so stagflation is sampled as a unit rather
# than as the product of two marginals.
# ---------------------------------------------------------------------------
_ALIGNED_START_YEAR = 1929
_ALIGNED_END_YEAR = 2024
_N_ALIGNED = _ALIGNED_END_YEAR - _ALIGNED_START_YEAR + 1   # 96 years


def _build_aligned_history() -> tuple[np.ndarray, np.ndarray]:
    """Realign the equity and inflation series onto their common 1929-2024 window.

    Equity is newest-first (index 0 = 2025), so calendar year Y sits at index
    ``2025 - Y``; inflation is oldest-first (index 0 = 1929), so year Y sits at
    index ``Y - 1929``. Returns ``(equity, inflation)`` arrays of length 96,
    both ordered oldest-first (1929 -> 2024) and index-aligned by calendar year.
    """
    equity = np.array([
        _SP500_HISTORICAL_RETURNS[2025 - y]
        for y in range(_ALIGNED_START_YEAR, _ALIGNED_END_YEAR + 1)
    ])
    inflation = np.array([
        _US_HISTORICAL_INFLATION[y - _ALIGNED_START_YEAR]
        for y in range(_ALIGNED_START_YEAR, _ALIGNED_END_YEAR + 1)
    ])
    return equity, inflation


_ALIGNED_EQUITY, _ALIGNED_INFLATION = _build_aligned_history()


def _stationary_block_indices(
    rng: np.random.Generator,
    n_sims: int,
    n_years: int,
    n_history: int,
    mean_block_years: float,
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices into a circular history.

    Returns an ``(n_sims, n_years)`` int array. Each simulation's path starts at
    a uniformly random year; each subsequent year either continues the current
    block (advance one year, wrapping around the end of the history) or, with
    probability ``1 / mean_block_years``, jumps to a fresh uniformly random year.
    The geometric block length (mean = ``mean_block_years``) preserves multi-year
    runs and mean reversion while keeping the sampler stationary -- unlike a
    fixed-length block bootstrap, there are no artificial seams at block
    boundaries.
    """
    p = 1.0 / max(mean_block_years, 1e-9)
    idx = np.empty((n_sims, n_years), dtype=np.int64)
    cur = rng.integers(0, n_history, size=n_sims)
    idx[:, 0] = cur
    for t in range(1, n_years):
        start_new = rng.random(n_sims) < p
        fresh = rng.integers(0, n_history, size=n_sims)
        cur = np.where(start_new, fresh, (cur + 1) % n_history)
        idx[:, t] = cur
    return idx


def _coupled_salary_growth(
    rng: np.random.Generator,
    inflation: np.ndarray,
    real_premium: float,
    std: float,
) -> np.ndarray:
    """Nominal salary growth tied to the sampled inflation.

    ``nominal = inflation + real_premium`` where ``real_premium ~ N(mean, std)``
    is small and far less volatile than inflation. In a high-inflation block
    nominal pay rises but real pay stagnates -- the wage-lags-inflation effect
    that an independent salary draw misses. Clipped to the same bounds the legacy
    independent draw used.
    """
    return np.clip(
        inflation + rng.normal(real_premium, std, inflation.shape), -0.10, 0.20)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Result of N Monte Carlo simulation runs."""
    years: list[int]

    # Net worth percentile bands
    p10_net_worth: list[float]
    p25_net_worth: list[float]
    p50_net_worth: list[float]
    p75_net_worth: list[float]
    p90_net_worth: list[float]
    mean_net_worth: list[float]

    # Liquidity risk: per-year probability that liquid assets (brokerage) go
    # negative in that simulation year.  Values in [0, 1].
    prob_negative_liquid: list[float]

    # Brokerage balance percentiles (same pool as liquid assets chart)
    p10_liquid: list[float]
    p50_liquid: list[float]
    p90_liquid: list[float]

    # Summary statistics
    prob_millionaire_10yr: float = 0.0
    num_simulations: int = 1_000

    # Simulation parameters (stored for display)
    use_historical_returns: bool = True
    use_historical_inflation: bool = True
    market_return_std: float = 0.15
    inflation_std: float = 0.015
    salary_growth_std: float = 0.02
    # Joint block bootstrap (equity+inflation sampled as aligned contiguous
    # blocks; salary tied to sampled inflation). True only when it was actually
    # used, i.e. block_bootstrap requested AND both historical series in play.
    block_bootstrap: bool = True
    mean_block_years: float = 5.0
    # Fraction of simulations in which each failsafe fired at least once,
    # keyed by failsafe name. Empty when the plan has no failsafes.
    failsafe_fire_rates: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Worker-process plumbing
# ---------------------------------------------------------------------------

def _mc_context():
    """Start method for Monte Carlo worker processes.

    macOS/Windows default to ``spawn``, which bootstraps each worker by
    re-importing the parent's ``__main__`` — under ``streamlit run`` that is the
    whole app, so every worker re-runs app.py's module-level Streamlit calls
    (with no runtime → warnings) and re-imports its heavy deps, which wrecks the
    speedup. ``fork`` inherits the parent's memory instead: no re-import, no app
    code re-run. The workers here are pure compute (no logging / inherited-lock
    use), so the usual fork-in-a-threaded-process hazard does not apply. Fall
    back to ``spawn`` only where ``fork`` is unavailable (e.g. Windows).
    """
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _mc_worker(plan, all_mkt, all_inf, all_sg, years, amort_cache):
    """Top-level (picklable) entry point for a Monte Carlo worker process.

    Rebuilds a fresh engine from the (picklable) plan and runs the ordinary
    per-sim loop over its slice of the pre-drawn RNG matrices. Kept module-level
    so it survives the ``spawn`` start method used on macOS.
    """
    from fintracker.projections import ProjectionEngine
    return ProjectionEngine(plan)._run_sim_rows(all_mkt, all_inf, all_sg, years, amort_cache)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """Samples economic paths and aggregates them into a ``MonteCarloResult``.

    Holds only the plan; each :meth:`run` builds the driving ``ProjectionEngine``
    (and the worker processes build their own), so the simulator itself stays
    trivially picklable and free of engine state.
    """

    def __init__(self, plan) -> None:
        self._plan = plan

    def run(
        self,
        n_simulations: int = 1_000,
        seed: Optional[int] = None,
        market_return_std: float = 0.15,
        inflation_std: float = 0.015,
        salary_growth_std: float = 0.02,
        use_historical_returns: bool = True,
        use_historical_inflation: bool = True,
        block_bootstrap: bool = True,
        mean_block_years: float = 5.0,
    ) -> MonteCarloResult:
        """
        Run N Monte Carlo simulations with randomized economic parameters.

        Parameters
        ----------
        n_simulations         : number of simulation runs
        seed                  : RNG seed for reproducibility (None = random)
        use_historical_returns: if True (default), sample market returns by
                                bootstrap from the historical S&P 500 dataset
                                (1926–2025).  This preserves the true empirical
                                distribution — fat tails, skewness, -43% crashes,
                                +54% booms — rather than assuming normality.
                                If False, draws from N(mean, market_return_std).
        market_return_std     : std dev used only when use_historical_returns=False
        inflation_std         : std dev of annual inflation (always normal)
        salary_growth_std     : std dev of annual salary growth (always normal)

        block_bootstrap       : if True (default) and both historical series are
                                in use, sample equity and inflation JOINTLY as
                                calendar-year-aligned pairs drawn in contiguous
                                blocks (Politis-Romano stationary bootstrap). This
                                preserves both the cross-series co-movement
                                (stagflation stays bundled) and the serial
                                correlation / mean reversion that independent
                                per-year draws erase. Salary growth is then tied
                                to the sampled inflation plus a small real premium.
                                If False, falls back to independent per-year draws.
        mean_block_years      : expected block length for the stationary bootstrap
                                (geometric, mean ~5 yrs).

        Outside the joint block-bootstrap path, inflation and salary growth remain
        normally distributed since those datasets are smaller and more symmetric.
        """
        from fintracker.projections import ProjectionEngine
        engine = ProjectionEngine(self._plan)
        rng = np.random.default_rng(seed)
        inv = self._plan.investments
        years = list(range(1, engine._horizon() + 1))

        hist = np.array(_SP500_HISTORICAL_RETURNS)

        amort_cache = self._precompute_amort_cache(engine)

        # Pre-draw all random matrices (n_simulations × n_years) at once.
        # Eliminates per-simulation RNG call overhead.
        n_years = len(years)
        # Joint block bootstrap: sample (equity, inflation) as calendar-year-
        # aligned pairs in contiguous blocks. Requires both historical series --
        # the alignment is what keeps stagflation years bundled. When either
        # series is normal-mode, fall back to the independent per-year draws.
        joint = block_bootstrap and use_historical_returns and use_historical_inflation
        if joint:
            idx = _stationary_block_indices(
                rng, n_simulations, n_years, _N_ALIGNED, mean_block_years)
            all_mkt = _ALIGNED_EQUITY[idx]
            all_inf = np.clip(_ALIGNED_INFLATION[idx], -0.15, 0.20)
            # Salary growth tracks the sampled inflation plus a small, less-
            # volatile real premium (the configured salary growth net of
            # configured inflation). In a high-inflation block nominal pay rises
            # but real pay stagnates -- the wage-lags-inflation stagflation
            # effect an independent salary draw misses entirely.
            real_premium = inv.annual_salary_growth_rate - inv.annual_inflation_rate
            all_sg = _coupled_salary_growth(rng, all_inf, real_premium, salary_growth_std)
        else:
            if use_historical_returns:
                all_mkt = rng.choice(hist, size=(n_simulations, n_years), replace=True)
            else:
                all_mkt = rng.normal(inv.annual_market_return, market_return_std, (n_simulations, n_years))
            if use_historical_inflation:
                all_inf = np.clip(rng.choice(np.array(_US_HISTORICAL_INFLATION),
                                             size=(n_simulations, n_years), replace=True), -0.15, 0.20)
            else:
                all_inf = np.clip(rng.normal(inv.annual_inflation_rate, inflation_std,
                                              (n_simulations, n_years)), 0, 0.15)
            all_sg = np.clip(rng.normal(inv.annual_salary_growth_rate, salary_growth_std,
                                         (n_simulations, n_years)), -0.10, 0.20)

        nw_arr, liq_arr, fired_list = self._run_paths(
            engine, all_mkt, all_inf, all_sg, years, amort_cache, n_simulations)

        return self._aggregate(
            years, nw_arr, liq_arr, fired_list, n_simulations,
            use_historical_returns, use_historical_inflation,
            market_return_std, inflation_std, salary_growth_std,
            joint, mean_block_years)

    def _precompute_amort_cache(self, engine) -> dict:
        """Mortgage amortization lookups for each home-purchase event.

        Identical across every simulation (path-independent), so compute once and
        share the lookup dicts with all paths / worker processes.
        """
        amort_cache: dict = {}
        for ev in self._plan.timeline_events:
            if ev.buy_home and ev.new_home_price and ev.new_home_interest_rate:
                price = ev.new_home_price
                down = ev.new_home_down_payment or price * 0.20
                rate = ev.new_home_interest_rate
                term = self._plan.housing.loan_term_years
                key = (price, down, rate, term)
                if key not in amort_cache:
                    hp = HousingProfile(home_price=price, down_payment=down,
                                        interest_rate=rate, loan_term_years=term)
                    mc = MortgageCalculator(hp, self._plan.investments.annual_home_appreciation_rate)
                    amort_cache[key] = engine._amort_lookup(mc)
        return amort_cache

    @staticmethod
    def _run_paths(engine, all_mkt, all_inf, all_sg, years, amort_cache, n_simulations):
        """Run every sampled path, fanning across processes for large batches.

        The sims are independent and the RNG is pre-drawn, so the parallel result
        is identical to the serial one; small batches (and the test suite) stay
        serial to skip process-startup overhead.
        """
        n_cpu = os.cpu_count() or 1
        if n_simulations >= _MC_PARALLEL_MIN_SIMS and n_cpu > 1:
            bounds = [(k * n_simulations) // n_cpu for k in range(n_cpu + 1)]
            slices = [(bounds[k], bounds[k + 1]) for k in range(n_cpu) if bounds[k] < bounds[k + 1]]
            nw_parts, liq_parts, fired_list = [], [], []
            with ProcessPoolExecutor(max_workers=len(slices), mp_context=_mc_context()) as ex:
                futures = [
                    ex.submit(_mc_worker, engine._plan,
                              all_mkt[s:e], all_inf[s:e], all_sg[s:e], years, amort_cache)
                    for (s, e) in slices
                ]
                for f in futures:                       # gathered in submission order
                    nw_c, liq_c, fired_c = f.result()
                    nw_parts.append(nw_c); liq_parts.append(liq_c); fired_list.extend(fired_c)
            return np.vstack(nw_parts), np.vstack(liq_parts), fired_list
        return engine._run_sim_rows(all_mkt, all_inf, all_sg, years, amort_cache)

    def _aggregate(
        self, years, nw_arr, liq_arr, fired_list, n_simulations,
        use_historical_returns, use_historical_inflation,
        market_return_std, inflation_std, salary_growth_std,
        joint, mean_block_years,
    ) -> MonteCarloResult:
        """Reduce the per-path trajectories to percentile bands + summary stats."""
        failsafe_counts: dict[str, int] = {fs.name: 0 for fs in self._plan.failsafes}
        for fired in fired_list:
            for name in fired:
                failsafe_counts[name] += 1
        failsafe_fire_rates = {name: c / n_simulations for name, c in failsafe_counts.items()}

        # Per-year percentiles / stats, vectorised across sims (axis 0).
        p_nw = np.percentile(nw_arr, [10, 25, 50, 75, 90], axis=0)
        p_liq = np.percentile(liq_arr, [10, 50, 90], axis=0)
        yr10 = nw_arr[:, 9] if nw_arr.shape[1] >= 10 else None

        return MonteCarloResult(
            years=years,
            p10_net_worth=[float(x) for x in p_nw[0]],
            p25_net_worth=[float(x) for x in p_nw[1]],
            p50_net_worth=[float(x) for x in p_nw[2]],
            p75_net_worth=[float(x) for x in p_nw[3]],
            p90_net_worth=[float(x) for x in p_nw[4]],
            mean_net_worth=[float(x) for x in nw_arr.mean(axis=0)],
            prob_negative_liquid=[float(x) for x in (liq_arr < 0).mean(axis=0)],
            p10_liquid=[float(x) for x in p_liq[0]],
            p50_liquid=[float(x) for x in p_liq[1]],
            p90_liquid=[float(x) for x in p_liq[2]],
            prob_millionaire_10yr=(
                float((yr10 >= 1_000_000).mean()) if yr10 is not None else 0.0
            ),
            num_simulations=n_simulations,
            use_historical_returns=use_historical_returns,
            use_historical_inflation=use_historical_inflation,
            market_return_std=market_return_std,
            inflation_std=inflation_std,
            salary_growth_std=salary_growth_std,
            block_bootstrap=joint,
            mean_block_years=mean_block_years,
            failsafe_fire_rates=failsafe_fire_rates,
        )
