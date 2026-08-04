"""
Long-term projection engine.

Architecture
------------
ProjectionEngine drives two public entry-points:

  run_deterministic()            → list[YearlySnapshot]
  run_monte_carlo()              → MonteCarloResult
  compute_retirement_readiness() → Optional[RetirementReadiness]

Internally, each year is processed as:

  1. _apply_timeline_events() — mutates EngineState for that year's events
  2. _compute_year()          — pure calculation; returns YearlySnapshot
       ├── _contributions()   — 401k / HSA / 529 amounts
       ├── _housing()         — cost, equity, amortisation
       ├── _lifestyle()       — medical / childcare / pets / parent care
       ├── _college()         — 529 drawdown, AOTC credit
       ├── _cars()            — loan payments, purchase/sale cash flows
       └── _asset_growth()    — new balances for all accounts
  3. _advance_state()         — rolls EngineState forward to next year
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Monte Carlo runs the independent per-simulation loop across processes once the
# batch is large enough to amortise process-startup overhead; below this it stays
# serial (small runs and the test suite skip the pool). Output is identical
# either way — the sims are independent and the RNG is pre-drawn.
_MC_PARALLEL_MIN_SIMS = 1000

from fintracker.constants import (
    HSA_LIMIT_SINGLE, HSA_LIMIT_FAMILY, LIMIT_SOLO_401K, ROTH_IRA_LIMIT,
    HOME_SALE_EXCLUSION_SINGLE, HOME_SALE_EXCLUSION_MFJ,
    irmaa_annual_surcharge, limit_401k,
)
from fintracker.finance_math import linear_phaseout, monthly_amortized_payment
from fintracker.models import (
    BusinessProfile, CarProfile, ChildcarePhase, ChildcareProfile, RothContributionPhase, EmployerMatch, KidCarProfile, MatchTier, CollegeProfile, FilingStatus, FinancialPlan,
    HousingProfile, IncomeProfile, InvestmentProfile,
    RetirementProfile, StrategyToggles, by_filing_status,
    Failsafe, FailsafeAction,
)
from fintracker.tax_engine import TaxEngine, TaxResult
from fintracker.mortgage import MortgageCalculator


# ---------------------------------------------------------------------------
# IRS / tax constants
# ---------------------------------------------------------------------------

_SE_TAX_RATE              = 0.1530
_SE_TAX_DEDUCTIBLE_SHARE  = 0.9235
_QBI_PHASEOUT_SINGLE      = 191_950
_QBI_PHASEOUT_MFJ         = 383_900
_AOTC_MAX_CREDIT          = 2_500
_AOTC_PHASEOUT_SINGLE_LOW  = 80_000
_AOTC_PHASEOUT_SINGLE_HIGH = 90_000
_AOTC_PHASEOUT_MFJ_LOW     = 160_000
_AOTC_PHASEOUT_MFJ_HIGH    = 180_000

_WEDDING_AGE = 26          # age a child's wedding is paid; saving runs through age 25

# Age at which traditional 401k/IRA withdrawals and Roth earnings become
# penalty-free (IRS 59½, floored to an integer since the model tracks whole-year
# ages). This is distinct from ``retirement_age`` (when earned income stops and
# the portfolio de-risks): you can retire before or after becoming penalty-free.
PENALTY_FREE_AGE = 59

# ---------------------------------------------------------------------------
# Retirement withdrawal waterfall
# ---------------------------------------------------------------------------
# Canonical set of accounts a cash-flow deficit can be funded from, in the
# fallback order used to append any source a configured order omits. Only
# ``retirement_401k`` is taxable (ordinary income); the rest are tax-free at the
# point of withdrawal in this model (brokerage cap-gains are handled elsewhere).
WITHDRAWAL_SOURCES: tuple[str, ...] = (
    "cash_buffer", "uninvested_cash", "retirement_401k", "brokerage", "roth_basis",
)

# Pre-retirement deficits keep the legacy waterfall and never touch the 401k
# (early-withdrawal penalties): cash → Roth basis → brokerage.
_PRE_RETIREMENT_ORDER: tuple[str, ...] = (
    "cash_buffer", "uninvested_cash", "roth_basis", "brokerage",
)

# Default retirement orders (cash first, Roth basis last); one is chosen from the
# starting balance mix when StrategyToggles.retirement_withdrawal_order is None.
_ORDER_BRACKET_FILL: tuple[str, ...] = (
    "cash_buffer", "uninvested_cash", "retirement_401k", "brokerage", "roth_basis",
)
_ORDER_CONVENTIONAL: tuple[str, ...] = (
    "cash_buffer", "uninvested_cash", "brokerage", "retirement_401k", "roth_basis",
)

# Lump-sum PURCHASES (down payments, one-off expenses, business, weddings) spend
# brokerage first — savings earmarked for the purchase — then cash, then (retired
# only) the 401k, then Roth basis. Uses the same waterfall engine as deficits.
_PURCHASE_ORDER: tuple[str, ...] = (
    "brokerage", "cash_buffer", "uninvested_cash", "retirement_401k", "roth_basis",
)


def _complete_order(order, fallback: tuple[str, ...]) -> list[str]:
    """Sanitise a withdrawal order: keep known keys, then append any omitted
    source in ``fallback`` order so a deficit can never be left unfunded while a
    source still has money. Unknown keys are dropped."""
    seen: list[str] = []
    for key in order or ():
        if key in WITHDRAWAL_SOURCES and key not in seen:
            seen.append(key)
    for key in fallback:
        if key not in seen:
            seen.append(key)
    return seen


def _deposit_growth_factor(annual_rate: float, period_months: float) -> float:
    """Growth multiplier for a year's deposits under sub-annual compounding.

    Annual rates are converted geometrically, so a *lump* starting balance grows by
    exactly ``(1 + annual_rate)`` regardless of the compounding period — that is the
    whole point of ``rate_period = (1 + annual)^(period/12) − 1``. The period only
    affects money paid in *during* the year: with finer compounding those deposits
    are dollar-cost-averaged across the sub-periods and earn part of a year's return
    instead of being dropped in as a year-end lump.

    Returns that factor: ``1.0`` at ``period_months == 12`` (annual — a deposit earns
    nothing extra), ``> 1`` for finer periods (e.g. monthly), ``< 1`` for coarser
    ones. Deposits are modelled as an ordinary annuity (equal instalments at the end
    of each sub-period). Derivation: FV of ``n`` end-of-period deposits summing to 1
    is ``[(1+rp)^n − 1] / (n·rp)``, and ``(1+rp)^n = 1 + annual_rate`` exactly, so the
    factor is ``annual_rate / (n·rp)``.
    """
    if period_months <= 0 or period_months == 12 or annual_rate <= -1.0:
        return 1.0
    n = 12.0 / period_months                       # compounding periods per year
    period_rate = (1.0 + annual_rate) ** (1.0 / n) - 1.0
    if abs(period_rate) < 1e-12:                    # zero return → deposits don't grow
        return 1.0
    return annual_rate / (n * period_rate)


def _available_sources(cash_buffer: float, uninvested_cash: float, ret_grown: float,
                       brok_grown: float, roth_basis: float, retired: bool) -> dict:
    """Post-growth balances a deficit can be funded from, keyed by source.

    The 401k/IRA is only offered once ``retired`` (age ≥ retirement_age); a
    negative grown brokerage cannot lend, so it is floored at 0 (a genuine
    shortfall is booked as negative brokerage by the caller).
    """
    return {
        "cash_buffer":     cash_buffer,
        "uninvested_cash": uninvested_cash,
        "retirement_401k": ret_grown if retired else 0.0,
        "brokerage":       max(0.0, brok_grown),
        "roth_basis":      roth_basis,
    }


def _fund_deficit(available: dict, order, net_deficit: float, wd_tax_rate: float):
    """Draw ``net_deficit`` net dollars from ``available`` balances in ``order``.

    ``available`` maps each :data:`WITHDRAWAL_SOURCES` key to its balance;
    ``retirement_401k`` is a *pre-tax* balance whose withdrawals are ordinary
    income, so covering $1 of deficit consumes $1/(1-t) of it (the tax is grossed
    up). Every other source is dollar-for-dollar.

    Returns ``(reductions, taxable_withdrawal, withdrawal_tax, shortfall)`` where
    ``reductions[key]`` is the balance drop for that account, ``taxable_withdrawal``
    is the pre-tax 401k draw (its MAGI contribution), ``withdrawal_tax`` the tax on
    it, and ``shortfall`` the deficit left unfunded after every source is exhausted.
    Pure — it mutates nothing.
    """
    reductions = {key: 0.0 for key in available}
    remaining  = max(0.0, net_deficit)
    taxable_withdrawal = withdrawal_tax = 0.0

    for key in order:
        if remaining <= 1e-9:
            break
        bal = available.get(key, 0.0)
        if bal <= 0:
            continue
        if key == "retirement_401k" and wd_tax_rate < 1.0:
            net_capacity = bal * (1.0 - wd_tax_rate)          # net cash this balance can deliver
            net_drawn    = min(remaining, net_capacity)
            pre_tax      = net_drawn / (1.0 - wd_tax_rate)    # gross-up for the tax
            reductions[key]     += pre_tax
            taxable_withdrawal  += pre_tax
            withdrawal_tax      += pre_tax - net_drawn
            remaining           -= net_drawn
        else:
            drawn = min(remaining, bal)
            reductions[key] += drawn
            remaining       -= drawn

    return reductions, taxable_withdrawal, withdrawal_tax, remaining


# ---------------------------------------------------------------------------
# Historical S&P 500 annual total returns (1926–2025)
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
# Failsafe metrics that are unitless ratios (not dollar amounts), so the
# present-value deflation applied to dollar metrics must be skipped for them.
_FS_RATIO_METRICS = frozenset({"medical_burden_ratio"})

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


from typing import NamedTuple

class _Growth(NamedTuple):
    """Return value of _asset_growth — named fields beat an 8-tuple."""
    retirement:      float
    hsa:             float
    col529:          float
    brokerage:       float
    uninvested:      float
    cash_buffer:     float
    roth_balance:    float
    roth_basis:      float
    brokerage_gains: float   # cumulative unrealized gains within `brokerage`
    taxable_withdrawal: float = 0.0  # pre-tax 401k/IRA drawn to fund a deficit (MAGI)
    withdrawal_tax:     float = 0.0  # ordinary-income tax on that withdrawal
    brokerage_withdrawal: float = 0.0  # brokerage drawn to fund a deficit this year
    roth_withdrawal:      float = 0.0  # Roth (basis + qualified earnings) drawn this year

@dataclass
class _ActiveFailsafe:
    """A triggered failsafe's in-flight action within one simulation path.

    ``start_year``/``end_year`` bound the active window (already offset by the
    failsafe's delay). The ``saved_*`` fields hold the earned-income baseline to
    restore when the window ends; ``activated``/``closed`` track the lifecycle.
    """
    action: FailsafeAction
    start_year: int
    end_year: Optional[int]
    saved_partner_income: float = 0.0
    saved_partner_working: bool = False
    saved_primary_income: float = 0.0
    saved_primary_working: bool = True
    activated: bool = False
    closed: bool = False


# ---------------------------------------------------------------------------
# Engine state — typed, explicit, no loose dicts
# ---------------------------------------------------------------------------

@dataclass
class EngineState:
    """
    Mutable state that rolls forward one year at a time.

    All income values are nominal dollars for the current projection year.
    `gross_income` is always derived as income_primary + income_partner and
    is kept in sync by every mutation that touches either component.
    """
    # Income
    income_primary: float
    income_partner: float
    filing_status: FilingStatus
    is_married: bool
    is_working: bool
    is_partner_working: bool

    # Family
    num_children: int
    num_pets: int
    child_birth_years: list[int]

    # Housing
    is_renting: bool
    monthly_rent: float
    mortgage_calc: Optional[MortgageCalculator]
    amort_lookup: dict[int, float]
    mortgage_year_offset: int
    mortgage_interest_rate: float
    home_price_ref: float
    home_value: float
    mortgage_balance: float

    # Balances
    retirement_balance: float
    brokerage_balance: float
    brokerage_gains: float          # cumulative unrealized gains within brokerage_balance
    hsa_balance: float
    college_529_balance: float
    uninvested_cash: float
    cash_buffer: float
    business_equity: float
    business_revenue: float

    # Flags
    parent_care_active: bool

    # Roth IRA — tracked separately because withdrawal rules differ from 401k.
    # roth_contribution_basis = cumulative post-tax contributions (total basis).
    # roth_vested_basis = portion of basis that is penalty-free to withdraw:
    #   under the 5-year rule for conversions, only contributions made ≥5
    #   projection years ago are penalty-free (conservative model).
    # roth_contrib_queue = ring buffer of last 5 years' contributions;
    #   the oldest entry vests each year.
    # roth_ira_balance = total Roth value including earnings (grows at market rate).
    roth_ira_balance: float
    roth_contribution_basis: float
    roth_vested_basis: float          # penalty-free portion (≥5 yrs old)
    roth_contrib_queue: list          # list[float] of last 5 contributions, FIFO

    # Cumulative inflation factor — rolling product of (1+inf_t) for t=1..year-1.
    # Tracked in state so each year's factor is correct regardless of whether inflation
    # is constant (deterministic) or sampled per-year (Monte Carlo).
    cumulative_inflation: float
    # Parallel factor for healthcare, which compounds at its own (higher) rate.
    cumulative_healthcare_inflation: float

    # Car loan state — one dict per car
    cars: list[dict]
    # Kid car loans — one entry per child who has received a car
    kid_car_loans: list[dict]
    # Wedding sinking fund — invested value accrued per child (parallel to
    # child_birth_years); held inside brokerage, spent at the wedding.
    wedding_fund: list[float]
    # Capital gains realized so far this projection year (reset annually); taxed
    # in _compute_year. Populated by sell_brokerage().
    realized_gains_ytd: float

    # Pre-tax 401k/IRA withdrawn to fund lump-sum PURCHASES this year (reset
    # annually), e.g. a car/home down payment when brokerage+cash are exhausted.
    # Feeds MAGI/IRMAA and the reported retirement withdrawal, parallel to the
    # operating-deficit withdrawal. Populated by _fund_purchase().
    purchase_taxable_wd: float = 0.0

    # Roth drawn to fund lump-sum PURCHASES this year (reset annually), reported
    # alongside the deficit-path Roth draw. Populated by _fund_purchase().
    purchase_roth_wd: float = 0.0

    # Failsafes (conditional events). `fired_failsafes` latches names that have
    # already triggered on this path (one-shot). `active_failsafes` holds the
    # in-flight actions layered onto this year's income. Both are per-path.
    fired_failsafes: set = field(default_factory=set)
    active_failsafes: list = field(default_factory=list)
    # Set by a failsafe action for the current year; consumed by _contributions.
    # Reset at the top of _evaluate_failsafes so it never persists across years.
    suspend_retirement_contributions: bool = False
    # Nominal vacation budget forced by a failsafe this year (None = use profile).
    vacation_override: Optional[float] = None
    # Multiplier applied to healthcare costs (OOP, health premium, self-LTC,
    # Medicare) this year; 1.0 = unchanged. A failsafe may cut it (e.g. 0.5).
    medical_cost_multiplier: float = 1.0

    @property
    def gross_income(self) -> float:
        return self.income_primary + self.income_partner

    def sell_brokerage(self, amount: float) -> None:
        """Withdraw ``amount`` from brokerage, realizing a pro-rata share of its
        unrealized gains (accumulated into realized_gains_ytd for cap-gains tax).

        Single chokepoint for brokerage debits so balance, basis, and gains stay
        consistent — deposits stay plain ``+=`` (new basis, no gain realized).
        """
        if amount > 0 and self.brokerage_balance > 0:
            realized = self.brokerage_gains * min(1.0, amount / self.brokerage_balance)
            self.brokerage_gains -= realized
            self.realized_gains_ytd += realized
        self.brokerage_balance -= amount


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class YearlySnapshot:
    """Complete financial picture for a single projection year."""
    year: int

    # Income
    gross_income: float
    net_income: float
    annual_tax_total: float

    # Expenses
    annual_housing_cost: float
    annual_lifestyle_cost: float
    annual_medical_oop: float
    annual_college_cost: float
    annual_529_drawdown: float
    annual_parent_care_cost: float
    annual_retirement_contributions: float
    annual_hsa_contributions: float
    annual_hsa_withdrawal: float           # HSA spent on qualified medical this year
    annual_brokerage_contribution: float
    annual_aotc_credit: float
    annual_car_payment: float
    annual_capital_gains_tax: float
    annual_wedding_save: float

    # Cash flow
    annual_breathing_room: float

    # Assets
    retirement_balance: float
    brokerage_balance: float
    brokerage_gains: float          # unrealized gains within brokerage_balance (for cap-gains tax)
    college_529_balance: float
    home_value: float
    home_equity: float
    hsa_balance: float
    uninvested_cash: float

    # Liabilities
    mortgage_balance: float

    # Net worth
    net_worth: float

    # Meta
    filing_status: FilingStatus
    num_children: int
    is_renting: bool
    is_married: bool
    is_working: bool
    is_partner_working: bool

    # Business income and equity
    roth_ira_balance: float
    roth_contribution_basis: float
    roth_vested_basis: float
    annual_roth_contribution: float
    annual_business_income: float = 0.0
    business_equity: float = 0.0
    # Car one-off costs (for display / debugging)
    car_purchase_cost: float = 0.0
    car_sale_proceeds: float = 0.0
    annual_wedding_spend: float = 0.0   # wedding paid from the sinking fund this year
    # Intentional cash buffer (earns 0%; maintained before sweeping to brokerage)
    cash_buffer: float = 0.0
    # Cumulative price level vs projection start (the engine's own inflation factor).
    # Divide any nominal figure in this year by it to express it in today's dollars.
    cumulative_inflation: float = 1.0

    # Per-year tax breakdown. Federal is net of education credits (AOTC) so the
    # three components always sum to annual_tax_total.
    annual_federal_tax: float = 0.0
    annual_fica_tax: float = 0.0
    annual_state_tax: float = 0.0

    # Newly-modelled recurring costs (all already included in the aggregate they
    # belong to — insurance premiums, self-LTC and Medicare are folded into
    # annual_lifestyle_cost; car operating cost is a separate car outflow). These
    # break them out for display and testing, mirroring how annual_medical_oop
    # itemises a slice of the lifestyle bucket.
    annual_insurance_premiums: float = 0.0   # health + disability + life
    annual_self_ltc_cost: float = 0.0        # your own long-term care
    annual_medicare_cost: float = 0.0        # base premium + IRMAA surcharge (65+)
    annual_car_operating_cost: float = 0.0   # insurance/fuel/maintenance/registration

    # Retirement drawdown: pre-tax 401k/IRA withdrawn to cover a deficit (ordinary
    # income, the IRMAA MAGI driver) and the ordinary-income tax paid on it.
    annual_retirement_withdrawal: float = 0.0
    annual_retirement_withdrawal_tax: float = 0.0

    # Brokerage drawn down to cover a deficit this year (already-taxed dollars).
    annual_brokerage_withdrawal: float = 0.0

    # Roth drawn this year (basis pre-retirement; basis + earnings once retired,
    # a qualified tax-free distribution). Deficit- and purchase-path draws combined.
    annual_roth_withdrawal: float = 0.0

    # Life-insurance death benefit paid into the estate in the year of death (0 otherwise).
    annual_life_insurance_payout: float = 0.0

    def to_todays_dollars(self, nominal: float) -> float:
        """Convert a nominal figure from this projection year into today's dollars."""
        return nominal / self.cumulative_inflation if self.cumulative_inflation else nominal

    @property
    def total_assets(self) -> float:
        return (
            self.retirement_balance + self.brokerage_balance
            + self.college_529_balance + self.home_equity
            + self.hsa_balance + self.uninvested_cash + self.cash_buffer
            + self.business_equity
        )

    @property
    def investable_assets(self) -> float:
        """Investable financial assets (pre-tax) — retirement, HSA, Roth, brokerage,
        and cash. Excludes home equity, the 529 (earmarked for college), and the
        business (illiquid). Single source of truth for the retirement-readiness
        pool and the app's "Total Investable Assets" line, so they cannot diverge.
        """
        return (
            self.retirement_balance + self.hsa_balance + self.roth_ira_balance
            + self.brokerage_balance + self.uninvested_cash + self.cash_buffer
        )

    @property
    def liquid_assets(self) -> float:
        return self.brokerage_balance


@dataclass
class RetirementReadiness:
    """Result of the retirement readiness analysis."""
    years_to_retirement: int
    retirement_year: int
    projected_balance_at_retirement: float        # tax-adjusted after-tax value
    projected_balance_pretax: float               # raw sum before tax haircut
    required_balance: float
    on_track: bool
    funded_pct: float
    annual_surplus_or_gap: float
    annual_cost_at_retirement: float   # first retirement-year cost of living (nominal)
    social_security_offset: float


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
# Main engine
# ---------------------------------------------------------------------------

def _legacy_car_down(car) -> float:
    """Sum of down payments for legacy-mode cars (no first_purchase_years)."""
    if not car:
        return 0.0
    if car.first_purchase_years:
        # All cars have explicit purchase years — no upfront deduction
        return 0.0
    return car.down_payment * car.num_cars


def _after_tax_value(balance: float, taxable_base: float, rate: float) -> float:
    """Account value after a withdrawal tax that applies only to `taxable_base`.

    Making the taxable base explicit keeps each account honest: a 401k's whole
    balance is ordinary income (base == balance), whereas a taxable brokerage is
    taxed only on its gains (base == gains, not the full balance).
    """
    return balance - taxable_base * rate


def _pay_loan_year(loan: dict, annual_rate: float, term_years: int) -> float:
    """Advance one amortising loan by a single year.

    Mutates ``loan['loan_balance']`` and ``loan['loan_year']`` in place and
    returns the annual payment made this year (0 if the loan is paid off or past
    its term).  Shared by household cars and kids' cars — previously two verbatim
    copies of this block lived in ``_cars``.

    ``loan`` must carry ``loan_balance``, ``loan_year`` and ``monthly_payment``.
    """
    if loan["loan_balance"] <= 0 or loan["loan_year"] > term_years:
        return 0.0
    monthly = loan["monthly_payment"]
    # Never pay more than what remains plus this year's accrued interest.
    annual_pmt = min(monthly * 12, loan["loan_balance"] * (1 + annual_rate / 12) * 12)
    r = annual_rate / 12
    if r > 0:
        n_remaining = (term_years - (loan["loan_year"] - 1)) * 12
        loan["loan_balance"] = max(0.0, monthly * (1 - (1 + r) ** -n_remaining) / r)
    else:
        loan["loan_balance"] = max(0.0, loan["loan_balance"] - monthly * 12)
    loan["loan_year"] += 1
    return annual_pmt


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
    return ProjectionEngine(plan)._run_sim_rows(all_mkt, all_inf, all_sg, years, amort_cache)


class ProjectionEngine:
    """
    Runs deterministic and Monte Carlo projections for a FinancialPlan.

    Usage::

        engine = ProjectionEngine(plan)
        snapshots = engine.run_deterministic()
        mc = engine.run_monte_carlo(n_simulations=1_000)
        rr = engine.compute_retirement_readiness(snapshots)
    """

    def __init__(self, plan: FinancialPlan) -> None:
        self._plan = plan
        self._tax = TaxEngine()
        # Memoises the (path-independent) present value of future medical bills
        # by year + the deterministic state inputs it depends on, so the medical-
        # burden failsafe forecast is computed once instead of once per sim/year.
        self._pv_medical_cache: dict = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_deterministic(self) -> list[YearlySnapshot]:
        state = self._initial_state()
        snapshots: list[YearlySnapshot] = []
        for year in range(1, self._horizon() + 1):
            self._apply_timeline_events(state, year)
            self._evaluate_failsafes(state, year)
            snap = self._compute_year(state, year)
            snapshots.append(snap)
            self._advance_state(state, snap)
        return snapshots

    def run_monte_carlo(
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
        rng = np.random.default_rng(seed)
        inv = self._plan.investments
        years = list(range(1, self._horizon() + 1))

        hist = np.array(_SP500_HISTORICAL_RETURNS)

        # Pre-compute mortgage amortization lookups — identical every simulation.
        _amort_cache: dict = {}
        for _ev in self._plan.timeline_events:
            if _ev.buy_home and _ev.new_home_price and _ev.new_home_interest_rate:
                _p = _ev.new_home_price
                _d = _ev.new_home_down_payment or _p * 0.20
                _r = _ev.new_home_interest_rate
                _t = self._plan.housing.loan_term_years
                _k = (_p, _d, _r, _t)
                if _k not in _amort_cache:
                    _hp_tmp = HousingProfile(home_price=_p, down_payment=_d,
                                             interest_rate=_r, loan_term_years=_t)
                    _mc_tmp = MortgageCalculator(_hp_tmp, self._plan.investments.annual_home_appreciation_rate)
                    _amort_cache[_k] = self._amort_lookup(_mc_tmp)

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

        # Run the independent per-sim loop. For large batches, fan the sims out
        # across processes (they are independent and the RNG is pre-drawn, so the
        # combined result is identical to the serial path); otherwise stay serial.
        n_cpu = os.cpu_count() or 1
        if n_simulations >= _MC_PARALLEL_MIN_SIMS and n_cpu > 1:
            bounds = [(k * n_simulations) // n_cpu for k in range(n_cpu + 1)]
            slices = [(bounds[k], bounds[k + 1]) for k in range(n_cpu) if bounds[k] < bounds[k + 1]]
            nw_parts, liq_parts, fired_list = [], [], []
            with ProcessPoolExecutor(max_workers=len(slices), mp_context=_mc_context()) as ex:
                futures = [
                    ex.submit(_mc_worker, self._plan,
                              all_mkt[s:e], all_inf[s:e], all_sg[s:e], years, _amort_cache)
                    for (s, e) in slices
                ]
                for f in futures:                       # gathered in submission order
                    nw_c, liq_c, fired_c = f.result()
                    nw_parts.append(nw_c); liq_parts.append(liq_c); fired_list.extend(fired_c)
            nw_arr = np.vstack(nw_parts); liq_arr = np.vstack(liq_parts)
        else:
            nw_arr, liq_arr, fired_list = self._run_sim_rows(
                all_mkt, all_inf, all_sg, years, _amort_cache)

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

    def _run_sim_rows(self, all_mkt, all_inf, all_sg, years, amort_cache):
        """Run the per-sim projection loop over a batch of pre-drawn RNG rows.

        Returns ``(nw, liq, fired_list)`` — net-worth and liquid (brokerage + cash
        buffer) trajectories as ``(n_rows, n_years)`` float arrays, and the failsafe
        names that fired on each path. Returning arrays (not nested lists) keeps the
        cross-process transfer and the percentile aggregation cheap. This is the sole
        per-sim loop; ``run_monte_carlo`` calls it directly (serial) or ships
        row-slices to worker processes, with identical aggregation either way.
        """
        all_nw: list[list[float]] = []
        all_liq: list[list[float]] = []
        fired_list: list[list[str]] = []
        for sim_idx in range(len(all_mkt)):
            mkt = all_mkt[sim_idx]
            inf = all_inf[sim_idx]
            sg  = all_sg[sim_idx]
            state = self._initial_state()
            sim_nw:  list[float] = []
            sim_liq: list[float] = []
            for i, year in enumerate(years):
                self._apply_timeline_events(state, year, amort_cache)
                self._evaluate_failsafes(state, year)
                snap = self._compute_year(
                    state, year,
                    market_return_override=float(mkt[i]),
                    inflation_override=float(inf[i]),
                    salary_growth_override=float(sg[i]),
                )
                sim_nw.append(snap.net_worth)
                # Liquid position = brokerage + cash_buffer: both are accessible.
                # Measuring brokerage alone understates liquidity when a buffer exists.
                sim_liq.append(snap.brokerage_balance + snap.cash_buffer)
                self._advance_state(state, snap,
                                    market_return=float(mkt[i]),
                                    inflation=float(inf[i]),
                                    salary_growth=float(sg[i]))
            all_nw.append(sim_nw)
            all_liq.append(sim_liq)
            fired_list.append(list(state.fired_failsafes))
        return np.array(all_nw), np.array(all_liq), fired_list

    def compute_retirement_readiness(
        self,
        snapshots: Optional[list[YearlySnapshot]] = None,
        withdrawal_tax_rate: Optional[float] = None,
        capital_gains_rate: Optional[float] = None,
    ) -> Optional[RetirementReadiness]:
        """
        withdrawal_tax_rate and capital_gains_rate override the values in
        RetirementProfile when provided — avoids rebuilding a new ProjectionEngine
        just to display after-tax retirement readiness in the UI.
        """
        """
        Returns None when no RetirementProfile is configured.

        Required balance = the present value of the ACTUAL projected cost of life
        after retirement. For every retirement year we take the real modelled
        outflow (housing + lifestyle + car) net of Social Security, then discount
        it back to the retirement year at ``expected_post_retirement_return``:

            required = Σ_t  max(0, cost_t - ss_t) / (1 + r)^(t - retirement_year)

        This tracks the true, non-flat cost curve — mortgage payoff, the spending
        smile, healthcare inflation, and end-of-life care all flow through — rather
        than assuming a flat ``desired_annual_income``.
        """
        rp = self._plan.retirement
        if rp is None:
            return None

        if snapshots is None:
            snapshots = self.run_deterministic()

        inflation    = self._plan.investments.annual_inflation_rate
        years_to_ret = rp.years_to_retirement

        # Projected balance = all investable assets at retirement year
        # (same pool used by the Retirement Readiness panel — single source of truth)
        snap = next((s for s in snapshots if s.year == years_to_ret), snapshots[-1])
        # Apply post-retirement withdrawal taxes to get the after-tax spendable value.
        # Roth and HSA are tax-free; 401k/IRA are taxed at ordinary income rates;
        # brokerage is taxed at capital gains rates on withdrawal.
        # Both tax rates default to 0% for backward compatibility.
        r401k_tax = (withdrawal_tax_rate if withdrawal_tax_rate is not None
                     else rp.retirement_withdrawal_tax_rate)
        # Retirement haircut uses the (typically lower) retirement drawdown rate.
        cap_gains  = (capital_gains_rate if capital_gains_rate is not None
                      else self._retirement_cap_gains_rate())
        projected_pretax = snap.investable_assets
        projected = (
            # 401k/IRA: entire balance is ordinary income on withdrawal.
            _after_tax_value(snap.retirement_balance, snap.retirement_balance, r401k_tax)
            + snap.hsa_balance                           # HSA: tax-free (medical)
            + snap.roth_ira_balance                      # Roth: fully tax-free
            # Brokerage: capital gains apply to the gains only, not the cost basis.
            + _after_tax_value(snap.brokerage_balance, snap.brokerage_gains, cap_gains)
            + snap.uninvested_cash + snap.cash_buffer    # Cash: already post-tax
        )

        # Required nest egg = PV of the ACTUAL projected cost of life after
        # retirement (housing + lifestyle + car each year), net of Social Security,
        # discounted back to the retirement year at the post-retirement return.
        r, g, n = rp.expected_post_retirement_return, inflation, rp.years_in_retirement
        retirement_snaps = [s for s in snapshots if s.year > years_to_ret]
        required = 0.0
        for s in retirement_snaps:
            ss_t     = rp.estimated_social_security_annual * (1 + inflation) ** (s.year - 1)
            cost_t   = (s.annual_housing_cost + s.annual_lifestyle_cost
                        + s.annual_car_operating_cost)
            net_need = max(0.0, cost_t - ss_t)
            required += net_need / (1 + r) ** (s.year - years_to_ret)

        # First retirement-year cost + SS at retirement, surfaced in the caption.
        first_ret = retirement_snaps[0] if retirement_snaps else snap
        annual_cost_at_retirement = (first_ret.annual_housing_cost
                                     + first_ret.annual_lifestyle_cost
                                     + first_ret.annual_car_operating_cost)
        ss_nominal = rp.estimated_social_security_annual * (1 + inflation) ** years_to_ret

        funded_pct  = projected / required if required > 0 else math.inf
        balance_gap = projected - required

        # Convert balance surplus/gap to annual income equivalent
        # Using the same growing annuity formula in reverse
        if r == 0:
            annual_gap = balance_gap / n if n > 0 else 0.0
        elif abs(r - g) < 1e-9:
            annual_gap = balance_gap * (1 + r) / n if n > 0 else 0.0
        else:
            annual_gap = balance_gap * (r - g) / (1 - ((1 + g) / (1 + r)) ** n)

        return RetirementReadiness(
            years_to_retirement=years_to_ret,
            retirement_year=snap.year,
            projected_balance_at_retirement=projected,
            projected_balance_pretax=projected_pretax,
            required_balance=required,
            on_track=projected >= required,
            funded_pct=funded_pct,
            annual_surplus_or_gap=annual_gap,
            annual_cost_at_retirement=annual_cost_at_retirement,
            social_security_offset=ss_nominal,
        )

    # ------------------------------------------------------------------ #
    # State initialisation                                                 #
    # ------------------------------------------------------------------ #

    def _initial_state(self) -> EngineState:
        p   = self._plan
        inv = p.investments
        is_married = p.income.filing_status == FilingStatus.MARRIED_FILING_JOINTLY

        mortgage_calc = None
        amort_lookup: dict[int, float] = {}
        if not p.housing.is_renting and p.housing.loan_amount > 0:
            mortgage_calc = MortgageCalculator(p.housing, inv.annual_home_appreciation_rate)
            amort_lookup  = self._amort_lookup(mortgage_calc)

        initial_brokerage = (
            inv.current_liquid_cash
            - inv.one_time_upcoming_expenses
            - (p.housing.down_payment if not p.housing.is_renting else 0.0)
            # Only deduct down payments for legacy cars (pre-purchased at projection start).
            # Cars with first_purchase_years will have their down payment deducted
            # in the year they are first bought via _cars().
            - (_legacy_car_down(p.car))
            + inv.current_brokerage_balance
        )

        return EngineState(
            income_primary=p.income.gross_annual_income,
            income_partner=p.income.spouse_gross_annual_income,
            filing_status=p.income.filing_status,
            is_married=is_married,
            is_working=True,
            is_partner_working=p.income.spouse_gross_annual_income > 0,
            num_children=p.lifestyle.num_children,
            num_pets=p.lifestyle.num_pets,
            child_birth_years=[0] * p.lifestyle.num_children,
            is_renting=p.housing.is_renting,
            monthly_rent=p.housing.monthly_rent,
            mortgage_calc=mortgage_calc,
            amort_lookup=amort_lookup,
            mortgage_year_offset=0,
            mortgage_interest_rate=p.housing.interest_rate,
            home_price_ref=p.housing.home_price,
            home_value=p.housing.home_price if not p.housing.is_renting else 0.0,
            mortgage_balance=p.housing.loan_amount if not p.housing.is_renting else 0.0,
            retirement_balance=inv.current_retirement_balance,
            brokerage_balance=initial_brokerage,
            # Starting brokerage is treated as all cost basis (no embedded gains);
            # gains accrue from projected market appreciation going forward.
            brokerage_gains=0.0,
            hsa_balance=0.0,
            college_529_balance=0.0,
            uninvested_cash=0.0,
            cash_buffer=0.0,
            parent_care_active=p.lifestyle.annual_parent_care_cost > 0,
            roth_ira_balance=p.investments.current_roth_ira_balance,
            roth_contribution_basis=p.investments.current_roth_ira_balance,
            # Existing balance is assumed to be fully vested (owner has had it ≥5 yrs)
            roth_vested_basis=p.investments.current_roth_ira_balance,
            roth_contrib_queue=[],
            cumulative_inflation=1.0,
            cumulative_healthcare_inflation=1.0,
            cars=self._init_cars(p.car),
            kid_car_loans=[],
            wedding_fund=[0.0] * p.lifestyle.num_children,
            realized_gains_ytd=0.0,
            business_equity=0.0,
            business_revenue=(p.business.annual_revenue if p.business else 0.0),
        )

    @staticmethod
    def _amort_lookup(mc: MortgageCalculator) -> dict[int, float]:
        return {row.year: row.balance for row in mc.full_schedule() if row.month % 12 == 0}

    @staticmethod
    def _init_cars(car: Optional[CarProfile]) -> list[dict]:
        """
        Initialise one state-dict per car.

        If first_purchase_years is configured, each car starts with no loan and
        no payments until its specified purchase year, at which point _cars()
        will buy it and start the loan.  Before that year the entry is inert.

        Legacy fallback: if first_purchase_years is None, uses the old stagger
        (car 0 bought at yr 1, car 1 at yr 0) so existing configs are unchanged.
        """
        if car is None:
            return []
        cars = []
        for i in range(car.num_cars):
            if car.first_purchase_years and i < len(car.first_purchase_years):
                # Explicit first purchase year — car hasn't been bought yet
                cars.append({
                    "loan_balance":    0.0,
                    "loan_year":       0,
                    "purchase_year":   None,          # None = not yet purchased
                    "first_buy_year":  car.first_purchase_years[i],
                    "monthly_payment": 0.0,
                })
            else:
                # Legacy: treat as already purchased and financed at projection start
                principal  = max(0.0, car.car_price - car.down_payment)
                monthly_pi = ProjectionEngine._car_monthly_pi(
                    principal, car.loan_rate, car.loan_term_years)
                cars.append({
                    "loan_balance":    principal,
                    "loan_year":       1,
                    "purchase_year":   1 - i,
                    "first_buy_year":  None,
                    "monthly_payment": monthly_pi,
                })
        return cars

    # ------------------------------------------------------------------ #
    # Timeline events                                                      #
    # ------------------------------------------------------------------ #

    def _apply_timeline_events(self, state: EngineState, year: int, _amort_cache: Optional[dict] = None) -> None:
        p = self._plan
        state.realized_gains_ytd = 0.0   # reset the year's realized-gains tally
        state.purchase_taxable_wd = 0.0  # reset the year's purchase-driven 401k draw
        state.purchase_roth_wd = 0.0     # reset the year's purchase-driven Roth draw
        for ev in p.events_for_year(year):

            if ev.marriage:
                state.filing_status = FilingStatus.MARRIED_FILING_JOINTLY
                state.is_married    = True

            if ev.new_child:
                birth = ev.child_birth_year_override if ev.child_birth_year_override is not None else year
                state.child_birth_years.append(birth)
                state.wedding_fund.append(0.0)
                state.num_children += 1

            if ev.new_pet:
                state.num_pets += 1

            # Work continuity
            if ev.stop_working:
                state.is_working    = False
                state.income_primary = 0.0
            if ev.resume_working:
                state.is_working    = True
            if ev.partner_stop_working:
                state.is_partner_working = False
                state.income_partner     = 0.0
            if ev.partner_resume_working:
                state.is_partner_working = True

            # Income changes (must come after stop/resume so resume + income_change works)
            if ev.income_change is not None:
                state.income_primary = ev.income_change
                if ev.resume_working:
                    state.is_working = True
            if ev.partner_income_change is not None:
                state.income_partner = ev.partner_income_change
                if ev.partner_resume_working:
                    state.is_partner_working = True

            # Parent care
            if ev.start_parent_care:
                state.parent_care_active = True
            if ev.stop_parent_care:
                state.parent_care_active = False

            # One-off cash
            state.brokerage_balance += ev.extra_one_time_income
            self._fund_purchase(state, ev.extra_one_time_expense, year)

            # Home purchase
            if ev.buy_home:
                self._apply_home_purchase(state, ev, _amort_cache)

        self._apply_auto_retirement(state, year)

    def _apply_auto_retirement(self, state: EngineState, year: int) -> None:
        """Stop earned income when the primary reaches retirement_age.

        Previously the projection kept paying a growing salary past
        retirement_age unless a manual stop_working event was added, decoupling
        the projection from the retirement-readiness view. This fires once, in
        the year the primary first reaches retirement_age (the crossing year, or
        year 1 if already at/past it), mirroring a stop_working event for both
        the primary and — since no separate partner age is modelled — the
        partner. A later resume_working event still overrides it. No-op unless a
        RetirementProfile is configured with auto_retire=True.
        """
        rp = self._plan.retirement
        if not rp or not rp.auto_retire:
            return
        age = self._age_in_year(year)
        if age is None or age < rp.retirement_age:
            return
        # Only at the crossing (first year age >= retirement_age); afterwards the
        # is_working=False state persists on its own and must not re-zero income
        # a resume_working event may have restored.
        if not (year == 1 or (age - 1) < rp.retirement_age):
            return
        if state.is_working:
            state.is_working = False
            state.income_primary = 0.0
        if state.is_partner_working:
            state.is_partner_working = False
            state.income_partner = 0.0

    # ------------------------------------------------------------------ #
    # Failsafes — conditional events checked against live state each year  #
    # ------------------------------------------------------------------ #

    def _failsafe_metric(self, state: EngineState, metric: str, year: int) -> float:
        """Value of a failsafe trigger metric, from live start-of-year state.

        Balance metrics read what's carried into the year (end of the prior
        year), the natural quantity for a threshold like "brokerage below $100k".
        ``net_worth`` mirrors the snapshot's composition. ``medical_burden_ratio``
        is a forward-looking, unitless ratio (see ``_pv_future_medical``) — set
        ``present_value: false`` on its condition since it is already unit-free.
        """
        home_equity = state.home_value - state.mortgage_balance
        cash = state.uninvested_cash + state.cash_buffer
        net_worth = (state.retirement_balance + state.hsa_balance
                     + state.college_529_balance + state.roth_ira_balance
                     + state.brokerage_balance + home_equity + cash
                     + state.business_equity)
        if metric == "brokerage_balance":
            return state.brokerage_balance
        if metric == "liquid_assets":
            return state.brokerage_balance + cash
        if metric == "investable_assets":
            return (state.retirement_balance + state.hsa_balance
                    + state.roth_ira_balance + state.brokerage_balance + cash)
        if metric == "retirement_balance":
            return state.retirement_balance
        if metric == "home_equity":
            return home_equity
        if metric == "net_worth":
            return net_worth
        if metric == "medical_burden_ratio":
            # PV of anticipated future medical bills as a fraction of net worth.
            # Non-positive net worth => any positive burden is "infinitely" large.
            if net_worth <= 0:
                return float("inf")
            return self._pv_future_medical(state, year) / net_worth
        raise ValueError(f"Unknown failsafe metric: {metric!r}")

    def _annual_medical_forecast(self, year: int, hc_f: float, is_married: bool,
                                 num_children: int, working: bool) -> float:
        """Anticipated healthcare cost for one future year (baseline, no cut).

        OOP + health premium (while working, pre-Medicare) + self-LTC (age-gated)
        + base Medicare (65+; IRMAA excluded as it is MAGI/path-dependent), all in
        year-``year`` nominal dollars via the healthcare factor ``hc_f``.
        """
        lif = self._plan.lifestyle
        rp = self._plan.retirement
        age = self._age_in_year(year)
        medicare_age = rp.medicare_start_age if rp else 65
        medical = lif.scaled_medical_oop(is_married, num_children) * hc_f
        health = (lif.annual_health_insurance_premium * hc_f
                  if working and (age is None or age < medicare_age) else 0.0)
        self_ltc = (lif.annual_self_ltc_cost * hc_f
                    if age is not None and age >= lif.self_ltc_start_age else 0.0)
        medicare = 0.0
        if rp and age is not None and age >= rp.medicare_start_age:
            enrolled = 2 if is_married else 1
            medicare = rp.annual_medicare_premium * hc_f * enrolled
        return medical + health + self_ltc + medicare

    def _pv_future_medical(self, state: EngineState, year: int) -> float:
        """Present value (at ``year``) of anticipated medical bills from ``year``
        through the horizon ("until death"), discounted at the expected return.

        A deterministic forecast: healthcare costs are driven by age and the
        (fixed) healthcare-inflation rate, not by market returns, so no simulation
        is needed. Uses baseline costs — it ignores any failsafe medical cut, so
        the trigger reflects the un-mitigated burden that would justify the move.

        The result depends only on ``year`` and the deterministic state inputs
        below (the healthcare-inflation factor is a fixed function of the year),
        so it is memoised and shared across every simulation path.
        """
        key = (year, state.is_working, state.is_married, state.num_children)
        cached = self._pv_medical_cache.get(key)
        if cached is not None:
            return cached
        inv = self._plan.investments
        rp = self._plan.retirement
        discount = rp.expected_post_retirement_return if rp else inv.annual_market_return
        hc_rate = inv.annual_healthcare_inflation_rate
        retire_age = rp.retirement_age if rp else None
        total = 0.0
        for t in range(year, self._horizon() + 1):
            hc_f_t = state.cumulative_healthcare_inflation * (1 + hc_rate) ** (t - year)
            age_t = self._age_in_year(t)
            working_t = state.is_working and (retire_age is None or age_t is None or age_t < retire_age)
            cost_t = self._annual_medical_forecast(t, hc_f_t, state.is_married,
                                                   state.num_children, working_t)
            total += cost_t / (1 + discount) ** (t - year)
        self._pv_medical_cache[key] = total
        return total

    def _failsafe_triggered(self, state: EngineState, year: int, fs: Failsafe) -> bool:
        results = []
        for c in fs.conditions:
            # end_year of None OR 0 (non-positive) means "to the horizon" — same
            # sentinel convention the UI uses (it sends 0 for "end"). Only a value
            # >= 1 bounds the window.
            end = c.end_year if (c.end_year and c.end_year >= 1) else self._horizon()
            if not (c.start_year <= year <= end):
                results.append(False)
                continue
            value = self._failsafe_metric(state, c.metric, year)
            # Ratio metrics are already unit-free; only deflate dollar metrics.
            if c.present_value and c.metric not in _FS_RATIO_METRICS and state.cumulative_inflation:
                value = value / state.cumulative_inflation
            if c.comparator == "below":
                results.append(value < c.threshold)
            elif c.comparator == "above":
                results.append(value > c.threshold)
            else:
                raise ValueError(f"Unknown failsafe comparator: {c.comparator!r}")
        if not results:
            return False
        return any(results) if fs.match == "any" else all(results)

    def _evaluate_failsafes(self, state: EngineState, year: int) -> None:
        """Arm any newly-triggered failsafes, then apply all active ones.

        Runs after ``_apply_timeline_events`` (so scripted events set the year's
        baseline first) and before ``_compute_year`` (so income overrides are
        taxed correctly). A triggered failsafe schedules its action to start at
        ``year + delay_years`` and, if ``duration_years`` is set, end after it.
        """
        if not self._plan.failsafes:
            return
        # Year-scoped action flags reset before re-evaluation so a suspension /
        # override only holds in years the trigger is actually active.
        state.suspend_retirement_contributions = False
        state.vacation_override = None
        state.medical_cost_multiplier = 1.0
        for fs in self._plan.failsafes:
            if fs.once and fs.name in state.fired_failsafes:
                continue
            if self._failsafe_triggered(state, year, fs):
                state.fired_failsafes.add(fs.name)
                start = year + fs.delay_years
                # duration_years of None OR 0 (or any non-positive) means
                # "permanent" — a single convention shared by the YAML, the UI
                # (which sends 0 as "permanent"), and the engine. Only a value
                # >= 1 bounds the window; otherwise it runs to the horizon.
                end = (start + fs.duration_years - 1
                       if fs.duration_years and fs.duration_years >= 1 else None)
                state.active_failsafes.append(
                    _ActiveFailsafe(action=fs.action, start_year=start, end_year=end))
        self._apply_active_failsafes(state, year)

    def _apply_active_failsafes(self, state: EngineState, year: int) -> None:
        """Apply, refresh, or close each in-flight failsafe action for this year.

        Sustained income is re-derived from present value every active year so it
        stays inflation-indexed, and *replaces* the target's earned income (the
        pre-failsafe value is saved and restored when the window ends).
        """
        for af in state.active_failsafes:
            if af.closed:
                continue
            active_now = af.start_year <= year and (af.end_year is None or year <= af.end_year)
            a = af.action
            if active_now:
                infl = state.cumulative_inflation if a.present_value else 1.0
                if not af.activated:
                    af.saved_partner_income = state.income_partner
                    af.saved_partner_working = state.is_partner_working
                    af.saved_primary_income = state.income_primary
                    af.saved_primary_working = state.is_working
                    af.activated = True
                    if a.one_time_income:
                        state.brokerage_balance += a.one_time_income * infl
                    if a.one_time_expense:
                        self._fund_purchase(state, a.one_time_expense * infl, year)
                if a.partner_income is not None:
                    state.income_partner = a.partner_income * infl
                    state.is_partner_working = True
                if a.primary_income is not None:
                    state.income_primary = a.primary_income * infl
                    state.is_working = True
                if a.suspend_retirement_contributions:
                    state.suspend_retirement_contributions = True
                if a.annual_vacation is not None:
                    state.vacation_override = a.annual_vacation * infl
                if a.medical_cost_multiplier is not None:
                    state.medical_cost_multiplier = a.medical_cost_multiplier
            elif af.activated:
                # Window ended: restore the saved earned-income baseline, once.
                if a.partner_income is not None:
                    state.income_partner = af.saved_partner_income
                    state.is_partner_working = af.saved_partner_working
                if a.primary_income is not None:
                    state.income_primary = af.saved_primary_income
                    state.is_working = af.saved_primary_working
                af.closed = True

    def _apply_home_purchase(self, state: EngineState, ev, _amort_cache: Optional[dict] = None) -> None:
        p          = self._plan
        new_price  = ev.new_home_price or ev.home_price_override or state.home_value
        new_down   = ev.new_home_down_payment or new_price * 0.20
        new_rate   = ev.new_home_interest_rate or state.mortgage_interest_rate

        if ev.sell_current_home and not state.is_renting:
            equity   = max(0.0, state.home_value - state.mortgage_balance)
            proceeds = equity - state.home_value * ev.seller_closing_cost_rate
            state.brokerage_balance += max(0.0, proceeds)
            # Capital gain on the sale = amount realised (net of selling costs) minus
            # the cost basis (home_price_ref = purchase price). The IRC §121 primary-
            # residence exclusion shelters the first $250k/$500k; the rest is a
            # long-term gain. Route it through realized_gains_ytd so the existing
            # cap-gains-tax line taxes it and it also lifts MAGI (→ IRMAA), exactly
            # like any other realised gain — rather than landing in brokerage untaxed.
            amount_realized = state.home_value * (1 - ev.seller_closing_cost_rate)
            gain            = max(0.0, amount_realized - state.home_price_ref)
            exclusion       = (HOME_SALE_EXCLUSION_MFJ if state.is_married
                               else HOME_SALE_EXCLUSION_SINGLE)
            state.realized_gains_ytd += max(0.0, gain - exclusion)

        self._fund_purchase(state, new_down + new_price * ev.buyer_closing_cost_rate, ev.year)

        new_hp = HousingProfile(
            home_price=new_price, down_payment=new_down, interest_rate=new_rate,
            loan_term_years=p.housing.loan_term_years,
            annual_property_tax_rate=p.housing.annual_property_tax_rate,
            annual_insurance=p.housing.annual_insurance,
            annual_maintenance_rate=p.housing.annual_maintenance_rate,
            pmi_annual_rate=p.housing.pmi_annual_rate,
        )
        new_calc = MortgageCalculator(new_hp, p.investments.annual_home_appreciation_rate)
        _ck = (new_price, new_down, new_rate, p.housing.loan_term_years)

        state.mortgage_calc        = new_calc
        state.amort_lookup         = (_amort_cache[_ck] if _amort_cache and _ck in _amort_cache
                                      else self._amort_lookup(new_calc))
        state.mortgage_year_offset = ev.year - 1
        state.mortgage_interest_rate = new_rate
        state.home_price_ref       = new_price
        state.home_value           = new_price
        state.mortgage_balance     = new_hp.loan_amount
        state.is_renting           = False

    # ------------------------------------------------------------------ #
    # Year computation — orchestrator + focused helpers                   #
    # ------------------------------------------------------------------ #

    def _compute_year(
        self,
        state: EngineState,
        year: int,
        market_return_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
        salary_growth_override: Optional[float] = None,
    ) -> YearlySnapshot:
        p   = self._plan
        inv = p.investments
        mkt = self._investment_return(year, market_return_override)
        inf = inflation_override if inflation_override is not None else inv.annual_inflation_rate
        # cumulative_inflation is the rolling product of (1+rate) for all prior years.
        # Using it directly — rather than (1+this_year_rate)^(year-1) — is correct
        # whether inflation is constant OR varies year-to-year (Monte Carlo).
        inf_f = state.cumulative_inflation
        hc_f  = state.cumulative_healthcare_inflation

        # --- Contributions & tax ---
        hsa, k401, partner_k401, r529, employer_match = self._contributions(state, year)
        biz_net, biz_se_tax, biz_equity, biz_solo_401k = self._business(state, year)
        tax, aotc, tax_breakdown = self._tax_and_credits(state, year, hsa, k401, r529, inf_f)

        # Backdoor Roth IRA contribution (post-tax — no deduction, reduces net income).
        # Limit: $7,000/person × (2 if married), always fixed nominal dollars.
        if self._plan.strategies.use_backdoor_roth:
            roth_limit   = ROTH_IRA_LIMIT * (2 if state.is_married else 1)
            roth_contrib = min(inv.roth_contribution_for_year(year), roth_limit)
        else:
            roth_contrib = 0.0

        net_income = state.gross_income + biz_net - tax - biz_se_tax - hsa - k401 - partner_k401 - biz_solo_401k - roth_contrib

        # --- Expenses ---
        housing_cost, home_equity, home_value, eoy_mortgage = self._housing(state, year, inf_f)
        lifestyle_base, medical_oop, parent_care, insurance_premiums, self_ltc = self._lifestyle(state, inf_f, hc_f, year)
        college_gross, drawdown_529, net_college, annual_529_save = self._college(state, year, inf_f, r529)
        wedding_save, wedding_spend = self._weddings(state, year, mkt)
        car_pmt, car_purchase, car_sale, car_operating = self._cars(state, year, inf_f)
        brokerage_earmark = inv.annual_brokerage_contribution
        # Wedding savings stay invested in brokerage until the wedding, so route
        # them into the brokerage inflow instead of letting them leave the books.
        brokerage_inflow = brokerage_earmark + wedding_save

        # Capital-gains tax on gains realized this year (home sale, plus brokerage
        # sold to fund purchases — accrued in realized_gains_ytd). Taxed through the
        # real 0/15/20% LTCG brackets stacked on this year's ordinary taxable income,
        # so a gain in a low-income year is taxed lightly and a large lumpy gain
        # reaches 20% automatically — no hand-set flat rate. (The flat
        # capital_gains_tax_rate now only feeds the retirement-readiness haircut.)
        tmp_inc, tmp_inv = self._tax_profiles(state, hsa, k401, r529)
        cap_gains_tax = self._tax.capital_gains_tax(
            state.realized_gains_ytd, tmp_inc, tmp_inv, p.strategies,
            num_children=state.num_children, inflation_factor=inf_f)

        # HSA pays qualified medical expenses tax-free, which the model previously
        # ignored (the HSA only ever grew, ballooning absurdly). We spend it in
        # RETIREMENT, not as incurred: with maximize_hsa the account is a retirement
        # health vehicle — you let it compound tax-free while working (paying medical
        # from income) and draw it down for the large medical/LTC costs of later
        # life. Out-of-pocket + long-term care are known here and claim the HSA
        # first; Medicare (which depends on MAGI) claims the remainder inside the
        # fixed point below. Drawn from the grown, post-contribution HSA and capped
        # at its balance, so it never goes negative.
        retired          = self._is_retired(year)
        hsa_available    = state.hsa_balance * (1 + mkt) + hsa
        hsa_medical_paid = (min(medical_oop + self_ltc, max(0.0, hsa_available))
                            if retired else 0.0)
        hsa_remaining    = max(0.0, hsa_available - hsa_medical_paid) if retired else 0.0

        # Breathing room excluding Medicare (which depends on MAGI, resolved below).
        # The HSA-funded slice of medical is added back: it is paid from the HSA,
        # so it does not draw on income or the cash waterfall.
        base_breathing_room = (
            net_income
            - housing_cost
            - lifestyle_base
            + hsa_medical_paid
            - annual_529_save
            - net_college
            - brokerage_inflow
            - car_pmt
            - car_operating
            - cap_gains_tax
        )

        # Medicare + IRMAA (65+) is a healthcare cost folded into the lifestyle
        # bucket. IRMAA depends on MAGI, which depends on the 401k/IRA withdrawal
        # used to fund the retirement deficit, which in turn depends on Medicare
        # (a spending item) — and the HSA can pay Medicare tax-free, shrinking that
        # withdrawal and hence MAGI. Resolve the whole loop with a short fixed-point
        # iteration: each pass previews the withdrawal for the current deficit (net
        # of the HSA's Medicare payment), recomputes MAGI → Medicare, and stops once
        # Medicare stabilises (Medicare is tiny next to the IRMAA bracket widths).
        medicare_cost, hsa_for_medicare = self._solve_medicare(
            state, year, inf_f, hc_f, mkt, biz_net, base_breathing_room,
            k401 + biz_solo_401k + employer_match, partner_k401, brokerage_inflow,
            hsa_remaining,
        )
        # Medicare is a healthcare cost too, so a medical-cost failsafe cuts it.
        # Keep the HSA-funded slice consistent — it can't exceed the reduced bill.
        medicare_cost *= state.medical_cost_multiplier
        hsa_for_medicare = min(hsa_for_medicare, medicare_cost)
        lifestyle_cost  = lifestyle_base + medicare_cost
        # The HSA-funded slice of Medicare is added back (paid from the HSA, not cash).
        breathing_room  = base_breathing_room - medicare_cost + hsa_for_medicare
        total_hsa_drawn = hsa_medical_paid + hsa_for_medicare

        # --- Asset growth ---
        g = self._asset_growth(
            state, year, mkt, hsa, k401 + biz_solo_401k + employer_match, partner_k401,
            annual_529_save, drawdown_529, brokerage_inflow, breathing_room,
            roth_contrib=roth_contrib,
            roth_basis_available=state.roth_vested_basis,
            annual_expenses=lifestyle_cost + housing_cost,
            hsa_medical_paid=total_hsa_drawn,
        )

        # Life-insurance death benefit: a fixed nominal payout into the estate in
        # the year the primary dies (the final projected year). Not inflated (level
        # term coverage) and not spendable income — it lands straight in net worth.
        life_payout = p.lifestyle.annual_life_insurance_death_benefit if self._is_death_year(year) else 0.0

        nw = (g.retirement + g.hsa + g.col529 + g.roth_balance + g.brokerage + home_equity
              + g.uninvested + g.cash_buffer + biz_equity + life_payout)

        return YearlySnapshot(
            year=year,
            gross_income=state.gross_income,
            net_income=net_income,
            annual_tax_total=tax,
            annual_housing_cost=housing_cost,
            annual_lifestyle_cost=lifestyle_cost,
            annual_medical_oop=medical_oop,
            annual_college_cost=college_gross,
            annual_529_drawdown=drawdown_529,
            annual_parent_care_cost=parent_care,
            annual_retirement_contributions=k401 + partner_k401 + employer_match,
            annual_hsa_contributions=hsa,
            annual_hsa_withdrawal=total_hsa_drawn,
            annual_brokerage_contribution=brokerage_earmark,
            annual_aotc_credit=aotc,
            annual_federal_tax=max(0.0, tax_breakdown.federal_income_tax - aotc),
            annual_fica_tax=tax_breakdown.total_fica,
            annual_state_tax=tax_breakdown.state_income_tax,
            annual_car_payment=car_pmt,
            annual_capital_gains_tax=cap_gains_tax,
            annual_wedding_save=wedding_save,
            annual_wedding_spend=wedding_spend,
            annual_breathing_room=breathing_room,
            retirement_balance=g.retirement,
            brokerage_balance=g.brokerage,
            brokerage_gains=g.brokerage_gains,
            college_529_balance=g.col529,
            home_value=home_value,
            home_equity=home_equity,
            hsa_balance=g.hsa,
            uninvested_cash=g.uninvested,
            cash_buffer=g.cash_buffer,
            mortgage_balance=eoy_mortgage,
            net_worth=nw,
            filing_status=state.filing_status,
            num_children=state.num_children,
            is_renting=state.is_renting,
            is_married=state.is_married,
            is_working=state.is_working,
            is_partner_working=state.is_partner_working,
            roth_ira_balance=g.roth_balance,
            roth_contribution_basis=g.roth_basis,
            roth_vested_basis=state.roth_vested_basis,
            annual_roth_contribution=roth_contrib,
            annual_business_income=biz_net,
            business_equity=biz_equity,
            car_purchase_cost=car_purchase,
            car_sale_proceeds=car_sale,
            cumulative_inflation=inf_f,
            annual_insurance_premiums=insurance_premiums,
            annual_self_ltc_cost=self_ltc,
            annual_medicare_cost=medicare_cost,
            annual_car_operating_cost=car_operating,
            annual_retirement_withdrawal=g.taxable_withdrawal + state.purchase_taxable_wd,
            annual_retirement_withdrawal_tax=(
                g.withdrawal_tax
                + state.purchase_taxable_wd * (
                    p.retirement.retirement_withdrawal_tax_rate if p.retirement else 0.0)
            ),
            annual_brokerage_withdrawal=g.brokerage_withdrawal,
            annual_roth_withdrawal=g.roth_withdrawal + state.purchase_roth_wd,
            annual_life_insurance_payout=life_payout,
        )

    # ------------------------------------------------------------------ #
    # Computation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _age_in_year(self, year: int) -> Optional[int]:
        """Primary's age in a projection year, or None without a RetirementProfile.

        Single source of age for every age-aware rule (401k catch-up, salary
        plateau) so the derivation lives in one place.
        """
        rp = self._plan.retirement
        return rp.current_age + (year - 1) if rp else None

    def _death_year(self) -> Optional[int]:
        """Projection year the primary dies (age == life_expectancy_age), or None.

        None when no RetirementProfile or no life_expectancy_age is set. Otherwise
        the death year *defines the projection endpoint*, whether that is earlier
        or later than projection_years — so the year-by-year tables always run
        right through death rather than stopping at an arbitrary horizon.
        """
        rp = self._plan.retirement
        if rp is None or rp.life_expectancy_age is None:
            return None
        return max(1, rp.life_expectancy_age - rp.current_age + 1)

    def _horizon(self) -> int:
        """Number of years to actually project. When a life_expectancy_age is set it
        governs the endpoint (extending past or truncating projection_years so the
        projection ends the year the primary dies); otherwise projection_years."""
        death = self._death_year()
        return death if death is not None else self._plan.projection_years

    def _is_death_year(self, year: int) -> bool:
        return year == self._death_year()

    def _is_retired(self, year: int) -> bool:
        """True once the primary reaches retirement_age (needs a RetirementProfile).

        Marks when earned income stops, the portfolio de-risks, and the HSA switches
        from accumulating to paying medical. Distinct from :meth:`_is_penalty_free`,
        which gates penalty-free access to the 401k/IRA and Roth earnings.
        """
        rp = self._plan.retirement
        if rp is None:
            return False
        age = self._age_in_year(year)
        return age is not None and age >= rp.retirement_age

    def _is_penalty_free(self, year: int) -> bool:
        """True once the primary is past 59½ — when the 401k/IRA and Roth earnings
        become penalty-free to withdraw, independent of ``retirement_age``.

        Someone who retires early (before 59½) still cannot tap those accounts
        without penalty; someone past 59½ but still working can. Needs a
        RetirementProfile for the age basis.
        """
        rp = self._plan.retirement
        if rp is None:
            return False
        age = self._age_in_year(year)
        return age is not None and age >= PENALTY_FREE_AGE

    def _investment_return(self, year: int, override: Optional[float]) -> float:
        """Single source of truth for this year's investment growth rate.

        A Monte-Carlo ``override`` (a sampled return) always wins. Otherwise the
        portfolio de-risks at retirement: retired years grow at the
        RetirementProfile's ``expected_post_retirement_return`` rather than the
        accumulation-phase ``annual_market_return``. This keeps the year-by-year
        projection consistent with the retirement-readiness calc, which discounts
        the cost stream at the same post-retirement rate.
        """
        if override is not None:
            return override
        rp = self._plan.retirement
        if rp is not None and self._is_retired(year):
            return rp.expected_post_retirement_return
        return self._plan.investments.annual_market_return

    def _default_retirement_order(self) -> list[str]:
        """Pick a withdrawal order from the starting balance mix.

        A large traditional 401k/IRA relative to the brokerage favours the
        "bracket-fill" order (draw the 401k before the brokerage) to shrink future
        RMDs and smooth taxes; otherwise the conventional order spends the
        already-taxed brokerage first and lets the 401k keep deferring.
        """
        inv = self._plan.investments
        if inv.current_retirement_balance >= inv.current_brokerage_balance:
            return list(_ORDER_BRACKET_FILL)
        return list(_ORDER_CONVENTIONAL)

    def _withdrawal_order(self, penalty_free: bool) -> list[str]:
        """Complete, sanitised source order for funding a deficit this year.

        Before 59½ the 401k is off-limits (penalty), so the pre-retirement order
        runs; once penalty-free the configured/default retirement order applies and
        the 401k participates.
        """
        if not penalty_free:
            return _complete_order(_PRE_RETIREMENT_ORDER, _PRE_RETIREMENT_ORDER)
        configured = self._plan.strategies.retirement_withdrawal_order
        base = configured if configured else self._default_retirement_order()
        return _complete_order(base, WITHDRAWAL_SOURCES)

    def _cap_gains_rate(self) -> float:
        """Long-term capital-gains rate: InvestmentProfile first, with
        RetirementProfile.capital_gains_tax_rate as a backward-compatible fallback."""
        rate = self._plan.investments.capital_gains_tax_rate
        if rate:
            return rate
        rp = self._plan.retirement
        return rp.capital_gains_tax_rate if rp else 0.0

    def _retirement_cap_gains_rate(self) -> float:
        """Effective cap-gains rate for the retirement-readiness haircut on the
        *remaining unrealized* gains.

        SIMPLIFIED ASSUMPTION: a retiree draws the brokerage down gradually and
        often sits in the 0%/15% LTCG bracket, so this can be set lower than the
        working-years rate. Falls back to the accumulation rate when unset.
        """
        rate = self._plan.investments.retirement_capital_gains_tax_rate
        return rate if rate is not None else self._cap_gains_rate()

    def _contributions(
        self, state: EngineState, year: int
    ) -> tuple[float, float, float, float]:
        """Returns (hsa, k401, partner_k401, r529)."""
        inv   = self._plan.investments
        strat = self._plan.strategies

        is_family = state.is_married or state.num_children > 0
        hsa_limit = HSA_LIMIT_FAMILY if is_family else HSA_LIMIT_SINGLE
        hsa       = min(inv.annual_hsa_contribution, hsa_limit) if strat.maximize_hsa else 0.0

        # Age-aware 401k ceiling: catch-up (age 50+) vs base, driven by the
        # primary's age this projection year.  Without a RetirementProfile the
        # age is unknown and limit_401k falls back to the catch-up ceiling.
        # The partner shares the primary's age basis (no separate partner age is
        # modelled); this only matters when a partner contributes above the base.
        k401_cap = limit_401k(self._age_in_year(year))
        k401         = min(inv.annual_401k_contribution, k401_cap)
        partner_k401 = (
            min(inv.partner_annual_401k_contribution, k401_cap)
            if state.income_partner > 0 else 0.0
        )

        # Employer match — free money, goes straight to retirement, no tax impact on employee
        employer_match = 0.0
        if inv.employer_match is not None:
            employer_match = inv.employer_match.compute_match(
                employee_contribution=k401,
                gross_salary=state.income_primary,
                projection_year=year,
            )

        # 529: stop contributing once all children have graduated (only enforced
        # when a CollegeProfile is configured)
        col = self._plan.college
        if col and state.child_birth_years:
            graduated = all(
                (year - by) >= col.start_age + col.years_per_child
                for by in state.child_birth_years
            )
            r529 = 0.0 if graduated else inv.annual_529_contribution
        else:
            r529 = inv.annual_529_contribution

        # A failsafe may suspend 401k/IRA deferrals this year; the employer match
        # is contingent on the employee contribution, so it goes too. HSA/529 are
        # separate account types and are left untouched.
        if state.suspend_retirement_contributions:
            k401 = partner_k401 = employer_match = 0.0

        return hsa, k401, partner_k401, r529, employer_match

    def _tax_profiles(self, state: EngineState, hsa: float, k401: float, r529: float):
        """Build the per-year (IncomeProfile, InvestmentProfile) the tax engine needs.

        Shared by the ordinary income-tax call and the capital-gains call so both
        see the same income, filing status, state, and deductions.
        """
        p = self._plan
        tmp_inc = IncomeProfile(
            gross_annual_income=state.gross_income,
            filing_status=state.filing_status,
            state=p.income.state,
            other_state_flat_rate=p.income.other_state_flat_rate,
        )
        tmp_inv = InvestmentProfile(
            annual_hsa_contribution=hsa,
            annual_401k_contribution=k401,
            annual_529_contribution=r529,
        )
        return tmp_inc, tmp_inv

    def _tax_and_credits(
        self,
        state: EngineState,
        year: int,
        hsa: float,
        k401: float,
        r529: float,
        inf_f: float,
    ) -> tuple[float, float, TaxResult]:
        """Returns (effective_tax, aotc_credit, tax_breakdown)."""
        p = self._plan
        tmp_inc, tmp_inv = self._tax_profiles(state, hsa, k401, r529)
        # inf_f inflation-indexes the tax brackets/deductions to this projection
        # year, mirroring the IRS's annual indexing (prevents nominal bracket creep).
        breakdown = self._tax.calculate(tmp_inc, tmp_inv, p.strategies,
                                        num_children=state.num_children,
                                        inflation_factor=inf_f)
        aotc     = self._aotc_credit(state, year, state.gross_income, state.is_married, inf_f)
        eff_tax  = max(0.0, breakdown.total_annual_tax - aotc)
        return eff_tax, aotc, breakdown

    def _housing(
        self, state: EngineState, year: int, inf_f: float
    ) -> tuple[float, float, float, float]:
        """Returns (annual_cost, home_equity, home_value, eoy_mortgage_balance)."""
        p = self._plan
        if state.is_renting:
            # monthly_rent is advanced each year by annual_rent_increase_rate
            # in _advance_state — already in nominal terms, no inf_f needed.
            return state.monthly_rent * 12, 0.0, 0.0, 0.0

        mc = state.mortgage_calc
        if mc:
            ref           = state.home_price_ref
            monthly_other = (
                ref * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
                + p.housing.annual_insurance
            ) / 12 * inf_f

            mortgage_yr   = year - state.mortgage_year_offset
            # Once the loan term is over the mortgage is paid off: stop charging
            # P&I (and PMI) and bill only the ongoing carrying costs. Without this
            # the fixed payment kept being charged for the whole projection.
            if mortgage_yr > mc._p.loan_term_years:
                monthly_pi = pmi = 0.0
            else:
                monthly_pi = mc.monthly_pi_payment()
                pmi = (
                    mc._pmi_payment(state.mortgage_balance)
                    if state.mortgage_balance / ref > 0.80 and mc._p.requires_pmi else 0.0
                )
            cost = (monthly_pi + monthly_other + pmi) * 12

            eoy_balance   = state.amort_lookup.get(mortgage_yr, state.mortgage_balance)
            equity        = max(0.0, state.home_value - eoy_balance)
            return cost, equity, state.home_value, eoy_balance

        # Owned outright
        ref  = state.home_price_ref
        cost = (
            ref * (p.housing.annual_property_tax_rate + p.housing.annual_maintenance_rate)
            + p.housing.annual_insurance
        ) * inf_f
        return cost, state.home_value, state.home_value, 0.0

    def _lifestyle(
        self, state: EngineState, inf_f: float, hc_f: float, year: int = 1
    ) -> tuple[float, float, float, float, float]:
        """Returns (annual_lifestyle, medical_oop, parent_care, insurance_premiums, self_ltc).

        ``inf_f`` is the general cumulative-inflation factor; ``hc_f`` is the
        (typically higher) healthcare factor applied to medical OOP, the health-
        insurance premium, and your own long-term care.
        """
        lif = self._plan.lifestyle
        med_mult = state.medical_cost_multiplier   # failsafe may cut healthcare costs

        medical     = lif.scaled_medical_oop(state.is_married, state.num_children) * hc_f * med_mult
        pets        = state.num_pets * lif.annual_pet_cost * inf_f
        vacation    = (state.vacation_override if state.vacation_override is not None
                       else lif.annual_vacation * inf_f)
        other       = lif.monthly_other_recurring * 12 * inf_f
        parent_care = lif.annual_parent_care_cost * inf_f if state.parent_care_active else 0.0

        # Insurance premiums (see LifestyleProfile). Disability replaces earned
        # income, so it lapses when you stop working; health (the employee share)
        # applies while working and pre-Medicare, after which the RetirementProfile
        # Medicare model takes over; life is charged every year until zeroed.
        # Only the health share tracks healthcare inflation; disability/life follow
        # the general rate.
        age          = self._age_in_year(year)
        medicare_age = self._plan.retirement.medicare_start_age if self._plan.retirement else 65
        health       = (lif.annual_health_insurance_premium * hc_f * med_mult
                        if state.is_working and (age is None or age < medicare_age) else 0.0)
        disability   = (lif.annual_disability_insurance_premium if state.is_working else 0.0) * inf_f
        premiums     = health + disability + lif.annual_life_insurance_premium * inf_f

        # Your own long-term care — age-gated, needs a RetirementProfile for age.
        self_ltc = (lif.annual_self_ltc_cost * hc_f * med_mult
                    if age is not None and age >= lif.self_ltc_start_age else 0.0)

        # Childcare: age-bracketed profile takes priority over flat monthly_childcare.
        # Each child's age is computed from their birth year for accurate per-child costs.
        if lif.childcare_profile and state.child_birth_years:
            childcare = sum(
                lif.childcare_profile.monthly_cost_at_age(year - by) * 12 * inf_f
                for by in state.child_birth_years
            )
        else:
            childcare = state.num_children * lif.monthly_childcare * 12 * inf_f

        # Retirement "spending smile": scale discretionary spending (vacation,
        # pets, monthly "other") down in later life. Healthcare, insurance, care,
        # LTC, and childcare are excluded — they keep tracking inflation/age.
        smile = (self._plan.retirement.discretionary_spending_factor(age)
                 if self._plan.retirement else 1.0)
        discretionary = (pets + vacation + other) * smile

        total = (medical + childcare + parent_care + premiums + self_ltc
                 + discretionary)
        return total, medical, parent_care, premiums, self_ltc

    def _magi_for_irmaa(self, state: EngineState, biz_net: float,
                        taxable_withdrawal: float = 0.0) -> float:
        """MAGI used to place the household in an IRMAA bracket, in nominal dollars.

        Built from income the engine actually recognises: earned + business
        income, realised capital gains this year, and the pre-tax 401k/IRA
        withdrawal used to fund a retirement deficit (ordinary income). In
        retirement, earned income is ~0 and the withdrawal is the dominant term,
        so IRMAA now tracks the real taxable draw rather than a proxy.
        """
        return (
            state.gross_income
            + max(0.0, biz_net)
            + max(0.0, state.realized_gains_ytd)
            + max(0.0, taxable_withdrawal)
            + max(0.0, state.purchase_taxable_wd)   # 401k tapped for a lump-sum purchase
        )

    def _medicare(self, state: EngineState, year: int, magi: float,
                  inf_f: float, hc_f: float) -> float:
        """Annual Medicare cost (base premium + IRMAA surcharge) once age >= start.

        Requires a RetirementProfile (for age and the premium); returns 0 before
        Medicare age or when no RetirementProfile is configured. A married couple
        is assumed to enrol together, so both the base premium and the per-person
        IRMAA surcharge are charged twice. Premium and surcharge dollars grow at
        the healthcare rate (``hc_f``); the IRMAA bracket thresholds are CPI-indexed
        at the general rate (``inf_f``), mirroring how the SSA re-indexes them.
        """
        rp  = self._plan.retirement
        age = self._age_in_year(year)
        if not rp or age is None or age < rp.medicare_start_age:
            return 0.0
        enrolled  = 2 if state.is_married else 1
        base      = rp.annual_medicare_premium * hc_f * enrolled
        surcharge = irmaa_annual_surcharge(magi, state.is_married, inf_f) * hc_f * enrolled
        return base + surcharge

    def _solve_medicare(self, state: EngineState, year: int, inf_f: float, hc_f: float,
                        mkt: float, biz_net: float, base_breathing_room: float,
                        k401_total: float, partner_k401: float, brokerage_inflow: float,
                        hsa_remaining: float = 0.0) -> tuple[float, float]:
        """Medicare cost, resolving the IRMAA↔withdrawal↔Medicare loop.

        Returns ``(medicare_cost, hsa_for_medicare)`` — the Medicare bill and the
        tax-free slice of it the HSA pays (0 when the HSA is exhausted by earlier
        medical or the household is not retired).

        Medicare (a spending item) enlarges any retirement deficit, which enlarges
        the taxable 401k withdrawal that funds it, which raises MAGI and hence
        IRMAA — and the HSA can pay Medicare tax-free, shrinking that withdrawal.
        This previews the withdrawal with the *same* waterfall the actual funding
        uses (so MAGI is self-consistent), nets out the HSA's Medicare payment, and
        iterates; it converges almost immediately because Medicare is small relative
        to the IRMAA bracket widths. Below Medicare age it short-circuits with no
        iteration.
        """
        rp = self._plan.retirement
        age = self._age_in_year(year)
        if rp is None or age is None or age < rp.medicare_start_age:
            # Below Medicare eligibility the cost is $0 regardless of MAGI, so skip
            # the MAGI/withdrawal machinery entirely (the common pre-65 years).
            return 0.0, 0.0
        if not self._is_retired(year) and self._medicare(state, year, 0.0, inf_f, hc_f) == 0.0:
            # Not old enough for Medicare and not tapping the 401k → no feedback.
            return self._medicare(state, year, self._magi_for_irmaa(state, biz_net), inf_f, hc_f), 0.0

        penalty_free = self._is_penalty_free(year)
        order    = self._withdrawal_order(penalty_free)
        wd_rate  = (self._plan.retirement.retirement_withdrawal_tax_rate
                    if self._plan.retirement else 0.0)
        ret_grown  = state.retirement_balance * (1 + mkt) + k401_total + partner_k401
        brok_grown = state.brokerage_balance * (1 + mkt) + brokerage_inflow
        roth_avail = state.roth_ira_balance if penalty_free else state.roth_vested_basis
        available  = _available_sources(state.cash_buffer, state.uninvested_cash, ret_grown,
                                        brok_grown, roth_avail, penalty_free)

        medicare = 0.0
        for _ in range(6):
            hsa_for_medicare = min(medicare, hsa_remaining)
            # HSA pays its slice of Medicare, so only the rest enlarges the deficit.
            deficit = max(0.0, -(base_breathing_room - medicare + hsa_for_medicare))
            _, taxable_wd, _, _ = _fund_deficit(available, order, deficit, wd_rate)
            magi = self._magi_for_irmaa(state, biz_net, taxable_wd)
            updated = self._medicare(state, year, magi, inf_f, hc_f)
            if abs(updated - medicare) < 1.0:
                return updated, min(updated, hsa_remaining)
            medicare = updated
        return medicare, min(medicare, hsa_remaining)

    def _college(
        self,
        state: EngineState,
        year: int,
        inf_f: float,
        r529: float,
    ) -> tuple[float, float, float, float]:
        """Returns (gross_cost, drawdown_529, net_from_brokerage, annual_529_save)."""
        col = self._plan.college
        annual_529_save = r529 * state.num_children

        if not col or not state.child_birth_years:
            return 0.0, 0.0, 0.0, annual_529_save

        # 529 available = current balance + this year's contributions (same-year drawdown allowed)
        available = state.college_529_balance + annual_529_save
        gross, drawdown, remaining = 0.0, 0.0, available

        for by in state.child_birth_years:
            age = year - by
            if col.start_age <= age < col.start_age + col.years_per_child:
                cost    = col.annual_cost_per_child * inf_f
                gross  += cost
                drawn   = min(remaining, cost)
                drawdown += drawn
                remaining -= drawn

        net_brokerage = max(0.0, gross - drawdown)
        return gross, drawdown, net_brokerage, annual_529_save

    def _aotc_credit(
        self,
        state: EngineState,
        year: int,
        gross_income: float,
        is_married: bool,
        inf_f: float,
    ) -> float:
        col = self._plan.college
        if not col or not col.use_aotc_credit:
            return 0.0

        low  = by_filing_status(is_married, _AOTC_PHASEOUT_SINGLE_LOW, _AOTC_PHASEOUT_MFJ_LOW)
        high = by_filing_status(is_married, _AOTC_PHASEOUT_SINGLE_HIGH, _AOTC_PHASEOUT_MFJ_HIGH)
        phase = linear_phaseout(gross_income, low, high)

        eligible = sum(
            1 for by in state.child_birth_years
            if col.start_age <= (year - by) < col.start_age + min(col.years_per_child, 4)
            and (year - by - col.start_age + 1) <= min(col.years_per_child, 4)
        )
        return eligible * _AOTC_MAX_CREDIT * phase

    def _weddings(self, state: EngineState, year: int, mkt: float) -> tuple[float, float]:
        """Wedding sinking fund — save, invested in brokerage, then spend at the wedding.

        Returns (annual_savings, wedding_spend). Each child's fund accrues the
        per-child rate and grows at the market rate while it waits; at the wedding
        it is paid out of brokerage (where it was held). Contributions run through
        age 25 (age < _WEDDING_AGE), matching the legacy stop age, so the yearly
        savings figure is unchanged; the payout lands when the child turns 26.
        """
        rate = self._plan.lifestyle.annual_wedding_fund_per_child
        if not rate:
            return 0.0, 0.0
        dep_f = _deposit_growth_factor(mkt, self._plan.investments.compounding_period_months)
        save = spend = 0.0
        for i, by in enumerate(state.child_birth_years):
            age = year - by
            state.wedding_fund[i] *= (1 + mkt)        # invested alongside brokerage
            if age < _WEDDING_AGE:
                state.wedding_fund[i] += rate * dep_f  # deposit earns partial-year growth
                save += rate                           # cash saved (reported nominal)
            elif age == _WEDDING_AGE:
                spend += state.wedding_fund[i]         # wedding paid from the accrued fund
                state.wedding_fund[i] = 0.0
        self._fund_purchase(state, spend, year)        # fund was held in brokerage
        return save, spend

    def _business(
        self,
        state: EngineState,
        year: int,
    ) -> tuple[float, float, float, float]:
        """
        Returns (net_income, se_tax, business_equity, solo_401k_contribution).

        net_income   — owner's draw after SE tax, QBI deduction, and health
                       insurance deduction; flows into breathing room
        se_tax       — self-employment tax owed (shown separately in cash flow)
        business_equity — current business asset value (net_profit × multiple)
        solo_401k_contribution — amount deposited to retirement this year

        Revenue is stored in state.business_revenue and grown in _advance_state.
        The initial_investment is deducted from brokerage in the start year.
        """
        biz = self._plan.business
        if biz is None or year < biz.start_year:
            return 0.0, 0.0, state.business_equity, 0.0

        # One-time initial investment in start year
        if year == biz.start_year and biz.initial_investment > 0:
            self._fund_purchase(state, biz.initial_investment, year)

        # Business sale: liquidate equity into brokerage once, then silence permanently.
        if biz.sale_year is not None and year >= biz.sale_year:
            if year == biz.sale_year:
                proceeds = state.business_equity  # already scaled by ownership_pct
                state.brokerage_balance += proceeds
            return 0.0, 0.0, 0.0, 0.0

        revenue    = state.business_revenue
        net_profit = revenue * (1.0 - biz.expense_ratio) * biz.ownership_pct

        # --- Self-employment tax ---
        # SE tax is 15.3% on 92.35% of net profit.
        # The employer half (7.65%) is deductible from AGI — reduces taxable income.
        se_base     = net_profit * _SE_TAX_DEDUCTIBLE_SHARE
        se_tax      = se_base * _SE_TAX_RATE
        employer_half_deduction = se_tax / 2.0

        # --- Health insurance deduction ---
        hi_deduction = min(biz.self_employed_health_insurance, net_profit)

        # --- QBI deduction ---
        # 20% of qualified business income, phased out above income thresholds.
        # Simplified: apply phase-out linearly over a $50k window above the limit.
        qbi_deduction = 0.0
        if biz.use_qbi_deduction:
            limit = by_filing_status(state.is_married, _QBI_PHASEOUT_SINGLE, _QBI_PHASEOUT_MFJ)
            phase = linear_phaseout(state.gross_income + net_profit, limit, limit + 50_000)
            qbi_deduction = net_profit * 0.20 * phase

        # --- Solo 401k ---
        # Capped at IRS limit and net profit (can't contribute more than earned)
        solo_k = min(biz.solo_401k_contribution, LIMIT_SOLO_401K, max(0.0, net_profit))

        # --- SEP-IRA ---
        # Up to 25% of net self-employment income (after SE tax deduction)
        sep_base = max(0.0, net_profit - employer_half_deduction)
        sep = min(biz.sep_ira_contribution, 0.25 * sep_base)
        # SEP flows into retirement alongside solo 401k
        solo_k_total = min(solo_k + sep, LIMIT_SOLO_401K)

        # --- Net income to owner ---
        # Gross profit minus all deductions; the actual tax impact on W-2 income
        # is handled in _tax_and_credits via the normal tax engine (which will see
        # a lower AGI because of employer_half_deduction + hi_deduction + qbi_deduction).
        # Here we return the owner's take-home after SE tax and retirement contributions.
        net_income = net_profit - se_tax - solo_k_total

        # --- Business equity ---
        biz_equity = net_profit * biz.equity_multiple

        return net_income, se_tax, biz_equity, solo_k_total

    def _fund_purchase(self, state: EngineState, amount: float, year: int) -> None:
        """Fund a lump-sum outflow from the full asset waterfall.

        Order: brokerage (realizing gains, as before) → cash buffer → uninvested
        cash → 401k/IRA (only once retired; grossed up for withdrawal tax) → Roth
        basis. This prevents a purchase from driving brokerage negative while other
        accounts still hold money — the bug where a car/home down payment overdrew
        an empty brokerage into an ever-compounding negative balance despite a large
        401k. Any 401k draw accrues to ``purchase_taxable_wd`` so it flows into
        MAGI/IRMAA and the reported withdrawal (same flat-rate tax model as deficit
        funding). A genuine, all-accounts-exhausted shortfall is booked as negative
        brokerage (real insolvency), matching the previous last-resort behaviour.
        """
        if amount <= 0:
            return
        penalty_free = self._is_penalty_free(year)
        wd_rate = (self._plan.retirement.retirement_withdrawal_tax_rate
                   if self._plan.retirement else 0.0)
        # Same pure waterfall engine as operating deficits, on the pre-growth
        # balances, in the purchase order (brokerage first). _available_sources
        # gates the 401k to 59½ and floors a negative brokerage at 0. Past 59½ the
        # whole Roth is qualified; before then only the basis is.
        roth_avail = state.roth_ira_balance if penalty_free else state.roth_vested_basis
        available = _available_sources(
            state.cash_buffer, state.uninvested_cash, state.retirement_balance,
            state.brokerage_balance, roth_avail, penalty_free,
        )
        reductions, taxable_wd, _tax, shortfall = _fund_deficit(
            available, _PURCHASE_ORDER, amount, wd_rate)

        # Apply the plan to state (brokerage via sell_brokerage so basis/gains stay
        # consistent; the 401k draw is grossed up inside _fund_deficit).
        if reductions["brokerage"] > 0:
            state.sell_brokerage(reductions["brokerage"])
        state.cash_buffer       -= reductions["cash_buffer"]
        state.uninvested_cash   -= reductions["uninvested_cash"]
        state.retirement_balance -= reductions["retirement_401k"]
        state.purchase_taxable_wd += taxable_wd
        roth_drawn = reductions["roth_basis"]
        if roth_drawn > 0:
            state.roth_ira_balance        = max(0.0, state.roth_ira_balance - roth_drawn)
            state.roth_contribution_basis = max(0.0, state.roth_contribution_basis - roth_drawn)
            state.roth_vested_basis       = max(0.0, state.roth_vested_basis - roth_drawn)
            state.purchase_roth_wd       += roth_drawn
        # Last resort — every account exhausted: book the shortfall as negative
        # brokerage (real insolvency), matching the operating-deficit behaviour.
        if shortfall > 1e-9:
            state.brokerage_balance -= shortfall

    def _finance_purchase(
        self, state: EngineState, nominal_price: float, nominal_down: float,
        rate: float, term_years: int, year: int,
    ) -> tuple[float, float]:
        """Fund ``nominal_down`` from the asset waterfall; return (principal, monthly).

        Single home for the "put money down, open a loan" step shared by first
        purchases, replacements, and kids' cars.
        """
        self._fund_purchase(state, nominal_down, year)
        principal = max(0.0, nominal_price - nominal_down)
        return principal, self._car_monthly_pi(principal, rate, term_years)

    def _cars(
        self, state: EngineState, year: int, inf_f: float
    ) -> tuple[float, float, float, float]:
        """Returns (annual_payment, purchase_cost, sale_proceeds, operating_cost)."""
        car = self._plan.car
        if not car:
            return 0.0, 0.0, 0.0, 0.0

        total_pmt, total_purchase, total_sale, total_operating = 0.0, 0.0, 0.0, 0.0

        for c in state.cars:
            # --- First purchase (explicit first_buy_year mode) ---
            if c["purchase_year"] is None:
                # Car hasn't been bought yet; wait for its first_buy_year
                if c.get("first_buy_year") == year:
                    nominal_down = car.down_payment * inf_f
                    principal, monthly = self._finance_purchase(
                        state, car.car_price * inf_f, nominal_down,
                        car.loan_rate, car.loan_term_years, year)
                    total_purchase += nominal_down
                    c.update(loan_balance=principal, loan_year=1,
                             purchase_year=year, monthly_payment=monthly)
                # Nothing to do before first_buy_year — skip to next car
                if c["purchase_year"] is None:
                    continue

            # --- Replacement cycle ---
            years_owned = year - c["purchase_year"]
            if years_owned > 0 and years_owned % car.replace_every_years == 0:
                proceeds = self._car_old_proceeds(state, car, year)
                state.brokerage_balance += proceeds
                total_sale += proceeds

                nominal_down = car.down_payment * inf_f
                principal, monthly = self._finance_purchase(
                    state, car.car_price * inf_f, nominal_down,
                    car.loan_rate, car.loan_term_years, year)
                total_purchase += nominal_down
                c.update(loan_balance=principal, loan_year=1,
                         purchase_year=year, monthly_payment=monthly)

            # --- Annual loan payment ---
            total_pmt += _pay_loan_year(c, car.loan_rate, car.loan_term_years)

            # --- Operating cost (owning the car, independent of the loan) ---
            # Charged every year the car exists (purchase_year is set above).
            total_operating += car.annual_operating_cost_per_car * inf_f

        # --- Kids' first cars ---
        if car and car.kids_car:
            col = self._plan.college  # may be None; buy_at_age defaulting handles it
            kc  = car.kids_car
            # buy_at_age default: graduation age if college configured, else 16
            if kc.buy_at_age is not None:
                buy_age = kc.buy_at_age
            elif col is not None:
                buy_age = col.start_age + col.years_per_child  # graduation age
            else:
                buy_age = 16

            for child_idx, birth_year in enumerate(state.child_birth_years):
                child_age = year - birth_year
                # Buy car in exactly the graduation year
                if child_age == buy_age:
                    # Check not already bought for this child
                    already = any(l["child_idx"] == child_idx for l in state.kid_car_loans)
                    if not already:
                        nominal_price = kc.car_price * inf_f
                        down          = nominal_price * kc.down_payment_pct
                        principal, monthly_pmt = self._finance_purchase(
                            state, nominal_price, down, kc.loan_rate, kc.loan_term_years, year)
                        total_purchase += down
                        state.kid_car_loans.append({
                            "child_idx":     child_idx,
                            "loan_balance":  principal,
                            "loan_year":     1,
                            "monthly_payment": monthly_pmt,
                        })

            # Annual payments on active kid car loans
            for loan in state.kid_car_loans:
                total_pmt += _pay_loan_year(loan, kc.loan_rate, kc.loan_term_years)

        return total_pmt, total_purchase, total_sale, total_operating

    def _car_old_proceeds(
        self, state: EngineState, car: CarProfile, year: int
    ) -> float:
        """Sell old car for residual_value, or hand down to an age-eligible child."""
        if not state.child_birth_years:
            return car.residual_value
        if any((year - by) >= car.hand_down_age for by in state.child_birth_years):
            return 0.0
        return car.residual_value

    def _asset_growth(
        self,
        state: EngineState,
        year: int,
        mkt: float,
        hsa: float,
        k401: float,
        partner_k401: float,
        annual_529_save: float,
        drawdown_529: float,
        brokerage_earmark: float,
        breathing_room: float,
        roth_contrib: float = 0.0,
        roth_basis_available: float = 0.0,
        annual_expenses: float = 0.0,
        hsa_medical_paid: float = 0.0,
    ) -> _Growth:
        """Returns named growth result — see _Growth for field docs."""
        col = self._plan.college
        inv = self._plan.investments

        # Deposits made during the year earn a partial year of return under sub-annual
        # compounding; dep_f scales them (1.0 when compounding is annual). The starting
        # balance always grows by exactly (1 + rate) — see _deposit_growth_factor.
        period = inv.compounding_period_months
        dep_f  = _deposit_growth_factor(mkt, period)

        ret_bal  = state.retirement_balance * (1 + mkt) + (k401 + partner_k401) * dep_f
        # HSA grows, takes this year's contribution, then pays qualified medical
        # (capped at balance by the caller, so this stays >= 0).
        hsa_bal  = state.hsa_balance * (1 + mkt) + hsa * dep_f - hsa_medical_paid

        r529_growth = (
            col.early_529_return if year <= col.glide_path_years else col.late_529_return
        ) if col else mkt
        col529_bal = max(0.0,
            state.college_529_balance * (1 + r529_growth)
            + annual_529_save * _deposit_growth_factor(r529_growth, period) - drawdown_529
        )

        # --- Cash buffer ---
        # Target floor = N months of annual expenses held as liquid cash (0% return).
        # Buffer is topped up from breathing room BEFORE surplus is swept to brokerage.
        # Deficits drain the buffer first, then brokerage (or uninvested_cash).
        buffer_floor = annual_expenses * inv.cash_buffer_months / 12
        current_buf  = state.cash_buffer

        # --- Roth IRA balance (grows at market rate, basis tracks contributions) ---
        # The balance grows the deposit; the basis records dollars actually put in.
        roth_bal   = state.roth_ira_balance * (1 + mkt) + roth_contrib * dep_f
        roth_basis = state.roth_contribution_basis + roth_contrib

        brok_grown = state.brokerage_balance * (1 + mkt) + brokerage_earmark * dep_f
        # Partial-year growth on brokerage deposits is unrealized gain (basis is the
        # dollars deposited, value is deposit·dep_f); track it alongside balance growth.
        brok_deposit_gain = brokerage_earmark * (dep_f - 1.0)
        taxable_withdrawal = withdrawal_tax = brokerage_withdrawal = roth_withdrawal = 0.0

        if breathing_room >= 0:
            # Surplus: top up the cash buffer to its floor, then invest the rest.
            topup      = min(breathing_room, max(0.0, buffer_floor - current_buf))
            new_buffer = current_buf + topup
            investable = breathing_room - topup
            if inv.auto_invest_surplus:
                brok_bal   = brok_grown + investable * dep_f
                brok_deposit_gain += investable * (dep_f - 1.0)
                uninvested = 0.0
            else:
                brok_bal   = brok_grown
                uninvested = state.uninvested_cash + investable
        else:
            # Deficit: fund it from accounts in the configured order. Before
            # retirement the 401k/IRA is excluded (penalties) and the legacy order
            # applies; in retirement the 401k participates and is taxed as ordinary
            # income (grossed up) — see _fund_deficit / _withdrawal_order.
            penalty_free = self._is_penalty_free(year)
            order   = self._withdrawal_order(penalty_free)
            wd_rate = (self._plan.retirement.retirement_withdrawal_tax_rate
                       if self._plan.retirement else 0.0)
            # Past 59½ the whole Roth (basis + earnings) is a qualified, tax-free
            # distribution; before then only the contribution basis is penalty-free,
            # so earnings stay locked.
            roth_avail = roth_bal if penalty_free else roth_basis_available
            available = _available_sources(current_buf, state.uninvested_cash, ret_bal,
                                           brok_grown, roth_avail, penalty_free)
            reductions, taxable_withdrawal, withdrawal_tax, shortfall = _fund_deficit(
                available, order, -breathing_room, wd_rate)

            new_buffer  = current_buf          - reductions["cash_buffer"]
            uninvested  = state.uninvested_cash - reductions["uninvested_cash"]
            ret_bal    -= reductions["retirement_401k"]
            roth_drawn  = reductions["roth_basis"]
            roth_withdrawal = roth_drawn
            roth_bal    = max(0.0, roth_bal   - roth_drawn)
            roth_basis  = max(0.0, roth_basis - roth_drawn)
            # Any deficit no source could cover shows as a negative brokerage
            # balance (insolvency), matching the previous last-resort behaviour.
            brokerage_withdrawal = reductions["brokerage"]
            brok_bal    = brok_grown - reductions["brokerage"] - shortfall

        # Track cumulative unrealized capital gains: market appreciation on the
        # starting balance, plus the partial-year growth earned by this year's
        # deposits (their basis is the dollars paid in). Contributions/withdrawals
        # otherwise move value and basis together, leaving gains untouched. Capped at
        # the balance so a large withdrawal effectively realizes gains.
        brok_gain = max(0.0, state.brokerage_balance) * mkt + brok_deposit_gain
        brokerage_gains = max(0.0, min(state.brokerage_gains + brok_gain, brok_bal))

        return _Growth(ret_bal, hsa_bal, col529_bal, brok_bal, uninvested, new_buffer,
                       roth_bal, roth_basis, brokerage_gains,
                       taxable_withdrawal, withdrawal_tax, brokerage_withdrawal,
                       roth_withdrawal)

    # ------------------------------------------------------------------ #
    # State advancement                                                    #
    # ------------------------------------------------------------------ #

    def _advance_state(
        self,
        state: EngineState,
        snap: YearlySnapshot,
        market_return: Optional[float] = None,
        inflation: Optional[float] = None,
        salary_growth: Optional[float] = None,
    ) -> None:
        p   = self._plan
        inv = p.investments
        inf = inflation if inflation is not None else inv.annual_inflation_rate

        # Salary real-growth plateau: once the primary reaches salary_growth_peak_age,
        # real raises stop — nominal growth is capped at inflation minus the configured
        # real decline (and never lifted above the underlying rate). Needs a
        # RetirementProfile for age; without one, growth is unchanged. Partner shares
        # the primary's age basis (no separate partner age is modelled).
        age = self._age_in_year(snap.year + 1)
        def _plateau(rate: float) -> float:
            if age is not None and age >= inv.salary_growth_peak_age:
                return min(rate, inf - inv.salary_real_decline_rate)
            return rate
        sg  = _plateau(salary_growth if salary_growth is not None else inv.annual_salary_growth_rate)
        psg = _plateau(salary_growth if salary_growth is not None else inv.partner_salary_growth_rate)

        if state.is_working:
            state.income_primary *= (1 + sg)
        if state.is_partner_working:
            state.income_partner *= (1 + psg)

        state.retirement_balance  = snap.retirement_balance
        state.brokerage_balance   = snap.brokerage_balance
        state.brokerage_gains     = snap.brokerage_gains
        state.hsa_balance         = snap.hsa_balance
        state.college_529_balance = snap.college_529_balance
        state.uninvested_cash       = snap.uninvested_cash
        state.cash_buffer           = snap.cash_buffer
        state.roth_ira_balance        = snap.roth_ira_balance
        state.roth_contribution_basis  = snap.roth_contribution_basis
        # 5-year vesting queue: push this year's contribution, pop the one
        # that is now 5 years old (it becomes penalty-free next year).
        # We use a simple list as FIFO (max len 5).
        queue = list(state.roth_contrib_queue)
        queue.append(snap.annual_roth_contribution)
        if len(queue) >= 5:
            vesting_now = queue.pop(0)   # oldest contribution, now ≥5 yrs old
        else:
            vesting_now = 0.0
        state.roth_contrib_queue  = queue
        state.roth_vested_basis  += vesting_now
        # Advance cumulative inflation: multiply by this year's rate
        state.cumulative_inflation *= (1 + inf)
        # Healthcare compounds at its own fixed rate (not the sampled inflation).
        state.cumulative_healthcare_inflation *= (1 + p.investments.annual_healthcare_inflation_rate)
        state.business_equity     = snap.business_equity
        if self._plan.business and snap.year >= self._plan.business.start_year:
            state.business_revenue *= (1 + self._plan.business.revenue_growth_rate)

        if state.is_renting:
            state.monthly_rent *= (1 + p.housing.annual_rent_increase_rate)
        else:
            state.home_value = snap.home_value * (1 + p.investments.annual_home_appreciation_rate)
            self._advance_mortgage(state, snap)

    def _advance_mortgage(self, state: EngineState, snap: YearlySnapshot) -> None:
        mc = state.mortgage_calc
        if not mc or state.mortgage_balance <= 0:
            return
        amort  = state.amort_lookup
        offset = state.mortgage_year_offset
        myr    = snap.year - offset
        if myr in amort:
            state.mortgage_balance = amort[myr]
        elif amort and myr > max(amort):
            state.mortgage_balance = 0.0
        else:
            rate    = state.mortgage_interest_rate
            ann_int = state.mortgage_balance * rate
            ann_pi  = mc.monthly_pi_payment() * 12
            state.mortgage_balance = max(0.0, state.mortgage_balance - max(0.0, ann_pi - ann_int))

    # ------------------------------------------------------------------ #
    # Car helpers                                                          #
    # ------------------------------------------------------------------ #

    # Amortising-loan payment — shared with the mortgage engine.
    _car_monthly_pi = staticmethod(monthly_amortized_payment)