"""
Failsafe subsystem for the projection engine.

A *failsafe* is a contingency rule ("if brokerage drops below $100k, partner
returns to work for 5 years") that watches a trigger metric each year and, when
tripped, arms an action that overrides income / spending for a window. This is a
distinct concern from advancing the plan's finances, so it lives in its own
collaborator: :class:`FailsafeController` reads and mutates the shared
``EngineState`` and leans on a few engine helpers (horizon, age, purchase
funding) via the engine reference it is constructed with.

``ProjectionEngine`` owns one controller for its lifetime and delegates the
handful of entry points (``_evaluate_failsafes``, ``_failsafe_metric``,
``_pv_future_medical``) to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from fintracker.models import Failsafe, FailsafeAction

if TYPE_CHECKING:                       # hints only — avoids an import cycle
    from fintracker.projections import EngineState, ProjectionEngine


# Failsafe metrics that are unitless ratios (not dollar amounts), so the
# present-value deflation applied to dollar metrics must be skipped for them.
_FS_RATIO_METRICS = frozenset({"medical_burden_ratio"})


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


class FailsafeController:
    """Evaluates and applies a plan's failsafes against a live ``EngineState``.

    Holds a back-reference to the driving :class:`ProjectionEngine` for the few
    cross-cutting helpers it needs (horizon, age-in-year, purchase funding) and
    memoises the deterministic present-value-of-future-medical forecast across
    simulation paths (it depends only on the year + deterministic state inputs).
    """

    def __init__(self, engine: "ProjectionEngine") -> None:
        self._engine = engine
        self._plan = engine._plan
        self._pv_medical_cache: dict = {}

    # ------------------------------------------------------------------ #
    # Trigger metrics                                                     #
    # ------------------------------------------------------------------ #

    def metric(self, state: "EngineState", metric: str, year: int) -> float:
        """Value of a failsafe trigger metric, from live start-of-year state.

        Balance metrics read what's carried into the year (end of the prior
        year), the natural quantity for a threshold like "brokerage below $100k".
        ``net_worth`` mirrors the snapshot's composition. ``medical_burden_ratio``
        is a forward-looking, unitless ratio (see ``pv_future_medical``) — set
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
            return self.pv_future_medical(state, year) / net_worth
        raise ValueError(f"Unknown failsafe metric: {metric!r}")

    def _annual_medical_forecast(self, year: int, hc_f: float, is_married: bool,
                                 num_children: int, working: bool) -> float:
        """Anticipated healthcare cost for one future year (baseline, no cut).

        OOP + health premium (while working, pre-Medicare) + self-LTC (final years
        of life) + base Medicare (65+; IRMAA excluded as it is MAGI/path-dependent),
        all in year-``year`` nominal dollars via the healthcare factor ``hc_f``.
        """
        lif = self._plan.lifestyle
        rp = self._plan.retirement
        age = self._engine._age_in_year(year)
        medicare_age = rp.medicare_start_age if rp else 65
        medical = lif.scaled_medical_oop(is_married, num_children) * hc_f
        health = (lif.annual_health_insurance_premium * hc_f
                  if working and (age is None or age < medicare_age) else 0.0)
        self_ltc = (lif.annual_self_ltc_cost * hc_f
                    if self._engine._self_ltc_active(year) else 0.0)
        medicare = 0.0
        if rp and age is not None and age >= rp.medicare_start_age:
            enrolled = 2 if is_married else 1
            medicare = rp.annual_medicare_premium * hc_f * enrolled
        return medical + health + self_ltc + medicare

    def pv_future_medical(self, state: "EngineState", year: int) -> float:
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
        horizon = self._engine._horizon()
        total = 0.0
        for t in range(year, horizon + 1):
            hc_f_t = state.cumulative_healthcare_inflation * (1 + hc_rate) ** (t - year)
            age_t = self._engine._age_in_year(t)
            working_t = state.is_working and (retire_age is None or age_t is None or age_t < retire_age)
            cost_t = self._annual_medical_forecast(t, hc_f_t, state.is_married,
                                                   state.num_children, working_t)
            total += cost_t / (1 + discount) ** (t - year)
        self._pv_medical_cache[key] = total
        return total

    # ------------------------------------------------------------------ #
    # Evaluation + application                                           #
    # ------------------------------------------------------------------ #

    def _triggered(self, state: "EngineState", year: int, fs: Failsafe) -> bool:
        results = []
        for c in fs.conditions:
            # end_year of None OR 0 (non-positive) means "to the horizon" — same
            # sentinel convention the UI uses (it sends 0 for "end"). Only a value
            # >= 1 bounds the window.
            end = c.end_year if (c.end_year and c.end_year >= 1) else self._engine._horizon()
            if not (c.start_year <= year <= end):
                results.append(False)
                continue
            value = self.metric(state, c.metric, year)
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

    def evaluate(self, state: "EngineState", year: int) -> None:
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
            if self._triggered(state, year, fs):
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
        self._apply_active(state, year)

    def _apply_active(self, state: "EngineState", year: int) -> None:
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
                        self._engine._fund_purchase(state, a.one_time_expense * infl, year)
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
