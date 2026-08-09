"""Social Security (primary AND partner) must flow through the year-by-year
projection and the Monte Carlo — not just the retirement-readiness gauge.

Guards the shared primitive ``_social_security_income`` and the household
roll-up ``ProjectionEngine._social_security_for_year`` that the gauge and the sim
both call, so the two can never drift apart.
"""
from fintracker.models import RetirementProfile
from fintracker.projections import ProjectionEngine, _social_security_income
from tests.builders import make_plan


def _retire_plan(*, ss=0.0, claim_age=None, partner_ss=0.0,
                 partner_claim_age=None, partner_age=None, years=10):
    """Baseline retiring at 65 in year 2 (primary age 64 now) so retirement years exist."""
    rp = RetirementProfile(
        current_age=64, retirement_age=65,
        estimated_social_security_annual=ss, social_security_claim_age=claim_age,
        partner_social_security_annual=partner_ss,
        partner_social_security_claim_age=partner_claim_age,
        partner_current_age=partner_age,
    )
    return make_plan(projection_years=years, retirement=rp)


def _final_nw(**kw):
    return ProjectionEngine(_retire_plan(**kw)).run_deterministic()[-1].net_worth


# ── Primary ────────────────────────────────────────────────────────────────
def test_social_security_raises_projected_net_worth():
    assert _final_nw(ss=30_000) > _final_nw(ss=0)


def test_social_security_raises_monte_carlo_median():
    """The exact thing the user hit: SS now changes the Monte Carlo outcome."""
    base = ProjectionEngine(_retire_plan(ss=0)).run_monte_carlo(n_simulations=200, seed=1)
    with_ss = ProjectionEngine(_retire_plan(ss=30_000)).run_monte_carlo(n_simulations=200, seed=1)
    assert with_ss.p50_net_worth[-1] > base.p50_net_worth[-1]


def test_later_claim_age_delays_benefit_and_lowers_net_worth():
    assert _final_nw(ss=30_000, claim_age=70) < _final_nw(ss=30_000, claim_age=65)


# ── Partner ──────────────────────────────────────────────────────────────────
def test_partner_social_security_raises_net_worth():
    assert _final_nw(partner_ss=20_000) > _final_nw(partner_ss=0)


def test_partner_and_primary_stack():
    both = _final_nw(ss=30_000, partner_ss=20_000)
    assert both > _final_nw(ss=30_000) and both > _final_nw(partner_ss=20_000)


def test_younger_partners_benefit_starts_later():
    # Partner age 55 now → reaches the default claim age (65) only in year 11,
    # past this 10-year horizon, so partner SS contributes nothing here.
    assert _final_nw(partner_ss=20_000, partner_age=55) == _final_nw(partner_ss=0)


# ── Shared primitives ────────────────────────────────────────────────────────
def test_shared_primitive_gate_inflation_tax():
    assert _social_security_income(40_000, 64, 65, 1.0, 0.0) == 0.0            # before claim
    assert _social_security_income(40_000, 65, 65, 1.0, 0.0) == 40_000        # at claim
    assert _social_security_income(40_000, 70, 65, 2.0, 0.0) == 80_000        # inflation factor
    assert _social_security_income(40_000, 65, 65, 1.0, 0.20) == 40_000 * (1 - 0.85 * 0.20)  # tax
    assert _social_security_income(0.0, 70, 65, 1.0, 0.0) == 0.0             # no benefit


def test_household_method_sums_primary_and_partner():
    eng = ProjectionEngine(_retire_plan(ss=40_000, partner_ss=20_000))
    # Year 2: both are 65 (≥ claim). No inflation, no tax → straight sum.
    assert eng._social_security_for_year(2, 1.0, 0.0) == 60_000


def test_household_method_zero_without_retirement_profile():
    assert ProjectionEngine(make_plan())._social_security_for_year(2, 1.0, 0.0) == 0.0


def test_ss_split_separates_primary_and_partner():
    eng = ProjectionEngine(_retire_plan(ss=40_000, partner_ss=20_000))
    # Year 2: both are 65 (≥ claim), no inflation/tax → the two benefits verbatim.
    assert eng._ss_split_for_year(2, 1.0, 0.0) == (40_000, 20_000)


# ── Auto-estimation at config load (dollar amounts are derived, not entered) ──
def test_auto_estimate_ss_fills_dollars_from_income_and_haircut():
    from fintracker.config import _auto_estimate_ss
    from fintracker.models import IncomeProfile, FilingStatus, State
    from fintracker.social_security import estimate_annual_benefit

    rp = RetirementProfile(current_age=40, work_start_age=22, retirement_age=67,
                           social_security_haircut=0.75)
    inc = IncomeProfile(150_000, FilingStatus.SINGLE, State.TEXAS,
                        spouse_gross_annual_income=80_000)
    _auto_estimate_ss(rp, inc)

    years, credits = 67 - 22, min(40, 4 * (67 - 22))
    assert rp.estimated_social_security_annual == estimate_annual_benefit(
        150_000, years_worked=years, claim_age=67, credits=credits, haircut=0.75)
    assert rp.partner_social_security_annual == estimate_annual_benefit(
        80_000, years_worked=years, claim_age=67, credits=credits, haircut=0.75)


def test_sample_config_autocalcs_social_security():
    """Loading the sample YAML (which sets no dollar amount) still yields a benefit."""
    import pathlib
    from fintracker.config import load_plan
    plan = load_plan(pathlib.Path(__file__).resolve().parent.parent / "config" / "sample.yaml")
    assert plan.retirement is not None
    assert plan.retirement.estimated_social_security_annual > 0
