"""
fintracker — Personal Long-Term Financial Planning Engine
=========================================================
Run with:  streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import pathlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from fintracker.models import (
    FilingStatus, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    BusinessProfile, CarProfile, ChildcarePhase, ChildcareProfile, EmployerMatch, MatchTier,
    RothContributionPhase, InvestmentProfile, StrategyToggles, FinancialPlan, TimelineEvent,
    Failsafe, FailsafeCondition, FailsafeAction,
)
from fintracker.tax_engine import TaxEngine, state_display_name
from fintracker.mortgage import MortgageCalculator
from fintracker.strategies import StrategyEngine
from fintracker.projections import ProjectionEngine, PENALTY_FREE_AGE
from fintracker.config import load_plan_or_sample, save_plan

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="fintracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Minimal CSS — refined dark-accented palette
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    color: #e0e0e0;
}
.metric-card .label { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { font-family: 'DM Mono', monospace; font-size: 1.75rem; font-weight: 500; color: #e8f4f8; margin-top: 0.25rem; }
.metric-card .delta-pos { font-size: 0.8rem; color: #4ade80; margin-top: 0.15rem; }
.metric-card .delta-neg { font-size: 0.8rem; color: #f87171; margin-top: 0.15rem; }

.strategy-card {
    background: #0d1117;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.875rem;
    color: #c9d1d9;
}
.tip-card {
    background: #0d1117;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.875rem;
    color: #c9d1d9;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e8f4f8;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(family="Inter", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=0, r=0, t=30, b=0),
)

COLORS = {
    "retirement": "#3b82f6",
    "brokerage": "#8b5cf6",
    "home_equity": "#10b981",
    "hsa": "#f59e0b",
    "taxes": "#ef4444",
    "housing": "#f97316",
    "lifestyle": "#06b6d4",
    "breathing": "#4ade80",
    "p10": "#374151",
    "p25": "#4b5563",
    "p50": "#3b82f6",
    "p75": "#4b5563",
    "p90": "#374151",
}


def fmt_dollar(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v:,.0f}"
    return f"${v:.0f}"


def hex_to_rgba(hex_color: str, alpha: float = 0.7) -> str:
    """Convert a #rrggbb hex string to a valid rgba(...) string for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_band(fig, x, upper, lower, fillcolor: str, name: str) -> None:
    """Add a shaded percentile band (upper→lower→back) to a Plotly figure."""
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor=fillcolor,
        line=dict(color="rgba(0,0,0,0)"),
        name=name, hoverinfo="skip",
    ))


def _add_line(fig, x, y, name: str, color: str, width: float = 2, dash: str | None = None) -> None:
    """Add a simple named line trace to a Plotly figure."""
    fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=color, width=width, dash=dash)))


def metric_card(label: str, value: str, delta: str = "", positive: bool = True) -> str:
    delta_cls = "delta-pos" if positive else "delta-neg"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {delta_html}
</div>"""


# ─────────────────────────────────────────────────────────────
# Sidebar — build the FinancialPlan
# ─────────────────────────────────────────────────────────────

def _wd(obj, attr, fallback, cast=None):
    """Widget default: obj.attr (optionally cast) when a loaded plan exists, else fallback.

    Collapses the ubiquitous ``value=cast(obj.attr) if obj else fallback`` idiom so
    each widget carries one fewer conditional.
    """
    if obj is None:
        return fallback
    val = getattr(obj, attr)
    return cast(val) if cast else val


def _seed(key: str, default):
    """Seed a widget's session_state value *once*, then return the key so the
    widget can own its state via ``key=`` (stable identity) instead of a
    ``value=``/``index=`` default.

    The sidebar's ``value=`` defaults are read from ``loaded_plan``, which is only
    written at the *end* of build_sidebar — so on the rerun that commits an edit,
    the default still holds the previous value and Streamlit reverts the widget to
    it on the first change (it sticks on the second). Keying the widget and seeding
    its state once removes that one-rerun lag. Config loads clear the ``w_*`` keys
    (see _load_config_expander) so a freshly loaded plan re-seeds these widgets.
    """
    st.session_state.setdefault(key, default)
    return key


def _childcare_phase_inputs(d_cp) -> ChildcareProfile:
    """Render age-based childcare cost inputs and return the profile."""
    st.sidebar.caption(
        "Define monthly costs per child at each age. "
        "Ages not covered default to $0. Costs inflate annually."
    )
    specs = [(0, 2, 2_500, "Infant/Toddler (0–2)"), (3, 4, 1_500, "Preschool (3–4)"),
             (5, 12, 600, "School-age (5–12)"), (13, 17, 150, "Teen (13–17)")]
    phases = []
    for i, (a_start, a_end, fallback, label) in enumerate(specs):
        default_cost = int(d_cp.phases[i].monthly_cost) if d_cp and len(d_cp.phases) > i else fallback
        cost = st.sidebar.number_input(
            f"{label} – monthly cost/child ($)", min_value=0, max_value=15_000,
            value=default_cost, step=50, key=f"cc_{a_start}_{a_end}",
        )
        phases.append(ChildcarePhase(age_start=a_start, age_end=a_end, monthly_cost=float(cost)))
    return ChildcareProfile(phases=phases)


def _income_section(defaults) -> IncomeProfile:
    d_inc = defaults.income if defaults else None
    st.sidebar.header("💵 Income")
    gross = st.sidebar.number_input(
        "Gross Annual Income  (own rate)",
        min_value=0, max_value=5_000_000, step=100,
        key=_seed("w_gross", _wd(d_inc, "gross_annual_income", 120_000, int)),
    )
    spouse = st.sidebar.number_input(
        "Spouse Gross Income  (own rate)",
        min_value=0, max_value=5_000_000, step=5_000,
        key=_seed("w_spouse", _wd(d_inc, "spouse_gross_annual_income", 0, int)),
    )
    filing_opts = [f.value for f in FilingStatus]
    filing = st.sidebar.selectbox(
        "Filing Status", options=filing_opts,
        format_func=lambda x: x.replace("_", " ").title(),
        key=_seed("w_filing", d_inc.filing_status.value if d_inc else filing_opts[0]),
    )
    state_options = [s.value for s in State]
    state_val = st.sidebar.selectbox(
        "State", options=state_options,
        key=_seed("w_state", d_inc.state.value if d_inc else "GA"),
    )
    other_rate = 0.05
    if state_val == "OTHER":
        other_rate = st.sidebar.slider("State Flat Tax Rate (%)", 0.0, 15.0, _wd(d_inc, "other_state_flat_rate", 5.0, lambda v: v * 100), 0.1) / 100

    income = IncomeProfile(
        gross_annual_income=float(gross),
        spouse_gross_annual_income=float(spouse),
        filing_status=FilingStatus(filing),
        state=State(state_val),
        other_state_flat_rate=other_rate,
    )
    return income


def _housing_section(defaults) -> HousingProfile:
    d_hou = defaults.housing if defaults else None
    st.sidebar.header("🏠 Housing")
    is_renting = st.sidebar.toggle("I'm Renting", value=_wd(d_hou, "is_renting", False))

    if is_renting:
        monthly_rent = st.sidebar.number_input(
            "Monthly Rent  (own rate)", min_value=0, max_value=20_000,
            value=_wd(d_hou, "monthly_rent", 2_000, int), step=100,
        )
        _hou_base = d_hou if d_hou else HousingProfile(home_price=0, down_payment=0, interest_rate=0.0)
        housing = dataclasses.replace(
            _hou_base,
            home_price=0.0, down_payment=0.0, interest_rate=0.0,
            is_renting=True, monthly_rent=float(monthly_rent),
        )
    else:
        home_price = st.sidebar.number_input(
            "Home Price  (current)", min_value=0, max_value=10_000_000,
            value=_wd(d_hou, "home_price", 400_000, int), step=10_000,
        )
        down_pmt = st.sidebar.number_input(
            "Down Payment  (current)", min_value=0, max_value=home_price,
            value=min(_wd(d_hou, "down_payment", 80_000, int), home_price), step=5_000,
        )
        rate = st.sidebar.slider(
            "Mortgage Rate (%)", 2.0, 12.0,
            float(d_hou.interest_rate * 100) if d_hou else 6.5, 0.125,
        )
        _hou_base = d_hou if d_hou else HousingProfile(home_price=0, down_payment=0, interest_rate=0.0)
        housing = dataclasses.replace(
            _hou_base,
            home_price=float(home_price),
            down_payment=float(down_pmt),
            interest_rate=rate / 100,
            is_renting=False,
            annual_property_tax_rate=_wd(d_hou, "annual_property_tax_rate", 0.012, float),
            annual_insurance=_wd(d_hou, "annual_insurance", 2_000, float),
        )
    return housing


def _lifestyle_section(defaults) -> LifestyleProfile:
    d_lif = defaults.lifestyle if defaults else None
    st.sidebar.header("🌿 Lifestyle")
    num_children = st.sidebar.number_input(
        "Current Children", min_value=0, max_value=10,
        value=_wd(d_lif, "num_children", 0, int),
    )
    d_cp = _wd(d_lif, "childcare_profile", None)
    childcare_mode = st.sidebar.radio(
        "Childcare cost model",
        ["Flat monthly rate", "Age-based schedule"],
        index=1 if d_cp else 0,
        horizontal=True,
        help="Age-based schedule reflects how costs change from infant daycare through school-age activities.",
    )
    childcare_profile = None
    if childcare_mode == "Flat monthly rate":
        monthly_childcare = st.sidebar.number_input(
            "Monthly Childcare per Child  (inflated yearly)",
            min_value=0, max_value=10_000,
            value=_wd(d_lif, "monthly_childcare", 0, int),
            step=100,
            help="Single rate applied to every child, every year. Inflated annually.",
        )
    else:
        monthly_childcare = 0  # unused when profile is set
        childcare_profile = _childcare_phase_inputs(d_cp)
    num_pets = st.sidebar.number_input(
        "Pets", min_value=0, max_value=10,
        value=_wd(d_lif, "num_pets", 0, int),
    )
    annual_pet = 0.0
    if num_pets > 0:
        annual_pet = st.sidebar.number_input(
            "Annual Pet Cost  (inflated yearly)", min_value=0, max_value=20_000,
            value=_wd(d_lif, "annual_pet_cost", 1_800, int), step=100,
        )
    medical = st.sidebar.number_input(
        "Annual Medical OOP  (inflated yearly)", min_value=0, max_value=50_000,
        value=_wd(d_lif, "annual_medical_oop", 3_000, int), step=500,
    )
    with st.sidebar.expander("🛡️ Insurance premiums & long-term care"):
        health_prem = st.number_input(
            "Health insurance premium — your share (annual, inflated)",
            min_value=0, max_value=60_000,
            value=_wd(d_lif, "annual_health_insurance_premium", 0, int), step=250,
            help="Employee/marketplace premium you actually pay. Applied while "
                 "working and before Medicare age; Medicare takes over after that.",
        )
        disability_prem = st.number_input(
            "Disability insurance premium (annual, inflated)",
            min_value=0, max_value=20_000,
            value=_wd(d_lif, "annual_disability_insurance_premium", 0, int), step=100,
            help="Only charged while working (it replaces earned income).",
        )
        life_prem = st.number_input(
            "Life insurance premium (annual, inflated)",
            min_value=0, max_value=20_000,
            value=_wd(d_lif, "annual_life_insurance_premium", 0, int), step=100,
            help="Charged every year. Set to 0 once a term policy lapses.",
        )
        life_benefit = st.number_input(
            "Life insurance death benefit (coverage)",
            min_value=0, max_value=10_000_000,
            value=_wd(d_lif, "annual_life_insurance_death_benefit", 0, int), step=50_000,
            help="Coverage amount paid into the estate/net worth in the year of death "
                 "(set a Life expectancy age below). Fixed nominal — not inflated.",
        )
        self_ltc = st.number_input(
            "Your own long-term care cost (annual, inflated)",
            min_value=0, max_value=500_000,
            value=_wd(d_lif, "annual_self_ltc_cost", 0, int), step=1_000,
            help="Your end-of-life / LTC costs (parents are handled separately). "
                 "Needs a retirement profile for age.",
        )
        self_ltc_start_age = st.number_input(
            "Long-term care starts at age",
            min_value=50, max_value=110,
            value=_wd(d_lif, "self_ltc_start_age", 80, int), step=1,
        )
    vacation = st.sidebar.number_input(
        "Annual Vacation  (inflated yearly)", min_value=0, max_value=100_000,
        value=_wd(d_lif, "annual_vacation", 5_000, int), step=1_000,
    )
    other_monthly = st.sidebar.number_input(
        "Other Monthly  (inflated yearly)", min_value=0, max_value=10_000,
        value=_wd(d_lif, "monthly_other_recurring", 500, int), step=100,
    )
    _lif_base = d_lif if d_lif else LifestyleProfile()
    lifestyle = dataclasses.replace(
        _lif_base,
        num_children=int(num_children),
        monthly_childcare=float(monthly_childcare),
        childcare_profile=childcare_profile,
        num_pets=int(num_pets),
        annual_pet_cost=float(annual_pet),
        annual_medical_oop=float(medical),
        annual_health_insurance_premium=float(health_prem),
        annual_disability_insurance_premium=float(disability_prem),
        annual_life_insurance_premium=float(life_prem),
        annual_life_insurance_death_benefit=float(life_benefit),
        annual_self_ltc_cost=float(self_ltc),
        self_ltc_start_age=int(self_ltc_start_age),
        annual_vacation=float(vacation),
        monthly_other_recurring=float(other_monthly),
    )
    return lifestyle


def _pct_default(loaded_amount, salary) -> float:
    """Default position for a '% of salary' slider from a loaded $ amount, else 6.0."""
    if loaded_amount is None or salary <= 0:
        return 6.0
    return round(loaded_amount / salary * 100, 1)


def _k401_contribution_input(label, salary, default_amount, default_pct, key, limit=30_500) -> float:
    """Render a `$ amount` / `% of salary` 401k input and return the annual dollars."""
    mode = st.sidebar.radio(
        f"{label} 401k input mode", ["$ amount", "% of salary"],
        horizontal=True, label_visibility="collapsed", key=f"{key}_mode",
    )
    if mode == "% of salary":
        pct = st.sidebar.slider(
            f"{label} 401k (% of gross salary)", 0.0, 30.0, default_pct, 0.5,
            help=f"Will be capped at the IRS limit (${limit:,}) automatically.", key=f"{key}_pct",
        )
        amt = min(float(salary) * pct / 100, limit)
        st.sidebar.caption(f"= ${amt:,.0f}/yr")
        return amt
    return float(st.sidebar.number_input(
        f"{label} Annual 401k Contribution  (fixed)",
        min_value=0, max_value=limit, value=int(default_amount), step=500, key=f"{key}_amt",
    ))


def _employer_match_inputs(d_em) -> 'EmployerMatch | None':
    """Render the employer 401k match formula; return an EmployerMatch or None."""
    st.sidebar.subheader("🏦 Employer 401k Match")
    if not st.sidebar.toggle("Employer offers 401k match", value=d_em is not None):
        return None
    with st.sidebar.expander("Match formula", expanded=True):
        st.caption(
            "Build your match formula tier by tier. "
            "Example: 100% on first 3% + 50% on next 2% = two tiers."
        )
        n_tiers = st.number_input(
            "Number of tiers", min_value=0, max_value=5,
            value=len(d_em.tiers) if d_em else 1, step=1,
            help="Most plans have 1–2 tiers. 0 = profit sharing only.",
        )
        tiers = []
        existing_tiers = d_em.tiers if d_em else []
        for ti in range(int(n_tiers)):
            prev = existing_tiers[ti] if ti < len(existing_tiers) else None
            tc1, tc2 = st.columns(2)
            mp = tc1.number_input(
                f"Tier {ti+1}: match %", 0, 200,
                int((prev.match_pct if prev else (1.0 if ti == 0 else 0.5)) * 100),
                key=f"em_mp_{ti}",
                help="Employer matches this % of your contribution in this tier.",
            ) / 100
            up = tc2.number_input(
                f"Tier {ti+1}: up to % salary", 1, 25,
                int((prev.up_to_pct_of_salary if prev else 0.06) * 100),
                key=f"em_up_{ti}",
                help="Employee contribution eligible for this tier (% of gross salary).",
            ) / 100
            tiers.append(MatchTier(match_pct=mp, up_to_pct_of_salary=up))
        em_cap = st.number_input(
            "Annual match cap  (fixed)", min_value=0, max_value=100_000,
            value=int(d_em.annual_cap) if (d_em and d_em.annual_cap) else 0, step=500,
            help="Absolute dollar ceiling on total employer match per year.",
        )
        em_vest = st.number_input(
            "Vesting (years, 0 = immediate)", min_value=0, max_value=10,
            value=_wd(d_em, "vesting_years", 0, int),
            help="Cliff vesting: match forfeited if you leave before this year.",
        )
        em_ps = st.number_input(
            "Profit sharing per year  (fixed)", min_value=0, max_value=100_000,
            value=_wd(d_em, "profit_sharing_annual", 0, int), step=500,
            help="Flat employer contribution regardless of your own contribution.",
        )
    return EmployerMatch(
        tiers=tiers,
        annual_cap=float(em_cap) if em_cap > 0 else None,
        vesting_years=int(em_vest),
        profit_sharing_annual=float(em_ps),
    )


def _investments_section(defaults) -> InvestmentProfile:
    d_inv = _wd(defaults, "investments", None)
    d_inc = _wd(defaults, "income", None)
    gross = _wd(d_inc, "gross_annual_income", 120_000, int)
    spouse = _wd(d_inc, "spouse_gross_annual_income", 0, int)
    roth_annual = _wd(d_inv, "annual_roth_ira_contribution", 7_000.0, float)
    roth_schedule = _wd(d_inv, "roth_contribution_schedule", None)
    st.sidebar.header("📊 Investments & Savings")
    liquid_cash = st.sidebar.number_input(
        "Current Liquid Cash  (current)", min_value=0, max_value=10_000_000,
        value=_wd(d_inv, "current_liquid_cash", 100_000, int), step=5_000,
    )
    retirement_bal = st.sidebar.number_input(
        "Current Retirement Balance  (current)", min_value=0, max_value=10_000_000,
        value=_wd(d_inv, "current_retirement_balance", 0, int), step=5_000,
    )
    brokerage_bal = st.sidebar.number_input(
        "Current Brokerage / Taxable Balance  (current)", min_value=0, max_value=10_000_000,
        value=_wd(d_inv, "current_brokerage_balance", 0, int), step=5_000,
        help="Existing taxable investment accounts (separate from 401k/IRA/HSA).",
    )
    roth_bal_current = st.sidebar.number_input(
        "Current Roth IRA Balance  (current)", min_value=0, max_value=10_000_000,
        value=_wd(d_inv, "current_roth_ira_balance", 0, int), step=5_000,
        help="Existing Roth IRA balance. The full amount is treated as contribution basis (withdrawable tax-free before brokerage when the Backdoor Roth strategy is on).",
    )
    one_time = st.sidebar.number_input(
        "Upcoming One-Time Expenses  (current)", min_value=0, max_value=1_000_000,
        value=_wd(d_inv, "one_time_upcoming_expenses", 0, int), step=5_000,
        help="Wedding, car purchase, etc. Subtracted from investable cash.",
    )
    k401 = _k401_contribution_input(
        "Your", gross,
        _wd(d_inv, "annual_401k_contribution", 23_000, int),
        _pct_default(_wd(d_inv, "annual_401k_contribution", None), gross),
        key="k401",
    )
    partner_k401 = 0.0
    if spouse > 0:
        partner_k401 = _k401_contribution_input(
            "Partner", spouse,
            _wd(d_inv, "partner_annual_401k_contribution", 0, int),
            _pct_default(_wd(d_inv, "partner_annual_401k_contribution", None), spouse),
            key="pk401",
        )
    hsa = st.sidebar.number_input(
        "Annual HSA Contribution  (fixed)", min_value=0, max_value=8_300,
        value=_wd(d_inv, "annual_hsa_contribution", 4_150, int), step=100,
    )
    c529 = st.sidebar.number_input(
        "Annual 529 Contribution per Child  (fixed)", min_value=0, max_value=50_000,
        value=_wd(d_inv, "annual_529_contribution", 0, int), step=500,
    )

    employer_match = _employer_match_inputs(_wd(d_inv, "employer_match", None))

    investments = InvestmentProfile(
        current_liquid_cash=float(liquid_cash),
        current_retirement_balance=float(retirement_bal),
        current_brokerage_balance=float(brokerage_bal),
        one_time_upcoming_expenses=float(one_time),
        annual_401k_contribution=float(k401),
        partner_annual_401k_contribution=float(partner_k401),
        annual_hsa_contribution=float(hsa),
        annual_529_contribution=float(c529),
        annual_market_return=float(st.sidebar.slider("Market Return (%)", 0.0, 15.0, _wd(d_inv, "annual_market_return", 8.0, lambda v: v * 100), 0.5)) / 100,
        annual_inflation_rate=float(st.sidebar.slider("Inflation (%)", 0.0, 10.0, _wd(d_inv, "annual_inflation_rate", 3.0, lambda v: v * 100), 0.25)) / 100,
        annual_healthcare_inflation_rate=float(st.sidebar.slider(
            "Healthcare Inflation (%)", 0.0, 12.0,
            _wd(d_inv, "annual_healthcare_inflation_rate", 5.0, lambda v: v * 100), 0.25,
            help="Applies to medical OOP, the health-insurance premium, your own long-term care, "
                 "and Medicare premiums + IRMAA. Historically ~5% vs ~3% general.")) / 100,
        annual_salary_growth_rate=float(st.sidebar.slider("Your Salary Growth (%)", 0.0, 15.0,
            _wd(d_inv, "annual_salary_growth_rate", 4.0, lambda v: v * 100), 0.5)) / 100,
        partner_salary_growth_rate=float(st.sidebar.slider("Partner Salary Growth (%)", 0.0, 15.0,
            _wd(d_inv, "partner_salary_growth_rate", 4.0, lambda v: v * 100), 0.5)) / 100
            if spouse > 0 else 0.04,
        annual_home_appreciation_rate=float(st.sidebar.slider("Home Appreciation (%)", 0.0, 10.0, _wd(d_inv, "annual_home_appreciation_rate", 3.5, lambda v: v * 100), 0.5)) / 100,
        auto_invest_surplus=st.sidebar.toggle(
            "Auto-Invest Surplus",
            value=_wd(d_inv, "auto_invest_surplus", True),
            help="ON: surplus swept into brokerage each year (earns market return). "
                 "OFF: surplus stays in cash (0% return).",
        ),
        cash_buffer_months=st.sidebar.slider(
            "Cash Buffer (months of expenses)",
            min_value=0.0, max_value=24.0,
            value=_wd(d_inv, "cash_buffer_months", 0.0, float),
            step=1.0,
            help="Keep this many months of living expenses as liquid cash (0% return) "
                 "before sweeping surplus to brokerage. Reduces liquidity risk in bad years. "
                 "Set to 0 to invest all surplus (default).",
        ),
        employer_match=employer_match,
        current_roth_ira_balance=float(roth_bal_current),
    )
    # Preserve fields not exposed in sidebar (partner_salary_growth_rate when solo,
    # annual_roth_ira_contribution, annual_brokerage_contribution)
    _inv_base = d_inv if d_inv else InvestmentProfile()
    investments = dataclasses.replace(
        investments,
        annual_roth_ira_contribution=roth_annual,
        roth_contribution_schedule=roth_schedule,
        annual_brokerage_contribution=_inv_base.annual_brokerage_contribution,
        # partner_salary_growth_rate: sidebar only shows it when spouse > 0;
        # preserve the loaded value when spouse income is 0 at sidebar time
        partner_salary_growth_rate=(
            investments.partner_salary_growth_rate
            if spouse > 0 else _inv_base.partner_salary_growth_rate
        ),
    )
    return investments


def _backdoor_roth_inputs(d_inc, roth_annual, roth_schedule, investments) -> InvestmentProfile:
    """Render backdoor-Roth contribution controls; return investments with roth fields set."""
    _roth_limit = 14_000 if (d_inc and d_inc.filing_status == FilingStatus.MARRIED_FILING_JOINTLY) else 7_000
    _roth_mode = st.sidebar.radio(
        "Contribution mode",
        ["Flat (same every year)", "Phase schedule (varies by year)"],
        index=1 if roth_schedule else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    if _roth_mode == "Flat (same every year)":
        roth_annual = st.sidebar.number_input(
            "Annual Roth IRA Contribution  (fixed)",
            min_value=0, max_value=_roth_limit,
            value=int(roth_annual) if roth_annual else _roth_limit,
            step=500,
            help=f"IRS limit: ${_roth_limit:,}/yr. Post-tax — no deduction. "
                 "Locks for 5 years then penalty-free to withdraw.",
        )
        return dataclasses.replace(investments,
            annual_roth_ira_contribution=float(roth_annual),
            roth_contribution_schedule=None)
    st.sidebar.caption(
        f"Define phases by projection year. Years not in any phase contribute $0. "
        f"IRS max: ${_roth_limit:,}/yr."
    )
    _existing = roth_schedule or []
    _n_phases = st.sidebar.number_input(
        "Number of phases", min_value=1, max_value=10,
        value=max(1, len(_existing)), step=1,
    )
    _new_phases = []
    for _pi in range(int(_n_phases)):
        _prev = _existing[_pi] if _pi < len(_existing) else None
        _pc1, _pc2, _pc3 = st.sidebar.columns(3)
        _ys = _pc1.number_input("Start yr", 1, 40,
            _prev.year_start if _prev else 1, key=f"rp_ys_{_pi}")
        _ye = _pc2.number_input("End yr", 1, 40,
            _prev.year_end if _prev else 5, key=f"rp_ye_{_pi}")
        _am = _pc3.number_input("$/yr", 0, _roth_limit,
            int(_prev.annual_amount if _prev else _roth_limit),
            step=500, key=f"rp_am_{_pi}")
        _new_phases.append(RothContributionPhase(
            year_start=int(_ys), year_end=int(_ye), annual_amount=float(_am)))
    return dataclasses.replace(investments,
        annual_roth_ira_contribution=0.0,
        roth_contribution_schedule=_new_phases)


def _strategies_section(defaults):
    d_str = defaults.strategies if defaults else None
    d_inc = defaults.income if defaults else None
    investments = defaults.investments if defaults else InvestmentProfile()
    roth_annual = float(defaults.investments.annual_roth_ira_contribution) if defaults else 7_000.0
    roth_schedule = defaults.investments.roth_contribution_schedule if defaults else None
    st.sidebar.header("🎯 Tax Strategies")
    _str_base = d_str if d_str else StrategyToggles()
    strategies = dataclasses.replace(
        _str_base,
        maximize_hsa=st.sidebar.toggle("Maximize HSA", value=_wd(d_str, "maximize_hsa", True)),
        maximize_401k=st.sidebar.toggle("Maximize 401k", value=_wd(d_str, "maximize_401k", True)),
        use_529_state_deduction=st.sidebar.toggle("Use 529 State Deduction", value=_wd(d_str, "use_529_state_deduction", False)),
        use_roth_ladder=st.sidebar.toggle("Roth Conversion Ladder", value=_wd(d_str, "use_roth_ladder", False)),
        use_backdoor_roth=st.sidebar.toggle(
            "Backdoor Roth IRA",
            value=_wd(d_str, "use_backdoor_roth", False),
            help="Contributes post-tax dollars to a Roth IRA each year (backdoor method — "
                 "no income limit). Contributions vest after 5 years and can then be "
                 "withdrawn tax-free before touching your brokerage account in a deficit.",
        ),
    )
    # Contribution schedule — shown only when toggle is ON
    if strategies.use_backdoor_roth:
        investments = _backdoor_roth_inputs(d_inc, roth_annual, roth_schedule, investments)

    strategies = dataclasses.replace(
        strategies,
        retirement_withdrawal_order=_withdrawal_order_input(d_str),
    )
    return strategies, investments


# Preset withdrawal orders (cash first, Roth basis last). None ⇒ engine derives
# from starting balances. Keys must match projections.WITHDRAWAL_SOURCES.
_WITHDRAWAL_ORDER_PRESETS = {
    "Auto (from balances)": None,
    "401k before brokerage (bracket-fill)":
        ["cash_buffer", "uninvested_cash", "retirement_401k", "brokerage", "roth_basis"],
    "Brokerage before 401k (conventional)":
        ["cash_buffer", "uninvested_cash", "brokerage", "retirement_401k", "roth_basis"],
}


def _withdrawal_order_input(d_str):
    """Retirement withdrawal-order preset selector; returns an order list or None."""
    labels = list(_WITHDRAWAL_ORDER_PRESETS)
    current = _wd(d_str, "retirement_withdrawal_order", None)
    try:
        idx = list(_WITHDRAWAL_ORDER_PRESETS.values()).index(current)
    except ValueError:
        idx = 0  # a custom/unrecognised order falls back to showing "Auto"
    with st.sidebar.expander("🏦 Retirement withdrawal order"):
        st.caption(
            "Which accounts fund spending once you're retired. 401k/IRA draws are "
            "taxed as ordinary income and count toward Medicare IRMAA. Cash is "
            "always first, Roth basis last."
        )
        choice = st.radio(
            "Order", labels, index=idx, label_visibility="collapsed",
        )
    return _WITHDRAWAL_ORDER_PRESETS[choice]


def _car_section(defaults) -> 'CarProfile | None':
    st.sidebar.header("🚗 Car")
    d_car = defaults.car if defaults else None
    has_car = st.sidebar.toggle("Model car purchases", value=d_car is not None)
    car = None
    if has_car:
        car = CarProfile(
            car_price=st.sidebar.number_input(
                "Car price  (inflated yearly)", min_value=0, max_value=200_000,
                value=_wd(d_car, "car_price", 25_000, int), step=1_000,
            ),
            down_payment=st.sidebar.number_input(
                "Down payment  (current)", min_value=0, max_value=100_000,
                value=_wd(d_car, "down_payment", 5_000, int), step=500,
            ),
            loan_rate=st.sidebar.slider(
                "Loan rate (%)", 0.0, 20.0,
                float(d_car.loan_rate * 100) if d_car else 6.5, 0.25,
            ) / 100,
            loan_term_years=st.sidebar.selectbox(
                "Loan term (years)", [3, 4, 5, 6, 7],
                index=[3,4,5,6,7].index(d_car.loan_term_years) if d_car else 2,
            ),
            replace_every_years=st.sidebar.selectbox(
                "Replace every (years)", [5, 7, 8, 10, 12, 15],
                index=[5,7,8,10,12,15].index(d_car.replace_every_years) if d_car else 3,
            ),
            residual_value=st.sidebar.number_input(
                "Sell old car for  (inflated yearly)", min_value=0, max_value=30_000,
                value=_wd(d_car, "residual_value", 5_000, int), step=500,
                help="Amount received when selling the old car if no child is old enough to receive it.",
            ),
            hand_down_age=st.sidebar.number_input(
                "Hand-down age (child)", min_value=14, max_value=25,
                value=_wd(d_car, "hand_down_age", 16, int), step=1,
                help="Minimum child age to receive the handed-down car instead of selling it.",
            ),
            num_cars=st.sidebar.selectbox(
                "Number of cars", [1, 2, 3],
                index=(d_car.num_cars - 1) if d_car else 0,
            ),
            annual_insurance_per_car=st.sidebar.number_input(
                "Insurance per car / yr  (inflated yearly)", min_value=0, max_value=20_000,
                value=_wd(d_car, "annual_insurance_per_car", 1_500, int), step=100,
                help="Recurring cost of owning each car, on top of the loan payment.",
            ),
            annual_maintenance_per_car=st.sidebar.number_input(
                "Maintenance per car / yr  (inflated yearly)", min_value=0, max_value=20_000,
                value=_wd(d_car, "annual_maintenance_per_car", 1_000, int), step=100,
            ),
            annual_fuel_per_car=st.sidebar.number_input(
                "Fuel per car / yr  (inflated yearly)", min_value=0, max_value=20_000,
                value=_wd(d_car, "annual_fuel_per_car", 2_000, int), step=100,
            ),
            annual_registration_per_car=st.sidebar.number_input(
                "Registration per car / yr  (inflated yearly)", min_value=0, max_value=5_000,
                value=_wd(d_car, "annual_registration_per_car", 200, int), step=50,
            ),
        )
    return car


def _business_section(defaults) -> 'BusinessProfile | None':
    st.sidebar.header("🏢 Business")
    d_biz = defaults.business if defaults else None
    has_business = st.sidebar.toggle("Model business ownership", value=d_biz is not None)
    business = None
    if has_business:
        with st.sidebar.expander("Business parameters", expanded=True):
            biz_revenue = st.number_input(
                "Annual gross revenue  (own rate)", min_value=0, max_value=10_000_000,
                value=_wd(d_biz, "annual_revenue", 200_000, int), step=5_000,
            )
            biz_expense_ratio = st.slider(
                "Operating expense ratio (%)", 0.0, 95.0,
                float(d_biz.expense_ratio * 100) if d_biz else 60.0, 1.0,
                help="Operating costs as % of revenue. Net profit = revenue × (1 − ratio).",
            ) / 100
            biz_growth = st.slider(
                "Revenue growth rate (%/yr)", 0.0, 30.0,
                float(d_biz.revenue_growth_rate * 100) if d_biz else 5.0, 0.5,
            ) / 100
            biz_start = st.number_input(
                "Start year", min_value=1, max_value=50,
                value=_wd(d_biz, "start_year", 1, int),
                help="Projection year the business starts generating income.",
            )
            biz_invest = st.number_input(
                "Initial investment  (current)", min_value=0, max_value=5_000_000,
                value=_wd(d_biz, "initial_investment", 0, int), step=5_000,
                help="One-time acquisition/startup cost drawn from brokerage in start year.",
            )
            biz_equity_mult = st.slider(
                "Equity multiple", 0.0, 10.0,
                _wd(d_biz, "equity_multiple", 3.0, float), 0.5,
                help="Business value = net profit × this. Set 0 to exclude from net worth.",
            )
            biz_sale_yr = st.number_input(
                "Sale year (0 = never sell)", min_value=0, max_value=50,
                value=int(d_biz.sale_year) if (d_biz and d_biz.sale_year) else 0,
                help="Sell business in this year; equity proceeds go to brokerage.",
            )
            st.markdown("**Tax & Retirement**")
            biz_qbi = st.toggle(
                "QBI deduction (20% pass-through)",
                value=_wd(d_biz, "use_qbi_deduction", True),
                help="20% deduction on qualified business income for pass-through entities.",
            )
            biz_health = st.number_input(
                "Self-employed health insurance  (fixed)", min_value=0, max_value=50_000,
                value=_wd(d_biz, "self_employed_health_insurance", 0, int), step=500,
                help="Annual premium — 100% deductible from AGI for self-employed.",
            )
            biz_solo_k = st.number_input(
                "Solo 401k contribution  (fixed)", min_value=0, max_value=69_000,
                value=_wd(d_biz, "solo_401k_contribution", 0, int), step=500,
                help="Owner solo 401k (up to $69k IRS limit). Tracked in retirement balance.",
            )
            biz_sep = st.number_input(
                "SEP-IRA contribution  (fixed)", min_value=0, max_value=69_000,
                value=_wd(d_biz, "sep_ira_contribution", 0, int), step=500,
                help="SEP-IRA (up to 25% of net self-employment income).",
            )
            biz_ownership = st.slider(
                "Your ownership share (%)", 1.0, 100.0,
                float((_wd(d_biz, "ownership_pct", 1.0)) * 100), 1.0,
                help="Your % stake in the business. 100% = sole owner. "
                     "50% = equal partnership. Profit, equity, and taxes all scale by this.",
            ) / 100
        _biz_base = d_biz if d_biz else BusinessProfile()
        business = dataclasses.replace(
            _biz_base,
            annual_revenue=float(biz_revenue),
            expense_ratio=float(biz_expense_ratio),
            revenue_growth_rate=float(biz_growth),
            start_year=int(biz_start),
            initial_investment=float(biz_invest),
            equity_multiple=float(biz_equity_mult),
            sale_year=int(biz_sale_yr) if biz_sale_yr > 0 else None,
            use_qbi_deduction=bool(biz_qbi),
            self_employed_health_insurance=float(biz_health),
            solo_401k_contribution=float(biz_solo_k),
            sep_ira_contribution=float(biz_sep),
            ownership_pct=float(biz_ownership),
        )
    return business


def _event_home_inputs(i, ev_def) -> dict:
    """Home-purchase sub-fields for one timeline event; returns replace() overrides."""
    price = st.number_input(
        "New home price ($)", min_value=0, max_value=10_000_000,
        value=int(ev_def.new_home_price) if (ev_def and ev_def.new_home_price) else 500_000,
        key=f"ev_hp_{i}",
    )
    down = st.number_input(
        "Down payment  (current)", min_value=0, max_value=int(price),
        value=int(ev_def.new_home_down_payment) if (ev_def and ev_def.new_home_down_payment) else int(price * 0.20),
        key=f"ev_hd_{i}",
    )
    rate = st.slider(
        "Mortgage rate (%)", 2.0, 12.0,
        float(ev_def.new_home_interest_rate * 100) if (ev_def and ev_def.new_home_interest_rate) else 6.5,
        0.125, key=f"ev_hr_{i}",
    )
    sell_current = st.checkbox(
        "Sell current home (add equity to cash)",
        value=_wd(ev_def, "sell_current_home", True), key=f"ev_sell_{i}",
    )
    buyer_closing = st.slider(
        "Buyer closing costs (% of price)", 0.0, 5.0,
        float(ev_def.buyer_closing_cost_rate * 100) if ev_def else 2.0, 0.25, key=f"ev_bcc_{i}",
        help="Title, lender fees, escrow, transfer tax — typically 1.5–3%",
    ) / 100
    seller_closing = 0.06
    if sell_current:
        seller_closing = st.slider(
            "Seller closing costs (% of sale price)", 0.0, 10.0,
            float(ev_def.seller_closing_cost_rate * 100) if ev_def else 6.0, 0.25, key=f"ev_scc_{i}",
            help="Agent commissions, transfer tax — typically 5–7%",
        ) / 100
    return dict(
        new_home_price=float(price) if price else None,
        new_home_down_payment=float(down) if down else None,
        new_home_interest_rate=rate / 100 if rate else None,
        sell_current_home=sell_current,
        buyer_closing_cost_rate=buyer_closing,
        seller_closing_cost_rate=seller_closing,
    )


def _event_inputs(i, ev_def) -> TimelineEvent:
    """Render one timeline event's widgets and return the assembled TimelineEvent."""
    yr = st.number_input(
        "Year", min_value=1, max_value=50,
        value=_wd(ev_def, "year", 1, int), key=f"ev_yr_{i}",
    )
    desc = st.text_input(
        "Description", value=_wd(ev_def, "description", ""), key=f"ev_desc_{i}",
    )
    ev_marriage = st.checkbox(
        "Marriage (→ MFJ filing)", value=_wd(ev_def, "marriage", False), key=f"ev_marry_{i}",
    )
    ev_child = st.checkbox("New child", value=_wd(ev_def, "new_child", False), key=f"ev_child_{i}")
    ev_pet = st.checkbox("New pet", value=_wd(ev_def, "new_pet", False), key=f"ev_pet_{i}")

    st.markdown("**Work changes**")
    ev_stop = st.checkbox("You stop working", value=_wd(ev_def, "stop_working", False), key=f"ev_stop_{i}")
    ev_resume = st.checkbox("You resume working", value=_wd(ev_def, "resume_working", False), key=f"ev_resume_{i}")
    ev_partner_stop = st.checkbox("Partner stops working", value=_wd(ev_def, "partner_stop_working", False), key=f"ev_pstop_{i}")
    ev_partner_resume = st.checkbox("Partner resumes working", value=_wd(ev_def, "partner_resume_working", False), key=f"ev_presume_{i}")

    ev_start_care = st.checkbox(
        "Start parent care", value=_wd(ev_def, "start_parent_care", False), key=f"ev_startcare_{i}",
        help="Activates annual_parent_care_cost from Lifestyle settings.",
    )
    ev_stop_care = st.checkbox("Stop parent care", value=_wd(ev_def, "stop_parent_care", False), key=f"ev_stopcare_{i}")
    ev_birth_yr_override = st.number_input(
        "Child birth year override (0 = this year)", min_value=-30, max_value=0,
        value=int(ev_def.child_birth_year_override) if (ev_def and ev_def.child_birth_year_override is not None) else 0,
        key=f"ev_birthyr_{i}",
        help="Set negative to indicate a child already born before the projection. "
             "0 means born in this event's year (default).",
    )

    ev_income = st.number_input(
        "Your new gross income (0 = no change)", min_value=0, max_value=5_000_000,
        value=int(ev_def.income_change) if (ev_def and ev_def.income_change) else 0, key=f"ev_inc_{i}",
    )
    ev_partner_income = st.number_input(
        "Partner new gross income (0 = no change)", min_value=0, max_value=5_000_000,
        value=int(ev_def.partner_income_change) if (ev_def and ev_def.partner_income_change) else 0, key=f"ev_pinc_{i}",
    )
    ev_expense = st.number_input(
        "One-time expense ($)", min_value=0, max_value=1_000_000,
        value=_wd(ev_def, "extra_one_time_expense", 0, int), key=f"ev_exp_{i}",
    )
    ev_bonus = st.number_input(
        "One-time income ($)", min_value=0, max_value=1_000_000,
        value=_wd(ev_def, "extra_one_time_income", 0, int), key=f"ev_bonus_{i}",
    )
    # Home purchase fields — shown only when buy_home is toggled on
    ev_buy_home = st.checkbox("Buy home", value=_wd(ev_def, "buy_home", False), key=f"ev_buyhome_{i}")
    home = dict(
        new_home_price=None, new_home_down_payment=None, new_home_interest_rate=None,
        sell_current_home=True, buyer_closing_cost_rate=0.02, seller_closing_cost_rate=0.06,
    )
    if ev_buy_home:
        home = _event_home_inputs(i, ev_def)

    _ev_base = ev_def if ev_def else TimelineEvent(year=int(yr), description=desc)
    return dataclasses.replace(
        _ev_base,
        year=int(yr),
        description=desc,
        marriage=ev_marriage,
        new_child=ev_child,
        new_pet=ev_pet,
        stop_working=ev_stop,
        resume_working=ev_resume,
        partner_stop_working=ev_partner_stop,
        partner_resume_working=ev_partner_resume,
        start_parent_care=ev_start_care,
        stop_parent_care=ev_stop_care,
        child_birth_year_override=int(ev_birth_yr_override) if ev_birth_yr_override != 0 else None,
        income_change=float(ev_income) if ev_income > 0 else None,
        partner_income_change=float(ev_partner_income) if ev_partner_income > 0 else None,
        extra_one_time_expense=float(ev_expense),
        extra_one_time_income=float(ev_bonus),
        buy_home=ev_buy_home,
        **home,
    )


def _events_section(defaults) -> 'list[TimelineEvent]':
    st.sidebar.header("🗓️ Timeline Events")
    st.sidebar.caption("Add life events that change your financial picture.")

    # Seed defaults from loaded plan so YAML events appear in the UI
    loaded_events = defaults.timeline_events if defaults else []
    default_n_events = len(loaded_events)

    n_events = st.sidebar.number_input(
        "Number of events", min_value=0, max_value=15, value=default_n_events
    )
    events: list[TimelineEvent] = []
    for i in range(int(n_events)):
        # Pull defaults for this slot from the loaded plan (if available)
        ev_def = loaded_events[i] if i < len(loaded_events) else None
        with st.sidebar.expander(
            f"Event {i+1}" + (f": {ev_def.description}" if ev_def and ev_def.description else ""),
            expanded=(i == 0),
        ):
            events.append(_event_inputs(i, ev_def))
    return events


_FS_METRICS = ["brokerage_balance", "liquid_assets", "investable_assets",
               "retirement_balance", "home_equity", "net_worth",
               "medical_burden_ratio"]
_FS_COMPARATORS = ["below", "above"]
_FS_MATCH = ["any", "all"]


def _failsafe_condition_inputs(i, j, cond_def) -> 'FailsafeCondition':
    """Widgets for one trigger condition (a `when:` item)."""
    c1, c2 = st.columns(2)
    metric = c1.selectbox(
        "Metric", _FS_METRICS,
        index=_FS_METRICS.index(cond_def.metric) if (cond_def and cond_def.metric in _FS_METRICS) else 0,
        key=f"fs_cm_{i}_{j}",
    )
    comparator = c2.selectbox(
        "Comparator", _FS_COMPARATORS,
        index=_FS_COMPARATORS.index(cond_def.comparator) if (cond_def and cond_def.comparator in _FS_COMPARATORS) else 0,
        key=f"fs_cc_{i}_{j}",
    )
    is_ratio = metric == "medical_burden_ratio"
    if is_ratio:
        # Unit-free ratio (PV of future medical bills ÷ net worth), e.g. 0.5.
        threshold = st.number_input(
            "Threshold (× net worth)", min_value=0.0, max_value=10.0, step=0.05,
            value=float(cond_def.threshold) if cond_def else 0.5, key=f"fs_ct_{i}_{j}",
        )
    else:
        threshold = float(st.number_input(
            "Threshold ($)", min_value=0, max_value=100_000_000, step=10_000,
            value=int(cond_def.threshold) if cond_def else 100_000, key=f"fs_ct_{i}_{j}",
        ))
    p1, p2, p3 = st.columns(3)
    present_value = p1.checkbox("Today's $", value=_wd(cond_def, "present_value", True),
                                key=f"fs_cpv_{i}_{j}", disabled=is_ratio,
                                help="Ignored for ratio metrics (already unit-free).")
    start_year = p2.number_input("Start yr", min_value=1, max_value=60,
                                 value=_wd(cond_def, "start_year", 1, int), key=f"fs_cs_{i}_{j}")
    end_year = p3.number_input(
        "End yr (0=end)", min_value=0, max_value=60,
        value=int(cond_def.end_year) if (cond_def and cond_def.end_year is not None) else 0,
        key=f"fs_ce_{i}_{j}",
    )
    return FailsafeCondition(
        metric=metric, comparator=comparator, threshold=float(threshold),
        present_value=present_value, start_year=int(start_year),
        end_year=int(end_year) if end_year > 0 else None,
    )


def _failsafe_action_inputs(i, act_def) -> 'FailsafeAction':
    """Widgets for a failsafe's action (the `then:` block)."""
    st.markdown("**Then — action**")
    present_value = st.checkbox("Amounts in today's $", value=_wd(act_def, "present_value", True), key=f"fs_apv_{i}")
    partner_income = st.number_input(
        "Partner income while active ($, 0 = none)", min_value=0, max_value=5_000_000, step=5_000,
        value=int(act_def.partner_income) if (act_def and act_def.partner_income) else 0, key=f"fs_api_{i}",
    )
    primary_income = st.number_input(
        "Your income while active ($, 0 = none)", min_value=0, max_value=5_000_000, step=5_000,
        value=int(act_def.primary_income) if (act_def and act_def.primary_income) else 0, key=f"fs_apri_{i}",
    )
    suspend = st.checkbox(
        "Suspend 401k/IRA contributions", value=_wd(act_def, "suspend_retirement_contributions", False),
        key=f"fs_asusp_{i}", help="Zeroes 401k/IRA deferrals (and the contingent employer match) while active.",
    )
    cut_vac = st.checkbox(
        "Override vacation budget", value=(act_def is not None and act_def.annual_vacation is not None),
        key=f"fs_acutv_{i}", help="Force the annual vacation budget to a set amount while active.",
    )
    annual_vacation = None
    if cut_vac:
        annual_vacation = float(st.number_input(
            "Vacation budget while active ($)", min_value=0, max_value=1_000_000, step=1_000,
            value=int(act_def.annual_vacation) if (act_def and act_def.annual_vacation is not None) else 4_000,
            key=f"fs_avac_{i}",
        ))
    cut_med = st.checkbox(
        "Reduce medical bills", value=(act_def is not None and act_def.medical_cost_multiplier is not None),
        key=f"fs_acutmed_{i}", help="Scale all healthcare costs (OOP, premiums, self-LTC, Medicare) while active, "
                                    "e.g. 50% for 'move abroad'.",
    )
    medical_cost_multiplier = None
    if cut_med:
        pct = st.slider("Medical bills while active (% of normal)", 0, 100,
                        value=int(round((act_def.medical_cost_multiplier if (act_def and act_def.medical_cost_multiplier is not None) else 0.5) * 100)),
                        step=5, key=f"fs_amed_{i}")
        medical_cost_multiplier = pct / 100.0
    a1, a2 = st.columns(2)
    one_time_income = a1.number_input("One-time income ($)", min_value=0, max_value=5_000_000, step=5_000,
                                      value=_wd(act_def, "one_time_income", 0, int), key=f"fs_aoti_{i}")
    one_time_expense = a2.number_input("One-time expense ($)", min_value=0, max_value=5_000_000, step=5_000,
                                       value=_wd(act_def, "one_time_expense", 0, int), key=f"fs_aote_{i}")
    return FailsafeAction(
        partner_income=float(partner_income) if partner_income > 0 else None,
        primary_income=float(primary_income) if primary_income > 0 else None,
        one_time_income=float(one_time_income),
        one_time_expense=float(one_time_expense),
        present_value=present_value,
        suspend_retirement_contributions=suspend,
        annual_vacation=annual_vacation,
        medical_cost_multiplier=medical_cost_multiplier,
    )


def _failsafe_inputs(i, fs_def) -> 'Failsafe':
    """Render one failsafe's widgets and return the assembled Failsafe."""
    name = st.text_input("Name", value=_wd(fs_def, "name", f"failsafe {i+1}"), key=f"fs_name_{i}")
    b1, b2 = st.columns(2)
    match = b1.selectbox("Match", _FS_MATCH,
                         index=_FS_MATCH.index(fs_def.match) if (fs_def and fs_def.match in _FS_MATCH) else 0,
                         key=f"fs_match_{i}", help="'any' fires when any condition is true; 'all' requires all.")
    once = b2.checkbox("Fire once", value=_wd(fs_def, "once", True), key=f"fs_once_{i}",
                       help="On: fires a single time per simulation. Off: re-evaluates every year — "
                            "pair with Duration = 1 for recurring belt-tightening (e.g. pause 401k any year cash is short).")
    d1, d2 = st.columns(2)
    delay = d1.number_input("Delay (yrs)", min_value=0, max_value=30,
                            value=_wd(fs_def, "delay_years", 0, int), key=f"fs_delay_{i}",
                            help="Lag between the trigger firing and the action taking effect.")
    duration = d2.number_input("Duration (yrs, 0 = permanent)", min_value=0, max_value=60,
                               value=int(fs_def.duration_years) if (fs_def and fs_def.duration_years is not None) else 0,
                               key=f"fs_dur_{i}")

    st.markdown(f"**When — triggers** (fires if **{match}** are true)")
    n_cond_default = len(fs_def.conditions) if (fs_def and fs_def.conditions) else 1
    n_cond = st.number_input("Number of conditions", min_value=1, max_value=4,
                             value=n_cond_default, key=f"fs_ncond_{i}")
    conditions = []
    for j in range(int(n_cond)):
        cond_def = fs_def.conditions[j] if (fs_def and j < len(fs_def.conditions)) else None
        st.caption(f"Condition {j+1}")
        conditions.append(_failsafe_condition_inputs(i, j, cond_def))

    action = _failsafe_action_inputs(i, fs_def.action if fs_def else None)
    return Failsafe(
        name=name, conditions=conditions, action=action, match=match,
        delay_years=int(delay), duration_years=int(duration) if duration > 0 else None, once=once,
    )


def _failsafes_section(defaults) -> 'list[Failsafe]':
    st.sidebar.header("🛟 Failsafes")
    st.sidebar.caption("Contingency actions that trigger when a metric crosses a threshold "
                       "(evaluated per simulation path).")
    loaded = defaults.failsafes if defaults else []
    n = st.sidebar.number_input("Number of failsafes", min_value=0, max_value=10,
                                value=len(loaded), key="fs_count")
    out: list = []
    for i in range(int(n)):
        fs_def = loaded[i] if i < len(loaded) else None
        with st.sidebar.expander(
            f"Failsafe {i+1}" + (f": {fs_def.name}" if (fs_def and fs_def.name) else ""),
            expanded=(i == 0),
        ):
            out.append(_failsafe_inputs(i, fs_def))
    return out


def _load_config_expander() -> None:
    """Render the 'Load config' expander; stash an uploaded plan in session_state."""
    with st.sidebar.expander("📂 Load / Save Config", expanded=False):
        uploaded = st.file_uploader("Load YAML config", type=["yaml", "yml"], label_visibility="collapsed")
        if not uploaded:
            return
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        from fintracker.config import load_plan
        try:
            # Only (re)load + re-seed widgets when a *new* file is uploaded, so
            # edits made after loading a config aren't wiped on every rerun.
            sig = (uploaded.name, uploaded.size)
            if st.session_state.get("_cfg_sig") != sig:
                st.session_state["_cfg_sig"] = sig
                st.session_state["loaded_plan"] = load_plan(tmp_path)
                for k in [k for k in st.session_state if k.startswith("w_")]:
                    del st.session_state[k]  # force keyed widgets to re-seed
            st.success("Config loaded!")
        except Exception as e:
            st.error(f"Error loading config: {e}")
        finally:
            os.unlink(tmp_path)


def _export_config_expander(plan) -> None:
    """Render the 'Export config' expander with a YAML download button."""
    with st.sidebar.expander("💾 Export Config", expanded=False):
        import io, yaml
        from fintracker.config import _plan_to_dict
        buf = io.StringIO()
        yaml.dump(_plan_to_dict(plan), buf, default_flow_style=False, sort_keys=False)
        st.download_button(
            "⬇️ Download personal.yaml",
            data=buf.getvalue(),
            file_name="personal.yaml",
            mime="text/yaml",
            width='stretch',
        )


# Single source of truth for sidebar sections: label → adapter that builds its
# component(s) from the loaded defaults and writes them into the components dict.
# Adding a section is a one-line entry here (the radio derives its options from
# these keys), so new sections don't spread churn across build_sidebar.
_SECTION_BUILDERS = {
    "💵 Income":      lambda d, c: c.update(income=_income_section(d)),
    "🏠 Housing":     lambda d, c: c.update(housing=_housing_section(d)),
    "🌿 Lifestyle":   lambda d, c: c.update(lifestyle=_lifestyle_section(d)),
    "📊 Investments": lambda d, c: c.update(investments=_investments_section(d)),
    "🎯 Strategies":  lambda d, c: c.update(zip(("strategies", "investments"), _strategies_section(d))),
    "🚗 Car":         lambda d, c: c.update(car=_car_section(d)),
    "🏢 Business":    lambda d, c: c.update(business=_business_section(d)),
    "🗓️ Events":      lambda d, c: c.update(timeline_events=_events_section(d)),
    "🛟 Failsafes":   lambda d, c: c.update(failsafes=_failsafes_section(d)),
}


def build_sidebar() -> FinancialPlan:
    st.sidebar.title("⚙️ Configure Your Plan")
    section = st.sidebar.radio(
        "section", list(_SECTION_BUILDERS), label_visibility="collapsed",
    )
    st.sidebar.markdown(
        "<small>"
        "<b>inflated yearly</b> — enter today's value; the engine increases it by your CPI assumption each year. &nbsp;"
        "<b>fixed</b> — stays the same nominal amount every year. &nbsp;"
        "<b>current</b> — enter the actual dollar amount as it stands today. &nbsp;"
        "<b>own rate</b> — grows at its own configured rate, not by CPI."
        "</small>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    _load_config_expander()

    # Defaults come from the loaded/persisted plan (None on a truly fresh start).
    defaults = st.session_state.get("loaded_plan", None)

    # Pre-initialise every plan component as a pass-through default; the active
    # section (dispatched below) overwrites the one(s) it owns. Sections not
    # currently shown keep these so FinancialPlan() always has all its args.
    components = dict(
        income=_wd(defaults, "income", IncomeProfile(gross_annual_income=120_000)),
        housing=_wd(defaults, "housing", HousingProfile(home_price=0, down_payment=0, interest_rate=0.0)),
        lifestyle=_wd(defaults, "lifestyle", LifestyleProfile()),
        investments=_wd(defaults, "investments", InvestmentProfile()),
        strategies=_wd(defaults, "strategies", StrategyToggles()),
        car=_wd(defaults, "car", None),
        business=_wd(defaults, "business", None),
        timeline_events=_wd(defaults, "timeline_events", []),
        failsafes=_wd(defaults, "failsafes", []),
    )
    _SECTION_BUILDERS[section](defaults, components)

    projection_years = st.sidebar.slider(
        "Projection Horizon (Years)", 5, 40, _wd(defaults, "projection_years", 30),
    )
    plan = FinancialPlan(
        **components,
        projection_years=int(projection_years),
        retirement=_wd(defaults, "retirement", None),
        college=_wd(defaults, "college", None),
    )

    # Persist edits so sections not currently shown retain their values
    # across reruns (they feed `defaults` above on the next render).
    st.session_state["loaded_plan"] = plan

    _export_config_expander(plan)
    return plan


# ─────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────

def _tab_cash_flow(plan, tax_result, monthly_housing, monthly_lifestyle, monthly_breathing, mortgage_calc) -> None:
    st.markdown('<div class="section-header">Monthly Cash Flow Breakdown</div>', unsafe_allow_html=True)

    col_chart, col_detail = st.columns([1, 1])

    with col_chart:
        monthly_k401 = plan.investments.annual_401k_contribution / 12
        monthly_hsa = plan.investments.annual_hsa_contribution / 12

        categories = ["Federal Tax", "FICA", "State Tax", "Housing", "Lifestyle",
                      "401k", "HSA", "Breathing Room"]
        values = [
            tax_result.federal_income_tax / 12,
            tax_result.total_fica / 12,
            tax_result.state_income_tax / 12,
            monthly_housing,
            monthly_lifestyle,
            monthly_k401,
            monthly_hsa,
            max(0, monthly_breathing),
        ]
        colors_list = ["#ef4444", "#f97316", "#fbbf24", "#f97316",
                       "#06b6d4", "#3b82f6", "#f59e0b", "#4ade80"]

        fig = go.Figure(go.Bar(
            x=categories, y=values,
            marker_color=colors_list,
            text=[fmt_dollar(v) for v in values],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            title="Monthly Dollar Allocation",
            **PLOTLY_DARK,
            yaxis_title="$ / month",
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig, width='stretch')

    with col_detail:
        st.markdown("#### Annual Tax Detail")
        tax_rows = [
            ("Federal Income Tax", tax_result.federal_income_tax),
            ("Social Security (6.2%)", tax_result.social_security_tax),
            ("Medicare (1.45%)", tax_result.medicare_tax),
            ("Additional Medicare", tax_result.additional_medicare_tax),
            ("State Income Tax", tax_result.state_income_tax),
            ("**Total Tax**", tax_result.total_annual_tax),
        ]
        df_tax = pd.DataFrame(tax_rows, columns=["Item", "Annual"])
        df_tax["Monthly"] = df_tax["Annual"] / 12
        df_tax["Annual"] = df_tax["Annual"].apply(lambda x: f"${x:,.0f}")
        df_tax["Monthly"] = df_tax["Monthly"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df_tax, hide_index=True, width='stretch')

        gross = plan.income.total_gross_income
        eff_rate = tax_result.total_annual_tax / gross if gross else 0
        st.markdown(f"**Effective Total Tax Rate:** `{eff_rate:.1%}`")

        if plan.housing.is_renting:
            st.info("🏠 Currently renting.")
        elif mortgage_calc:
            summary = mortgage_calc.summary()
            st.markdown(f"""
**Mortgage Summary**
- Monthly P&I: `{fmt_dollar(summary.monthly_pi)}`
- PMI: `{fmt_dollar(summary.monthly_pmi_initial)}/mo` {'(drops off month ' + str(summary.pmi_removal_month) + ')' if summary.pmi_removal_month else '(N/A — 20%+ down)'}
- Total interest over life of loan: `{fmt_dollar(summary.total_interest_paid)}`
""")

# ── TAB 4: Mortgage ──────────────────────────────────────


def _tab_mortgage(plan, mortgage_calc) -> None:
    if plan.housing.is_renting:
        st.info("🏠 You're currently renting. Configure a home purchase to see amortization details.")
    elif mortgage_calc:
        st.markdown('<div class="section-header">Full Amortization Schedule</div>', unsafe_allow_html=True)
        summary = mortgage_calc.summary()
        schedule = mortgage_calc.full_schedule()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Loan Amount", fmt_dollar(summary.loan_amount))
        m2.metric("Monthly P&I", fmt_dollar(summary.monthly_pi))
        m3.metric("Total Interest Paid", fmt_dollar(summary.total_interest_paid))
        m4.metric("PMI Removal", f"Month {summary.pmi_removal_month}" if summary.pmi_removal_month else "N/A")

        # Annual snapshots chart
        annual_sched = [r for r in schedule if r.month % 12 == 0]
        years_amort = [r.year for r in annual_sched]

        fig_amort = go.Figure()
        fig_amort.add_trace(go.Scatter(
            x=years_amort, y=[r.home_value for r in annual_sched],
            name="Home Value", fill="tozeroy",
            line=dict(color=COLORS["home_equity"], width=2),
            fillcolor="rgba(16,185,129,0.15)",
        ))
        fig_amort.add_trace(go.Scatter(
            x=years_amort, y=[r.balance for r in annual_sched],
            name="Loan Balance", fill="tozeroy",
            line=dict(color=COLORS["taxes"], width=2),
            fillcolor="rgba(239,68,68,0.2)",
        ))
        fig_amort.add_trace(go.Scatter(
            x=years_amort, y=[r.equity for r in annual_sched],
            name="Your Equity",
            line=dict(color=COLORS["brokerage"], width=2, dash="dash"),
        ))
        fig_amort.update_layout(
            title="Home Value vs Loan Balance vs Equity",
            **PLOTLY_DARK, height=380, yaxis_title="$",
        )
        st.plotly_chart(fig_amort, width='stretch')

        # P&I breakdown over time
        fig_pi = go.Figure()
        fig_pi.add_trace(go.Bar(
            x=years_amort, y=[r.interest for r in annual_sched],
            name="Interest", marker_color=COLORS["taxes"],
        ))
        fig_pi.add_trace(go.Bar(
            x=years_amort, y=[r.principal for r in annual_sched],
            name="Principal", marker_color=COLORS["home_equity"],
        ))
        fig_pi.update_layout(
            barmode="stack", title="Annual Interest vs Principal Paid",
            **PLOTLY_DARK, height=300, yaxis_title="$ / year",
        )
        st.plotly_chart(fig_pi, width='stretch')

        # Full table (collapsed by default)
        with st.expander("📋 Full Amortization Table (month-by-month)"):
            df_sched = pd.DataFrame([{
                "Month": r.month,
                "Year": r.year,
                "Payment": f"${r.payment:,.0f}",
                "Principal": f"${r.principal:,.0f}",
                "Interest": f"${r.interest:,.0f}",
                "PMI": f"${r.pmi:,.2f}",
                "Balance": f"${r.balance:,.0f}",
                "Cumulative Interest": f"${r.cumulative_interest:,.0f}",
                "Home Value": f"${r.home_value:,.0f}",
                "Equity": f"${r.equity:,.0f}",
            } for r in schedule])
            st.dataframe(df_sched, hide_index=True, width='stretch', height=400)

# ── TAB 5: Tax Strategies ────────────────────────────────


def _tab_tax(plan, snapshots, strategy_result, yr1, tax_result, marginal_rate=0.0) -> None:
    st.markdown('<div class="section-header">Tax Optimization Analysis</div>', unsafe_allow_html=True)

    sa1, sa2, sa3, sa4 = st.columns(4)
    sa1.metric("HSA Savings", fmt_dollar(strategy_result.hsa_annual_savings) + "/yr")
    sa2.metric("401k Savings", fmt_dollar(strategy_result.k401_annual_savings) + "/yr")
    sa3.metric("529 State Savings", fmt_dollar(strategy_result.state_529_annual_savings) + "/yr")
    sa4.metric("Total Tax Alpha", fmt_dollar(strategy_result.total_annual_savings) + "/yr",
               delta=f"≈ {fmt_dollar(strategy_result.total_annual_savings * 10)} over 10 yrs (uninvested)")

    st.markdown("#### Strategy Insights")
    for note in strategy_result.notes:
        is_tip = note.startswith("💡") or note.startswith("⚠️")
        card_class = "tip-card" if is_tip else "strategy-card"
        st.markdown(f'<div class="{card_class}">{note}</div>', unsafe_allow_html=True)

    # Waterfall chart: income → take-home
    # Use year-1 snapshot values so the filing status, HSA tier, and tax
    # all match what the projection table shows for year 1.
    # Waterfall uses tax_result (already computed from yr1 state above)
    st.markdown("#### Where Does Your Gross Income Go?")
    gross = yr1.gross_income
    waterfall_cats = [
        "Gross Income",
        "Federal Tax",
        "FICA",
        "State Tax",
        "401k",
        "HSA",
        "Take-Home",
    ]
    waterfall_vals = [
        gross,
        -tax_result.federal_income_tax,
        -tax_result.total_fica,
        -tax_result.state_income_tax,
        -yr1.annual_retirement_contributions,
        -yr1.annual_hsa_contributions,
        0,  # calculated
    ]
    take_home = gross + sum(waterfall_vals[1:-1])
    waterfall_vals[-1] = take_home

    measures = ["absolute"] + ["relative"] * (len(waterfall_cats) - 2) + ["total"]
    wf_colors = (
        ["#4ade80"]
        + ["#ef4444"] * 3
        + ["#3b82f6", "#f59e0b"]
        + ["#4ade80"]
    )

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=waterfall_cats,
        y=waterfall_vals,
        connector=dict(line=dict(color="#21262d", width=1)),
        increasing=dict(marker_color="#4ade80"),
        decreasing=dict(marker_color="#ef4444"),
        totals=dict(marker_color="#4ade80"),
        text=[fmt_dollar(abs(v)) for v in waterfall_vals],
        textposition="outside",
    ))
    fig_wf.update_layout(
        title="Annual Income Waterfall",
        **PLOTLY_DARK, height=400, yaxis_title="$",
        showlegend=False,
    )
    st.plotly_chart(fig_wf, width='stretch')

    # Year-1 rates the waterfall doesn't show (effective + marginal, with the
    # state named so the state's contribution is traceable).
    state_name = state_display_name(plan.income.state)
    eff_rate = tax_result.total_annual_tax / gross if gross else 0.0
    st.caption(
        f"Year 1: **{eff_rate:.1%}** effective · **{marginal_rate:.1%}** marginal tax rate "
        f"(federal + {state_name} state)."
    )

    # ── Per-year tax detail ──────────────────────────────────
    st.markdown(f"#### Taxes by Year — {state_name}")
    st.caption(
        f"State tax reflects **{state_name}** rules. Filing status is shown per year "
        "(it changes at a marriage event). Federal is net of education credits; "
        "federal and state brackets are inflation-indexed each projection year."
    )
    _filing_label = {
        FilingStatus.SINGLE: "Single",
        FilingStatus.MARRIED_FILING_JOINTLY: "MFJ",
        FilingStatus.HEAD_OF_HOUSEHOLD: "HoH",
    }
    state_col = f"State ({state_name})"
    per_year = pd.DataFrame([{
        "Year": s.year,
        "Filing": _filing_label.get(s.filing_status, s.filing_status.value),
        "Gross Income": s.gross_income,
        "Federal": s.annual_federal_tax,
        "FICA": s.annual_fica_tax,
        state_col: s.annual_state_tax,
        "Total Tax": s.annual_tax_total,
        "Effective Rate": (s.annual_tax_total / s.gross_income) if s.gross_income else 0.0,
    } for s in snapshots])
    display_df = per_year.copy()
    for c in ["Gross Income", "Federal", "FICA", state_col, "Total Tax"]:
        display_df[c] = display_df[c].apply(fmt_dollar)
    display_df["Effective Rate"] = per_year["Effective Rate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, hide_index=True, width='stretch', height=360)

# ── TAB 1: Projections ───────────────────────────────────


# Ages at which tax-advantaged accounts become penalty-free to withdraw, so a
# retiree's "liquid" assets realistically include them:
#   59 — traditional 401k/IRA and Roth (≈ the IRS 59½ penalty-free age)
#   65 — HSA for any purpose (always penalty-free for medical)
_PENALTY_FREE_RETIREMENT_AGE = PENALTY_FREE_AGE   # shared with the engine's waterfall gate
_HSA_PENALTY_FREE_AGE = 65


def _liquid_assets(plan, snapshots) -> list[float]:
    """Per-year liquid assets, age-aware — what you could realistically spend.

    Always liquid: brokerage + uninvested cash + cash buffer. Once penalty-free to
    withdraw, retirement accounts are added: the traditional 401k/IRA at 59½ (net
    of income tax, since withdrawals are taxable), Roth at 59½ (tax-free), and the
    HSA at 65 (any-purpose). Without a RetirementProfile there is no age basis, so
    retirement accounts are never counted.
    """
    rp = plan.retirement
    current_age = rp.current_age if rp else None
    wd_tax = rp.retirement_withdrawal_tax_rate if rp else 0.0
    out = []
    for s in snapshots:
        liquid = s.brokerage_balance + s.uninvested_cash + s.cash_buffer
        age = current_age + s.year - 1 if current_age is not None else None
        if age is not None and age >= _PENALTY_FREE_RETIREMENT_AGE:
            liquid += s.retirement_balance * (1.0 - wd_tax) + s.roth_ira_balance
        if age is not None and age >= _HSA_PENALTY_FREE_AGE:
            liquid += s.hsa_balance
        out.append(liquid)
    return out


def _projection_income_chart(plan, snapshots, df, projection_engine) -> None:
    """Income vs expenses + balances chart with retirement/Roth/529 overlays."""
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(
        x=df["Year"], y=df["Net Income"], name="Net Income",
        line=dict(color="#4ade80", width=2),
        yaxis="y1",
    ))
    fig_cf.add_trace(go.Scatter(
        x=df["Year"],
        y=df["Housing Cost"] + df["Lifestyle Cost"],
        name="Total Expenses",
        line=dict(color="#ef4444", width=2),
        yaxis="y1",
    ))
    fig_cf.add_trace(go.Scatter(
        x=df["Year"], y=_liquid_assets(plan, snapshots),
        name="Liquid Assets (brokerage + cash; +401k net-of-tax & Roth at 59½, HSA at 65)",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        yaxis="y2",
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.08)",
    ))

    # ── Total investable assets + retirement target ─────────────
    # Total investable = retirement + HSA + Roth + brokerage + cash (all accessible
    # funds). Uses YearlySnapshot.investable_assets — the same pool the retirement-
    # readiness calc scores against, so the chart line and the readiness % agree.
    total_investable = [s.investable_assets for s in snapshots]
    fig_cf.add_trace(go.Scatter(
        x=df["Year"], y=total_investable,
        name="Total Investable Assets (401k + HSA + Roth + brokerage + cash)",
        line=dict(color="#818cf8", width=2, dash="dashdot"),
        yaxis="y2",
    ))
    # Retirement target — only when RetirementProfile is configured
    if plan.retirement:
        rr = projection_engine.compute_retirement_readiness(snapshots)
        if rr:
            fig_cf.add_hline(
                y=rr.required_balance,
                line=dict(color="#818cf8", width=1, dash="dot"),
                annotation_text=f"Retirement target {fmt_dollar(rr.required_balance)}",
                annotation_position="top left",
                annotation_font=dict(color="#818cf8", size=10),
                yref="y2",
            )

    # ── Roth IRA balance line ─────────────────────────────────
    if any(s.roth_ira_balance > 0 for s in snapshots):
        fig_cf.add_trace(go.Scatter(
            x=df["Year"],
            y=[s.roth_ira_balance for s in snapshots],
            name="Roth IRA Balance",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            yaxis="y2",
        ))

    # ── 529 college fund line + target ──────────────────────────
    if plan.college and any(s.college_529_balance > 0 for s in snapshots):
        col529_vals = [s.college_529_balance for s in snapshots]
        fig_cf.add_trace(go.Scatter(
            x=df["Year"], y=col529_vals,
            name="529 Balance",
            line=dict(color="#34d399", width=2, dash="dash"),
            yaxis="y2",
        ))

        # College target: sum of all nominal college costs from first
        # college year onward — this is what you need saved by then
        college_costs = [(s.year, s.annual_college_cost)
                         for s in snapshots if s.annual_college_cost > 0]
        if college_costs:
            first_college_yr = college_costs[0][0]
            last_college_yr  = college_costs[-1][0]
            total_college_cost = sum(c for _, c in college_costs)

            # Horizontal target: total nominal cost (a rough but intuitive benchmark)
            fig_cf.add_hline(
                y=total_college_cost,
                line=dict(color="#34d399", width=1, dash="dot"),
                annotation_text=f"College total {fmt_dollar(total_college_cost)}",
                annotation_position="bottom right",
                annotation_font=dict(color="#34d399", size=10),
                yref="y2",
            )

            # Vertical band marking active college years
            fig_cf.add_vrect(
                x0=first_college_yr - 0.5,
                x1=last_college_yr + 0.5,
                fillcolor="rgba(52,211,153,0.07)",
                line_width=0,
                annotation_text="College years",
                annotation_position="top left",
                annotation_font=dict(color="#34d399", size=9),
            )

    # Can't use **PLOTLY_DARK here because it already defines 'yaxis';
    # passing yaxis= again would cause a duplicate keyword error.
    fig_cf.update_layout(
        title="Income vs Expenses + Liquid Assets",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(family="Inter", color="#c9d1d9"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=360,
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(title="$ / year", gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis2=dict(
            title="Balances ($)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor="#374151",
            zerolinewidth=1,
            tickformat="$,.0f",
        ),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    st.plotly_chart(fig_cf, width='stretch')


def _liquidity_warnings(plan, snapshots) -> None:
    """Surface negative / low liquid-asset years as an error or warning banner.

    Liquid assets are age-aware (see _liquid_assets): retirement accounts count
    only once penalty-free to withdraw, so a retiree funding expenses from a large
    401k/Roth/HSA is not flagged as illiquid.
    """
    pairs = list(zip(snapshots, _liquid_assets(plan, snapshots)))
    negative_years = [(s, v) for s, v in pairs if v < 0]
    low_years = [(s, v) for s, v in pairs if 0 <= v < 10_000]
    if negative_years:
        first, _ = negative_years[0]
        worst, worst_v = min(negative_years, key=lambda p: p[1])
        st.error(
            f"⚠️ **Liquid assets go negative in Year {first.year}** — "
            f"worst point is **{fmt_dollar(worst_v)}** in Year {worst.year}. "
            f"You would need to sell investments, take on debt, or reduce expenses to cover the shortfall."
        )
    elif low_years:
        first, _ = low_years[0]
        st.warning(
            f"⚠️ **Liquid assets fall below $10,000 in Year {first.year}** "
            f"(lowest: **{fmt_dollar(min(v for _, v in low_years))}**). "
            f"Consider building a larger emergency buffer."
        )

    # Brokerage-specific insolvency: the taxable account can go negative even when
    # age-aware liquid assets (which count a 59½+ retiree's 401k/Roth) stay positive.
    # A negative brokerage is an implicit margin loan the model keeps compounding at
    # the market rate rather than a real borrowing cost — an optimistic simplification
    # worth flagging honestly so the projected net worth isn't taken at face value.
    brokerage_negative = [s for s in snapshots if s.brokerage_balance < 0]
    if brokerage_negative:
        first = brokerage_negative[0]
        worst = min(brokerage_negative, key=lambda s: s.brokerage_balance)
        st.warning(
            f"⚠️ **Taxable brokerage goes negative in Year {first.year}** "
            f"(worst: **{fmt_dollar(worst.brokerage_balance)}** in Year {worst.year}). "
            "This means a purchase or shortfall was funded by implicitly borrowing against "
            "the account. The model keeps compounding that negative balance at the market "
            "return instead of charging a real loan rate, so the projected net worth in and "
            "after these years is optimistic — treat it as a signal to hold more cash or "
            "spread out large outflows."
        )


def _retirement_readiness_panel(plan, snapshots, projection_engine) -> None:
    """Withdrawal-tax controls + retirement readiness metrics (no-op if unconfigured)."""
    if not plan.retirement:
        return
    # ── Withdrawal tax rates (inline controls, not sidebar) ──────
    # These let the user model the real tax advantage of Roth vs 401k/brokerage.
    with st.expander("⚙️ Withdrawal Tax Assumptions (affects Retirement Readiness)", expanded=False):
        st.caption(
            "Set estimated tax rates to see your **after-tax** projected balance — "
            "which makes Roth dollars worth more than 401k or brokerage dollars. "
            "Leave at 0% to see the raw balance (no tax adjustment, same as before)."
        )
        _tc1, _tc2 = st.columns(2)
        _401k_tax = _tc1.slider(
            "401k / IRA withdrawal tax rate (%)",
            0, 45,
            int(plan.retirement.retirement_withdrawal_tax_rate * 100),
            1,
            help="Tax on 401k/IRA withdrawals in retirement (ordinary income rate). "
                 "Typical: 22–32% for most retirees.",
        ) / 100
        _cg_tax = _tc2.slider(
            "Brokerage capital gains rate (%)",
            0, 25,
            int(plan.retirement.capital_gains_tax_rate * 100),
            1,
            help="Tax on brokerage gains in retirement (long-term capital gains rate). "
                 "Typical: 15–20% for most retirees. Roth IRA is always 0%."
        ) / 100
        if _401k_tax > 0 or _cg_tax > 0:
            st.caption(
                f"After-tax value: 401k/IRA discounted {_401k_tax:.0%}, "
                f"brokerage discounted {_cg_tax:.0%}, Roth IRA untouched (tax-free)."
            )
    rr_panel = projection_engine.compute_retirement_readiness(
        snapshots,
        withdrawal_tax_rate=_401k_tax,
        capital_gains_rate=_cg_tax,
    )
    if not rr_panel:
        return
    st.markdown("#### 🎯 Retirement Readiness")
    _tax_adjusted = (_401k_tax > 0 or _cg_tax > 0)
    status  = "✅ On Track" if rr_panel.on_track else "⚠️ Off Track"
    funded  = f"{rr_panel.funded_pct:.0%}"
    gap_lbl = "Annual Surplus" if rr_panel.annual_surplus_or_gap >= 0 else "Annual Gap"
    gap_val = fmt_dollar(abs(rr_panel.annual_surplus_or_gap))
    gap_sign = "+" if rr_panel.annual_surplus_or_gap >= 0 else "-"
    proj_label = "Projected at Retirement (after-tax)" if _tax_adjusted else "Projected at Retirement"

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Status", status)
    rc2.metric("Funded", funded,
               delta=f"target {fmt_dollar(rr_panel.required_balance)}",
               delta_color="normal" if rr_panel.on_track else "inverse")
    rc3.metric(proj_label, fmt_dollar(rr_panel.projected_balance_at_retirement),
               delta=f"pre-tax {fmt_dollar(rr_panel.projected_balance_pretax)}" if _tax_adjusted else None)
    rc4.metric(gap_lbl, f"{gap_sign}{gap_val} /yr",
               delta=f"over {plan.retirement.years_in_retirement}yr @ {plan.retirement.expected_post_retirement_return:.0%}")

    st.caption(
        f"Retire at age {plan.retirement.retirement_age} (year {rr_panel.years_to_retirement}) · "
        f"First-year retirement cost: {fmt_dollar(rr_panel.annual_cost_at_retirement)}/yr (nominal) · "
        + (f"SS offset: {fmt_dollar(rr_panel.social_security_offset)}/yr · " if rr_panel.social_security_offset > 0 else "")
        + f"Post-retirement return: {plan.retirement.expected_post_retirement_return:.0%}"
        + (f" · 401k tax {_401k_tax:.0%} · cap gains {_cg_tax:.0%}" if _tax_adjusted else "")
    )
    st.divider()


def _compute_milestones(plan, snapshots) -> list:
    """Pure logic: first-crossing net-worth / mortgage-payoff / cash-deficit milestones.

    Returns a list of (label, value, delta_note) tuples ready for rendering.
    """
    # Scan snapshots once and record the FIRST year each threshold is crossed.
    nw_thresholds = [
        (250_000,  "Net Worth $250k"),
        (500_000,  "Net Worth $500k"),
        (1_000_000, "Net Worth $1M 🎉"),
        (2_000_000, "Net Worth $2M"),
        (5_000_000, "Net Worth $5M"),
    ]
    reached = {}  # label -> YearlySnapshot
    for s in snapshots:
        for threshold, label in nw_thresholds:
            if label not in reached and s.net_worth >= threshold:
                reached[label] = s

    milestones = [
        (label, f"Year {reached[label].year}", f"NW {fmt_dollar(reached[label].net_worth)}")
        for _, label in nw_thresholds if label in reached
    ]

    mortgage_paid = next((s for s in snapshots if s.mortgage_balance == 0 and not s.is_renting), None)
    if mortgage_paid:
        milestones.append(("🏠 Mortgage Paid Off", f"Year {mortgage_paid.year}", ""))

    worst_liquid = min(snapshots, key=lambda s: s.brokerage_balance)
    if worst_liquid.brokerage_balance < -50_000:
        milestones.append(("⚠️ Peak Cash Deficit", f"Year {worst_liquid.year}",
                           fmt_dollar(worst_liquid.brokerage_balance)))

    # Wedding fund paid out — one card per wedding year, showing the amount saved
    # (contributions plus investment growth) at the time the child marries, in that
    # year's dollars, with the equivalent in today's dollars for context.
    weddings = [s for s in snapshots if s.annual_wedding_spend > 0]
    for idx, s in enumerate(weddings, 1):
        label = "💍 Child's Wedding Fund" if len(weddings) == 1 else f"💍 Child #{idx}'s Wedding Fund"
        today = s.to_todays_dollars(s.annual_wedding_spend)
        milestones.append((label, fmt_dollar(s.annual_wedding_spend),
                           f"Year {s.year} · {fmt_dollar(today)} in today's $"))

    if not milestones:
        milestones.append(("Net Worth $1M", f"Not reached in {plan.projection_years} years", ""))
    return milestones


def _render_milestones(plan, snapshots) -> None:
    """Render the milestone metrics computed by _compute_milestones in rows of 3."""
    st.markdown("#### Key Milestones")
    milestones = _compute_milestones(plan, snapshots)
    for row_start in range(0, len(milestones), 3):
        row = milestones[row_start:row_start + 3]
        cols = st.columns(len(row))
        for col, (label, value, delta) in zip(cols, row):
            col.metric(label, value, delta if delta else None)


def _projection_data_table(df) -> None:
    """Year-by-year cash-flow and balance tables inside an expander."""
    with st.expander("📋 Full Year-by-Year Projection Table"):
        # ── Annual Cash Flows (what happened this year) ──────────
        st.markdown("**Annual Cash Flows** — income, costs, and surplus for each year")
        flow_cols = ["Year", "Gross Income", "Taxes", "Net Income", "Housing Cost",
                     "Lifestyle Cost", "Car Operating", "401k Withdrawal",
                     "Brokerage Withdrawal", "Roth Withdrawal", "Breathing Room"]
        flow_df = df[flow_cols].copy()
        flow_df["Gross Income"]  = flow_df["Gross Income"].apply(fmt_dollar)
        flow_df["Net Income"]    = flow_df["Net Income"].apply(fmt_dollar)
        flow_df["Taxes"]         = flow_df["Taxes"].apply(fmt_dollar)
        flow_df["Housing Cost"]  = flow_df["Housing Cost"].apply(fmt_dollar)
        flow_df["Lifestyle Cost"]= flow_df["Lifestyle Cost"].apply(fmt_dollar)
        flow_df["Car Operating"] = flow_df["Car Operating"].apply(fmt_dollar)
        flow_df["401k Withdrawal"] = flow_df["401k Withdrawal"].apply(fmt_dollar)
        flow_df["Brokerage Withdrawal"] = flow_df["Brokerage Withdrawal"].apply(fmt_dollar)
        flow_df["Roth Withdrawal"] = flow_df["Roth Withdrawal"].apply(fmt_dollar)
        flow_df["Breathing Room"]= flow_df["Breathing Room"].apply(
            lambda x: f"{'▲ ' if x >= 0 else '▼ '}{fmt_dollar(x)}"
        )
        st.dataframe(flow_df, hide_index=True, width='stretch')

        st.markdown("")

        # ── End-of-Year Balances (where you stand) ───────────────
        st.markdown("**End-of-Year Balances** — cumulative wealth position at end of each year")
        bal_cols = ["Year", "Retirement", "Brokerage", "HSA",
                    "Home Equity", "Mortgage Balance", "Net Worth"]
        bal_df = df[bal_cols].copy()
        for col in ["Retirement", "Brokerage", "HSA",
                    "Home Equity", "Mortgage Balance", "Net Worth"]:
            bal_df[col] = bal_df[col].apply(fmt_dollar)
        st.dataframe(bal_df, hide_index=True, width='stretch')


def _tab_projections(plan, snapshots, projection_engine) -> None:
    st.markdown('<div class="section-header">Long-Term Wealth Projection</div>', unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Year": s.year,
        "Gross Income": s.gross_income,
        "Net Income": s.net_income,
        "Housing Cost": s.annual_housing_cost,
        "Lifestyle Cost": s.annual_lifestyle_cost,
        "Car Operating": s.annual_car_operating_cost,
        "Breathing Room": s.annual_breathing_room,
        "401k Withdrawal": s.annual_retirement_withdrawal,
        "Brokerage Withdrawal": s.annual_brokerage_withdrawal,
        "Roth Withdrawal": s.annual_roth_withdrawal,
        "Retirement": s.retirement_balance,
        "Brokerage": s.brokerage_balance,
        "Home Equity": s.home_equity,
        "HSA": s.hsa_balance,
        "Net Worth": s.net_worth,
        "Mortgage Balance": s.mortgage_balance,
        "Taxes": s.annual_tax_total,
    } for s in snapshots])

    # Net worth composition stacked area
    fig_nw = go.Figure()
    for key, label, color in [
        ("Retirement", "Retirement", COLORS["retirement"]),
        ("Brokerage", "Brokerage", COLORS["brokerage"]),
        ("Home Equity", "Home Equity", COLORS["home_equity"]),
        ("HSA", "HSA", COLORS["hsa"]),
    ]:
        fig_nw.add_trace(go.Scatter(
            x=df["Year"], y=df[key], name=label,
            mode="lines", stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=hex_to_rgba(color, 0.7) if color.startswith("#") else color,
        ))
    fig_nw.update_layout(
        title="Net Worth Composition Over Time",
        **PLOTLY_DARK, height=420, yaxis_title="$",
    )
    st.plotly_chart(fig_nw, width='stretch')

    col_l, col_r = st.columns(2)
    with col_l:
        _projection_income_chart(plan, snapshots, df, projection_engine)
    with col_r:
        # Breathing room bar
        br_colors = ["#4ade80" if v >= 0 else "#ef4444" for v in df["Breathing Room"]]
        fig_br = go.Figure(go.Bar(
            x=df["Year"], y=df["Breathing Room"],
            marker_color=br_colors,
            text=[fmt_dollar(v) for v in df["Breathing Room"]],
            textposition="outside",
        ))
        fig_br.update_layout(
            title="Annual Breathing Room (Cash Surplus)", **PLOTLY_DARK, height=300, yaxis_title="$",
        )
        st.plotly_chart(fig_br, width='stretch')

    _liquidity_warnings(plan, snapshots)
    _retirement_readiness_panel(plan, snapshots, projection_engine)
    _render_milestones(plan, snapshots)
    _projection_data_table(df)

# ── TAB 2: Monte Carlo ───────────────────────────────────


def _mc_simulation_params() -> dict:
    """Render the Monte Carlo parameter controls; return run_monte_carlo kwargs."""
    with st.expander("⚙️ Simulation Parameters", expanded=False):
        col_t1, col_t2 = st.columns(2)
        use_hist = col_t1.toggle(
            "Historical S&P 500 returns",
            value=True,
            help="ON (recommended): bootstrap from 100 years of actual S&P 500 data "
                 "(1926–2025), capturing fat tails, -43% crashes, and +54% booms. "
                 "OFF: draws from a normal distribution.",
        )
        use_hist_inf = col_t2.toggle(
            "Historical US inflation",
            value=True,
            help="ON (recommended): bootstrap from 96 years of actual CPI data "
                 "(1929–2024), including deflation, 1970s stagflation (13.3%), "
                 "and 2021–22 surge (7%). OFF: draws from a normal distribution.",
        )
        both_hist = use_hist and use_hist_inf
        bcol1, bcol2 = st.columns(2)
        block_bs = bcol1.toggle(
            "Joint block bootstrap",
            value=True,
            disabled=not both_hist,
            help="ON (recommended): sample equity and inflation JOINTLY as "
                 "calendar-year-aligned pairs drawn in multi-year blocks, so "
                 "stagflation (bad stocks + high inflation) stays bundled and "
                 "multi-year regimes (sticky inflation, crash-then-recovery) are "
                 "preserved. Salary growth is tied to the sampled inflation. "
                 "OFF: draw each series independently, one year at a time. "
                 "Requires both historical toggles ON.",
        )
        mean_block = bcol2.slider(
            "Mean block length (yrs)", 1.0, 10.0, 2.0, 1.0,
            disabled=not (both_hist and block_bs),
            help="Average length of each contiguous historical run in the block "
                 "bootstrap (stationary bootstrap; block lengths are random with "
                 "this mean).",
        )
        mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
        n_sims = mc_col1.number_input(
            "Simulations", min_value=100, max_value=10_000,
            value=5_000, step=100,
            help="More simulations = smoother percentile bands but slower.",
        )
        mkt_std = mc_col2.slider(
            "Market Return Std Dev (%)", 1.0, 30.0, 15.0, 1.0,
            disabled=use_hist,
            help="Only used when historical returns are OFF. "
                 "Historical S&P 500 std dev is ~19.6%.",
        ) / 100
        inf_std = mc_col3.slider(
            "Inflation Std Dev (%)", 0.0, 5.0, 1.5, 0.25,
            disabled=use_hist_inf,
            help="Only used when historical inflation is OFF. "
                 "Historical CPI std dev is ~3.9%.",
        ) / 100
        sg_std = mc_col4.slider(
            "Salary Growth Std Dev (%)", 0.0, 10.0, 2.0, 0.5,
            help="Year-to-year variation in salary growth.",
        ) / 100
        mc_seed = st.checkbox("Fix random seed (reproducible)", value=True)
    return dict(
        n_sims=int(n_sims), use_hist=use_hist, use_hist_inf=use_hist_inf,
        mkt_std=mkt_std, inf_std=inf_std, sg_std=sg_std, mc_seed=mc_seed,
        block_bs=block_bs, mean_block=float(mean_block),
    )


def _mc_networth_fan_chart(mc, snapshots, n_sims) -> None:
    """Net-worth percentile fan chart with the deterministic overlay."""
    years_mc = mc.years
    fig_mc = go.Figure()
    _add_band(fig_mc, years_mc, mc.p90_net_worth, mc.p10_net_worth, "rgba(59,130,246,0.1)", "p10–p90 band")
    _add_band(fig_mc, years_mc, mc.p75_net_worth, mc.p25_net_worth, "rgba(59,130,246,0.2)", "p25–p75 band")
    _add_line(fig_mc, years_mc, mc.p50_net_worth, "Median (p50)", "#3b82f6", 2.5)
    _add_line(fig_mc, years_mc, mc.p90_net_worth, "Optimistic (p90)", "#4ade80", 1.5, "dot")
    _add_line(fig_mc, years_mc, mc.p10_net_worth, "Pessimistic (p10)", "#f87171", 1.5, "dot")
    _add_line(fig_mc, [s.year for s in snapshots], [s.net_worth for s in snapshots],
              "Deterministic", "#f59e0b", 2, "dash")
    hist_parts = []
    if mc.use_historical_returns:   hist_parts.append("hist. returns")
    if mc.use_historical_inflation: hist_parts.append("hist. inflation")
    if mc.block_bootstrap:          hist_parts.append("block bootstrap")
    mode_label = (", ".join(hist_parts) if hist_parts
                  else f"Normal(σ_mkt={mc.market_return_std:.0%}, σ_inf={mc.inflation_std:.0%})")
    fig_mc.update_layout(
        title=f"Net Worth Distribution — {n_sims:,} Simulations ({mode_label})",
        **PLOTLY_DARK, height=460, yaxis_title="Net Worth ($)", xaxis_title="Year",
    )
    st.plotly_chart(fig_mc, width='stretch')


def _mc_liquidity_warning(mc) -> None:
    """Error / warning / success banner summarising liquidity risk across sims."""
    high_risk_years = [(yr, p) for yr, p in zip(mc.years, mc.prob_negative_liquid) if p > 0.10]
    if high_risk_years:
        first_yr = high_risk_years[0][0]
        peak_yr, peak_p = max(high_risk_years, key=lambda t: t[1])
        count = len(high_risk_years)
        span = (
            f"Year {first_yr}" if count == 1
            else f"{count} of your projected years, starting in Year {first_yr}"
        )
        st.error(
            f"⚠️ **Significant liquidity risk detected.** In more than 10% of simulations, "
            f"liquid assets go negative in {span} — peaking at Year {peak_yr} ({peak_p:.0%}). "
            f"Consider building a larger cash buffer or reducing fixed expenses."
        )
    elif any(p > 0 for p in mc.prob_negative_liquid):
        st.warning(
            "⚠️ **Low but non-zero liquidity risk.** Some simulations produce negative "
            "liquid assets in at least one year. Your plan is resilient but not bulletproof."
        )
    else:
        st.success("✅ **No liquidity risk.** Liquid assets stayed positive in all simulations.")


def _mc_failsafe_metrics(plan, mc) -> None:
    """One tile per failsafe: the % of simulations in which it fired.

    Makes it obvious at a glance whether a failsafe is actually triggering on
    the current numbers (0% = the condition is never hit; a high % = it is
    doing a lot of the work holding up the downside)."""
    if not plan.failsafes:
        return
    st.markdown("**Failsafes** — share of simulations in which each one triggered")
    cols = st.columns(min(len(plan.failsafes), 4))
    for i, fs in enumerate(plan.failsafes):
        rate = mc.failsafe_fire_rates.get(fs.name, 0.0)
        help_txt = (
            "Never triggered in these simulations — the trigger condition wasn't "
            "met on any path. Check the threshold and year window."
            if rate == 0 else
            "Fraction of simulated futures in which this failsafe's condition was "
            "met and its action kicked in."
        )
        cols[i % len(cols)].metric(f"🛟 {fs.name}", f"{rate:.0%}", help=help_txt)


@st.cache_data(show_spinner=False)
def _cached_monte_carlo(plan, **params):
    """Memoize the ~5s simulation so re-selecting the Monte Carlo tab with an
    unchanged plan/params is instant. Keyed on plan + params; only ever called
    with a fixed seed (unseeded runs bypass this — see call site)."""
    return ProjectionEngine(plan).run_monte_carlo(**params)


def _tab_monte_carlo(plan, snapshots, projection_engine) -> None:
    st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        "Runs N simulations with randomized annual shocks. "
        "**p10 / p50 / p90** = the 10th, 50th (median), and 90th percentile outcomes — "
        "p10 is a bad-luck scenario, p50 is the middle outcome, p90 is a good-luck scenario. "
        "**Liquidity risk** = probability of brokerage + cash going negative in a given year "
        "(having to liquidate retirement accounts or take on debt). "

        "**Historical mode** (recommended): market returns and inflation are sampled from "
        "~100 years of actuals (S&P 500 1926–2025, CPI 1929–2024) — preserving fat tails, "
        "crash years, and boom years as they really happened. With **joint block bootstrap** "
        "on, the two are drawn together in multi-year blocks so stagflation stays bundled and "
        "multi-year regimes are preserved, and salary growth tracks the sampled inflation. "
        "Shows the full range of outcomes including liquidity risk."
    )

    # ── Simulation parameters ────────────────────────────────
    p = _mc_simulation_params()
    n_sims = p["n_sims"]
    params = dict(
        n_simulations=n_sims, seed=42 if p["mc_seed"] else None,
        use_historical_returns=p["use_hist"], use_historical_inflation=p["use_hist_inf"],
        market_return_std=p["mkt_std"], inflation_std=p["inf_std"],
        salary_growth_std=p["sg_std"],
        block_bootstrap=p["block_bs"], mean_block_years=p["mean_block"],
    )
    with st.spinner(f"Running {n_sims:,} simulations…"):
        # Unseeded (non-reproducible) runs bypass the cache so each run reshuffles.
        mc = (_cached_monte_carlo(plan, **params) if p["mc_seed"]
              else ProjectionEngine(plan).run_monte_carlo(**params))

    # ── Summary KPIs ─────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Median Net Worth (Final Year)", fmt_dollar(mc.p50_net_worth[-1]))
    mc2.metric("Best 10% (p90)", fmt_dollar(mc.p90_net_worth[-1]))
    mc3.metric("Worst 10% (p10)", fmt_dollar(mc.p10_net_worth[-1]))
    worst_liq_prob = max(mc.prob_negative_liquid)
    worst_liq_yr   = mc.years[mc.prob_negative_liquid.index(worst_liq_prob)]
    mc4.metric(
        "Peak Liquidity Risk (worst-year chance of running out of liquid cash)",
        f"{worst_liq_prob:.1%}",
        delta=f"worst in year {worst_liq_yr}",
        delta_color="inverse",
    )

    _mc_failsafe_metrics(plan, mc)

    # ── Net worth fan chart ───────────────────────────────────
    years_mc = mc.years
    _mc_networth_fan_chart(mc, snapshots, n_sims)

    # ── Liquidity risk ────────────────────────────────────────
    st.markdown("#### Liquidity Risk — Probability of Negative Liquid Assets by Year")
    st.caption(
        "Each bar shows the fraction of simulations where your brokerage balance "
        "went negative in that year — meaning you ran out of accessible cash "
        "and would need to liquidate retirement accounts or take on debt."
    )

    # Colour bars by severity
    liq_colors = [
        "#ef4444" if p > 0.20
        else "#f97316" if p > 0.10
        else "#fbbf24" if p > 0.05
        else "#4ade80"
        for p in mc.prob_negative_liquid
    ]
    fig_liq = go.Figure(go.Bar(
        x=years_mc,
        y=[p * 100 for p in mc.prob_negative_liquid],
        marker_color=liq_colors,
        hovertemplate="Year %{x}: %{y:.1f}% of simulations went negative<extra></extra>",
    ))
    # Zero-risk reference line
    fig_liq.add_hline(y=0, line=dict(color="#374151", width=1))
    fig_liq.update_layout(
        **PLOTLY_DARK, height=280,
        yaxis_title="% of simulations",
        xaxis_title="Year",
        yaxis_ticksuffix="%",
        showlegend=False,
    )
    st.plotly_chart(fig_liq, width='stretch')

    # Warn if any year has >10% liquidity risk
    _mc_liquidity_warning(mc)

    # ── Liquid assets fan chart ───────────────────────────────
    st.markdown("#### Liquid Assets (Brokerage) Distribution")
    fig_liq_fan = go.Figure()
    _add_band(fig_liq_fan, years_mc, mc.p90_liquid, mc.p10_liquid, "rgba(245,158,11,0.12)", "p10–p90 band")
    _add_line(fig_liq_fan, years_mc, mc.p50_liquid, "Median liquid", "#f59e0b", 2)
    _add_line(fig_liq_fan, years_mc, mc.p10_liquid, "Pessimistic (p10)", "#f87171", 1.5, "dot")
    _add_line(fig_liq_fan, years_mc, [s.brokerage_balance for s in snapshots],
              "Deterministic", "#4ade80", 1.5, "dash")
    # Zero line — going below this means illiquid
    fig_liq_fan.add_hline(
        y=0, line=dict(color="#ef4444", width=1.5, dash="dot"),
        annotation_text="Illiquid threshold", annotation_position="right",
        annotation_font=dict(color="#ef4444", size=9),
    )
    fig_liq_fan.update_layout(
        **PLOTLY_DARK, height=320,
        yaxis_title="Brokerage Balance ($)", xaxis_title="Year",
        yaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_liq_fan, width='stretch')

    # ── Final-year percentile bar ─────────────────────────────
    st.markdown("#### Net Worth Percentiles at Final Year")
    pct_labels = ["p10", "p25", "p50", "p75", "p90"]
    pct_values = [
        mc.p10_net_worth[-1], mc.p25_net_worth[-1],
        mc.p50_net_worth[-1], mc.p75_net_worth[-1],
        mc.p90_net_worth[-1],
    ]
    fig_hist = go.Figure(go.Bar(
        x=pct_labels, y=pct_values,
        marker_color=["#f87171", "#fb923c", "#3b82f6", "#34d399", "#4ade80"],
        text=[fmt_dollar(v) for v in pct_values],
        textposition="outside",
    ))
    fig_hist.update_layout(
        title=f"Net Worth Percentiles at Year {plan.projection_years}",
        **PLOTLY_DARK, height=300, yaxis_title="$", showlegend=False,
    )
    st.plotly_chart(fig_hist, width='stretch')

    col_prob1, col_prob2 = st.columns(2)
    col_prob1.metric("Probability of $1M+ (Year 10)", f"{mc.prob_millionaire_10yr:.1%}")
    col_prob2.metric("Simulations Run", f"{mc.num_simulations:,}")


def render_dashboard(plan: FinancialPlan) -> None:
    # ── Single source of truth ────────────────────────────────────────────────
    # All financial figures flow from one place:
    #   1. ProjectionEngine runs first — applies year-1 events (marriage, home
    #      purchase, etc.) before computing taxes, so filing status, HSA tier,
    #      and family size are all correct.
    #   2. snapshots[0] is the authoritative year-1 result. Every tab reads
    #      from it — no independent tax recalculation elsewhere.
    #   3. tax_engine.calculate() is called exactly ONCE, using the year-1
    #      state from snapshots[0], to get the detailed breakdown (fed/FICA/
    #      state split) that the Cash Flow and Waterfall charts need.
    #   4. mortgage_calc is constructed once, only for the Mortgage tab display.
    # ─────────────────────────────────────────────────────────────────────────

    tax_engine = TaxEngine()
    strategy_engine = StrategyEngine()
    projection_engine = ProjectionEngine(plan)

    snapshots = projection_engine.run_deterministic()
    yr1 = snapshots[0]

    # Derive year-1 income profile from the snapshot (events already applied)
    yr1_income_profile = IncomeProfile(
        gross_annual_income=yr1.gross_income,
        filing_status=yr1.filing_status,
        state=plan.income.state,
        other_state_flat_rate=plan.income.other_state_flat_rate,
    )
    yr1_inv_profile = InvestmentProfile(
        annual_hsa_contribution=yr1.annual_hsa_contributions,
        annual_401k_contribution=yr1.annual_retirement_contributions,
    )

    # ONE tax calculation — used by all tabs
    tax_result = tax_engine.calculate(
        yr1_income_profile, yr1_inv_profile, plan.strategies,
        num_children=yr1.num_children,
    )
    # Combined federal + state marginal rate for the year-1 tax picture.
    marginal_rate = tax_engine.marginal_rate(
        yr1_income_profile, yr1_inv_profile, plan.strategies,
    )

    # Strategy analysis uses the same year-1 state
    strategy_result = strategy_engine.analyze(
        yr1_income_profile, yr1_inv_profile, plan.strategies,
        num_children=yr1.num_children,
    )

    # Monthly figures derived from year-1 snapshot — divide annual by 12
    monthly_net       = yr1.net_income / 12
    monthly_housing   = yr1.annual_housing_cost / 12
    monthly_lifestyle = yr1.annual_lifestyle_cost / 12
    monthly_breathing = yr1.annual_breathing_room / 12

    # Mortgage calculator — constructed once, only used for Tab 2 detail display
    mortgage_calc = (
        MortgageCalculator(plan.housing, plan.investments.annual_home_appreciation_rate)
        if not plan.housing.is_renting else None
    )

    # ── Header ───────────────────────────────────────────────
    st.markdown("# 📈 fintracker")
    st.markdown("*Personal long-term financial planning — tax-aware, scenario-driven, Monte Carlo enabled.*")
    st.divider()

    # ── Top KPIs ─────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("Monthly Take-Home", fmt_dollar(monthly_net)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Monthly Housing", fmt_dollar(monthly_housing)), unsafe_allow_html=True)
    with c3:
        delta_sign = monthly_breathing >= 0
        st.markdown(metric_card(
            "Monthly Breathing Room", fmt_dollar(monthly_breathing),
            delta=("▲ Positive cash flow" if delta_sign else "▼ Cash flow deficit"),
            positive=delta_sign,
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "Tax Strategy Savings", fmt_dollar(strategy_result.total_annual_savings) + "/yr",
        ), unsafe_allow_html=True)
    with c5:
        final_nw = snapshots[-1].net_worth
        st.markdown(metric_card(
            f"Net Worth (Yr {plan.projection_years})", fmt_dollar(final_nw),
        ), unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ─────────────────────────────────────────────────
    # A segmented control (not st.tabs) so ONLY the active view's body runs each
    # rerun. st.tabs executes every tab body on every rerun, which forced the ~5s
    # Monte Carlo simulation to run even while the user was on Projections. Here
    # the heavy MC only runs when its view is actually selected.
    tab_renderers = {
        "📈 Projections":   lambda: _tab_projections(plan=plan, snapshots=snapshots, projection_engine=projection_engine),
        "🎲 Monte Carlo":   lambda: _tab_monte_carlo(plan=plan, snapshots=snapshots, projection_engine=projection_engine),
        "💰 Cash Flow":     lambda: _tab_cash_flow(plan=plan, tax_result=tax_result, monthly_housing=monthly_housing, monthly_lifestyle=monthly_lifestyle, monthly_breathing=monthly_breathing, mortgage_calc=mortgage_calc),
        "🏠 Mortgage":      lambda: _tab_mortgage(plan=plan, mortgage_calc=mortgage_calc),
        "🎯 Tax Strategies": lambda: _tab_tax(plan=plan, snapshots=snapshots, strategy_result=strategy_result, yr1=yr1, tax_result=tax_result, marginal_rate=marginal_rate),
    }
    labels = list(tab_renderers)
    active = st.segmented_control("View", labels, default=labels[0], label_visibility="collapsed") or labels[0]
    tab_renderers[active]()

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def _auto_load_personal_config() -> None:
    """
    On first page load, look for personal.yaml next to app.py and
    pre-populate session_state so the sidebar reflects the user's real numbers.
    Only runs once per session; manual uploads/changes override it.
    """
    if "loaded_plan" in st.session_state:
        return  # already loaded (either auto or by user upload)

    app_dir = pathlib.Path(__file__).parent
    candidates = [
        app_dir / "config" / "personal.yaml",
        app_dir / "config" / "personal.yml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        from fintracker.config import load_plan
        try:
            st.session_state["loaded_plan"] = load_plan(path)
        except Exception:
            pass  # malformed YAML — fall through to defaults silently
        return


def main():
    _auto_load_personal_config()
    plan = build_sidebar()
    render_dashboard(plan)


if __name__ == "__main__":
    main()