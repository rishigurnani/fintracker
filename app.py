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
    InvestmentProfile, StrategyToggles, FinancialPlan, TimelineEvent,
)
from fintracker.tax_engine import TaxEngine
from fintracker.mortgage import MortgageCalculator
from fintracker.strategies import StrategyEngine
from fintracker.projections import ProjectionEngine
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

def build_sidebar() -> FinancialPlan:
    st.sidebar.title("⚙️ Configure Your Plan")
    section = st.sidebar.radio(
        "section",
        ["💵 Income", "🏠 Housing", "🌿 Lifestyle", "📊 Investments",
         "🎯 Strategies", "🚗 Car", "🏢 Business", "🗓️ Events"],
        label_visibility="collapsed",
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

    # --- Load from file ---
    with st.sidebar.expander("📂 Load Config", expanded=False):
        uploaded = st.file_uploader("Load YAML config", type=["yaml", "yml"], label_visibility="collapsed")
        if uploaded:
            import yaml, tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            from fintracker.config import load_plan
            try:
                st.session_state["loaded_plan"] = load_plan(tmp_path)
                st.success("Config loaded!")
            except Exception as e:
                st.error(f"Error loading config: {e}")
            finally:
                os.unlink(tmp_path)

    # Default starting values (from loaded plan or defaults)
    defaults = st.session_state.get("loaded_plan", None)
    d_inc = defaults.income if defaults else None
    d_hou = defaults.housing if defaults else None
    d_lif = defaults.lifestyle if defaults else None
    d_inv = defaults.investments if defaults else None
    d_str = defaults.strategies if defaults else None

    # Pre-initialise all plan components from the loaded plan (or defaults).
    # Each section block may override its own variable; variables for
    # sections not currently shown stay as these pass-through values so
    # FinancialPlan() below always has all required arguments.
    income      = defaults.income      if defaults else IncomeProfile()
    housing     = defaults.housing     if defaults else HousingProfile(home_price=0, down_payment=0, interest_rate=0.0)
    lifestyle   = defaults.lifestyle   if defaults else LifestyleProfile()
    investments = defaults.investments if defaults else InvestmentProfile()
    strategies  = defaults.strategies  if defaults else StrategyToggles()
    car         = defaults.car         if defaults else None
    business    = defaults.business    if defaults else None
    events: list[TimelineEvent] = defaults.timeline_events if defaults else []
    # Cross-section scalar values — defined here so every section can reference them
    gross  = int(defaults.income.gross_annual_income)             if defaults else 120_000
    spouse = int(defaults.income.spouse_gross_annual_income)      if defaults else 0
    employer_match = defaults.investments.employer_match          if defaults else None

    if section == "💵 Income":
        # ── Income ───────────────────────────────────────────────
        st.sidebar.header("💵 Income")
        gross = st.sidebar.number_input(
            "Gross Annual Income  (own rate)",
            min_value=0, max_value=5_000_000,
            value=int(d_inc.gross_annual_income) if d_inc else 120_000, step=5_000,
        )
        spouse = st.sidebar.number_input(
            "Spouse Gross Income  (own rate)",
            min_value=0, max_value=5_000_000,
            value=int(d_inc.spouse_gross_annual_income) if d_inc else 0, step=5_000,
        )
        filing = st.sidebar.selectbox(
            "Filing Status",
            options=[f.value for f in FilingStatus],
            index=[f.value for f in FilingStatus].index(d_inc.filing_status.value) if d_inc else 0,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        state_options = [s.value for s in State]
        state_val = st.sidebar.selectbox(
            "State",
            options=state_options,
            index=state_options.index(d_inc.state.value) if d_inc else state_options.index("GA"),
        )
        other_rate = 0.05
        if state_val == "OTHER":
            other_rate = st.sidebar.slider("State Flat Tax Rate (%)", 0.0, 15.0, 5.0, 0.1) / 100

        income = IncomeProfile(
            gross_annual_income=float(gross),
            spouse_gross_annual_income=float(spouse),
            filing_status=FilingStatus(filing),
            state=State(state_val),
            other_state_flat_rate=other_rate,
        )


    if section == "🏠 Housing":
        # ── Housing ──────────────────────────────────────────────
        st.sidebar.header("🏠 Housing")
        is_renting = st.sidebar.toggle("I'm Renting", value=d_hou.is_renting if d_hou else False)

        if is_renting:
            monthly_rent = st.sidebar.number_input(
                "Monthly Rent  (own rate)", min_value=0, max_value=20_000,
                value=int(d_hou.monthly_rent) if d_hou else 2_000, step=100,
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
                value=int(d_hou.home_price) if d_hou else 400_000, step=10_000,
            )
            down_pmt = st.sidebar.number_input(
                "Down Payment  (current)", min_value=0, max_value=home_price,
                value=min(int(d_hou.down_payment) if d_hou else 80_000, home_price), step=5_000,
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
                annual_property_tax_rate=float(d_hou.annual_property_tax_rate) if d_hou else 0.012,
                annual_insurance=float(d_hou.annual_insurance) if d_hou else 2_000,
            )


    if section == "🌿 Lifestyle":
        # ── Lifestyle ────────────────────────────────────────────
        st.sidebar.header("🌿 Lifestyle")
        num_children = st.sidebar.number_input(
            "Current Children", min_value=0, max_value=10,
            value=int(d_lif.num_children) if d_lif else 0,
        )
        d_cp = d_lif.childcare_profile if d_lif else None
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
                value=int(d_lif.monthly_childcare) if d_lif else 0,
                step=100,
                help="Single rate applied to every child, every year. Inflated annually.",
            )
        else:
            monthly_childcare = 0  # unused when profile is set
            st.sidebar.caption(
                "Define monthly costs per child at each age. "
                "Ages not covered default to $0. Costs inflate annually."
            )
            _default_phases = [
                (0,  2,  int(d_cp.phases[0].monthly_cost) if d_cp and len(d_cp.phases) > 0 else 2_500),
                (3,  4,  int(d_cp.phases[1].monthly_cost) if d_cp and len(d_cp.phases) > 1 else 1_500),
                (5,  12, int(d_cp.phases[2].monthly_cost) if d_cp and len(d_cp.phases) > 2 else 600),
                (13, 17, int(d_cp.phases[3].monthly_cost) if d_cp and len(d_cp.phases) > 3 else 150),
            ]
            _phase_labels = ["Infant/Toddler (0–2)", "Preschool (3–4)",
                             "School-age (5–12)", "Teen (13–17)"]
            phases = []
            for (a_start, a_end, default_cost), label in zip(_default_phases, _phase_labels):
                cost = st.sidebar.number_input(
                    f"{label} – monthly cost/child ($)",
                    min_value=0, max_value=15_000,
                    value=default_cost, step=50,
                    key=f"cc_{a_start}_{a_end}",
                )
                phases.append(ChildcarePhase(age_start=a_start, age_end=a_end, monthly_cost=float(cost)))
            childcare_profile = ChildcareProfile(phases=phases)
        num_pets = st.sidebar.number_input(
            "Pets", min_value=0, max_value=10,
            value=int(d_lif.num_pets) if d_lif else 0,
        )
        annual_pet = 0.0
        if num_pets > 0:
            annual_pet = st.sidebar.number_input(
                "Annual Pet Cost  (inflated yearly)", min_value=0, max_value=20_000,
                value=int(d_lif.annual_pet_cost) if d_lif else 1_800, step=100,
            )
        medical = st.sidebar.number_input(
            "Annual Medical OOP  (inflated yearly)", min_value=0, max_value=50_000,
            value=int(d_lif.annual_medical_oop) if d_lif else 3_000, step=500,
        )
        vacation = st.sidebar.number_input(
            "Annual Vacation  (inflated yearly)", min_value=0, max_value=100_000,
            value=int(d_lif.annual_vacation) if d_lif else 5_000, step=1_000,
        )
        other_monthly = st.sidebar.number_input(
            "Other Monthly  (inflated yearly)", min_value=0, max_value=10_000,
            value=int(d_lif.monthly_other_recurring) if d_lif else 500, step=100,
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
            annual_vacation=float(vacation),
            monthly_other_recurring=float(other_monthly),
        )


    if section == "📊 Investments":
        # ── Investments ──────────────────────────────────────────
        st.sidebar.header("📊 Investments & Savings")
        liquid_cash = st.sidebar.number_input(
            "Current Liquid Cash  (current)", min_value=0, max_value=10_000_000,
            value=int(d_inv.current_liquid_cash) if d_inv else 100_000, step=5_000,
        )
        retirement_bal = st.sidebar.number_input(
            "Current Retirement Balance  (current)", min_value=0, max_value=10_000_000,
            value=int(d_inv.current_retirement_balance) if d_inv else 0, step=5_000,
        )
        brokerage_bal = st.sidebar.number_input(
            "Current Brokerage / Taxable Balance  (current)", min_value=0, max_value=10_000_000,
            value=int(d_inv.current_brokerage_balance) if d_inv else 0, step=5_000,
            help="Existing taxable investment accounts (separate from 401k/IRA/HSA).",
        )
        one_time = st.sidebar.number_input(
            "Upcoming One-Time Expenses  (current)", min_value=0, max_value=1_000_000,
            value=int(d_inv.one_time_upcoming_expenses) if d_inv else 0, step=5_000,
            help="Wedding, car purchase, etc. Subtracted from investable cash.",
        )
        _401k_mode = st.sidebar.radio(
            "401k contribution input mode",
            ["$ amount", "% of salary"],
            horizontal=True,
            label_visibility="collapsed",
        )
        _IRS_401K_LIMIT = 30_500
        if _401k_mode == "% of salary":
            _k401_pct = st.sidebar.slider(
                "Your 401k (% of gross salary)", 0.0, 30.0,
                round(d_inv.annual_401k_contribution / float(gross) * 100, 1) if (d_inv and gross > 0) else 6.0,
                0.5,
                help=f"Will be capped at the IRS limit (${_IRS_401K_LIMIT:,}) automatically.",
            )
            k401 = min(float(gross) * _k401_pct / 100, _IRS_401K_LIMIT)
            st.sidebar.caption(f"= ${k401:,.0f}/yr")
        else:
            k401 = st.sidebar.number_input(
                "Your Annual 401k Contribution  (fixed)", min_value=0, max_value=_IRS_401K_LIMIT,
                value=int(d_inv.annual_401k_contribution) if d_inv else 23_000, step=500,
            )
        partner_k401 = 0
        if spouse > 0:
            _pk401_mode = st.sidebar.radio(
                "Partner 401k input mode",
                ["$ amount", "% of salary"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if _pk401_mode == "% of salary":
                _pk401_pct = st.sidebar.slider(
                    "Partner 401k (% of gross salary)", 0.0, 30.0,
                    round(d_inv.partner_annual_401k_contribution / float(spouse) * 100, 1) if (d_inv and spouse > 0) else 6.0,
                    0.5,
                    help=f"Will be capped at the IRS limit (${_IRS_401K_LIMIT:,}) automatically.",
                )
                partner_k401 = min(float(spouse) * _pk401_pct / 100, _IRS_401K_LIMIT)
                st.sidebar.caption(f"= ${partner_k401:,.0f}/yr")
            else:
                partner_k401 = st.sidebar.number_input(
                    "Partner Annual 401k Contribution  (fixed)", min_value=0, max_value=_IRS_401K_LIMIT,
                    value=int(d_inv.partner_annual_401k_contribution) if d_inv else 0, step=500,
                    help="Partner's independent 401k — each person has their own IRS limit.",
                )
        hsa = st.sidebar.number_input(
            "Annual HSA Contribution  (fixed)", min_value=0, max_value=8_300,
            value=int(d_inv.annual_hsa_contribution) if d_inv else 4_150, step=100,
        )
        c529 = st.sidebar.number_input(
            "Annual 529 Contribution per Child  (fixed)", min_value=0, max_value=50_000,
            value=int(d_inv.annual_529_contribution) if d_inv else 0, step=500,
        )

        # ── Employer 401k Match ─────────────────────────────────
        st.sidebar.subheader("🏦 Employer 401k Match")
        d_em = d_inv.employer_match if d_inv else None
        has_match = st.sidebar.toggle("Employer offers 401k match", value=d_em is not None)
        employer_match = None
        if has_match:
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
                    value=int(d_em.vesting_years) if d_em else 0,
                    help="Cliff vesting: match forfeited if you leave before this year.",
                )
                em_ps = st.number_input(
                    "Profit sharing per year  (fixed)", min_value=0, max_value=100_000,
                    value=int(d_em.profit_sharing_annual) if d_em else 0, step=500,
                    help="Flat employer contribution regardless of your own contribution.",
                )
            employer_match = EmployerMatch(
                tiers=tiers,
                annual_cap=float(em_cap) if em_cap > 0 else None,
                vesting_years=int(em_vest),
                profit_sharing_annual=float(em_ps),
            )

        investments = InvestmentProfile(
            current_liquid_cash=float(liquid_cash),
            current_retirement_balance=float(retirement_bal),
            current_brokerage_balance=float(brokerage_bal),
            one_time_upcoming_expenses=float(one_time),
            annual_401k_contribution=float(k401),
            partner_annual_401k_contribution=float(partner_k401),
            annual_hsa_contribution=float(hsa),
            annual_529_contribution=float(c529),
            annual_market_return=float(st.sidebar.slider("Market Return (%)", 0.0, 15.0, 8.0, 0.5)) / 100,
            annual_inflation_rate=float(st.sidebar.slider("Inflation (%)", 0.0, 10.0, 3.0, 0.25)) / 100,
            annual_salary_growth_rate=float(st.sidebar.slider("Your Salary Growth (%)", 0.0, 15.0,
                float(d_inv.annual_salary_growth_rate * 100) if d_inv else 4.0, 0.5)) / 100,
            partner_salary_growth_rate=float(st.sidebar.slider("Partner Salary Growth (%)", 0.0, 15.0,
                float(d_inv.partner_salary_growth_rate * 100) if d_inv else 4.0, 0.5)) / 100
                if spouse > 0 else 0.04,
            annual_home_appreciation_rate=float(st.sidebar.slider("Home Appreciation (%)", 0.0, 10.0, 3.5, 0.5)) / 100,
            auto_invest_surplus=st.sidebar.toggle(
                "Auto-Invest Surplus",
                value=d_inv.auto_invest_surplus if d_inv else True,
                help="ON: surplus swept into brokerage each year (earns market return). "
                     "OFF: surplus stays in cash (0% return).",
            ),
            cash_buffer_months=st.sidebar.slider(
                "Cash Buffer (months of expenses)",
                min_value=0.0, max_value=24.0,
                value=float(d_inv.cash_buffer_months) if d_inv else 0.0,
                step=1.0,
                help="Keep this many months of living expenses as liquid cash (0% return) "
                     "before sweeping surplus to brokerage. Reduces liquidity risk in bad years. "
                     "Set to 0 to invest all surplus (default).",
            ),
            employer_match=employer_match,
        )
        # Preserve fields not exposed in sidebar (partner_salary_growth_rate when solo,
        # annual_roth_ira_contribution, annual_brokerage_contribution)
        _inv_base = d_inv if d_inv else InvestmentProfile()
        investments = dataclasses.replace(
            investments,
            annual_roth_ira_contribution=_inv_base.annual_roth_ira_contribution,
            annual_brokerage_contribution=_inv_base.annual_brokerage_contribution,
            # partner_salary_growth_rate: sidebar only shows it when spouse > 0;
            # preserve the loaded value when spouse income is 0 at sidebar time
            partner_salary_growth_rate=(
                investments.partner_salary_growth_rate
                if spouse > 0 else _inv_base.partner_salary_growth_rate
            ),
        )


    if section == "🎯 Strategies":
        # ── Strategies ───────────────────────────────────────────
        st.sidebar.header("🎯 Tax Strategies")
        _str_base = d_str if d_str else StrategyToggles()
        strategies = dataclasses.replace(
            _str_base,
            maximize_hsa=st.sidebar.toggle("Maximize HSA", value=d_str.maximize_hsa if d_str else True),
            maximize_401k=st.sidebar.toggle("Maximize 401k", value=d_str.maximize_401k if d_str else True),
            use_529_state_deduction=st.sidebar.toggle("Use 529 State Deduction", value=d_str.use_529_state_deduction if d_str else False),
            use_roth_ladder=st.sidebar.toggle("Roth Conversion Ladder", value=d_str.use_roth_ladder if d_str else False),
        )


    if section == "🚗 Car":
        # ── Car ─────────────────────────────────────────────────
        st.sidebar.header("🚗 Car")
        d_car = defaults.car if defaults else None
        has_car = st.sidebar.toggle("Model car purchases", value=d_car is not None)
        car = None
        if has_car:
            car = CarProfile(
                car_price=st.sidebar.number_input(
                    "Car price  (inflated yearly)", min_value=0, max_value=200_000,
                    value=int(d_car.car_price) if d_car else 25_000, step=1_000,
                ),
                down_payment=st.sidebar.number_input(
                    "Down payment  (current)", min_value=0, max_value=100_000,
                    value=int(d_car.down_payment) if d_car else 5_000, step=500,
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
                    value=int(d_car.residual_value) if d_car else 5_000, step=500,
                    help="Amount received when selling the old car if no child is old enough to receive it.",
                ),
                hand_down_age=st.sidebar.number_input(
                    "Hand-down age (child)", min_value=14, max_value=25,
                    value=int(d_car.hand_down_age) if d_car else 16, step=1,
                    help="Minimum child age to receive the handed-down car instead of selling it.",
                ),
                num_cars=st.sidebar.selectbox(
                    "Number of cars", [1, 2, 3],
                    index=(d_car.num_cars - 1) if d_car else 0,
                ),
            )


    if section == "🏢 Business":
        # ── Business ─────────────────────────────────────────────
        st.sidebar.header("🏢 Business")
        d_biz = defaults.business if defaults else None
        has_business = st.sidebar.toggle("Model business ownership", value=d_biz is not None)
        business = None
        if has_business:
            with st.sidebar.expander("Business parameters", expanded=True):
                biz_revenue = st.number_input(
                    "Annual gross revenue  (own rate)", min_value=0, max_value=10_000_000,
                    value=int(d_biz.annual_revenue) if d_biz else 200_000, step=5_000,
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
                    value=int(d_biz.start_year) if d_biz else 1,
                    help="Projection year the business starts generating income.",
                )
                biz_invest = st.number_input(
                    "Initial investment  (current)", min_value=0, max_value=5_000_000,
                    value=int(d_biz.initial_investment) if d_biz else 0, step=5_000,
                    help="One-time acquisition/startup cost drawn from brokerage in start year.",
                )
                biz_equity_mult = st.slider(
                    "Equity multiple", 0.0, 10.0,
                    float(d_biz.equity_multiple) if d_biz else 3.0, 0.5,
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
                    value=d_biz.use_qbi_deduction if d_biz else True,
                    help="20% deduction on qualified business income for pass-through entities.",
                )
                biz_health = st.number_input(
                    "Self-employed health insurance  (fixed)", min_value=0, max_value=50_000,
                    value=int(d_biz.self_employed_health_insurance) if d_biz else 0, step=500,
                    help="Annual premium — 100% deductible from AGI for self-employed.",
                )
                biz_solo_k = st.number_input(
                    "Solo 401k contribution  (fixed)", min_value=0, max_value=69_000,
                    value=int(d_biz.solo_401k_contribution) if d_biz else 0, step=500,
                    help="Owner solo 401k (up to $69k IRS limit). Tracked in retirement balance.",
                )
                biz_sep = st.number_input(
                    "SEP-IRA contribution  (fixed)", min_value=0, max_value=69_000,
                    value=int(d_biz.sep_ira_contribution) if d_biz else 0, step=500,
                    help="SEP-IRA (up to 25% of net self-employment income).",
                )
                biz_ownership = st.slider(
                    "Your ownership share (%)", 1.0, 100.0,
                    float((d_biz.ownership_pct if d_biz else 1.0) * 100), 1.0,
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


    if section == "🗓️ Events":
        # ── Timeline Events ──────────────────────────────────────
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
                yr = st.number_input(
                    "Year", min_value=1, max_value=50,
                    value=int(ev_def.year) if ev_def else 1,
                    key=f"ev_yr_{i}",
                )
                desc = st.text_input(
                    "Description",
                    value=ev_def.description if ev_def else "",
                    key=f"ev_desc_{i}",
                )
                ev_marriage = st.checkbox(
                    "Marriage (→ MFJ filing)",
                    value=ev_def.marriage if ev_def else False,
                    key=f"ev_marry_{i}",
                )
                ev_child = st.checkbox(
                    "New child",
                    value=ev_def.new_child if ev_def else False,
                    key=f"ev_child_{i}",
                )
                ev_pet = st.checkbox(
                    "New pet",
                    value=ev_def.new_pet if ev_def else False,
                    key=f"ev_pet_{i}",
                )

                st.markdown("**Work changes**")
                ev_stop = st.checkbox(
                    "You stop working",
                    value=ev_def.stop_working if ev_def else False,
                    key=f"ev_stop_{i}",
                )
                ev_resume = st.checkbox(
                    "You resume working",
                    value=ev_def.resume_working if ev_def else False,
                    key=f"ev_resume_{i}",
                )
                ev_partner_stop = st.checkbox(
                    "Partner stops working",
                    value=ev_def.partner_stop_working if ev_def else False,
                    key=f"ev_pstop_{i}",
                )
                ev_partner_resume = st.checkbox(
                    "Partner resumes working",
                    value=ev_def.partner_resume_working if ev_def else False,
                    key=f"ev_presume_{i}",
                )

                ev_start_care = st.checkbox(
                    "Start parent care",
                    value=ev_def.start_parent_care if ev_def else False,
                    key=f"ev_startcare_{i}",
                    help="Activates annual_parent_care_cost from Lifestyle settings.",
                )
                ev_stop_care = st.checkbox(
                    "Stop parent care",
                    value=ev_def.stop_parent_care if ev_def else False,
                    key=f"ev_stopcare_{i}",
                )
                ev_birth_yr_override = st.number_input(
                    "Child birth year override (0 = this year)",
                    min_value=-30, max_value=0,
                    value=int(ev_def.child_birth_year_override) if (ev_def and ev_def.child_birth_year_override is not None) else 0,
                    key=f"ev_birthyr_{i}",
                    help="Set negative to indicate a child already born before the projection. "
                         "0 means born in this event's year (default).",
                )

                ev_income = st.number_input(
                    "Your new gross income (0 = no change)",
                    min_value=0, max_value=5_000_000,
                    value=int(ev_def.income_change) if (ev_def and ev_def.income_change) else 0,
                    key=f"ev_inc_{i}",
                )
                ev_partner_income = st.number_input(
                    "Partner new gross income (0 = no change)",
                    min_value=0, max_value=5_000_000,
                    value=int(ev_def.partner_income_change) if (ev_def and ev_def.partner_income_change) else 0,
                    key=f"ev_pinc_{i}",
                )
                ev_expense = st.number_input(
                    "One-time expense ($)",
                    min_value=0, max_value=1_000_000,
                    value=int(ev_def.extra_one_time_expense) if ev_def else 0,
                    key=f"ev_exp_{i}",
                )
                ev_bonus = st.number_input(
                    "One-time income ($)",
                    min_value=0, max_value=1_000_000,
                    value=int(ev_def.extra_one_time_income) if ev_def else 0,
                    key=f"ev_bonus_{i}",
                )
                # Home purchase fields — shown only when buy_home is set in YAML
                # or if the user explicitly toggles it on
                ev_buy_home = st.checkbox(
                    "Buy home",
                    value=ev_def.buy_home if ev_def else False,
                    key=f"ev_buyhome_{i}",
                )
                ev_new_home_price = None
                ev_new_home_down = None
                ev_new_home_rate = None
                ev_sell_current = True
                ev_buyer_closing = 0.02   # default; only overridden when ev_buy_home=True
                ev_seller_closing = 0.06  # default; only overridden when ev_buy_home=True
                if ev_buy_home:
                    ev_new_home_price = st.number_input(
                        "New home price ($)",
                        min_value=0, max_value=10_000_000,
                        value=int(ev_def.new_home_price) if (ev_def and ev_def.new_home_price) else 500_000,
                        key=f"ev_hp_{i}",
                    )
                    ev_new_home_down = st.number_input(
                        "Down payment  (current)",
                        min_value=0, max_value=int(ev_new_home_price),
                        value=int(ev_def.new_home_down_payment) if (ev_def and ev_def.new_home_down_payment) else int(ev_new_home_price * 0.20),
                        key=f"ev_hd_{i}",
                    )
                    ev_new_home_rate = st.slider(
                        "Mortgage rate (%)",
                        2.0, 12.0,
                        float(ev_def.new_home_interest_rate * 100) if (ev_def and ev_def.new_home_interest_rate) else 6.5,
                        0.125,
                        key=f"ev_hr_{i}",
                    )
                    ev_sell_current = st.checkbox(
                        "Sell current home (add equity to cash)",
                        value=ev_def.sell_current_home if ev_def else True,
                        key=f"ev_sell_{i}",
                    )
                    ev_buyer_closing = st.slider(
                        "Buyer closing costs (% of price)",
                        0.0, 5.0,
                        float(ev_def.buyer_closing_cost_rate * 100) if ev_def else 2.0,
                        0.25,
                        key=f"ev_bcc_{i}",
                        help="Title, lender fees, escrow, transfer tax — typically 1.5–3%",
                    ) / 100
                    ev_seller_closing = 0.06
                    if ev_sell_current:
                        ev_seller_closing = st.slider(
                            "Seller closing costs (% of sale price)",
                            0.0, 10.0,
                            float(ev_def.seller_closing_cost_rate * 100) if ev_def else 6.0,
                            0.25,
                            key=f"ev_scc_{i}",
                            help="Agent commissions, transfer tax — typically 5–7%",
                        ) / 100

                _ev_base = ev_def if ev_def else TimelineEvent(year=int(yr), description=desc)
                events.append(dataclasses.replace(
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
                    new_home_price=float(ev_new_home_price) if ev_new_home_price else None,
                    new_home_down_payment=float(ev_new_home_down) if ev_new_home_down else None,
                    new_home_interest_rate=float(ev_new_home_rate) / 100 if ev_new_home_rate else None,
                    sell_current_home=ev_sell_current,
                    buyer_closing_cost_rate=ev_buyer_closing,
                    seller_closing_cost_rate=ev_seller_closing,
                ))


    projection_years = st.sidebar.slider(
        "Projection Horizon (Years)", 5, 40,
        defaults.projection_years if defaults else 30,
    )

    plan = FinancialPlan(
        income=income, housing=housing, lifestyle=lifestyle,
        investments=investments, strategies=strategies,
        timeline_events=events, projection_years=int(projection_years),
        retirement=defaults.retirement if defaults else None,
        college=defaults.college if defaults else None,
        car=car,
        business=business,
    )

    # Save config button
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
            use_container_width=True,
        )

    return plan


# ─────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────

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
    tabs = st.tabs(["📈 Projections", "🎲 Monte Carlo", "💰 Cash Flow", "🏠 Mortgage", "🎯 Tax Strategies"])

    # ── TAB 3: Cash Flow ─────────────────────────────────────
    with tabs[2]:
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.dataframe(df_tax, hide_index=True, use_container_width=True)

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
    with tabs[3]:
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
            st.plotly_chart(fig_amort, use_container_width=True)

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
            st.plotly_chart(fig_pi, use_container_width=True)

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
                st.dataframe(df_sched, hide_index=True, use_container_width=True, height=400)

    # ── TAB 5: Tax Strategies ────────────────────────────────
    with tabs[4]:
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
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── TAB 1: Projections ───────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header">Long-Term Wealth Projection</div>', unsafe_allow_html=True)

        df = pd.DataFrame([{
            "Year": s.year,
            "Gross Income": s.gross_income,
            "Net Income": s.net_income,
            "Housing Cost": s.annual_housing_cost,
            "Lifestyle Cost": s.annual_lifestyle_cost,
            "Breathing Room": s.annual_breathing_room,
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
        st.plotly_chart(fig_nw, use_container_width=True)

        col_l, col_r = st.columns(2)

        with col_l:
            # Income vs costs + liquid assets on secondary axis
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
                x=df["Year"], y=df["Brokerage"], name="Liquid Assets (brokerage + cash, excl. retirement accounts)",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                yaxis="y2",
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.08)",
            ))

            # ── Total investable assets + retirement target ─────────────
            # Total investable = retirement + HSA + brokerage + cash (all accessible funds).
            total_investable = [
                s.retirement_balance + s.hsa_balance + s.brokerage_balance
                + s.uninvested_cash + s.cash_buffer
                for s in snapshots
            ]
            fig_cf.add_trace(go.Scatter(
                x=df["Year"], y=total_investable,
                name="Total Investable Assets (401k + HSA + brokerage + cash)",
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
            st.plotly_chart(fig_cf, use_container_width=True)

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
            st.plotly_chart(fig_br, use_container_width=True)

        # ── Liquidity warnings ───────────────────────────────────
        negative_years = [s for s in snapshots if s.brokerage_balance < 0]
        low_years = [s for s in snapshots if 0 <= s.brokerage_balance < 10_000]
        if negative_years:
            first = negative_years[0]
            worst = min(negative_years, key=lambda s: s.brokerage_balance)
            st.error(
                f"⚠️ **Liquid assets go negative in Year {first.year}** — "
                f"worst point is **{fmt_dollar(worst.brokerage_balance)}** in Year {worst.year}. "
                f"You would need to sell investments, take on debt, or reduce expenses to cover the shortfall."
            )
        elif low_years:
            first = low_years[0]
            st.warning(
                f"⚠️ **Liquid assets fall below $10,000 in Year {first.year}** "
                f"(lowest: **{fmt_dollar(min(s.brokerage_balance for s in low_years))}**). "
                f"Consider building a larger emergency buffer."
            )

        # Key milestones — deduplicated, dynamically laid out
        # ── Retirement Readiness Panel ────────────────────────────────
        if plan.retirement:
            rr_panel = projection_engine.compute_retirement_readiness(snapshots)
            if rr_panel:
                st.markdown("#### 🎯 Retirement Readiness")
                color   = "#4ade80" if rr_panel.on_track else "#ef4444"
                status  = "✅ On Track" if rr_panel.on_track else "⚠️ Off Track"
                funded  = f"{rr_panel.funded_pct:.0%}"
                gap_lbl = "Annual Surplus" if rr_panel.annual_surplus_or_gap >= 0 else "Annual Gap"
                gap_val = fmt_dollar(abs(rr_panel.annual_surplus_or_gap))
                gap_sign = "+" if rr_panel.annual_surplus_or_gap >= 0 else "-"

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Status", status)
                rc2.metric("Funded", funded,
                           delta=f"target {fmt_dollar(rr_panel.required_balance)}",
                           delta_color="normal" if rr_panel.on_track else "inverse")
                rc3.metric("Projected at Retirement", fmt_dollar(rr_panel.projected_balance_at_retirement))
                rc4.metric(gap_lbl, f"{gap_sign}{gap_val} /yr",
                           delta=f"over {plan.retirement.years_in_retirement}yr @ {plan.retirement.expected_post_retirement_return:.0%}")

                st.caption(
                    f"Retire at age {plan.retirement.retirement_age} (year {rr_panel.years_to_retirement}) · "
                    f"Desired income: {fmt_dollar(rr_panel.desired_income_nominal)}/yr (nominal) · "
                    + (f"SS offset: {fmt_dollar(rr_panel.social_security_offset)}/yr · " if rr_panel.social_security_offset > 0 else "")
                    + f"Post-retirement return: {plan.retirement.expected_post_retirement_return:.0%}"
                )
                st.divider()

        st.markdown("#### Key Milestones")

        # Each entry: (label, value, delta_note)
        # We scan snapshots once and record the FIRST year each threshold is crossed.
        # Thresholds are scaled to the projection horizon so they're always meaningful.
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

        milestones = []
        for _, label in nw_thresholds:
            if label in reached:
                s = reached[label]
                milestones.append((label, f"Year {s.year}", f"NW {fmt_dollar(s.net_worth)}"))

        # Add mortgage payoff milestone
        mortgage_paid = next((s for s in snapshots if s.mortgage_balance == 0 and not s.is_renting), None)
        if mortgage_paid:
            milestones.append(("🏠 Mortgage Paid Off", f"Year {mortgage_paid.year}", ""))

        # Add "debt-free" note if brokerage ever goes deeply negative
        worst_liquid = min(snapshots, key=lambda s: s.brokerage_balance)
        if worst_liquid.brokerage_balance < -50_000:
            milestones.append(("⚠️ Peak Cash Deficit", f"Year {worst_liquid.year}",
                               fmt_dollar(worst_liquid.brokerage_balance)))

        if not milestones:
            milestones.append(("Net Worth $1M", f"Not reached in {plan.projection_years} years", ""))

        # Render in rows of 3
        for row_start in range(0, len(milestones), 3):
            row = milestones[row_start:row_start + 3]
            cols = st.columns(len(row))
            for col, (label, value, delta) in zip(cols, row):
                col.metric(label, value, delta if delta else None)

        # Data table
        with st.expander("📋 Full Year-by-Year Projection Table"):
            import streamlit as _st

            # ── Annual Cash Flows (what happened this year) ──────────
            st.markdown("**Annual Cash Flows** — income, costs, and surplus for each year")
            flow_cols = ["Year", "Gross Income", "Taxes", "Net Income",
                         "Housing Cost", "Lifestyle Cost", "Breathing Room"]
            flow_df = df[flow_cols].copy()
            flow_df["Gross Income"]  = flow_df["Gross Income"].apply(fmt_dollar)
            flow_df["Net Income"]    = flow_df["Net Income"].apply(fmt_dollar)
            flow_df["Taxes"]         = flow_df["Taxes"].apply(fmt_dollar)
            flow_df["Housing Cost"]  = flow_df["Housing Cost"].apply(fmt_dollar)
            flow_df["Lifestyle Cost"]= flow_df["Lifestyle Cost"].apply(fmt_dollar)
            flow_df["Breathing Room"]= flow_df["Breathing Room"].apply(
                lambda x: f"{'▲ ' if x >= 0 else '▼ '}{fmt_dollar(x)}"
            )
            st.dataframe(flow_df, hide_index=True, use_container_width=True)

            st.markdown("")

            # ── End-of-Year Balances (where you stand) ───────────────
            st.markdown("**End-of-Year Balances** — cumulative wealth position at end of each year")
            bal_cols = ["Year", "Retirement", "Brokerage", "HSA",
                        "Home Equity", "Mortgage Balance", "Net Worth"]
            bal_df = df[bal_cols].copy()
            for col in ["Retirement", "Brokerage", "HSA",
                        "Home Equity", "Mortgage Balance", "Net Worth"]:
                bal_df[col] = bal_df[col].apply(fmt_dollar)
            st.dataframe(bal_df, hide_index=True, use_container_width=True)

    # ── TAB 2: Monte Carlo ───────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)
        st.markdown(
            "Runs N simulations with randomized annual shocks. "
            "**p10 / p50 / p90** = the 10th, 50th (median), and 90th percentile outcomes — "
            "p10 is a bad-luck scenario, p50 is the middle outcome, p90 is a good-luck scenario. "
            "**Liquidity risk** = probability of brokerage + cash going negative in a given year "
            "(having to liquidate retirement accounts or take on debt). "

            "**Historical mode** (recommended): market returns sampled from 100 years of S&P 500 "
            "actuals (1926–2025) — preserving fat tails, crash years, and boom years as they "
            "really happened. "
            "Inflation and salary growth are always drawn from normal distributions. "
            "Shows the full range of outcomes including liquidity risk."
        )

        # ── Simulation parameters ────────────────────────────────
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

        with st.spinner(f"Running {n_sims:,} simulations…"):
            mc = projection_engine.run_monte_carlo(
                n_simulations=int(n_sims),
                seed=42 if mc_seed else None,
                use_historical_returns=use_hist,
                use_historical_inflation=use_hist_inf,
                market_return_std=mkt_std,
                inflation_std=inf_std,
                salary_growth_std=sg_std,
            )

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

        # ── Net worth fan chart ───────────────────────────────────
        years_mc = mc.years
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=years_mc + years_mc[::-1],
            y=mc.p90_net_worth + mc.p10_net_worth[::-1],
            fill="toself", fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p10–p90 band", hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_mc + years_mc[::-1],
            y=mc.p75_net_worth + mc.p25_net_worth[::-1],
            fill="toself", fillcolor="rgba(59,130,246,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p25–p75 band", hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_mc, y=mc.p50_net_worth,
            name="Median (p50)", line=dict(color="#3b82f6", width=2.5),
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_mc, y=mc.p90_net_worth,
            name="Optimistic (p90)", line=dict(color="#4ade80", width=1.5, dash="dot"),
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_mc, y=mc.p10_net_worth,
            name="Pessimistic (p10)", line=dict(color="#f87171", width=1.5, dash="dot"),
        ))
        fig_mc.add_trace(go.Scatter(
            x=[s.year for s in snapshots], y=[s.net_worth for s in snapshots],
            name="Deterministic", line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        hist_parts = []
        if mc.use_historical_returns:   hist_parts.append("hist. returns")
        if mc.use_historical_inflation: hist_parts.append("hist. inflation")
        mode_label = (", ".join(hist_parts) if hist_parts
                      else f"Normal(σ_mkt={mc.market_return_std:.0%}, σ_inf={mc.inflation_std:.0%})")
        fig_mc.update_layout(
            title=f"Net Worth Distribution — {n_sims:,} Simulations ({mode_label})",
            **PLOTLY_DARK, height=460, yaxis_title="Net Worth ($)", xaxis_title="Year",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

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
        st.plotly_chart(fig_liq, use_container_width=True)

        # Warn if any year has >10% liquidity risk
        high_risk_years = [(yr, p) for yr, p in zip(mc.years, mc.prob_negative_liquid) if p > 0.10]
        if high_risk_years:
            yr_list = ", ".join(f"Year {yr} ({p:.0%})" for yr, p in high_risk_years[:5])
            st.error(
                f"⚠️ **Significant liquidity risk detected.** In more than 10% of simulations, "
                f"liquid assets go negative in: {yr_list}. "
                f"Consider building a larger cash buffer or reducing fixed expenses."
            )
        elif any(p > 0 for p in mc.prob_negative_liquid):
            st.warning(
                "⚠️ **Low but non-zero liquidity risk.** Some simulations produce negative "
                "liquid assets in at least one year. Your plan is resilient but not bulletproof."
            )
        else:
            st.success("✅ **No liquidity risk.** Liquid assets stayed positive in all simulations.")

        # ── Liquid assets fan chart ───────────────────────────────
        st.markdown("#### Liquid Assets (Brokerage) Distribution")
        fig_liq_fan = go.Figure()
        fig_liq_fan.add_trace(go.Scatter(
            x=years_mc + years_mc[::-1],
            y=mc.p90_liquid + mc.p10_liquid[::-1],
            fill="toself", fillcolor="rgba(245,158,11,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="p10–p90 band", hoverinfo="skip",
        ))
        fig_liq_fan.add_trace(go.Scatter(
            x=years_mc, y=mc.p50_liquid,
            name="Median liquid", line=dict(color="#f59e0b", width=2),
        ))
        fig_liq_fan.add_trace(go.Scatter(
            x=years_mc, y=mc.p10_liquid,
            name="Pessimistic (p10)", line=dict(color="#f87171", width=1.5, dash="dot"),
        ))
        fig_liq_fan.add_trace(go.Scatter(
            x=years_mc, y=[s.brokerage_balance for s in snapshots],
            name="Deterministic", line=dict(color="#4ade80", width=1.5, dash="dash"),
        ))
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
        st.plotly_chart(fig_liq_fan, use_container_width=True)

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
        st.plotly_chart(fig_hist, use_container_width=True)

        col_prob1, col_prob2 = st.columns(2)
        col_prob1.metric("Probability of $1M+ (Year 10)", f"{mc.prob_millionaire_10yr:.1%}")
        col_prob2.metric("Simulations Run", f"{mc.num_simulations:,}")


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
        if path.exists():
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