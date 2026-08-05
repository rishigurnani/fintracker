"""
Configuration loader / saver.

Personal config lives in config/personal.yaml (gitignored).
A sample config is provided in config/sample.yaml (tracked in git).

Usage::

    plan = load_plan("config/personal.yaml")
    save_plan(plan, "config/personal.yaml")

Design
------
Each profile is described once by a *field spec* — a list of
``(name, load_cast, default[, dump_transform])`` tuples — and the generic
:func:`_build` / :func:`_dump` helpers drive both directions.  This replaces the
old hand-written mirror where every field was spelled out twice (once to parse,
once to serialise) and the two lists could silently drift apart.

Only genuinely special fields (enums, nested objects, phase lists, and a couple
of back-compat aliases) need bespoke handling on top of the specs.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from fintracker.models import (
    BusinessProfile, CarProfile, ChildcarePhase, ChildcareProfile, EmployerMatch, MatchTier, KidCarProfile, CollegeProfile, RothContributionPhase, FilingStatus, RetirementProfile, State,
    IncomeProfile, HousingProfile, LifestyleProfile,
    InvestmentProfile, StrategyToggles, TimelineEvent, FinancialPlan,
    Failsafe, FailsafeCondition, FailsafeAction,
)


# ---------------------------------------------------------------------------
# Spec-driven build / dump
# ---------------------------------------------------------------------------

def _raw(v):
    """Pass a value through untouched (optional fields with no coercion)."""
    return v


def _opt(cast):
    """Optional cast: ``None`` stays ``None``, otherwise apply ``cast``."""
    return lambda v: None if v is None else cast(v)


def _opt_falsy(cast):
    """Optional cast treating falsy input (0/None/'') as ``None``.

    Matches the historical ``float(d[k]) if d.get(k) else None`` idiom used for
    optional home-purchase amounts.
    """
    return lambda v: cast(v) if v else None


def _build(cls, d, spec, **extra):
    """Instantiate ``cls`` from dict ``d`` using ``spec`` (+ any ``extra`` kwargs).

    An *omitted* key falls back to the spec default. A key that is *present but
    blank* in YAML parses as ``None`` — that is almost always a mistake (a stray
    ``foo:`` line), so for a non-optional field it raises a clear, field-naming
    error instead of crashing deep inside the cast (e.g. ``float(None)``) or
    silently defaulting to a value the user never intended. Optional fields (spec
    default ``None`` + an ``_opt`` cast) legitimately accept ``None`` and pass through.
    """
    def value(name, default):
        if name in d:
            v = d[name]
            if v is None and default is not None:
                raise ValueError(
                    f"{cls.__name__}.{name} is present but blank in the config — "
                    f"give it a value (e.g. {default!r}) or remove the line."
                )
            return v
        return default
    kwargs = {name: cast(value(name, default)) for name, cast, default, *_ in spec}
    kwargs.update(extra)
    return cls(**kwargs)


def _dump(obj, spec):
    """Serialise ``obj`` to a dict of ``{field: value}`` per ``spec``.

    A 4th spec element, if present, is a transform applied to non-None values on
    the way out (e.g. ``FilingStatus`` → its string value).
    """
    out = {}
    for name, _cast, _default, *rest in spec:
        val = getattr(obj, name)
        dump_fn = rest[0] if rest else None
        out[name] = dump_fn(val) if (dump_fn and val is not None) else val
    return out


# ---------------------------------------------------------------------------
# Field specs — each field declared once, drives both load and save.
# Tuple: (name, load_cast, default[, dump_transform])
# ---------------------------------------------------------------------------

_INCOME_SPEC = [
    ("gross_annual_income", float, 100_000),
    ("filing_status", FilingStatus, "single", lambda e: e.value),
    ("state", State, "GA", lambda e: e.value),
    ("other_state_flat_rate", float, 0.05),
    ("spouse_gross_annual_income", float, 0),
]

_HOUSING_SPEC = [
    ("home_price", float, 400_000),
    ("down_payment", float, 80_000),
    ("interest_rate", float, 0.065),
    ("loan_term_years", int, 30),
    ("annual_property_tax_rate", float, 0.012),
    ("annual_insurance", float, 2_000),
    ("annual_maintenance_rate", float, 0.01),
    ("pmi_annual_rate", float, 0.005),
    ("is_renting", bool, False),
    ("monthly_rent", float, 0),
    ("annual_rent_increase_rate", float, 0.03),
]

# Flat lifestyle fields; childcare_profile is nested (handled separately).
_LIFESTYLE_SPEC = [
    ("monthly_childcare", float, 0),
    ("num_children", int, 0),
    ("num_pets", int, 0),
    ("annual_pet_cost", float, 0),
    ("annual_medical_oop", float, 3_000),
    ("medical_auto_scale", bool, True),
    ("medical_spouse_multiplier", float, 1.8),
    ("medical_per_child_annual", float, 1_500),
    ("annual_health_insurance_premium", float, 0),
    ("annual_disability_insurance_premium", float, 0),
    ("annual_life_insurance_premium", float, 0),
    ("annual_life_insurance_death_benefit", float, 0),
    ("annual_self_ltc_cost", float, 0),
    ("self_ltc_years_before_death", int, 3),
    ("annual_vacation", float, 5_000),
    ("monthly_other_recurring", float, 500),
    ("annual_parent_care_cost", float, 0),
    ("annual_wedding_fund_per_child", float, 0),
]

# Flat investment fields; auto_invest_surplus, employer_match and the roth
# schedule need bespoke handling (see _dict_to_plan / _plan_to_dict).
_INVESTMENTS_SPEC = [
    ("current_liquid_cash", float, 50_000),
    ("current_retirement_balance", float, 0),
    ("current_brokerage_balance", float, 0),
    ("one_time_upcoming_expenses", float, 0),
    ("annual_401k_contribution", float, 23_000),
    ("partner_annual_401k_contribution", float, 0),
    ("annual_roth_ira_contribution", float, 0),
    ("annual_hsa_contribution", float, 4_150),
    ("annual_529_contribution", float, 0),
    ("annual_brokerage_contribution", float, 0),
    ("annual_market_return", float, 0.08),
    ("annual_inflation_rate", float, 0.03),
    ("annual_healthcare_inflation_rate", float, 0.05),
    ("annual_salary_growth_rate", float, 0.04),
    ("partner_salary_growth_rate", float, 0.04),
    ("annual_home_appreciation_rate", float, 0.035),
    ("salary_growth_peak_age", int, 55),
    ("salary_real_decline_rate", float, 0.0),
    ("capital_gains_tax_rate", float, 0.0),
    ("retirement_capital_gains_tax_rate", _opt(float), None),
    ("taxable_dividend_yield", float, 0.02),
    ("cash_buffer_months", float, 0.0),
    ("compounding_period_months", float, 12.0),
    ("current_roth_ira_balance", float, 0.0),
]

_STRATEGIES_SPEC = [
    ("maximize_hsa", bool, True),
    ("use_529_state_deduction", bool, False),
    ("maximize_401k", bool, True),
    ("use_roth_ladder", bool, False),
    ("use_backdoor_roth", bool, False),
    ("roth_conversion_annual_amount", float, 0),
    # Optional list[str] of account keys; None → engine derives from balances.
    ("retirement_withdrawal_order", _raw, None),
]

_RETIREMENT_SPEC = [
    ("current_age", int, 35),
    ("retirement_age", int, 65),
    ("desired_annual_income", float, 80_000),
    ("years_in_retirement", int, 30),
    ("expected_post_retirement_return", float, 0.05),
    ("estimated_social_security_annual", float, 0),
    ("retirement_withdrawal_tax_rate", float, 0.0),
    ("capital_gains_tax_rate", float, 0.0),
    ("medicare_start_age", int, 65),
    ("annual_medicare_premium", float, 2_100),
    ("auto_retire", bool, True),
    ("life_expectancy_age", _opt(int), None),
    ("spending_smile_slowgo_age", int, 75),
    ("spending_smile_slowgo_factor", float, 0.90),
    ("spending_smile_nogo_age", int, 85),
    ("spending_smile_nogo_factor", float, 0.80),
]

_COLLEGE_SPEC = [
    ("annual_cost_per_child", float, 35_000),
    ("years_per_child", int, 4),
    ("start_age", int, 18),
    ("use_aotc_credit", bool, True),
    ("early_529_return", float, 0.08),
    ("late_529_return", float, 0.04),
    ("glide_path_years", int, 10),
]

_BUSINESS_SPEC = [
    ("annual_revenue", float, 0.0),
    ("expense_ratio", float, 0.60),
    ("revenue_growth_rate", float, 0.05),
    ("initial_investment", float, 0.0),
    ("start_year", int, 1),
    ("use_qbi_deduction", bool, True),
    ("self_employed_health_insurance", float, 0.0),
    ("solo_401k_contribution", float, 0.0),
    ("sep_ira_contribution", float, 0.0),
    ("equity_multiple", float, 3.0),
    ("sale_year", _opt(int), None),
    ("ownership_pct", float, 1.0),
]

# Flat car fields; kids_car (nested) and first_purchase_years (list) are extra.
_CAR_SPEC = [
    ("car_price", float, 25_000),
    ("down_payment", float, 5_000),
    ("loan_rate", float, 0.065),
    ("loan_term_years", int, 5),
    ("replace_every_years", int, 10),
    ("residual_value", float, 5_000),
    ("hand_down_age", int, 16),
    ("num_cars", int, 1),
    ("annual_insurance_per_car", float, 1_500),
    ("annual_maintenance_per_car", float, 1_000),
    ("annual_fuel_per_car", float, 2_000),
    ("annual_registration_per_car", float, 200),
]

_KID_CAR_SPEC = [
    ("car_price", float, 15_000),
    ("down_payment_pct", float, 0.20),
    ("loan_rate", float, 0.07),
    ("loan_term_years", int, 5),
    ("buy_at_age", _raw, None),
]

# home_price_override is a load-only back-compat alias (not serialised).
_EVENT_SPEC = [
    ("year", int, 0),
    ("description", str, ""),
    ("income_change", _raw, None),
    ("partner_income_change", _raw, None),
    ("stop_working", bool, False),
    ("resume_working", bool, False),
    ("partner_stop_working", bool, False),
    ("partner_resume_working", bool, False),
    ("start_parent_care", bool, False),
    ("stop_parent_care", bool, False),
    ("child_birth_year_override", _raw, None),
    ("new_child", bool, False),
    ("new_pet", bool, False),
    ("marriage", bool, False),
    ("buy_home", bool, False),
    ("new_home_price", _opt_falsy(float), None),
    ("new_home_down_payment", _opt_falsy(float), None),
    ("new_home_interest_rate", _opt_falsy(float), None),
    ("sell_current_home", bool, True),
    ("buyer_closing_cost_rate", float, 0.02),
    ("seller_closing_cost_rate", float, 0.06),
    ("extra_one_time_expense", float, 0),
    ("extra_one_time_income", float, 0),
]

# Failsafes: a `when:` list of conditions + a single `action:`, plus top-level
# scalars. Conditions and action nest, so they are built separately and passed
# to the outer _build as extra kwargs.
_FAILSAFE_COND_SPEC = [
    ("metric", str, "brokerage_balance"),
    ("comparator", str, "below"),
    ("threshold", float, 0.0),
    ("present_value", bool, True),
    ("start_year", int, 1),
    ("end_year", _opt(int), None),
]
_FAILSAFE_ACTION_SPEC = [
    ("partner_income", _opt(float), None),
    ("primary_income", _opt(float), None),
    ("one_time_income", float, 0.0),
    ("one_time_expense", float, 0.0),
    ("present_value", bool, True),
    ("suspend_retirement_contributions", bool, False),
    ("annual_vacation", _opt(float), None),
    ("medical_cost_multiplier", _opt(float), None),
]
_FAILSAFE_SPEC = [
    ("name", str, "failsafe"),
    ("match", str, "any"),
    ("delay_years", int, 0),
    ("duration_years", _opt(int), None),
    ("once", bool, True),
]

_CHILDCARE_PHASE_SPEC = [("age_start", int, 0), ("age_end", int, 0), ("monthly_cost", float, 0.0)]
_ROTH_PHASE_SPEC = [("year_start", int, 0), ("year_end", int, 0), ("annual_amount", float, 0.0)]
_MATCH_TIER_SPEC = [("match_pct", float, 0.0), ("up_to_pct_of_salary", float, 0.0)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_plan(path: str | Path) -> FinancialPlan:
    """Load a FinancialPlan from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return _dict_to_plan(yaml.safe_load(f))


def save_plan(plan: FinancialPlan, path: str | Path) -> None:
    """Serialize a FinancialPlan to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_plan_to_dict(plan), f, default_flow_style=False, sort_keys=False)


def load_plan_or_sample(path: str | Path = "config/personal.yaml") -> FinancialPlan:
    """Load personal config; fall back to sample.yaml then hard-coded defaults."""
    try:
        return load_plan(path)
    except FileNotFoundError:
        sample = Path(__file__).parent.parent / "config" / "sample.yaml"
        return load_plan(sample) if sample.exists() else _default_plan()


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

def _dict_to_plan(d: dict) -> FinancialPlan:
    income   = _build(IncomeProfile, d.get("income", {}), _INCOME_SPEC)
    housing  = _build(HousingProfile, d.get("housing", {}), _HOUSING_SPEC)

    l_d = d.get("lifestyle", {})
    lifestyle = _build(
        LifestyleProfile, l_d, _LIFESTYLE_SPEC,
        childcare_profile=(_dict_to_childcare_profile(l_d["childcare_profile"])
                           if "childcare_profile" in l_d else None),
    )

    inv_d = d.get("investments", {})
    s_d   = d.get("strategies", {})
    # auto_invest_surplus lives in investments:; read from strategies: for back-compat
    auto_invest = inv_d.get("auto_invest_surplus", s_d.get("auto_invest_surplus", True))
    investments = _build(
        InvestmentProfile, inv_d, _INVESTMENTS_SPEC,
        auto_invest_surplus=bool(auto_invest),
        employer_match=(_dict_to_employer_match(inv_d["employer_match"])
                        if "employer_match" in inv_d else None),
        roth_contribution_schedule=(
            [_build(RothContributionPhase, p, _ROTH_PHASE_SPEC)
             for p in inv_d["roth_contribution_schedule"]]
            if "roth_contribution_schedule" in inv_d else None
        ),
    )

    strategies = _build(StrategyToggles, s_d, _STRATEGIES_SPEC)
    events = [_dict_to_event(e) for e in d.get("timeline_events", [])]
    failsafes = [_dict_to_failsafe(f) for f in d.get("failsafes", [])]

    return FinancialPlan(
        income=income,
        housing=housing,
        lifestyle=lifestyle,
        investments=investments,
        strategies=strategies,
        timeline_events=events,
        failsafes=failsafes,
        projection_years=int(d.get("projection_years", 30)),
        retirement=_build(RetirementProfile, d["retirement"], _RETIREMENT_SPEC) if "retirement" in d else None,
        college=_build(CollegeProfile, d["college"], _COLLEGE_SPEC) if "college" in d else None,
        car=_dict_to_car(d["car"]) if "car" in d else None,
        business=_build(BusinessProfile, d["business"], _BUSINESS_SPEC) if "business" in d else None,
    )


def _dict_to_event(e: dict) -> TimelineEvent:
    # home_price_override is a load-only back-compat alias, so it lives outside the spec.
    return _build(TimelineEvent, e, _EVENT_SPEC,
                  home_price_override=e.get("home_price_override"))


def _dict_to_failsafe(f: dict) -> Failsafe:
    conditions = [_build(FailsafeCondition, c, _FAILSAFE_COND_SPEC)
                  for c in f.get("when", [])]
    action = _build(FailsafeAction, f.get("action", {}), _FAILSAFE_ACTION_SPEC)
    return _build(Failsafe, f, _FAILSAFE_SPEC, conditions=conditions, action=action)


def _failsafe_to_dict(fs: Failsafe) -> dict:
    out = _dump(fs, _FAILSAFE_SPEC)
    out["when"] = [_dump(c, _FAILSAFE_COND_SPEC) for c in fs.conditions]
    out["action"] = _dump(fs.action, _FAILSAFE_ACTION_SPEC)
    return out


def _dict_to_childcare_profile(cp: dict) -> ChildcareProfile:
    """
    Parse a childcare_profile dict into a ChildcareProfile.

    Validates each phase and raises a descriptive ValueError if a phase is
    missing required fields (age_start, age_end, monthly_cost).  A common
    YAML mistake is splitting one phase across two list items:

        Bad (two separate items):       Good (one item with all fields):
          - age_start: 3                  - age_start: 3
          - age_end:   4                    age_end:   4
            monthly_cost: 1500              monthly_cost: 1500

    The bad form is valid YAML but produces {age_start: 3} and
    {age_end: 4, monthly_cost: 1500} as separate dicts — both missing fields.
    """
    phases = []
    for i, p in enumerate(cp.get("phases", [])):
        missing = [f for f in ("age_start", "age_end", "monthly_cost") if f not in p]
        if missing:
            raise ValueError(
                f"childcare_profile phase {i+1} is missing: {missing}. "
                f"Got: {p}. "
                f"Common fix: ensure age_start, age_end, and monthly_cost are all "
                f"indented under the same '- ' list marker (not separate list items)."
            )
        phases.append(_build(ChildcarePhase, p, _CHILDCARE_PHASE_SPEC))
    return ChildcareProfile(phases=phases)


def _dict_to_employer_match(em: dict) -> EmployerMatch:
    return EmployerMatch(
        tiers=[_build(MatchTier, t, _MATCH_TIER_SPEC) for t in em.get("tiers", [])],
        annual_cap=em.get("annual_cap"),          # None = no cap
        vesting_years=int(em.get("vesting_years", 0)),
        profit_sharing_annual=float(em.get("profit_sharing_annual", 0.0)),
    )


def _dict_to_car(c: dict) -> CarProfile:
    kc_d = c.get("kids_car")
    fpy = c.get("first_purchase_years")
    return _build(
        CarProfile, c, _CAR_SPEC,
        kids_car=_build(KidCarProfile, kc_d, _KID_CAR_SPEC) if kc_d else None,
        first_purchase_years=[int(y) for y in fpy] if fpy else None,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _plan_to_dict(plan: FinancialPlan) -> dict:
    investments = _dump(plan.investments, _INVESTMENTS_SPEC)
    investments["auto_invest_surplus"] = plan.investments.auto_invest_surplus
    if plan.investments.roth_contribution_schedule:
        investments["roth_contribution_schedule"] = [
            _dump(p, _ROTH_PHASE_SPEC) for p in plan.investments.roth_contribution_schedule
        ]
    if plan.investments.employer_match:
        em = plan.investments.employer_match
        investments["employer_match"] = {
            "tiers": [_dump(t, _MATCH_TIER_SPEC) for t in em.tiers],
            "annual_cap": em.annual_cap,
            "vesting_years": em.vesting_years,
            "profit_sharing_annual": em.profit_sharing_annual,
        }

    lifestyle = _dump(plan.lifestyle, _LIFESTYLE_SPEC)
    if plan.lifestyle.childcare_profile:
        lifestyle["childcare_profile"] = {
            "phases": [_dump(p, _CHILDCARE_PHASE_SPEC) for p in plan.lifestyle.childcare_profile.phases]
        }

    d: dict = {
        "projection_years": plan.projection_years,
        "income": _dump(plan.income, _INCOME_SPEC),
        "housing": _dump(plan.housing, _HOUSING_SPEC),
        "lifestyle": lifestyle,
        "investments": investments,
        "strategies": _dump(plan.strategies, _STRATEGIES_SPEC),
        "timeline_events": [_dump(e, _EVENT_SPEC) for e in plan.timeline_events],
    }

    if plan.retirement:
        d["retirement"] = _dump(plan.retirement, _RETIREMENT_SPEC)
    if plan.college:
        d["college"] = _dump(plan.college, _COLLEGE_SPEC)
    if plan.business:
        d["business"] = _dump(plan.business, _BUSINESS_SPEC)
    if plan.car:
        car_d = _dump(plan.car, _CAR_SPEC)
        car_d["first_purchase_years"] = plan.car.first_purchase_years
        if plan.car.kids_car:
            car_d["kids_car"] = _dump(plan.car.kids_car, _KID_CAR_SPEC)
        d["car"] = car_d
    if plan.failsafes:
        d["failsafes"] = [_failsafe_to_dict(f) for f in plan.failsafes]

    return d


# ---------------------------------------------------------------------------
# Default plan
# ---------------------------------------------------------------------------

def _default_plan() -> FinancialPlan:
    return FinancialPlan(
        income=IncomeProfile(gross_annual_income=120_000,
                             filing_status=FilingStatus.SINGLE, state=State.GEORGIA),
        housing=HousingProfile(home_price=400_000, down_payment=80_000, interest_rate=0.065),
        lifestyle=LifestyleProfile(annual_medical_oop=3_000, annual_vacation=5_000,
                                   monthly_other_recurring=500),
        investments=InvestmentProfile(current_liquid_cash=100_000, annual_401k_contribution=23_000,
                                      annual_hsa_contribution=4_150, annual_market_return=0.08,
                                      annual_inflation_rate=0.03, annual_salary_growth_rate=0.04),
        strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
        projection_years=30,
    )
