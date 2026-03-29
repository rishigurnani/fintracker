# 📈 fintracker

> **Personal long-term financial planning — tax-aware, scenario-driven, Monte Carlo enabled.**

A self-hosted Streamlit app and Python library for people who want to understand where their money is *actually* going — and where it's headed.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-360%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-~85%25-brightgreen.svg)](tests/)

---

## Why fintracker?

Most free financial tools give you a single-number answer. fintracker gives you *understanding*:

| Feature | fintracker | Typical Free Tool |
|---|---|---|
| Full amortization schedule (exact math) | ✅ | ❌ Simplified |
| Multi-state income tax engine | ✅ GA, CA, NY, TX, FL, WA, IL, NC, VA, CO | ❌ |
| FICA + Additional Medicare Tax | ✅ | ❌ |
| HSA saves FICA *and* income tax (correctly) | ✅ | ❌ |
| 401k vs Roth vs HSA strategy comparison | ✅ | ❌ |
| Employer 401k match (tiered, vested, profit sharing) | ✅ | ❌ |
| 529 state deduction by state | ✅ | ❌ |
| College costs with 529 drawdown + AOTC credit | ✅ | Rare |
| Age-based childcare cost schedule | ✅ | ❌ |
| Retirement readiness (growing annuity formula) | ✅ | Rare |
| Dual income with independent salary growth | ✅ | ❌ |
| Stop/resume work (sabbatical, caregiving) | ✅ | ❌ |
| Business ownership (SE tax, QBI, solo 401k, equity) | ✅ | ❌ |
| Car financing with replacement cycles and kids' cars | ✅ | ❌ |
| Parent care costs with timeline events | ✅ | ❌ |
| PMI with automatic removal at 80% LTV | ✅ | ❌ |
| Monte Carlo — 5,000 simulations | ✅ | Rare |
| Historical bootstrap (100yr S&P + 96yr CPI data) | ✅ | ❌ |
| Cash buffer modelling (emergency fund) | ✅ | ❌ |
| Timeline events (marriage, children, raise, home) | ✅ | ❌ |
| YAML config — reproducible, git-friendly | ✅ | N/A |
| 100% local — your data never leaves your machine | ✅ | ❌ |
| Open source, hackable | ✅ | ❌ |

---

## Screenshots

<img src="./docs/img1.png" width="800">

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/rishigurnani/fintracker.git
cd fintracker

# 2. Install (tested with python3.11)
pip install -e ".[dev]"

# 3. Copy sample config and fill in your real numbers
cp config/sample.yaml config/personal.yaml
# Edit config/personal.yaml — it's gitignored, so your data stays private

# 4. Run
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

---

## Configuration

All personal data lives in `config/personal.yaml`, which is **gitignored** — it will never be accidentally committed. A fully documented sample with all available options is in [`config/sample.yaml`](config/sample.yaml).

### Dollar amounts in the YAML

There are four types of dollar values — the sidebar labels each one clearly:

| Label | Meaning | Examples |
|---|---|---|
| **inflated yearly** | Enter today's value; engine multiplies by CPI each year | Lifestyle costs, college costs, car prices, retirement income goal |
| **fixed** | Same nominal amount every year regardless of inflation | All contributions: 401k, HSA, 529, Roth; employer match amounts |
| **current** | Enter the actual amount as it stands today | Balances (liquid cash, retirement, brokerage), home price |
| **own rate** | Grows at its own configured rate, not CPI | Income (salary growth), rent (rent increase rate), business revenue |

### Minimal example

```yaml
projection_years: 30

income:
  gross_annual_income: 120000          # current
  filing_status: single                # single | married_filing_jointly | head_of_household
  state: GA                            # GA | CA | NY | TX | FL | WA | IL | NC | VA | CO | OTHER

housing:
  is_renting: false
  home_price: 400000                   # current
  down_payment: 80000                  # current
  interest_rate: 0.065

investments:
  current_liquid_cash: 100000          # current
  annual_401k_contribution: 23000      # fixed — stays $23k/yr
  annual_hsa_contribution: 4150        # fixed
  annual_market_return: 0.08
  annual_inflation_rate: 0.03
  annual_salary_growth_rate: 0.04

strategies:
  maximize_hsa: true
  maximize_401k: true

timeline_events:
  - year: 3
    description: "Get married"
    marriage: true
  - year: 4
    description: "First child"
    new_child: true
```

### Employer 401k match

Supports any match structure: simple, tiered, capped, cliff vesting, profit sharing.

```yaml
investments:
  annual_401k_contribution: 21000      # fixed
  employer_match:
    tiers:
      - match_pct: 1.00
        up_to_pct_of_salary: 0.03      # 100% match on first 3% of salary
      - match_pct: 0.50
        up_to_pct_of_salary: 0.02      # 50% match on next 2%
    annual_cap: null                   # no dollar cap
    vesting_years: 3                   # cliff vesting: no match if you leave before yr 3
    profit_sharing_annual: 0           # flat annual employer add
```

Common configurations:
- **Simple 50% on first 6%:** `tiers: [{match_pct: 0.50, up_to_pct_of_salary: 0.06}]`
- **Dollar-for-dollar on first 3%:** `tiers: [{match_pct: 1.00, up_to_pct_of_salary: 0.03}]`
- **Capped at $5k:** add `annual_cap: 5000`
- **Profit sharing only:** `tiers: []`, `profit_sharing_annual: 3000`

### Age-based childcare costs

Replace the flat `monthly_childcare` with a realistic cost curve per life stage:

```yaml
lifestyle:
  childcare_profile:
    phases:
      - age_start: 0
        age_end:   2
        monthly_cost: 2500    # inflated yearly
      - age_start: 3
        age_end:   4
        monthly_cost: 1500
      - age_start: 5
        age_end:   12
        monthly_cost: 600
      - age_start: 13
        age_end:   17
        monthly_cost: 150
      # age 18+ handled by CollegeProfile; defaults to $0 here
```

> **YAML formatting note:** Each phase's three fields (`age_start`, `age_end`, `monthly_cost`) must all be indented under the **same** `- ` list marker. A common mistake is splitting them across separate `- ` items — the loader catches this and names the broken phase.

The flat `monthly_childcare` field is retained for backward compatibility.

### Business ownership

```yaml
business:
  annual_revenue: 500000              # own rate — grows at revenue_growth_rate
  expense_ratio: 0.65                 # costs as fraction of revenue
  revenue_growth_rate: 0.06
  initial_investment: 150000          # current — one-time startup cost
  start_year: 2
  ownership_pct: 0.60                 # your share (e.g. 0.60 = 60%; partner owns 40%)
  use_qbi_deduction: true
  self_employed_health_insurance: 18000   # fixed
  solo_401k_contribution: 40000           # fixed
  equity_multiple: 3.0
  sale_year: 20
```

### Car financing

```yaml
car:
  car_price: 35000                    # inflated yearly
  down_payment: 7000                  # current
  loan_rate: 0.065
  loan_term_years: 5
  replace_every_years: 12
  residual_value: 6000                # inflated yearly
  hand_down_age: 16
  num_cars: 2
  first_purchase_years: [3, 5]        # projection years of first purchase for each car
  kids_car:
    car_price: 15000                  # inflated yearly
    down_payment_pct: 0.20
    loan_rate: 0.07
    loan_term_years: 5
    buy_at_age: 16                    # 16 = driving age, 22 (or null) = graduation
```

### Retirement readiness

```yaml
retirement:
  current_age: 32
  retirement_age: 65
  desired_annual_income: 80000        # inflated yearly — today's dollars
  years_in_retirement: 30
  expected_post_retirement_return: 0.05
  estimated_social_security_annual: 24000   # inflated yearly
```

The engine uses the **growing annuity formula** — spending inflates throughout retirement, not a flat annuity. Projected balance at retirement includes all accounts: 401k/IRA + HSA + brokerage + cash.

### College costs

```yaml
college:
  annual_cost_per_child: 35000        # inflated yearly
  years_per_child: 4
  start_age: 18
  use_aotc_credit: true
  early_529_return: 0.08
  late_529_return: 0.04
  glide_path_years: 10
```

### Cash buffer

```yaml
investments:
  cash_buffer_months: 3    # hold 3 months of expenses in cash at 0% before investing surplus
```

### Stop/resume work

```yaml
timeline_events:
  - year: 4
    description: "Sabbatical"
    stop_working: true
  - year: 5
    description: "Return to work"
    resume_working: true
    income_change: 145000
```

---

## Features In Depth

### 🧮 Tax Engine

- **2024/2025 federal brackets** — MFJ, Single, Head of Household
- **FICA** — Social Security (capped at $168,600), Medicare, Additional Medicare Tax
- **HSA** — reduces Federal + FICA + State (except CA)
- **401k** — reduces Federal + State (not FICA — correct)
- **AOTC** — direct tax credit; correct phase-out by filing status
- **SE tax** — 15.3% on 92.35% of net profit × ownership share; employer half deductible
- **QBI deduction** — 20% pass-through, phased out above $191,950 (single) / $383,900 (MFJ)

### 🎲 Monte Carlo Simulation

5,000 simulations by default:

| Variable | Historical mode (default) | Normal mode |
|---|---|---|
| Market returns | Bootstrap from 100yr S&P 500 (1926–2025) | N(μ, σ) |
| Inflation | Bootstrap from 96yr US CPI (1929–2024) | N(μ, σ) |
| Salary growth | N(μ, σ) | N(μ, σ) |

Historical bootstrap captures deflation (1930s), stagflation (1970s, peak 13.3%), and post-WWII spikes (18.1%) that normal distributions miss — leading to materially more accurate liquidity risk estimates.

**p10 / p50 / p90** = 10th, 50th, 90th percentile outcomes. p10 is a bad-luck scenario; p90 is good luck. **Liquidity risk** = probability of brokerage + cash going negative in a given year.

Performance: ~1.2ms/simulation; 5,000 sims in ~6s.

---

## Test Suite

```
360 tests · 7 files · ~85% coverage on core engine (excl. Streamlit UI)
```

```bash
pytest tests/ -v --tb=short
```

| File | Tests | Covers |
|---|---|---|
| `test_accounting.py` | 68 | Deficit spending, mortgage, childcare, salary growth, medical scaling, marriage, dual income |
| `test_new_features.py` | 182 | Retirement readiness, stop/resume, college, parent care, auto-invest, cars, MC liquidity, cash buffer, export fidelity, business, employer match, childcare profile, historical bootstrap, ownership_pct, cumulative inflation fix, MC audit |
| `test_tax_engine.py` | 31 | Federal brackets, FICA, HSA/401k deductions, state taxes |
| `test_projections.py` | 33 | Deterministic engine, timeline events, Monte Carlo, home purchase, HSA tiers |
| `test_mortgage.py` | 24 | Monthly payment, amortization, PMI |
| `test_config.py` | 11 | YAML round-trip for all profile types |
| `test_strategies.py` | 11 | Strategy savings quantification |

Every known bug has a regression test. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full list.

---

## Project Structure

```
fintracker/
├── app.py                  # Streamlit UI
├── fintracker/
│   ├── models.py           # All dataclasses
│   ├── tax_engine.py       # Federal + multi-state + SE tax
│   ├── mortgage.py         # Exact amortization + PMI
│   ├── strategies.py       # Strategy analyzer
│   ├── projections.py      # Deterministic + Monte Carlo engine
│   └── config.py           # YAML load/save
├── tests/                  # 360 tests
├── config/
│   ├── sample.yaml         # Fully documented example (tracked)
│   └── personal.yaml       # Your private numbers (gitignored)
├── CONTRIBUTING.md
├── pyproject.toml
└── .gitignore
```

---

## Supported States

| State | Tax Type | HSA | 529 |
|---|---|---|---|
| Georgia (GA) | Flat 5.39% | ✅ | ✅ $8k/beneficiary |
| California (CA) | Progressive | ❌ | ❌ |
| New York (NY) | Progressive | ✅ | ✅ $5k/beneficiary |
| Texas (TX) | None | N/A | N/A |
| Florida (FL) | None | N/A | N/A |
| Washington (WA) | None | N/A | N/A |
| Illinois (IL) | Flat 4.95% | ✅ | ✅ $10k/beneficiary |
| North Carolina (NC) | Flat 4.5% | ✅ | ❌ |
| Virginia (VA) | Progressive | ✅ | ❌ |
| Colorado (CO) | Flat 4.4% | ✅ | ✅ $20k/beneficiary |

> **Disclaimer:** Tax laws change annually. fintracker is for planning and scenario analysis — not tax advice. Consult a CPA for your specific situation.

---

## Roadmap

- [ ] Backdoor Roth IRA modelling
- [ ] Scenario A vs B side-by-side comparison
- [ ] Rent vs Buy breakeven analysis
- [ ] Export to PDF report
- [ ] More states (NJ, MA, AZ, MN)
- [ ] Graded 401k vesting schedule

---

## Timeline Events Reference

All fields are optional unless noted. Multiple events can fire in the same year.

| Field | Type | Description |
|---|---|---|
| `year` | int | **Required.** Fires at the start of this projection year |
| `description` | str | **Required.** Label shown in the UI |
| `marriage` | bool | Switches to MFJ filing, upgrades HSA to family limit, scales medical costs |
| `new_child` | bool | Increments child count; activates childcare and 529 contributions |
| `child_birth_year_override` | int | Birth year offset for college cost timing (negative = already born) |
| `new_pet` | bool | Increments pet count |
| `income_change` | float | New gross income for primary person |
| `partner_income_change` | float | New gross income for partner |
| `stop_working` | bool | Primary income → 0; salary growth paused |
| `resume_working` | bool | Primary resumes; pair with `income_change` for new salary |
| `partner_stop_working` | bool | Partner income → 0 |
| `partner_resume_working` | bool | Partner resumes; pair with `partner_income_change` |
| `start_parent_care` | bool | Activates `annual_parent_care_cost` from lifestyle |
| `stop_parent_care` | bool | Deactivates parent care cost |
| `buy_home` | bool | Purchases a new home; optionally sells current |
| `sell_current_home` | bool | Captures equity (minus seller closing costs) into brokerage |
| `buyer_closing_cost_rate` | float | Default 0.02 (2% of purchase price) |
| `seller_closing_cost_rate` | float | Default 0.06 (6% of sale price) |
| `extra_one_time_expense` | float | One-time cash outflow (drains brokerage) |
| `extra_one_time_income` | float | One-time cash inflow (adds to brokerage) |

---

## Using fintracker as a Library

The core engine is decoupled from Streamlit:

```python
from fintracker.models import *
from fintracker.projections import ProjectionEngine

plan = FinancialPlan(
    income=IncomeProfile(150_000, FilingStatus.SINGLE, State.GEORGIA),
    housing=HousingProfile(0, 0, 0.0, is_renting=True, monthly_rent=1_800),
    lifestyle=LifestyleProfile(annual_vacation=8_000, annual_medical_oop=3_000),
    investments=InvestmentProfile(
        current_liquid_cash=80_000,
        current_retirement_balance=120_000,
        annual_401k_contribution=23_000,
        annual_market_return=0.08,
    ),
    strategies=StrategyToggles(maximize_401k=True, maximize_hsa=True),
    retirement=RetirementProfile(
        current_age=33, retirement_age=65,
        desired_annual_income=90_000,
        estimated_social_security_annual=24_000,
    ),
    college=CollegeProfile(annual_cost_per_child=35_000, use_aotc_credit=True),
    timeline_events=[
        TimelineEvent(year=2, description="Buy home", buy_home=True,
                      new_home_price=550_000, new_home_down_payment=110_000,
                      new_home_interest_rate=0.068, sell_current_home=False),
        TimelineEvent(year=3, description="Get married", marriage=True),
        TimelineEvent(year=4, description="First child", new_child=True),
    ],
    projection_years=30,
)

engine = ProjectionEngine(plan)
snapshots = engine.run_deterministic()

print(f"Year 30 net worth: ${snapshots[-1].net_worth:,.0f}")

# Retirement readiness
rr = engine.compute_retirement_readiness(snapshots)
status = "✅ On track" if rr.on_track else "⚠️ Off track"
print(f"{status} — {rr.funded_pct:.0%} funded")
print(f"Projected at retirement: ${rr.projected_balance_at_retirement:,.0f}")
print(f"Required: ${rr.required_balance:,.0f}")

# Monte Carlo
mc = engine.run_monte_carlo(n_simulations=1_000, seed=42)
print(f"Median net worth year 30: ${mc.p50_net_worth[-1]:,.0f}")
print(f"Probability of $1M by year 10: {mc.prob_millionaire_10yr:.1%}")
```

---

## Contributing

PRs welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before making changes.

---

## License

MIT — use it, fork it, build on it.