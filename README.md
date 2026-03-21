# 📈 fintracker

> **Personal long-term financial planning — tax-aware, scenario-driven, Monte Carlo enabled.**

A self-hosted Streamlit app and Python library for people who want to understand where their money is *actually* going — and where it's headed.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

---

## Why fintracker?

Most free financial tools give you a single-number answer. fintracker gives you *understanding*:

| Feature | fintracker | Typical Free Tool |
|---|---|---|
| Full amortization schedule (exact math) | ✅ | ❌ Simplified |
| Multi-state income tax engine | ✅ GA, CA, NY, TX, FL, WA, IL, NC, VA, CO | ❌ Single state or flat rate |
| FICA + Additional Medicare Tax | ✅ | ❌ |
| HSA saves FICA *and* income tax (correctly) | ✅ | ❌ |
| 401k vs Roth vs HSA strategy comparison | ✅ | ❌ |
| 529 state deduction by state | ✅ | ❌ |
| PMI with automatic removal at 80% LTV | ✅ | ❌ |
| Monte Carlo simulation (1,000 runs) | ✅ | Rare |
| Timeline events (marriage, children, raise) | ✅ | ❌ |
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

# 5. Run tests (optional)
pytest tests/ -v
```

Your browser will open at `http://localhost:8501`.

---

## Configuration

All personal data lives in `config/personal.yaml`, which is **gitignored** — it will never be accidentally committed to your repository.

A fully documented sample is provided in [`config/sample.yaml`](config/sample.yaml).

```yaml
income:
  gross_annual_income: 120000
  filing_status: single          # single | married_filing_jointly | head_of_household
  state: GA                      # GA | CA | NY | TX | FL | WA | IL | NC | VA | CO | OTHER

housing:
  home_price: 400000
  down_payment: 80000
  interest_rate: 0.065

investments:
  current_liquid_cash: 100000
  annual_401k_contribution: 23000
  annual_hsa_contribution: 4150
  annual_market_return: 0.08

strategies:
  maximize_hsa: true
  maximize_401k: true
  use_529_state_deduction: false

timeline_events:
  - year: 3
    description: "Get married"
    marriage: true
  - year: 4
    description: "First child"
    new_child: true
```

---

## Features In Depth

### 🧮 Tax Engine

- **2024/2025 federal brackets** — MFJ, Single, Head of Household
- **FICA** — Social Security (capped at $168,600 wage base), Medicare (1.45%), Additional Medicare Tax (0.9% above $200k/$250k)
- **HSA deduction** reduces Federal + FICA + State (except California, which correctly does not recognize HSA)
- **401k deduction** reduces Federal + State (not FICA — also correct)
- **State taxes** — progressive or flat, with state-specific standard deductions and 529 deduction rules

### 🏠 Mortgage Calculator

Exact amortization (not approximations):
- Month-by-month P&I split
- PMI charged while LTV > 80%, automatically removed
- Home value appreciation applied each year
- Full table with cumulative interest, equity, and home value

### 🎯 Strategy Analyzer

Quantifies the dollar value of each strategy toggle, in isolation:
- HSA: "Contributing $4,150 saves ~$1,537/yr in Federal + FICA + State taxes"
- 401k: "Contributing $23,000 saves ~$6,397/yr in Federal + State taxes"
- 529: "Deducting $8,000 for 1 child saves ~$431/yr in Georgia state tax"
- Roth IRA eligibility and phase-out detection
- Backdoor Roth warning for high earners

### 📈 Projections

Year-by-year simulation over up to 40 years:
- Salary grows at configurable annual rate
- Expenses inflate at configurable inflation rate
- Contributions auto-scale with inflation (capped at IRS limits)
- Filing status, family size, and income can change via Timeline Events
- Tracks retirement balance, brokerage, home equity, HSA, and mortgage balance separately

### 🎲 Monte Carlo

1,000 simulations with randomized:
- **Market returns**: N(μ=your assumption, σ=15%) — typical equity volatility
- **Inflation**: N(μ=your assumption, σ=1.5%) — clipped to [0%, 15%]
- **Salary growth**: N(μ=your assumption, σ=2%)

Outputs:
- Percentile bands (p10/p25/p50/p75/p90) for net worth
- Probability of reaching $1M by year 10
- Comparison of deterministic vs median outcome

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Federal bracket math (verified against IRS published values)
- FICA with SS wage base cap and Additional Medicare Tax
- State tax for all 10 supported states
- HSA/401k deduction effects on FICA vs income tax
- Full amortization schedule integrity (principal sums to loan, balance reaches zero)
- PMI removal at 80% LTV
- Projection net worth component accounting
- Timeline event mutations (marriage, children, income changes)
- Monte Carlo reproducibility and percentile ordering
- YAML config round-trip serialization

---

## Project Structure

```
fintracker/
├── app.py                  # Streamlit UI
├── fintracker/
│   ├── models.py           # Dataclasses: FinancialPlan and all sub-models
│   ├── tax_engine.py       # Federal + multi-state tax calculation
│   ├── mortgage.py         # Full amortization schedule
│   ├── strategies.py       # HSA, 401k, 529, Roth strategy analyzer
│   ├── projections.py      # Deterministic + Monte Carlo projection engine
│   └── config.py           # YAML load/save
├── tests/
│   ├── conftest.py         # Shared pytest fixtures
│   ├── test_tax_engine.py
│   ├── test_mortgage.py
│   ├── test_projections.py
│   ├── test_strategies.py
│   └── test_config.py
├── config/
│   ├── sample.yaml         # ✅ Tracked in git — documented example
│   └── personal.yaml       # 🔒 Gitignored — your private numbers
├── pyproject.toml
└── .gitignore
```

---

## Extending fintracker

The core library is decoupled from Streamlit — you can use it in scripts, notebooks, or other UIs:

```python
from fintracker.models import *
from fintracker.tax_engine import TaxEngine
from fintracker.projections import ProjectionEngine

plan = FinancialPlan(
    income=IncomeProfile(gross_annual_income=150_000, filing_status=FilingStatus.SINGLE, state=State.GEORGIA),
    housing=HousingProfile(home_price=500_000, down_payment=100_000, interest_rate=0.065),
    lifestyle=LifestyleProfile(annual_vacation=8_000),
    investments=InvestmentProfile(current_liquid_cash=80_000, annual_401k_contribution=23_000),
    strategies=StrategyToggles(maximize_hsa=True, maximize_401k=True),
    projection_years=30,
)

snapshots = ProjectionEngine(plan).run_deterministic()
print(f"Year 30 net worth: ${snapshots[-1].net_worth:,.0f}")

mc = ProjectionEngine(plan).run_monte_carlo(n_simulations=1_000)
print(f"Median net worth: ${mc.p50_net_worth[-1]:,.0f}")
print(f"Probability of $1M by year 10: {mc.prob_millionaire_10yr:.1%}")
```

---

## Supported States

| State | Tax Type | HSA Deduction | 529 Deduction |
|---|---|---|---|
| Georgia (GA) | Flat 5.39% | ✅ | ✅ $8k/beneficiary (MFJ) |
| California (CA) | Progressive | ❌ | ❌ |
| New York (NY) | Progressive | ✅ | ✅ $5k/beneficiary |
| Texas (TX) | None | N/A | N/A |
| Florida (FL) | None | N/A | N/A |
| Washington (WA) | None | N/A | N/A |
| Illinois (IL) | Flat 4.95% | ✅ | ✅ $10k/beneficiary |
| North Carolina (NC) | Flat 4.5% | ✅ | ❌ |
| Virginia (VA) | Progressive | ✅ | ❌ |
| Colorado (CO) | Flat 4.4% | ✅ | ✅ $20k/beneficiary |

> **Disclaimer:** Tax laws change annually. Always consult a CPA or financial advisor for your specific situation. This tool is for planning and scenario analysis — not tax advice.

---

## Roadmap

- [ ] Social Security benefit estimation
- [ ] Backdoor Roth IRA modeling
- [ ] Scenario A vs B side-by-side comparison
- [ ] Rent vs Buy breakeven analysis
- [ ] FIRE number calculator (4% rule)
- [ ] Export to PDF report
- [ ] More states

---

## Contributing

PRs welcome. Please add tests for any new financial logic — correctness is the top priority.

```bash
# Run the full test suite before submitting
pytest tests/ -v --tb=short
```

---

## License

MIT — use it, fork it, build on it.
