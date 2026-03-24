# Contributing to fintracker

## Rules for AI assistants working on this codebase

**These rules exist because bugs have been introduced and re-introduced. Read them before writing any code.**

### Every code change requires a test

No exceptions. If you fix a bug, add a regression test that would have caught it.
If you add a feature, add tests that verify its behavior.
If you refactor, confirm existing tests still pass.

This is not optional and should not be skipped for "simple" changes.
Simple changes have caused the most damage in this project.

### Known bugs that were introduced and fixed — each has a test

These bugs were real, caused incorrect financial projections, and must never regress:

| Bug | Test location |
|---|---|
| `max(0, breathing_room)` silently ignored deficits | `test_accounting.py::TestDeficitSpending` |
| Mortgage paydown used approximation (~$35k drift over 30yr) | `test_accounting.py::TestMortgagePaydown` |
| Home equity used start-of-year balance instead of end-of-year | `test_accounting.py::TestMortgagePaydown::test_home_equity_uses_end_of_year_balance` |
| `end_of_year_balance` unbound when `is_renting=True` | `test_projections.py` (renting plan tests) |
| `maximize_401k=True` overrode stated contribution to IRS max | `test_accounting.py::TestMaximizeContributions` |
| `maximize_hsa=True` overrode stated contribution to IRS max | `test_accounting.py::TestMaximizeContributions` |
| `current_brokerage_balance` not passed from sidebar to plan | `test_accounting.py` |
| `monthly_childcare` hidden when `num_children=0`, defaulting to $0 | `test_accounting.py::TestChildcare` |
| Marriage event didn't upgrade HSA to family tier | `test_accounting.py::TestHSAFamilyTier` |
| Dual-income partner salary growth clobbered single-income gross | `test_accounting.py` (dual income tests) |

### The test-writing workflow

1. Write the fix
2. Write the test that would have caught the bug **before** the fix
3. Confirm the test **fails** on the old code and **passes** on the new code
4. Run the full suite and confirm nothing else broke

### Running tests

```bash
cd fintracker
pytest tests/ -v
```

All tests must pass before any change is considered complete.

### Contribution amounts are nominal, not inflated

`annual_401k_contribution`, `annual_hsa_contribution`, and `annual_529_contribution`
in `InvestmentProfile` are the user's stated nominal amounts.
The projection engine must use these exactly as stated (capped at IRS limits).
These values must **not** be multiplied by an inflation factor.
`maximize_401k` and `maximize_hsa` flags control tax treatment and the strategy
analyzer only — they do not override the contribution amount.