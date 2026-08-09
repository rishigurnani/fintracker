"""Rough Social Security retirement-benefit estimator (today's dollars).

Kept self-contained so it can feed ``RetirementProfile.estimated_social_security_annual``
without entangling the tax or projection engines. The PIA "bend point" formula is
just a progressive bracket calc, so it reuses :func:`progressive_tax` rather than
re-deriving bracket math (same primitive the tax engine uses).

Simplifications (documented so callers know the error bars):
  * AIME is derived from one steady real salary rather than a full 35-year
    indexed earnings history; missing years toward 35 count as $0.
  * dollar constants track the ``SS_WAGE_BASE`` tax year for consistency.
"""
from __future__ import annotations

from fintracker.constants import SS_WAGE_BASE
from fintracker.finance_math import progressive_tax

FULL_RETIREMENT_AGE = 67          # for anyone born 1960 or later
CREDITS_TO_QUALIFY = 40           # ~10 years of covered work
_MAX_YEARS = 35                   # AIME averages your top-35 earning years

# Monthly PIA bend points (same tax year as SS_WAGE_BASE): 90% / 32% / 15%.
_PIA_BRACKETS: list[tuple[float, float]] = [
    (1_174.0, 0.90), (7_078.0, 0.32), (float("inf"), 0.15),
]


def estimate_aime(annual_earnings: float, years_worked: float = _MAX_YEARS) -> float:
    """Average Indexed Monthly Earnings from a steady real salary (today's $).

    Earnings are capped at the SS wage base (income above it earns no benefit);
    working fewer than 35 years averages the shortfall in as $0.
    """
    capped = min(max(annual_earnings, 0.0), SS_WAGE_BASE)
    years = min(max(years_worked, 0.0), _MAX_YEARS)
    return capped * years / _MAX_YEARS / 12.0


def claim_factor(claim_age: float, fra: float = FULL_RETIREMENT_AGE) -> float:
    """Benefit multiplier vs. PIA for claiming at ``claim_age`` (clamped 62–70).

    Early: −5/9%/mo for the first 36 months, −5/12%/mo beyond (→ 0.70 at 62 for
    an FRA of 67). Delayed: +2/3%/mo up to age 70 (→ 1.24 at 70). 1.0 at FRA.
    """
    age = min(max(claim_age, 62.0), 70.0)
    months = round((age - fra) * 12)
    if months >= 0:
        return 1.0 + months * (2.0 / 3.0) / 100.0
    early = -months
    return 1.0 - (min(early, 36) * (5.0 / 9.0) + max(early - 36, 0) * (5.0 / 12.0)) / 100.0


def estimate_annual_benefit(
    annual_earnings: float, *, years_worked: float = _MAX_YEARS,
    claim_age: float = FULL_RETIREMENT_AGE, credits: int = CREDITS_TO_QUALIFY,
    haircut: float = 1.0,
) -> float:
    """Estimated annual benefit in today's dollars — 0 if under 40 credits.

    ``haircut`` (0–1) applies a trust-fund-shortfall discount (e.g. 0.75 mirrors
    the official pessimistic case); 1.0 assumes benefits are paid in full.
    """
    if credits < CREDITS_TO_QUALIFY:
        return 0.0
    pia_monthly = progressive_tax(estimate_aime(annual_earnings, years_worked), _PIA_BRACKETS)
    return pia_monthly * 12.0 * claim_factor(claim_age) * haircut
