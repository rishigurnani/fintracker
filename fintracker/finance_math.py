"""
Pure financial math primitives.

Domain-agnostic building blocks shared across the mortgage, car-loan, tax, and
strategy engines.  Everything here is a plain function with no dependency on the
``fintracker`` data models, so it can be unit-tested and reused freely.

Consolidating these here removes several copies of the same formula that had
drifted into slightly different algebraic forms (e.g. the amortising-loan
payment appeared once in ``mortgage.py`` and once in ``projections.py``).
"""
from __future__ import annotations


def monthly_amortized_payment(principal: float, annual_rate: float, term_years: int) -> float:
    """Level monthly payment that fully amortises ``principal`` over the term.

    Works for mortgages and car loans alike.  Returns 0 for a non-positive
    principal and falls back to straight-line repayment at a 0% rate.
    """
    if principal <= 0:
        return 0.0
    n = term_years * 12
    if n <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / n
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** -n)


def progressive_tax(taxable_income: float, brackets: list[tuple[float, float]]) -> float:
    """Total tax owed applying progressive ``(upper_bound, marginal_rate)`` brackets.

    The final bracket's upper bound is expected to be ``float("inf")``.  An empty
    bracket list means no tax.
    """
    tax = 0.0
    prev_bound = 0.0
    for upper_bound, rate in brackets:
        if taxable_income <= prev_bound:
            break
        taxable_slice = min(taxable_income, upper_bound) - prev_bound
        tax += taxable_slice * rate
        prev_bound = upper_bound
    return tax


def marginal_rate_at(taxable_income: float, brackets: list[tuple[float, float]]) -> float:
    """Marginal rate that applies at ``taxable_income`` for the given brackets.

    Returns 0 when there are no brackets (e.g. a no-income-tax state).
    """
    if not brackets:
        return 0.0
    rate = brackets[-1][1]
    for upper_bound, bracket_rate in brackets:
        if taxable_income <= upper_bound:
            return bracket_rate
    return rate


def linear_phaseout(x: float, low: float, high: float) -> float:
    """Fraction remaining as ``x`` moves across a phase-out window.

    Returns 1.0 at or below ``low``, 0.0 at or above ``high``, and a linear
    taper in between.  Used for AOTC, QBI, and Roth-IRA income phase-outs.
    """
    if x <= low:
        return 1.0
    if x >= high or high <= low:
        return 0.0
    return 1.0 - (x - low) / (high - low)
