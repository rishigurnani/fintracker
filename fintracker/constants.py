"""
IRS contribution limits and income thresholds (2024 tax year).

Single source of truth for the limits that were previously duplicated across
``strategies.py`` and ``projections.py`` (and had already drifted — the 401k cap
was defined as both 23,000 and 30,500 in different modules).

Update these once per year; every engine reads from here.
"""
from __future__ import annotations

# --- HSA (2024) ---
HSA_LIMIT_SINGLE = 4_150
HSA_LIMIT_FAMILY = 8_300

# --- 401k / solo 401k (2024) ---
LIMIT_401K = 23_000            # elective deferral, under age 50
LIMIT_401K_CATCHUP = 30_500    # age 50+ (base + catch-up)
LIMIT_SOLO_401K = 69_000       # solo 401k / SEP combined owner limit


def limit_401k(age: int | None) -> float:
    """401k elective-deferral ceiling for a person of the given age.

    The catch-up ceiling applies at age 50+.  When ``age`` is ``None`` (no age
    information — e.g. a plan without a RetirementProfile) the catch-up ceiling
    is used: it is the most permissive legal cap and keeps age-agnostic plans
    unchanged.
    """
    if age is None or age >= 50:
        return LIMIT_401K_CATCHUP
    return LIMIT_401K

# --- Roth IRA (2024) ---
ROTH_IRA_LIMIT = 7_000                     # per person, per year
ROTH_PHASEOUT_SINGLE = (146_000, 161_000)  # MAGI phase-out (single/HoH)
ROTH_PHASEOUT_MFJ = (230_000, 240_000)     # MAGI phase-out (married filing jointly)


# --- Medicare IRMAA (2024) ---
# Income-Related Monthly Adjustment Amount: a MAGI-tiered surcharge added on top
# of the base Medicare Part B + Part D premium, charged *per enrolled person*.
# Each tuple is (MAGI upper bound, annual surcharge per person). The surcharge is
# the annualised Part B + Part D adjustment for that bracket (2024 figures). The
# final bracket's bound is +inf. Thresholds are inflation-indexed by the caller,
# mirroring how the SSA re-indexes them each year.
_IRMAA_SURCHARGE_SINGLE: list[tuple[float, float]] = [
    (103_000,       0.0),
    (129_000,     994.80),   # Part B +$69.90/mo + Part D +$12.90/mo
    (161_000,   2_496.00),   # +$174.70/mo + $33.30/mo
    (193_000,   3_999.60),   # +$279.50/mo + $53.80/mo
    (500_000,   5_502.00),   # +$384.30/mo + $74.20/mo
    (float("inf"), 6_003.60),  # +$419.30/mo + $81.00/mo
]

# MFJ thresholds are exactly double the single thresholds (same surcharges).
_IRMAA_SURCHARGE_MFJ: list[tuple[float, float]] = [
    (bound * 2 if bound != float("inf") else bound, surcharge)
    for bound, surcharge in _IRMAA_SURCHARGE_SINGLE
]


def irmaa_annual_surcharge(magi: float, is_married: bool,
                           inflation_factor: float = 1.0) -> float:
    """Annual IRMAA surcharge *per enrolled person* for a given MAGI.

    Returns the surcharge that sits on top of the base Medicare premium. Married
    filers use the doubled MFJ thresholds; the returned figure is still per
    person (a couple where both are enrolled pays it twice). ``inflation_factor``
    scales the bracket bounds to the projection year (1.0 = base year), so a
    retiree stays in the same *real* bracket instead of creeping upward.
    """
    table = _IRMAA_SURCHARGE_MFJ if is_married else _IRMAA_SURCHARGE_SINGLE
    for bound, surcharge in table:
        if magi <= bound * inflation_factor:
            return surcharge
    return table[-1][1]
