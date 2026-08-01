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
