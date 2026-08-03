"""
Tests for the retirement "spending smile": discretionary lifestyle (vacation,
pets, monthly "other") is scaled down in later life — full spend through 74,
−10% for 75–84, −20% at 85+ — while medical, Medicare, insurance, care and LTC
costs are left untouched.

Inflation/market growth are pinned to 0 (via the shared builders) so the smile
shows up as an exact, isolated reduction in the discretionary bucket.
"""
import pytest

from fintracker.models import (
    FilingStatus, IncomeProfile, LifestyleProfile, RetirementProfile, State,
)
from fintracker.projections import ProjectionEngine
from tests.builders import make_plan, investments, zero_lifestyle


def _project(plan):
    return ProjectionEngine(plan).run_deterministic()


# ── Unit: the multiplier itself ──────────────────────────────────────────────

class TestDiscretionaryFactor:
    rp = RetirementProfile()

    @pytest.mark.parametrize("age,expected", [
        (None, 1.0),
        (64, 1.0),
        (74, 1.0),   # last full-spend year
        (75, 0.90),  # slow-go band starts
        (84, 0.90),  # last slow-go year
        (85, 0.80),  # no-go band starts
        (100, 0.80),
    ])
    def test_factor_by_age(self, age, expected):
        assert self.rp.discretionary_spending_factor(age) == pytest.approx(expected)

    def test_factors_are_configurable(self):
        rp = RetirementProfile(
            spending_smile_slowgo_age=70, spending_smile_slowgo_factor=0.5,
            spending_smile_nogo_age=90, spending_smile_nogo_factor=0.25,
        )
        assert rp.discretionary_spending_factor(69) == pytest.approx(1.0)
        assert rp.discretionary_spending_factor(70) == pytest.approx(0.5)
        assert rp.discretionary_spending_factor(90) == pytest.approx(0.25)

    def test_smile_disabled_when_factors_are_one(self):
        rp = RetirementProfile(
            spending_smile_slowgo_factor=1.0, spending_smile_nogo_factor=1.0,
        )
        assert rp.discretionary_spending_factor(85) == pytest.approx(1.0)


# ── Integration: the smile flows into annual_lifestyle_cost ───────────────────

class TestSmileInProjection:
    # Discretionary bucket = vacation + pets + other = 10k + 2k + 12k = 24k.
    DISCRETIONARY = 24_000

    def _plan(self):
        return make_plan(
            income=IncomeProfile(120_000, FilingStatus.SINGLE, State.TEXAS),
            lifestyle=zero_lifestyle(
                annual_vacation=10_000,
                num_pets=1, annual_pet_cost=2_000,
                monthly_other_recurring=1_000,   # 12k/yr
            ),
            # Big cash pile funds retirement deficits without 401k withdrawals,
            # so MAGI (and thus IRMAA) stays at 0.
            investments=investments(current_liquid_cash=5_000_000),
            # Already retired at the start; Medicare premium zeroed so lifestyle
            # equals the discretionary bucket exactly.
            retirement=RetirementProfile(
                current_age=70, retirement_age=65, annual_medicare_premium=0.0,
            ),
            projection_years=20,
        )

    def _at_age(self, snaps, age):
        return snaps[age - 70]  # year 1 == age 70

    def test_full_spend_through_74(self):
        snaps = _project(self._plan())
        assert self._at_age(snaps, 74).annual_lifestyle_cost == pytest.approx(self.DISCRETIONARY)

    def test_slowgo_10pct_cut_at_75(self):
        snaps = _project(self._plan())
        assert self._at_age(snaps, 75).annual_lifestyle_cost == pytest.approx(self.DISCRETIONARY * 0.90)
        assert self._at_age(snaps, 84).annual_lifestyle_cost == pytest.approx(self.DISCRETIONARY * 0.90)

    def test_nogo_20pct_cut_at_85(self):
        snaps = _project(self._plan())
        assert self._at_age(snaps, 85).annual_lifestyle_cost == pytest.approx(self.DISCRETIONARY * 0.80)

    def test_smile_does_not_touch_medical(self):
        # With a medical OOP cost present, only the discretionary slice is scaled;
        # medical passes through unreduced at every age.
        plan = self._plan()
        plan.lifestyle = LifestyleProfile(
            annual_vacation=10_000, num_pets=1, annual_pet_cost=2_000,
            monthly_other_recurring=1_000,
            annual_medical_oop=8_000, medical_auto_scale=False,
        )
        snaps = _project(plan)
        # age 85: discretionary * 0.8 + full medical
        expected = self.DISCRETIONARY * 0.80 + 8_000
        s85 = self._at_age(snaps, 85)
        assert s85.annual_lifestyle_cost == pytest.approx(expected)
        assert s85.annual_medical_oop == pytest.approx(8_000)
