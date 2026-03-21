"""
Strategy analysis engine.

Quantifies the dollar value of each tax-optimization strategy so users can
see exactly how much each toggle is worth.
"""
from __future__ import annotations

from dataclasses import dataclass

from fintracker.models import (
    FilingStatus, IncomeProfile, InvestmentProfile, StrategyToggles, State
)
from fintracker.tax_engine import TaxEngine, _STATE_TAX_CONFIGS


@dataclass
class StrategyResult:
    """Annual dollar value of each active strategy."""
    hsa_annual_savings: float         # Federal + State + FICA saved via HSA
    k401_annual_savings: float        # Federal + State saved via 401k
    state_529_annual_savings: float   # State-only savings from 529 deductions
    roth_ladder_annual_benefit: float # Estimated future tax savings from Roth conversions
    total_annual_savings: float

    # Recommended contribution amounts
    recommended_hsa_contribution: float
    recommended_401k_contribution: float
    recommended_roth_ira_contribution: float

    notes: list[str]


# 2024 IRS contribution limits
_HSA_LIMITS_2024 = {
    FilingStatus.SINGLE: 4_150,
    FilingStatus.MARRIED_FILING_JOINTLY: 8_300,
    FilingStatus.HEAD_OF_HOUSEHOLD: 4_150,
}
_401K_LIMIT_2024 = 23_000
_401K_CATCHUP_LIMIT_2024 = 30_500  # Age 50+
_ROTH_IRA_LIMIT_2024 = {
    FilingStatus.SINGLE: 7_000,
    FilingStatus.MARRIED_FILING_JOINTLY: 14_000,
    FilingStatus.HEAD_OF_HOUSEHOLD: 7_000,
}
_ROTH_PHASEOUT_SINGLE = (146_000, 161_000)
_ROTH_PHASEOUT_MFJ = (230_000, 240_000)


class StrategyEngine:
    """Calculates value of each financial strategy."""

    def __init__(self):
        self._tax_engine = TaxEngine()

    def analyze(
        self,
        income: IncomeProfile,
        investments: InvestmentProfile,
        strategies: StrategyToggles,
        num_children: int = 0,
        age: int = 35,
    ) -> StrategyResult:
        notes: list[str] = []

        # Baseline tax (no strategies)
        baseline_inv = InvestmentProfile(
            annual_401k_contribution=0,
            annual_hsa_contribution=0,
            annual_529_contribution=0,
            annual_roth_ira_contribution=0,
        )
        baseline_strat = StrategyToggles(
            maximize_hsa=False,
            use_529_state_deduction=False,
            maximize_401k=False,
            use_roth_ladder=False,
        )
        baseline = self._tax_engine.calculate(income, baseline_inv, baseline_strat, num_children)

        # --- HSA savings ---
        hsa_limit = _HSA_LIMITS_2024.get(income.filing_status, 4_150)
        hsa_inv = InvestmentProfile(annual_hsa_contribution=hsa_limit)
        hsa_strat = StrategyToggles(maximize_hsa=True, maximize_401k=False, use_529_state_deduction=False)
        hsa_result = self._tax_engine.calculate(income, hsa_inv, hsa_strat, num_children)
        hsa_savings = baseline.total_annual_tax - hsa_result.total_annual_tax

        if strategies.maximize_hsa:
            notes.append(
                f"HSA: Contributing ${hsa_limit:,} saves ~${hsa_savings:,.0f}/yr "
                f"(Federal + FICA + State)."
            )
        else:
            notes.append(
                f"💡 Tip: Maximizing your HSA (${hsa_limit:,}) could save ~${hsa_savings:,.0f}/yr in taxes."
            )

        # --- 401k savings ---
        k401_limit = _401K_CATCHUP_LIMIT_2024 if age >= 50 else _401K_LIMIT_2024
        k401_inv = InvestmentProfile(annual_401k_contribution=k401_limit)
        k401_strat = StrategyToggles(maximize_401k=True, maximize_hsa=False, use_529_state_deduction=False)
        k401_result = self._tax_engine.calculate(income, k401_inv, k401_strat, num_children)
        k401_savings = baseline.total_annual_tax - k401_result.total_annual_tax

        if strategies.maximize_401k:
            notes.append(
                f"401k: Contributing ${k401_limit:,} saves ~${k401_savings:,.0f}/yr "
                f"(Federal + State)."
            )

        # --- 529 savings (state-level only) ---
        state_529_savings = 0.0
        if num_children > 0 and income.state != State.OTHER:
            config = _STATE_TAX_CONFIGS.get(income.state)
            if config and config.allows_529_deduction:
                max_529_deduction = config.per_beneficiary_529_deduction * num_children
                state_rate = config.brackets[-1][1] if config.brackets else 0.0
                state_529_savings = max_529_deduction * state_rate

                if strategies.use_529_state_deduction:
                    notes.append(
                        f"529: Deducting ${max_529_deduction:,} for {num_children} child(ren) "
                        f"saves ~${state_529_savings:,.0f}/yr in {config.name} state tax."
                    )
                else:
                    notes.append(
                        f"💡 Tip: Using {config.name}'s 529 deduction could save "
                        f"~${state_529_savings:,.0f}/yr in state taxes."
                    )

        # --- Roth conversion ladder ---
        roth_benefit = 0.0
        if strategies.use_roth_ladder and strategies.roth_conversion_annual_amount > 0:
            marginal = self._tax_engine.marginal_rate(income, investments, strategies)
            # Rough estimate: converting now at current rate avoids RMD taxes in retirement
            # Assumes 20+ years of tax-free compounding
            years_to_retirement = max(1, 65 - age)
            future_value_factor = (1 + 0.08) ** years_to_retirement
            converted = strategies.roth_conversion_annual_amount
            future_balance = converted * future_value_factor
            # Estimated retirement marginal rate at 24%
            est_retirement_rate = 0.24
            roth_benefit = future_balance * (est_retirement_rate - marginal) / years_to_retirement
            notes.append(
                f"Roth ladder: Converting ${converted:,}/yr now at {marginal:.1%} marginal rate "
                f"could yield ~${roth_benefit:,.0f}/yr in average annual tax benefit."
            )

        # --- Roth IRA eligibility check ---
        gross = income.total_gross_income
        roth_limit = _ROTH_IRA_LIMIT_2024.get(income.filing_status, 7_000)
        phaseout = (
            _ROTH_PHASEOUT_MFJ
            if income.filing_status == FilingStatus.MARRIED_FILING_JOINTLY
            else _ROTH_PHASEOUT_SINGLE
        )
        if gross > phaseout[1]:
            notes.append(
                "⚠️  Your income exceeds the Roth IRA limit. Consider a Backdoor Roth IRA."
            )
            recommended_roth = 0.0
        elif gross > phaseout[0]:
            reduced_pct = 1 - (gross - phaseout[0]) / (phaseout[1] - phaseout[0])
            recommended_roth = roth_limit * reduced_pct
            notes.append(
                f"Your Roth IRA contribution is phased out. Reduced limit: ~${recommended_roth:,.0f}."
            )
        else:
            recommended_roth = roth_limit

        total = hsa_savings + k401_savings + state_529_savings + roth_benefit

        return StrategyResult(
            hsa_annual_savings=hsa_savings,
            k401_annual_savings=k401_savings,
            state_529_annual_savings=state_529_savings,
            roth_ladder_annual_benefit=roth_benefit,
            total_annual_savings=total,
            recommended_hsa_contribution=hsa_limit,
            recommended_401k_contribution=k401_limit,
            recommended_roth_ira_contribution=recommended_roth,
            notes=notes,
        )
