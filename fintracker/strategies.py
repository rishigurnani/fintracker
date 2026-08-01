"""
Strategy analysis engine.

Quantifies the dollar value of each tax-optimization strategy so users can
see exactly how much each toggle is worth.
"""
from __future__ import annotations

from dataclasses import dataclass

from fintracker.constants import (
    HSA_LIMIT_SINGLE, HSA_LIMIT_FAMILY, LIMIT_401K, LIMIT_401K_CATCHUP,
    ROTH_IRA_LIMIT, ROTH_PHASEOUT_SINGLE, ROTH_PHASEOUT_MFJ,
)
from fintracker.finance_math import linear_phaseout
from fintracker.models import (
    IncomeProfile, InvestmentProfile, StrategyToggles, State,
    by_filing_status,
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


class StrategyEngine:
    """Calculates value of each financial strategy."""

    def __init__(self):
        self._tax_engine = TaxEngine()

    def _savings_from(self, income, num_children, baseline, inv, strat) -> float:
        """Annual tax reduction of a maximized-contribution scenario vs the baseline."""
        result = self._tax_engine.calculate(income, inv, strat, num_children)
        return baseline.total_annual_tax - result.total_annual_tax

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
        hsa_limit = by_filing_status(
            income.filing_status, HSA_LIMIT_SINGLE, HSA_LIMIT_FAMILY, hoh=HSA_LIMIT_SINGLE)
        hsa_savings = self._savings_from(
            income, num_children, baseline,
            InvestmentProfile(annual_hsa_contribution=hsa_limit),
            StrategyToggles(maximize_hsa=True, maximize_401k=False, use_529_state_deduction=False),
        )

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
        k401_limit = LIMIT_401K_CATCHUP if age >= 50 else LIMIT_401K
        k401_savings = self._savings_from(
            income, num_children, baseline,
            InvestmentProfile(annual_401k_contribution=k401_limit),
            StrategyToggles(maximize_401k=True, maximize_hsa=False, use_529_state_deduction=False),
        )

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
        roth_limit = by_filing_status(
            income.filing_status, ROTH_IRA_LIMIT, ROTH_IRA_LIMIT * 2, hoh=ROTH_IRA_LIMIT)
        phaseout = by_filing_status(
            income.filing_status, ROTH_PHASEOUT_SINGLE, ROTH_PHASEOUT_MFJ)
        if gross > phaseout[1]:
            notes.append(
                "⚠️  Your income exceeds the Roth IRA limit. Consider a Backdoor Roth IRA."
            )
            recommended_roth = 0.0
        elif gross > phaseout[0]:
            recommended_roth = roth_limit * linear_phaseout(gross, phaseout[0], phaseout[1])
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
