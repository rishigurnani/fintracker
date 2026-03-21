"""Tests for StrategyEngine."""
import pytest
from fintracker.models import FilingStatus, State, IncomeProfile, InvestmentProfile, StrategyToggles
from fintracker.strategies import StrategyEngine


@pytest.fixture
def engine():
    return StrategyEngine()


@pytest.fixture
def ga_income():
    return IncomeProfile(
        gross_annual_income=120_000,
        filing_status=FilingStatus.SINGLE,
        state=State.GEORGIA,
    )


@pytest.fixture
def mfj_income():
    return IncomeProfile(
        gross_annual_income=180_000,
        filing_status=FilingStatus.MARRIED_FILING_JOINTLY,
        state=State.GEORGIA,
    )


@pytest.fixture
def standard_investments():
    return InvestmentProfile(
        annual_401k_contribution=23_000,
        annual_hsa_contribution=4_150,
        annual_529_contribution=8_000,
    )


@pytest.fixture
def all_on():
    return StrategyToggles(
        maximize_hsa=True,
        maximize_401k=True,
        use_529_state_deduction=True,
        use_roth_ladder=False,
    )


class TestStrategyAnalysis:

    def test_hsa_savings_positive(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on)
        assert result.hsa_annual_savings > 0

    def test_401k_savings_positive(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on)
        assert result.k401_annual_savings > 0

    def test_529_savings_zero_no_children(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on, num_children=0)
        assert result.state_529_annual_savings == 0.0

    def test_529_savings_positive_with_children(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on, num_children=2)
        assert result.state_529_annual_savings > 0

    def test_529_savings_scales_with_children(self, engine, ga_income, standard_investments, all_on):
        r1 = engine.analyze(ga_income, standard_investments, all_on, num_children=1)
        r2 = engine.analyze(ga_income, standard_investments, all_on, num_children=2)
        assert r2.state_529_annual_savings > r1.state_529_annual_savings

    def test_total_savings_sum_of_components(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on, num_children=1)
        expected_total = (
            result.hsa_annual_savings
            + result.k401_annual_savings
            + result.state_529_annual_savings
            + result.roth_ladder_annual_benefit
        )
        assert result.total_annual_savings == pytest.approx(expected_total, abs=0.01)

    def test_texas_no_529_savings(self, engine, standard_investments, all_on):
        """Texas has no income tax → no 529 state deduction savings."""
        tx_income = IncomeProfile(
            gross_annual_income=120_000,
            filing_status=FilingStatus.SINGLE,
            state=State.TEXAS,
        )
        result = engine.analyze(tx_income, standard_investments, all_on, num_children=2)
        assert result.state_529_annual_savings == 0.0

    def test_high_income_roth_phaseout_note(self, engine, standard_investments, all_on):
        """High income should trigger a Roth phase-out warning."""
        high_income = IncomeProfile(
            gross_annual_income=200_000,
            filing_status=FilingStatus.SINGLE,
            state=State.GEORGIA,
        )
        result = engine.analyze(high_income, standard_investments, all_on)
        notes_text = " ".join(result.notes)
        assert "Roth" in notes_text or "backdoor" in notes_text.lower() or "phased" in notes_text.lower()

    def test_recommended_contributions_positive(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on)
        assert result.recommended_hsa_contribution > 0
        assert result.recommended_401k_contribution > 0

    def test_mfj_higher_hsa_limit_than_single(self, engine, mfj_income, standard_investments, all_on):
        """MFJ filer gets higher HSA limit."""
        ga_single = IncomeProfile(
            gross_annual_income=120_000,
            filing_status=FilingStatus.SINGLE,
            state=State.GEORGIA,
        )
        r_single = engine.analyze(ga_single, standard_investments, all_on)
        r_mfj = engine.analyze(mfj_income, standard_investments, all_on)
        assert r_mfj.recommended_hsa_contribution > r_single.recommended_hsa_contribution

    def test_notes_list_is_not_empty(self, engine, ga_income, standard_investments, all_on):
        result = engine.analyze(ga_income, standard_investments, all_on)
        assert len(result.notes) > 0
