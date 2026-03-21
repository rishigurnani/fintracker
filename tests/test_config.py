"""Tests for YAML config serialization/deserialization."""
import os
import tempfile
import pytest

from fintracker.models import FilingStatus, State, StrategyToggles, TimelineEvent
from fintracker.config import save_plan, load_plan, load_plan_or_sample, _default_plan


class TestConfigRoundTrip:

    def _round_trip(self, plan):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        try:
            save_plan(plan, path)
            return load_plan(path)
        finally:
            os.unlink(path)

    def test_default_plan_round_trips(self):
        plan = _default_plan()
        loaded = self._round_trip(plan)
        assert loaded.income.gross_annual_income == plan.income.gross_annual_income
        assert loaded.housing.home_price == plan.housing.home_price
        assert loaded.housing.interest_rate == plan.housing.interest_rate
        assert loaded.investments.annual_market_return == plan.investments.annual_market_return

    def test_filing_status_preserved(self):
        plan = _default_plan()
        plan.income.filing_status = FilingStatus.MARRIED_FILING_JOINTLY
        loaded = self._round_trip(plan)
        assert loaded.income.filing_status == FilingStatus.MARRIED_FILING_JOINTLY

    def test_state_preserved(self):
        plan = _default_plan()
        plan.income.state = State.CALIFORNIA
        loaded = self._round_trip(plan)
        assert loaded.income.state == State.CALIFORNIA

    def test_strategy_toggles_preserved(self):
        plan = _default_plan()
        plan.strategies = StrategyToggles(
            maximize_hsa=False,
            maximize_401k=True,
            use_529_state_deduction=True,
            use_roth_ladder=True,
            roth_conversion_annual_amount=15_000,
        )
        loaded = self._round_trip(plan)
        assert loaded.strategies.maximize_hsa is False
        assert loaded.strategies.maximize_401k is True
        assert loaded.strategies.use_529_state_deduction is True
        assert loaded.strategies.roth_conversion_annual_amount == 15_000.0

    def test_timeline_events_preserved(self):
        plan = _default_plan()
        plan.timeline_events = [
            TimelineEvent(year=2, description="Marry", marriage=True, extra_one_time_expense=20_000),
            TimelineEvent(year=3, description="Child", new_child=True),
        ]
        loaded = self._round_trip(plan)
        assert len(loaded.timeline_events) == 2
        assert loaded.timeline_events[0].year == 2
        assert loaded.timeline_events[0].marriage is True
        assert loaded.timeline_events[0].extra_one_time_expense == 20_000.0
        assert loaded.timeline_events[1].new_child is True

    def test_projection_years_preserved(self):
        plan = _default_plan()
        plan.projection_years = 25
        loaded = self._round_trip(plan)
        assert loaded.projection_years == 25

    def test_numeric_precision_maintained(self):
        plan = _default_plan()
        plan.income.gross_annual_income = 137_500.50
        loaded = self._round_trip(plan)
        assert loaded.income.gross_annual_income == pytest.approx(137_500.50, abs=0.01)

    def test_renting_plan_round_trips(self):
        plan = _default_plan()
        plan.housing.is_renting = True
        plan.housing.monthly_rent = 2_500
        loaded = self._round_trip(plan)
        assert loaded.housing.is_renting is True
        assert loaded.housing.monthly_rent == 2_500.0


class TestLoadOrSample:

    def test_missing_file_returns_default(self):
        plan = load_plan_or_sample("/nonexistent/path/config.yaml")
        assert plan is not None
        assert plan.income.gross_annual_income > 0

    def test_load_plan_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_plan("/nonexistent/config.yaml")

    def test_sample_yaml_is_loadable(self):
        """The tracked sample.yaml must always be parseable."""
        import pathlib
        sample_path = pathlib.Path(__file__).parent.parent / "config" / "sample.yaml"
        if sample_path.exists():
            plan = load_plan(sample_path)
            assert plan.income.gross_annual_income > 0
            assert plan.projection_years > 0
