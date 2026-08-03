"""Smoke tests for the Streamlit UI (app.py).

Regression guard for the crash where ``build_sidebar`` constructed
``IncomeProfile()`` with no arguments when no config was loaded
(``st.session_state["loaded_plan"]`` unset / None) — e.g. a fresh clone with
no ``config/personal.yaml``. That raised
``TypeError: IncomeProfile.__init__() missing 1 required positional argument``
and took down the whole app on first load.
"""
import pathlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app.py")

SECTIONS = [
    "💵 Income", "🏠 Housing", "🌿 Lifestyle", "📊 Investments",
    "🎯 Strategies", "🚗 Car", "🏢 Business", "🗓️ Events", "🛟 Failsafes",
]

# Runs only build_sidebar() — skips the heavy Monte Carlo dashboard so the
# per-section sweep stays fast while still exercising every no-config fallback.
_SIDEBAR_ONLY = (
    "import sys\n"
    f"sys.path.insert(0, {REPO_ROOT.as_posix()!r})\n"
    "import app\n"
    "app.build_sidebar()\n"
)


def _errors(at):
    return [str(e.value) for e in at.exception]


def test_full_app_renders_without_config():
    """End-to-end: with no loaded plan the whole app must render, not crash."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["loaded_plan"] = None  # simulate: no personal.yaml present
    at.run()
    assert not at.exception, _errors(at)


@pytest.mark.parametrize("section", SECTIONS)
def test_sidebar_section_renders_without_config(section):
    """Each sidebar section builds from defaults (defaults is None) without error.

    Guards every ``_wd(None, ...)`` fallback and each ``_*_section`` helper.
    """
    at = AppTest.from_string(_SIDEBAR_ONLY, default_timeout=60)
    at.session_state["loaded_plan"] = None
    at.run()
    # Switch to the target section, keeping the no-config state on the re-run.
    at.sidebar.radio[0].set_value(section)
    at.session_state["loaded_plan"] = None
    at.run()
    assert not at.exception, _errors(at)


def test_failsafes_section_seeds_and_roundtrips_loaded_plan():
    """The Failsafes section renders a loaded failsafe and rebuilds it into the plan."""
    at = AppTest.from_string(_SIDEBAR_ONLY, default_timeout=60)
    at.run()  # first run establishes the app + widgets

    import sys, pathlib as _pl
    sys.path.insert(0, _pl.Path(__file__).resolve().parent.parent.as_posix())
    from tests.builders import make_plan
    from fintracker.models import Failsafe, FailsafeCondition, FailsafeAction

    fs = Failsafe(
        name="partner returns to work", match="any", delay_years=1, duration_years=5, once=True,
        conditions=[FailsafeCondition("brokerage_balance", "below", 100_000, True, 15, 30)],
        action=FailsafeAction(partner_income=100_000, present_value=True),
    )
    plan = make_plan()
    plan.failsafes = [fs]
    at.session_state["loaded_plan"] = plan
    at.sidebar.radio[0].set_value("🛟 Failsafes")
    at.run()
    assert not at.exception, _errors(at)

    rebuilt = at.session_state["loaded_plan"]
    assert len(rebuilt.failsafes) == 1
    got = rebuilt.failsafes[0]
    assert got.name == "partner returns to work"
    assert got.delay_years == 1 and got.duration_years == 5 and got.once is True
    assert got.conditions[0].metric == "brokerage_balance"
    assert got.conditions[0].threshold == 100_000
    assert got.conditions[0].start_year == 15 and got.conditions[0].end_year == 30
    assert got.action.partner_income == 100_000


def test_build_sidebar_returns_valid_plan_without_config():
    """The no-config path yields a usable FinancialPlan (income default applied)."""
    at = AppTest.from_string(_SIDEBAR_ONLY, default_timeout=60)
    at.session_state["loaded_plan"] = None
    at.run()
    assert not at.exception, _errors(at)
    plan = at.session_state["loaded_plan"]  # build_sidebar persists the plan
    assert plan is not None
    assert plan.income.gross_annual_income > 0
