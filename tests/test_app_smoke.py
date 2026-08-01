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
    "🎯 Strategies", "🚗 Car", "🏢 Business", "🗓️ Events",
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


def test_build_sidebar_returns_valid_plan_without_config():
    """The no-config path yields a usable FinancialPlan (income default applied)."""
    at = AppTest.from_string(_SIDEBAR_ONLY, default_timeout=60)
    at.session_state["loaded_plan"] = None
    at.run()
    assert not at.exception, _errors(at)
    plan = at.session_state["loaded_plan"]  # build_sidebar persists the plan
    assert plan is not None
    assert plan.income.gross_annual_income > 0
