"""Regression tests for the Monte Carlo parameter panel (`_mc_simulation_params`).

The panel was simplified: the returns and inflation controls now share one
`_series_sampling` helper, and each normal-distribution σ slider is shown only
when its series is set to parametric — instead of a permanently greyed-out
slider. These tests guard that behaviour and that the run-kwargs contract the
caller depends on stays intact (no functionality lost in the cleanup).
"""
import pathlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_PARAMS_ONLY = (
    "import sys\n"
    f"sys.path.insert(0, {REPO_ROOT.as_posix()!r})\n"
    "import streamlit as st\n"
    "from tests.builders import make_plan\n"
    "import app\n"
    "st.session_state['mc'] = app._mc_simulation_params(make_plan())\n"
)

# Every keyword the caller unpacks into run_monte_carlo (+ the death-age range).
_CONTRACT = {
    "n_sims", "use_hist", "use_hist_inf", "mkt_std", "inf_std", "sg_std",
    "mc_seed", "block_bs", "mean_block", "death_age_min", "death_age_max",
}


def _errors(at):
    return [str(e.value) for e in at.exception]


def test_params_return_contract_intact():
    at = AppTest.from_string(_PARAMS_ONLY, default_timeout=60).run()
    assert not at.exception, _errors(at)
    assert _CONTRACT <= set(at.session_state["mc"])


def test_sigma_sliders_hidden_when_historical_shown_when_parametric():
    """Default (both historical) hides the σ sliders; going parametric reveals them.

    This is the simplification's core: no inapplicable greyed-out controls, but
    the parametric knobs are still reachable — so nothing is actually removed.
    """
    at = AppTest.from_string(_PARAMS_ONLY, default_timeout=60).run()
    assert not at.exception, _errors(at)

    # Default state: both series historical → neither σ slider is rendered.
    labels = [s.label for s in at.slider]
    assert not any("Market return" in l for l in labels), labels
    assert not any("Inflation" in l for l in labels), labels
    assert at.session_state["mc"]["use_hist"] is True

    # Switch returns to parametric → its σ slider appears (and is used).
    returns_toggle = next(t for t in at.toggle if "Historical S&P 500 returns" in t.label)
    returns_toggle.set_value(False).run()
    assert not at.exception, _errors(at)

    labels2 = [s.label for s in at.slider]
    assert any("Market return" in l for l in labels2), labels2
    assert at.session_state["mc"]["use_hist"] is False
