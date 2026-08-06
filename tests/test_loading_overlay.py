"""Regression tests for the Monte Carlo loading overlay.

The bug: switching to the Monte Carlo tab (or any slow recompute) left the
*previous* view's charts visible underneath the spinner. Streamlit only prunes
a run's leftover elements at end-of-run, so while the sim blocks mid-render the
old tab's content lingers on screen.

The fix (`app._loading_overlay`) paints an opaque, fixed-position overlay
*before* the blocking sim and clears it afterward, so nothing stale shows
through. These tests guard both halves:

  * the overlay actually covers the viewport opaquely (else stale content would
    show through), and
  * the Monte Carlo tab clears it once the sim finishes (else it would cover the
    results forever).
"""
import pathlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app.py")

_OVERLAY_ONLY = (
    "import sys\n"
    f"sys.path.insert(0, {REPO_ROOT.as_posix()!r})\n"
    "import app\n"
    "app._loading_overlay('Running 5,000 simulations…')\n"
)


def _errors(at):
    return [str(e.value) for e in at.exception]


def test_loading_overlay_is_opaque_fullscreen_cover():
    """The overlay must fully hide whatever is behind it while a sim runs.

    If any of these properties regress (not fixed-position, not spanning the
    viewport, or a transparent background) the previous tab's charts would show
    through again — the exact bug this guards against.
    """
    at = AppTest.from_string(_OVERLAY_ONLY, default_timeout=60).run()
    assert not at.exception, _errors(at)

    html = "\n".join(m.value for m in at.markdown)
    assert "fintracker-loading" in html
    assert "position: fixed" in html          # positioned against the viewport
    assert "inset: 0" in html                 # spans the whole area
    assert "#0f1117" in html                  # opaque (matches app background)
    assert "z-index: 9990" in html            # painted above the stale content
    assert "Running 5,000 simulations" in html


def test_monte_carlo_tab_clears_overlay_and_renders_results():
    """End-to-end: selecting Monte Carlo shows results, with no overlay left over.

    The overlay is created before the sim and cleared after; if the clear is
    dropped, the overlay markdown would remain in the final element tree (and
    would cover the results on screen). Asserting it is *absent* after the run
    guards the clear.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.session_state["loaded_plan"] = None      # no personal.yaml → sample plan
    at.run()
    assert not at.exception, _errors(at)

    # Select the Monte Carlo view (segmented_control has no AppTest accessor, so
    # drive it via its keyed session-state value) and re-run.
    at.session_state["dashboard_tab"] = "🎲 Monte Carlo"
    at.run()
    assert not at.exception, _errors(at)

    # Monte Carlo results rendered (a KPI unique to that tab).
    metric_labels = [m.label for m in at.metric]
    assert any("Median Net Worth" in lbl for lbl in metric_labels), metric_labels

    # ...and the loading overlay was cleared — nothing left covering them.
    leftover = [m.value for m in at.markdown if "fintracker-loading" in m.value]
    assert leftover == [], "loading overlay was not cleared after the sim"
