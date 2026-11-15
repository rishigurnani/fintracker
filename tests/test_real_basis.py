"""Tests for the real-dollars rebase (`app._rebase_networth`).

The Monte Carlo view can be shown in today's purchasing power by deflating
every net-worth band at the assumed inflation rate. These guard that the
deflation is per-year-correct, that rate 0 is a true identity (the nominal
default), that liquidity series are left untouched, and that the cached source
result is never mutated.
"""
import pytest

pytest.importorskip("streamlit")
import app
from fintracker.montecarlo import MonteCarloResult


def _mc():
    """A tiny 3-year result with distinct net-worth and liquid bands."""
    return MonteCarloResult(
        years=[1, 2, 3],
        p10_net_worth=[100.0, 200.0, 300.0],
        p25_net_worth=[110.0, 210.0, 310.0],
        p50_net_worth=[120.0, 220.0, 320.0],
        p75_net_worth=[130.0, 230.0, 330.0],
        p90_net_worth=[140.0, 240.0, 340.0],
        mean_net_worth=[125.0, 225.0, 325.0],
        prob_negative_liquid=[0.0, 0.1, 0.2],
        p10_liquid=[10.0, 20.0, 30.0],
        p50_liquid=[11.0, 21.0, 31.0],
        p90_liquid=[12.0, 22.0, 32.0],
    )


def test_rate_zero_is_identity():
    mc = _mc()
    out = app._rebase_networth(mc, 0.0)
    for band in ("p10", "p25", "p50", "p75", "p90", "mean"):
        attr = f"{band}_net_worth"
        assert getattr(out, attr) == getattr(mc, attr)


def test_deflates_each_year_by_assumed_rate():
    mc = _mc()
    rate = 0.03
    out = app._rebase_networth(mc, rate)
    # value(year y) / (1+rate)**y, year taken from mc.years (1-indexed).
    expected = [v / (1 + rate) ** y for v, y in zip(mc.p50_net_worth, mc.years)]
    assert out.p50_net_worth == pytest.approx(expected)
    # Later years lose more purchasing power: the deflation factor grows with y.
    factors = [n / r for n, r in zip(mc.p50_net_worth, out.p50_net_worth)]
    assert factors[0] < factors[1] < factors[2]


def test_liquidity_and_metadata_left_nominal():
    mc = _mc()
    out = app._rebase_networth(mc, 0.05)
    assert out.p10_liquid == mc.p10_liquid
    assert out.p50_liquid == mc.p50_liquid
    assert out.p90_liquid == mc.p90_liquid
    assert out.prob_negative_liquid == mc.prob_negative_liquid
    assert out.years == mc.years


def test_source_result_not_mutated():
    mc = _mc()
    before = list(mc.p90_net_worth)
    app._rebase_networth(mc, 0.04)
    assert mc.p90_net_worth == before
