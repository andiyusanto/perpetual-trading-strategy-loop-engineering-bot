"""Tests for the signal screening harness.

The two load-bearing tests are the power pair: the screen must DETECT a signal
with a real injected edge, and must REJECT pure noise. A screen that never fires
is useless; one that always fires is worse than useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.screen import (
    DEFAULT_COST_BPS,
    SignalSet,
    forward_returns,
    screen_signal,
)

TF = 300_000  # 5m


def _bars(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "open_time": [i * TF for i in range(n)],
            "close_time": [(i + 1) * TF for i in range(n)],
            "open": close, "high": close, "low": close, "close": close,
            "vwap": close, "volume": 1.0, "buy_vol": 0.0, "sell_vol": 0.0,
            "delta": 0.0, "num_trades": 1,
        }
    )


def _random_walk(n=6000, seed=0, sigma=8.0):
    rng = np.random.default_rng(seed)
    return 100_000 * np.exp(np.cumsum(rng.normal(0, sigma * 1e-4, n)))


# ---------------------------------------------------------------- mechanics

def test_forward_return_is_direction_signed():
    close = np.array([100.0, 101.0, 102.0, 103.0])
    bars = _bars(close)
    t = np.array([bars.close_time.iloc[0]])
    up = forward_returns(bars, t, np.array([1]), horizon_min=5)
    down = forward_returns(bars, t, np.array([-1]), horizon_min=5)
    assert up[0] == pytest.approx(100.0, abs=1e-6)   # +1% = 100 bps
    assert down[0] == pytest.approx(-100.0, abs=1e-6)


def test_forward_return_uses_only_closed_bars():
    """Entry price is the last bar CLOSED at the signal time, never a live one."""
    close = np.array([100.0, 200.0, 300.0])
    bars = _bars(close)
    # signal 1ms before bar 0 closes -> no closed bar yet -> NaN
    r = forward_returns(bars, np.array([TF - 1]), np.array([1]), 5)
    assert np.isnan(r[0])
    # signal exactly at bar 0's close -> entry uses bar 0 (price 100)
    r = forward_returns(bars, np.array([TF]), np.array([1]), 5)
    assert r[0] == pytest.approx((200.0 / 100.0 - 1) * 1e4)


def test_horizon_past_end_of_data_is_nan_not_zero():
    bars = _bars(np.array([100.0, 101.0, 102.0]))
    r = forward_returns(bars, np.array([bars.close_time.iloc[-1]]), np.array([1]), 60)
    assert np.isnan(r[0])


def test_signalset_rejects_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        SignalSet("x", np.array([1]), np.array([0]))
    with pytest.raises(ValueError, match="same length"):
        SignalSet("x", np.array([1, 2]), np.array([1]))


# ---------------------------------------------------------------- power pair

def test_detects_an_injected_edge():
    """POWER: a signal with a real, cost-clearing edge must be found."""
    n = 6000
    close = _random_walk(n, seed=1)
    times, dirs = [], []
    rng = np.random.default_rng(2)
    idx = rng.choice(np.arange(100, n - 100), size=250, replace=False)
    # inject a +60bps drift over the 12 bars (60 min) following each signal
    close = close.copy()
    for i in sorted(idx):
        close[i + 1 : i + 13] *= np.linspace(1.0, 1.006, 12)
        close[i + 13 :] *= 1.006
    bars = _bars(close)
    for i in sorted(idx):
        times.append(bars.close_time.iloc[i])
        dirs.append(1)
    sig = SignalSet("injected_edge", np.array(times), np.array(dirs))

    rep = screen_signal(bars, sig, horizons_min=(60,), n_perm=60, log=False)
    h = rep["horizons"]["60m"]
    assert h["mean_bps"] > DEFAULT_COST_BPS
    assert h["ci95_bps"][0] > DEFAULT_COST_BPS   # CI clears the cost hurdle
    assert h["perm_p"] < 0.05
    assert rep["verdict"] == "TRADEABLE EDGE"


def test_rejects_pure_noise():
    """SPECIFICITY: random timing on a random walk must NOT be called tradeable."""
    n = 6000
    bars = _bars(_random_walk(n, seed=3))
    rng = np.random.default_rng(4)
    idx = rng.choice(np.arange(100, n - 300), size=250, replace=False)
    sig = SignalSet(
        "pure_noise",
        bars.close_time.to_numpy()[np.sort(idx)],
        rng.choice([-1, 1], size=250),
    )
    rep = screen_signal(bars, sig, horizons_min=(60, 240), n_perm=60, log=False)
    assert rep["verdict"] == "NO TRADEABLE EDGE"
    for h in rep["horizons"].values():
        assert not h["clears_cost"]


def test_real_but_sub_cost_edge_is_not_tradeable():
    """The distinction that would have saved the last cycle: a genuine edge
    smaller than the round trip is real and still untradeable."""
    n = 6000
    close = _random_walk(n, seed=5, sigma=2.0)
    rng = np.random.default_rng(6)
    idx = np.sort(rng.choice(np.arange(100, n - 100), size=300, replace=False))
    close = close.copy()
    for i in idx:
        close[i + 1 :] *= 1.00035  # +3.5bps: real, but well under 16bps cost
    bars = _bars(close)
    sig = SignalSet("tiny_edge", bars.close_time.to_numpy()[idx], np.ones(len(idx), int))
    rep = screen_signal(bars, sig, horizons_min=(60,), n_perm=60, log=False)
    h = rep["horizons"]["60m"]
    assert h["mean_bps"] > 0          # the edge is genuinely there ...
    assert not h["clears_cost"]       # ... and still not worth trading
    assert rep["verdict"] == "NO TRADEABLE EDGE"


# ---------------------------------------------------------------- bookkeeping

def test_every_screen_is_logged(tmp_path, monkeypatch):
    import src.research.screen as sc
    monkeypatch.setattr(sc, "SCREENING_LOG", tmp_path / "screening_log.jsonl")
    bars = _bars(_random_walk(2000, seed=7))
    rng = np.random.default_rng(8)
    idx = np.sort(rng.choice(np.arange(50, 1500), size=40, replace=False))
    sig = SignalSet("logged", bars.close_time.to_numpy()[idx], np.ones(len(idx), int))
    sc.screen_signal(bars, sig, horizons_min=(60,), n_perm=10, log=True)
    sc.screen_signal(bars, sig, horizons_min=(60,), n_perm=10, log=True)
    assert sc.screens_run() == 2, "failures must be logged too, not just passes"


def test_reports_clustering_not_just_raw_n():
    """Raw n overstates independence when signals cluster in time."""
    bars = _bars(_random_walk(3000, seed=9))
    # 50 signals all inside one day
    t = bars.close_time.to_numpy()[100:150]
    sig = SignalSet("clustered", t, np.ones(50, int))
    rep = screen_signal(bars, sig, horizons_min=(60,), n_perm=10, log=False)
    assert rep["distinct_days"] >= 1
    assert rep["signals_per_day"] > 1


def test_empty_signal_set_is_handled():
    bars = _bars(_random_walk(500, seed=10))
    sig = SignalSet("none", np.array([], dtype="int64"), np.array([], dtype=int))
    rep = screen_signal(bars, sig, log=False)
    assert rep["verdict"] == "NO SIGNALS"


def test_underpowered_null_is_inconclusive_not_a_rejection():
    """A tiny sample must NOT be reported as 'no edge' — it cannot see one."""
    n = 4000
    bars = _bars(_random_walk(n, seed=21, sigma=25.0))  # noisy -> wide SE
    rng = np.random.default_rng(22)
    idx = np.sort(rng.choice(np.arange(50, n - 400), size=12, replace=False))
    sig = SignalSet("tiny_sample", bars.close_time.to_numpy()[idx],
                    rng.choice([-1, 1], size=12))
    rep = screen_signal(bars, sig, horizons_min=(240,), n_perm=30, log=False)
    assert rep["verdict"] == "INCONCLUSIVE (UNDERPOWERED)"
    assert rep["any_horizon_powered"] is False
    assert rep["horizons"]["240m"]["mde_bps"] > DEFAULT_COST_BPS


def test_well_powered_null_is_a_real_rejection():
    """With enough samples and low noise, a null IS a rejection."""
    n = 20000
    bars = _bars(_random_walk(n, seed=23, sigma=1.0))
    rng = np.random.default_rng(24)
    idx = np.sort(rng.choice(np.arange(50, n - 100), size=3000, replace=False))
    sig = SignalSet("powered_noise", bars.close_time.to_numpy()[idx],
                    rng.choice([-1, 1], size=3000))
    rep = screen_signal(bars, sig, horizons_min=(60,), n_perm=30, log=False)
    assert rep["horizons"]["60m"]["powered"] is True
    assert rep["verdict"] == "NO TRADEABLE EDGE"
