"""Tests for the positioning-divergence signal.

The episode rule is what stands between us and a false pass: raw 5-min data
produces tens of thousands of threshold-exceeding rows from a few hundred real
episodes. These tests pin that behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.positioning import (
    BAR_MS, PositioningParams, build_episodes, divergence, trailing_pct_rank,
)


def _metrics(retail, top, start=0):
    n = len(retail)
    return pd.DataFrame({
        "create_time": [start + i * BAR_MS for i in range(n)],
        "global_account_ratio": retail,
        "top_account_ratio": top,
    })


def test_divergence_is_log_ratio_difference():
    m = _metrics([2.0], [1.0])
    assert divergence(m).iloc[0] == pytest.approx(np.log(2.0))
    m2 = _metrics([1.0], [1.0])
    assert divergence(m2).iloc[0] == pytest.approx(0.0)  # cohorts agree -> 0


def test_divergence_rejects_nonpositive_ratios():
    m = _metrics([0.0, -1.0, 2.0], [1.0, 1.0, 1.0])
    d = divergence(m)
    assert d.isna().iloc[0] and d.isna().iloc[1] and not d.isna().iloc[2]


def test_trailing_rank_is_strictly_trailing():
    s = pd.Series(np.arange(10, dtype=float))
    r = trailing_pct_rank(s, 5)
    assert r.iloc[:4].isna().all()          # warm-up
    assert r.iloc[4] == pytest.approx(1.0)  # rising series -> last is the max
    # appending future data must not change earlier values
    s2 = pd.concat([s, pd.Series([-99.0, -98.0])], ignore_index=True)
    r2 = trailing_pct_rank(s2, 5)
    assert np.allclose(r.to_numpy(), r2.to_numpy()[:10], equal_nan=True)


def _ramp_then(pattern, lookback=20):
    """Constant warm-up then an explicit pattern of divergence values.

    The warm-up is CONSTANT on purpose: all-tied values have midrank ~0.5, which
    sits in the neutral band and cannot fire. Random warm-up would put ~10% of
    bars in the extreme decile by construction and generate its own episodes,
    which would mask the behaviour under test.
    """
    warm = [0.0] * (lookback * 3)
    vals = warm + list(pattern)
    retail = np.exp(np.array(vals))
    top = np.ones(len(vals))
    return _metrics(retail, top)


def test_episode_fires_once_per_crossing_not_once_per_bar():
    """A sustained extreme must yield ONE episode, not one per 5-min row."""
    p = PositioningParams(lookback_bars=20, min_gap_ms=0)
    m = _ramp_then([5.0] * 50, lookback=20)   # 50 consecutive extreme bars
    ep = build_episodes(m, p)
    assert len(ep) == 1, f"expected 1 episode, got {len(ep)}"
    assert ep.iloc[0]["direction"] == -1      # retail extremely long -> short


def test_must_return_to_neutral_before_refiring():
    p = PositioningParams(lookback_bars=20, min_gap_ms=0)
    # extreme, then neutral, then extreme again -> two episodes
    m = _ramp_then([5.0] * 10 + [0.0] * 40 + [5.0] * 10, lookback=20)
    ep = build_episodes(m, p)
    assert len(ep) == 2


def test_direction_convention():
    p = PositioningParams(lookback_bars=20, min_gap_ms=0)
    hi = build_episodes(_ramp_then([5.0] * 30, lookback=20), p)
    lo = build_episodes(_ramp_then([-5.0] * 30, lookback=20), p)
    assert hi.iloc[0]["direction"] == -1   # retail crowded long -> fade -> short
    assert lo.iloc[0]["direction"] == 1    # retail crowded short -> long


def test_same_direction_rate_limit():
    """Bursts separated by enough neutral bars to clear the trailing window.

    (Short gaps would not work: a repeated value dilutes its own percentile
    below the decile, so it stops being extreme — which is correct behaviour of
    a trailing rank, but useless for exercising the rate limit.)
    """
    p24 = PositioningParams(lookback_bars=20, min_gap_ms=24 * 3600 * 1000)
    p0 = PositioningParams(lookback_bars=20, min_gap_ms=0)
    pattern = [5.0] * 3 + [0.0] * 60 + [5.0] * 3 + [0.0] * 60 + [5.0] * 3
    m = _ramp_then(pattern, lookback=20)
    n0, n24 = len(build_episodes(m, p0)), len(build_episodes(m, p24))
    assert n0 == 3, f"expected 3 unrestricted episodes, got {n0}"
    assert n24 == 1, f"24h limit should collapse them to 1, got {n24}"


def test_trailing_percentile_adapts_so_a_persistent_level_stops_being_extreme():
    """A value only stays 'extreme' relative to its own recent history."""
    p = PositioningParams(lookback_bars=20, min_gap_ms=0)
    m = _ramp_then([5.0] * 200, lookback=20)
    ep = build_episodes(m, p)
    assert len(ep) == 1  # fires on the crossing, then the level becomes normal


def test_publication_lag_is_applied():
    p = PositioningParams(lookback_bars=20, min_gap_ms=0)
    m = _ramp_then([5.0] * 30, lookback=20)
    ep = build_episodes(m, p)
    assert ep.iloc[0]["signal_time"] == ep.iloc[0]["raw_time"] + BAR_MS


def test_no_episodes_during_warmup():
    p = PositioningParams(lookback_bars=500, min_gap_ms=0)
    m = _ramp_then([5.0] * 30, lookback=20)   # never reaches the lookback
    assert build_episodes(m, p).empty


def test_metrics_timestamps_are_milliseconds_not_seconds():
    """Regression: pandas>=2 parses these strings to datetime64[us], so
    astype('int64')//1e6 silently yields SECONDS -- a 1000x error that renders
    every timestamp as 1970 and makes every downstream time join return nothing.
    It produced a fabricated 'INDEPENDENT' verdict from an empty match set."""
    import pandas as pd
    from pathlib import Path
    d = Path("data/screening/BTCUSDT/metrics")
    files = sorted(d.glob("*.parquet"))
    if not files:
        pytest.skip("no metrics sample downloaded")
    df = pd.read_parquet(files[0])
    # 2020+ in ms is ~1.6e12; in seconds it would be ~1.6e9
    assert df["create_time"].min() > 1e12, "timestamps look like seconds, not ms"
    yr = pd.to_datetime(df["create_time"].min(), unit="ms", utc=True).year
    assert 2019 <= yr <= 2027, f"implausible year {yr}"
