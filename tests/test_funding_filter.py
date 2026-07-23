"""Tests for the funding-rate gate.

The look-ahead tests matter most: at decision time T the gate may only see
funding that had already SETTLED, never the settlement that happens later —
including the one landing exactly at T+1ms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.divergence import BEARISH, BULLISH
from src.strategy.funding_filter import (
    MS_PER_DAY,
    FundingGateParams,
    evaluate_funding_gate,
    gate_divergences,
)

EIGHT_H = 8 * 3600 * 1000


def _funding(rates: list[float], start: int = 0, step: int = EIGHT_H) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["TESTUSDT"] * len(rates),
            "funding_time": [start + i * step for i in range(len(rates))],
            "funding_rate": rates,
            "mark_price": [100.0] * len(rates),
        }
    )


def _ramp(n: int = 200) -> pd.DataFrame:
    """Monotonically increasing funding, so rank position is unambiguous."""
    return _funding([i * 1e-6 for i in range(n)])


def test_uses_only_settled_funding_not_the_future():
    f = _ramp(200)
    # decide 1ms BEFORE the 100th settlement -> must see the 99th
    t = int(f.funding_time.iloc[100]) - 1
    r = evaluate_funding_gate(f, [t], FundingGateParams(min_obs=1))
    assert r.loc[0, "last_funding_time"] == f.funding_time.iloc[99]
    assert r.loc[0, "last_funding_rate"] == pytest.approx(f.funding_rate.iloc[99])


def test_settlement_exactly_at_decision_time_is_visible():
    f = _ramp(200)
    t = int(f.funding_time.iloc[100])
    r = evaluate_funding_gate(f, [t], FundingGateParams(min_obs=1))
    assert r.loc[0, "last_funding_time"] == t


def test_abstains_before_anything_settled():
    f = _ramp(50)
    r = evaluate_funding_gate(f, [int(f.funding_time.iloc[0]) - 1],
                              FundingGateParams(min_obs=1))
    assert np.isnan(r.loc[0, "last_funding_rate"])
    assert not r.loc[0, "is_high_extreme"]
    assert not r.loc[0, "is_low_extreme"]


def test_abstains_when_window_too_thin():
    """A newly listed pair must not manufacture an 'extreme' from a few points."""
    f = _ramp(10)
    r = evaluate_funding_gate(f, [int(f.funding_time.iloc[-1])],
                              FundingGateParams(min_obs=60))
    assert r.loc[0, "n_obs"] == 10
    assert np.isnan(r.loc[0, "pct_rank"])
    assert not r.loc[0, "is_high_extreme"]
    assert not r.loc[0, "is_low_extreme"]


def test_high_and_low_extremes_are_detected():
    f = _ramp(200)
    p = FundingGateParams(min_obs=1, extreme_pct=0.10)
    # newest rate is the max of a rising ramp -> top of its window
    hi = evaluate_funding_gate(f, [int(f.funding_time.iloc[-1])], p)
    assert hi.loc[0, "is_high_extreme"] and not hi.loc[0, "is_low_extreme"]

    # falling ramp -> newest is the minimum
    f2 = _funding([-i * 1e-6 for i in range(200)])
    lo = evaluate_funding_gate(f2, [int(f2.funding_time.iloc[-1])], p)
    assert lo.loc[0, "is_low_extreme"] and not lo.loc[0, "is_high_extreme"]


def test_mid_distribution_is_not_an_extreme():
    f = _funding([0.0] * 100 + [1e-6] + [0.0] * 100)
    t = int(f.funding_time.iloc[-1])
    r = evaluate_funding_gate(f, [t], FundingGateParams(min_obs=1))
    assert not r.loc[0, "is_high_extreme"]
    assert not r.loc[0, "is_low_extreme"]


def test_lookback_window_excludes_old_events():
    """Events older than lookback_days must not shape the distribution."""
    old = _funding([9.0] * 100, start=0)                       # ancient, huge
    recent = _funding([0.0] * 100, start=200 * MS_PER_DAY)     # recent, flat
    f = pd.concat([old, recent], ignore_index=True)
    t = int(recent.funding_time.iloc[-1])
    r = evaluate_funding_gate(f, [t], FundingGateParams(lookback_days=90, min_obs=1))
    assert r.loc[0, "n_obs"] == 100  # only the recent block is in-window


def test_percentile_is_relative_per_pair_not_absolute():
    """The same absolute rate is extreme for one pair and ordinary for another."""
    calm = _funding([0.0] * 199 + [0.0005])     # 0.05% is huge here
    wild = _funding(list(np.linspace(-0.01, 0.01, 200)))  # 0.05% is mid-range
    p = FundingGateParams(min_obs=1)
    t_calm = int(calm.funding_time.iloc[-1])
    assert evaluate_funding_gate(calm, [t_calm], p).loc[0, "is_high_extreme"]

    wild2 = wild.copy()
    wild2.loc[wild2.index[-1], "funding_rate"] = 0.0005  # same absolute value
    t_wild = int(wild2.funding_time.iloc[-1])
    assert not evaluate_funding_gate(wild2, [t_wild], p).loc[0, "is_high_extreme"]


def _divs(kinds: list[str], t: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": kinds,
            "idx_prev": [0] * len(kinds),
            "idx_curr": [1] * len(kinds),
            "price_prev": [1.0] * len(kinds),
            "price_curr": [2.0] * len(kinds),
            "cvd_prev": [1.0] * len(kinds),
            "cvd_curr": [0.0] * len(kinds),
            "confirmed_at_idx": [5] * len(kinds),
            "confirmed_at_time": [t] * len(kinds),
        }
    )


def test_gate_direction_bearish_needs_high_extreme():
    f = _ramp(200)  # newest = high extreme (crowded longs)
    t = int(f.funding_time.iloc[-1])
    p = FundingGateParams(min_obs=1)
    out = gate_divergences(_divs([BEARISH, BULLISH], t), f, p)
    assert bool(out.loc[out.kind == BEARISH, "funding_gate_open"].iloc[0]) is True
    # a HIGH extreme must NOT open a bullish (crowded-short) setup
    assert bool(out.loc[out.kind == BULLISH, "funding_gate_open"].iloc[0]) is False


def test_gate_direction_bullish_needs_low_extreme():
    f = _funding([-i * 1e-6 for i in range(200)])  # newest = low extreme
    t = int(f.funding_time.iloc[-1])
    p = FundingGateParams(min_obs=1)
    out = gate_divergences(_divs([BEARISH, BULLISH], t), f, p)
    assert bool(out.loc[out.kind == BULLISH, "funding_gate_open"].iloc[0]) is True
    assert bool(out.loc[out.kind == BEARISH, "funding_gate_open"].iloc[0]) is False


def test_gate_closed_when_funding_is_ordinary():
    f = _funding([0.0] * 200)
    t = int(f.funding_time.iloc[-1])
    out = gate_divergences(_divs([BEARISH, BULLISH], t), f, FundingGateParams(min_obs=1))
    assert not out["funding_gate_open"].any()


def test_empty_divergences_returns_empty():
    f = _ramp(200)
    empty = _divs([], 0)
    assert gate_divergences(empty, f).empty


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="extreme_pct"):
        FundingGateParams(extreme_pct=0.9)
    with pytest.raises(ValueError, match="lookback_days"):
        FundingGateParams(lookback_days=0)
