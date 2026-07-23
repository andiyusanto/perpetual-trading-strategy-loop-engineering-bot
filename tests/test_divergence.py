"""Tests for the three divergence layers.

The truncation-invariance tests are the important ones: they are the executable
form of methodology rule 3 (no look-ahead bias). If a layer could see the
future, computing it on a truncated series would give different answers for
bars that both series share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.divergence import (
    BEARISH,
    BULLISH,
    SWING_HIGH,
    SWING_LOW,
    SwingParams,
    cvd_roc,
    detect_regular_divergence,
    find_swing_points,
    roc_deceleration,
    rolling_spearman,
    spearman_breakdown,
)

TF = 900_000  # 15m


def _bars(highs, lows, closes=None, cvd=None) -> pd.DataFrame:
    n = len(highs)
    closes = closes if closes is not None else list(highs)
    df = pd.DataFrame(
        {
            "open_time": [i * TF for i in range(n)],
            "close_time": [(i + 1) * TF for i in range(n)],
            "open": closes, "high": highs, "low": lows, "close": closes,
            "vwap": closes, "volume": 1.0, "buy_vol": 0.0, "sell_vol": 0.0,
            "delta": 0.0, "num_trades": 1,
        }
    )
    if cvd is not None:
        df["cvd"] = cvd
    return df


# ---------------------------------------------------------------- Layer 1

def test_finds_swing_high_and_low():
    #            0  1  2  3(H) 4  5  6
    highs =     [1, 2, 3, 9,   3, 2, 1]
    lows =      [5, 4, 3, 1,   3, 4, 5]  # index 3 is also the low extreme
    sw = find_swing_points(_bars(highs, lows), SwingParams(3, 3))
    highs_found = sw[sw.kind == SWING_HIGH]
    lows_found = sw[sw.kind == SWING_LOW]
    assert highs_found["idx"].tolist() == [3]
    assert lows_found["idx"].tolist() == [3]
    assert highs_found.iloc[0]["price"] == 9


def test_swing_confirmation_is_delayed_by_right_bars():
    """The core look-ahead guard: a swing at i is only knowable at i+right."""
    highs = [1, 2, 3, 9, 3, 2, 1]
    lows = [5, 4, 3, 1, 3, 4, 5]
    sw = find_swing_points(_bars(highs, lows), SwingParams(3, 3))
    row = sw[sw.kind == SWING_HIGH].iloc[0]
    assert row["idx"] == 3
    assert row["confirmed_at_idx"] == 6          # 3 + right(3)
    assert row["confirmed_at_time"] == 7 * TF    # close_time of bar 6


def test_flat_plateau_is_not_a_swing():
    """Strict inequality: equal neighbours must not register as a swing."""
    highs = [1, 2, 3, 9, 9, 2, 1]  # tie at idx 3/4
    lows = [5, 4, 3, 2, 2, 4, 5]
    sw = find_swing_points(_bars(highs, lows), SwingParams(3, 3))
    assert sw[sw.kind == SWING_HIGH].empty


def test_no_swings_when_series_too_short():
    sw = find_swing_points(_bars([1, 2, 3], [3, 2, 1]), SwingParams(3, 3))
    assert sw.empty


def test_swings_are_truncation_invariant():
    """No look-ahead: truncating later bars must not change earlier swings."""
    rng = np.random.default_rng(7)
    highs = list(np.cumsum(rng.normal(size=200)) + 100)
    lows = [h - 1.0 for h in highs]
    full = find_swing_points(_bars(highs, lows), SwingParams(3, 3))
    cut = 150
    trunc = find_swing_points(_bars(highs[:cut], lows[:cut]), SwingParams(3, 3))
    # every swing confirmed strictly before the cut must appear identically
    a = full[full.confirmed_at_idx < cut - 3].reset_index(drop=True)
    b = trunc[trunc.confirmed_at_idx < cut - 3].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_bearish_regular_divergence_price_hh_cvd_lh():
    # two swing highs: idx 3 (price 9, cvd 100), idx 9 (price 10 HH, cvd 50 LH)
    highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 10, 3, 2, 1]
    lows = [h - 5 for h in highs]
    cvd = [0, 0, 0, 100, 0, 0, 0, 0, 0, 50, 0, 0, 0]
    bars = _bars(highs, lows, cvd=cvd)
    sw = find_swing_points(bars, SwingParams(3, 3))
    div = detect_regular_divergence(bars, sw)
    bear = div[div.kind == BEARISH]
    assert len(bear) == 1
    r = bear.iloc[0]
    assert (r["idx_prev"], r["idx_curr"]) == (3, 9)
    assert r["price_curr"] > r["price_prev"]   # higher high
    assert r["cvd_curr"] < r["cvd_prev"]       # lower CVD high
    assert r["confirmed_at_idx"] == 12         # later swing (9) + right(3)


def test_bullish_regular_divergence_price_ll_cvd_hl():
    lows = [9, 8, 7, 1, 7, 8, 9, 8, 7, 0, 7, 8, 9]
    highs = [l + 5 for l in lows]
    cvd = [0, 0, 0, -100, 0, 0, 0, 0, 0, -50, 0, 0, 0]
    bars = _bars(highs, lows, cvd=cvd)
    sw = find_swing_points(bars, SwingParams(3, 3))
    div = detect_regular_divergence(bars, sw)
    bull = div[div.kind == BULLISH]
    assert len(bull) == 1
    r = bull.iloc[0]
    assert r["price_curr"] < r["price_prev"]   # lower low
    assert r["cvd_curr"] > r["cvd_prev"]       # higher CVD low


def test_no_divergence_when_cvd_confirms_price():
    """Price HH + CVD HH is confirmation, not divergence - must not fire."""
    highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 10, 3, 2, 1]
    lows = [h - 5 for h in highs]
    cvd = [0, 0, 0, 100, 0, 0, 0, 0, 0, 150, 0, 0, 0]  # CVD also higher
    bars = _bars(highs, lows, cvd=cvd)
    sw = find_swing_points(bars, SwingParams(3, 3))
    div = detect_regular_divergence(bars, sw)
    assert div[div.kind == BEARISH].empty


def test_divergence_skips_nan_cvd_warmup():
    highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 10, 3, 2, 1]
    lows = [h - 5 for h in highs]
    cvd = [np.nan] * 4 + [0, 0, 0, 0, 0, 50, 0, 0, 0]  # first swing in warm-up
    bars = _bars(highs, lows, cvd=cvd)
    sw = find_swing_points(bars, SwingParams(3, 3))
    assert detect_regular_divergence(bars, sw).empty


def test_max_bars_between_rejects_distant_swings():
    highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 10, 3, 2, 1]
    lows = [h - 5 for h in highs]
    cvd = [0, 0, 0, 100, 0, 0, 0, 0, 0, 50, 0, 0, 0]
    bars = _bars(highs, lows, cvd=cvd)
    sw = find_swing_points(bars, SwingParams(3, 3))
    assert detect_regular_divergence(bars, sw, max_bars_between=3).empty
    assert not detect_regular_divergence(bars, sw, max_bars_between=6).empty


def test_divergence_requires_cvd_column():
    bars = _bars([1, 2, 3], [3, 2, 1])
    with pytest.raises(KeyError, match="cvd"):
        detect_regular_divergence(bars, _dummy_swings())


def _dummy_swings() -> pd.DataFrame:
    return pd.DataFrame(
        {"idx": [0], "kind": [SWING_HIGH], "price": [1.0],
         "open_time": [0], "confirmed_at_idx": [3], "confirmed_at_time": [4 * TF]}
    )


# ---------------------------------------------------------------- Layer 2

def test_rolling_spearman_perfect_monotonic_relationships():
    x = pd.Series([1.0, 2, 3, 4, 5, 6])
    up = pd.Series([10.0, 20, 30, 40, 50, 60])     # perfectly increasing
    down = pd.Series([60.0, 50, 40, 30, 20, 10])   # perfectly decreasing
    assert rolling_spearman(x, up, 4).dropna().eq(1.0).all()
    assert rolling_spearman(x, down, 4).dropna().eq(-1.0).all()


def test_rolling_spearman_is_rank_based_not_linear():
    """A monotonic but strongly non-linear relation is still rank-corr 1."""
    x = pd.Series([1.0, 2, 3, 4, 5])
    y = pd.Series([1.0, 4, 9, 16, 25])  # x^2, monotonic
    assert rolling_spearman(x, y, 5).iloc[-1] == pytest.approx(1.0)


def test_rolling_spearman_warmup_is_nan():
    x = pd.Series(np.arange(10, dtype=float))
    out = rolling_spearman(x, x, 4)
    assert out.iloc[:3].isna().all()
    assert not out.iloc[3:].isna().any()


def test_rolling_spearman_constant_window_is_nan_not_zero():
    x = pd.Series([5.0] * 6)          # zero variance -> undefined, not 0
    y = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert rolling_spearman(x, y, 4).dropna().empty


def test_rolling_spearman_truncation_invariant():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=120))
    y = pd.Series(rng.normal(size=120))
    full = rolling_spearman(x, y, 20).to_numpy()
    trunc = rolling_spearman(x.iloc[:80], y.iloc[:80], 20).to_numpy()
    assert np.allclose(full[:80], trunc, equal_nan=True)


def test_spearman_breakdown_flags_below_threshold():
    bars = _bars([1.0] * 6, [1.0] * 6, closes=[1.0, 2, 3, 4, 5, 6],
                 cvd=[6.0, 5, 4, 3, 2, 1])  # perfectly anti-correlated
    out = spearman_breakdown(bars, window=4, threshold=0.3)
    assert out["spearman_breakdown"].iloc[3:].all()
    # NaN warm-up must not be flagged as a signal
    assert not out["spearman_breakdown"].iloc[:3].any()


# ---------------------------------------------------------------- Layer 3

def test_cvd_roc_is_average_change_per_bar():
    cvd = pd.Series([0.0, 10, 20, 30, 40])
    roc = cvd_roc(cvd, window=2)
    assert roc.iloc[2] == pytest.approx(10.0)  # (20-0)/2
    assert roc.iloc[:2].isna().all()


def test_roc_deceleration_fires_when_flow_slows():
    # fast accumulation then a near-flat stretch
    cvd = pd.Series([0, 100, 200, 300, 400, 500, 505, 508, 510, 511, 512, 513.0])
    bars = _bars([1.0] * len(cvd), [1.0] * len(cvd), cvd=list(cvd))
    out = roc_deceleration(bars, roc_window=2, peak_lookback=3, decel_ratio=0.5)
    assert bool(out["roc_decelerating"].iloc[-1])


def test_roc_prior_peak_excludes_current_bar():
    """The current bar must not set the peak it is compared against."""
    cvd = pd.Series(np.arange(12, dtype=float) * 10)  # constant ROC
    bars = _bars([1.0] * 12, [1.0] * 12, cvd=list(cvd))
    out = roc_deceleration(bars, roc_window=2, peak_lookback=3, decel_ratio=0.99)
    # constant ROC never decelerates below 99% of the prior peak
    assert not out["roc_decelerating"].any()


def test_roc_deceleration_truncation_invariant():
    rng = np.random.default_rng(11)
    cvd = list(np.cumsum(rng.normal(size=100)))
    bars = _bars([1.0] * 100, [1.0] * 100, cvd=cvd)
    full = roc_deceleration(bars, roc_window=5, peak_lookback=10)
    trunc = roc_deceleration(bars.iloc[:60].copy(), roc_window=5, peak_lookback=10)
    assert np.allclose(
        full["cvd_roc"].to_numpy()[:60], trunc["cvd_roc"].to_numpy(), equal_nan=True
    )
    assert (
        full["roc_decelerating"].to_numpy()[:60]
        == trunc["roc_decelerating"].to_numpy()
    ).all()


def test_roc_rejects_bad_ratio():
    bars = _bars([1.0] * 5, [1.0] * 5, cvd=[1.0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="decel_ratio"):
        roc_deceleration(bars, decel_ratio=1.5)
