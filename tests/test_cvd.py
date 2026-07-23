"""Tests for the CVD bar-aggregation foundation.

The taker-side convention is the highest-stakes detail in this module: if
``is_buyer_maker`` were read backwards, CVD would invert and every divergence
result downstream would look plausible while being exactly wrong. It is pinned
explicitly here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.cvd import (
    DEFAULT_CVD_WINDOW,
    TIMEFRAME_MS,
    _finalize,
    _partial_bars,
    bars_from_trades,
    reindex_to_grid,
    rolling_cvd,
)

TF_5M = TIMEFRAME_MS["5m"]


def _trades(rows: list[tuple[int, float, float, bool]]) -> pd.DataFrame:
    """rows: (timestamp_ms, price, quantity, is_buyer_maker)"""
    return pd.DataFrame(
        rows, columns=["timestamp", "price", "quantity", "is_buyer_maker"]
    )


def test_taker_side_convention_buy_is_buyer_maker_false():
    """is_buyer_maker=False -> taker BOUGHT -> buy volume (positive delta)."""
    df = _trades([(0, 100.0, 2.0, False)])
    bar = bars_from_trades(df, "5m").iloc[0]
    assert bar["buy_vol"] == 2.0
    assert bar["sell_vol"] == 0.0
    assert bar["delta"] == 2.0


def test_taker_side_convention_sell_is_buyer_maker_true():
    """is_buyer_maker=True -> taker SOLD -> sell volume (negative delta)."""
    df = _trades([(0, 100.0, 3.0, True)])
    bar = bars_from_trades(df, "5m").iloc[0]
    assert bar["buy_vol"] == 0.0
    assert bar["sell_vol"] == 3.0
    assert bar["delta"] == -3.0


def test_delta_is_buy_minus_sell_and_volume_is_total():
    df = _trades(
        [
            (0, 100.0, 5.0, False),  # buy 5
            (1, 101.0, 2.0, True),   # sell 2
        ]
    )
    bar = bars_from_trades(df, "5m").iloc[0]
    assert bar["buy_vol"] == 5.0
    assert bar["sell_vol"] == 2.0
    assert bar["delta"] == 3.0
    assert bar["volume"] == 7.0
    assert bar["num_trades"] == 2


def test_ohlc_uses_time_order_not_price_order():
    df = _trades(
        [
            (0, 100.0, 1.0, False),
            (1, 105.0, 1.0, False),  # high
            (2, 95.0, 1.0, True),    # low
            (3, 99.0, 1.0, True),    # last -> close
        ]
    )
    bar = bars_from_trades(df, "5m").iloc[0]
    assert bar["open"] == 100.0
    assert bar["high"] == 105.0
    assert bar["low"] == 95.0
    assert bar["close"] == 99.0


def test_bars_land_on_utc_aligned_grid():
    # 5m grid: a trade at 04:59.999 and one at 05:00.000 are in different bars.
    df = _trades(
        [
            (TF_5M - 1, 100.0, 1.0, False),
            (TF_5M, 200.0, 1.0, False),
        ]
    )
    bars = bars_from_trades(df, "5m")
    assert list(bars["open_time"]) == [0, TF_5M]
    assert list(bars["close_time"]) == [TF_5M, 2 * TF_5M]
    assert (bars["open_time"] % TF_5M == 0).all()


def test_vwap_is_volume_weighted():
    df = _trades(
        [
            (0, 100.0, 1.0, False),
            (1, 200.0, 3.0, False),  # weighted toward 200
        ]
    )
    bar = bars_from_trades(df, "5m").iloc[0]
    assert bar["vwap"] == pytest.approx((100.0 * 1 + 200.0 * 3) / 4)


def test_empty_bars_are_not_emitted():
    """A gap with no trades produces no bar (rather than a zero-volume bar)."""
    df = _trades([(0, 100.0, 1.0, False), (2 * TF_5M, 100.0, 1.0, False)])
    bars = bars_from_trades(df, "5m")
    assert list(bars["open_time"]) == [0, 2 * TF_5M]  # middle bar absent


def test_batch_split_matches_single_pass():
    """Streaming in batches must equal aggregating in one pass.

    This is what makes ``bars_for_month`` safe: a bar straddling a batch
    boundary must still get the earliest open and the latest close.
    """
    rows = [(i * 1000, 100.0 + i, 1.0 + i * 0.1, i % 2 == 0) for i in range(600)]
    df = _trades(rows)

    single = bars_from_trades(df, "5m")

    # split mid-bar so bars straddle the boundary
    parts = [df.iloc[:137], df.iloc[137:410], df.iloc[410:]]
    streamed = _finalize([_partial_bars(p, TF_5M) for p in parts], TF_5M)

    pd.testing.assert_frame_equal(single, streamed)


def test_deterministic_repeat_runs_identical():
    rows = [(i * 700, 100.0 + (i % 7), 0.5 + (i % 3), i % 3 == 0) for i in range(500)]
    df = _trades(rows)
    pd.testing.assert_frame_equal(
        bars_from_trades(df, "15m"), bars_from_trades(df, "15m")
    )


def test_unsorted_input_is_handled():
    """Out-of-order timestamps must not corrupt open/close."""
    ordered = _trades(
        [
            (0, 100.0, 1.0, False),
            (1, 110.0, 1.0, False),
            (2, 120.0, 1.0, False),
        ]
    )
    shuffled = ordered.iloc[[2, 0, 1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        bars_from_trades(ordered, "5m"), bars_from_trades(shuffled, "5m")
    )


def test_unknown_timeframe_raises():
    with pytest.raises(ValueError, match="unknown timeframe"):
        bars_from_trades(_trades([(0, 1.0, 1.0, False)]), "7m")


# --------------------------------------------------------------------------
# rolling-window CVD (the v1 "rolling window reset" anchoring)
# --------------------------------------------------------------------------


def _bars_with_deltas(deltas: list[float], tf: int = TF_5M) -> pd.DataFrame:
    """Minimal contiguous bar frame carrying the given per-bar deltas."""
    n = len(deltas)
    return pd.DataFrame(
        {
            "open_time": [i * tf for i in range(n)],
            "close_time": [(i + 1) * tf for i in range(n)],
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "vwap": 100.0, "volume": [abs(d) for d in deltas],
            "buy_vol": 0.0, "sell_vol": 0.0,
            "delta": deltas, "num_trades": 1,
        }
    )


def test_rolling_cvd_is_trailing_sum_of_delta():
    bars = _bars_with_deltas([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_cvd(bars, "5m", window=3)
    # first two bars incomplete -> NaN; then trailing 3-bar sums
    assert out["cvd"].isna().tolist() == [True, True, False, False, False]
    assert out["cvd"].dropna().tolist() == [6.0, 9.0, 12.0]


def test_rolling_cvd_partial_window_is_nan_not_partial_sum():
    """A partial window is a different statistic - it must not be emitted."""
    bars = _bars_with_deltas([5.0, 5.0, 5.0])
    out = rolling_cvd(bars, "5m", window=3)
    assert out["cvd"].iloc[0] != out["cvd"].iloc[0]  # NaN
    assert out["cvd"].iloc[1] != out["cvd"].iloc[1]  # NaN
    assert out["cvd"].iloc[2] == 15.0


def test_rolling_cvd_has_no_look_ahead():
    """CVD at bar i must not change when future bars are appended."""
    base = _bars_with_deltas([1.0, -2.0, 3.0, -4.0])
    extended = _bars_with_deltas([1.0, -2.0, 3.0, -4.0, 99.0, -99.0])
    a = rolling_cvd(base, "5m", window=2)["cvd"].tolist()
    b = rolling_cvd(extended, "5m", window=2)["cvd"].tolist()[: len(a)]
    assert a[1:] == b[1:]  # index 0 is NaN in both


def test_rolling_cvd_rejects_gapped_series():
    """Gaps would make an N-bar window span more than N*timeframe of real time."""
    bars = _bars_with_deltas([1.0, 2.0, 3.0])
    gapped = bars.drop(index=1).reset_index(drop=True)
    with pytest.raises(ValueError, match="gap"):
        rolling_cvd(gapped, "5m", window=2)


def test_reindex_to_grid_fills_missing_bars_with_zero_flow():
    bars = _bars_with_deltas([1.0, 2.0, 3.0])
    gapped = bars.drop(index=1).reset_index(drop=True)
    filled = reindex_to_grid(gapped, "5m")
    assert len(filled) == 3
    assert filled["open_time"].tolist() == [0, TF_5M, 2 * TF_5M]
    # the synthetic bar carries no flow ...
    assert filled.loc[1, "delta"] == 0.0
    assert filled.loc[1, "volume"] == 0.0
    assert filled.loc[1, "num_trades"] == 0.0
    # ... and is flat at the previous close
    assert filled.loc[1, "close"] == 100.0
    # and it is now safe to roll
    rolling_cvd(filled, "5m", window=2)


def test_rolling_cvd_default_window_matches_documented_value():
    assert DEFAULT_CVD_WINDOW == 20


def test_rolling_cvd_rejects_bad_window():
    with pytest.raises(ValueError, match="window must be >= 1"):
        rolling_cvd(_bars_with_deltas([1.0, 2.0]), "5m", window=0)
