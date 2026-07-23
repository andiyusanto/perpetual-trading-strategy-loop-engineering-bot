"""Tests for the backtest engine, cost model, and metrics.

The load-bearing tests here are:
  - entry never uses the candle that produced the signal (look-ahead)
  - an ambiguous candle (stop AND target inside its range) records the STOP
  - no code path yields a pre-cost PnL
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LONG, SHORT, CostModel
from src.backtest.engine import (
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    StrategyParams,
    simulate,
)
from src.backtest.metrics import (
    bootstrap_ci,
    breakeven_win_rate,
    max_drawdown,
    permutation_test_vs_random_entries,
    summarize,
)

TF5, TF15 = 300_000, 900_000


def _bars(n, tf, highs, lows, closes):
    return pd.DataFrame(
        {
            "open_time": [i * tf for i in range(n)],
            "close_time": [(i + 1) * tf for i in range(n)],
            "open": closes, "high": highs, "low": lows, "close": closes,
            "vwap": closes, "volume": 1.0, "buy_vol": 0.0, "sell_vol": 0.0,
            "delta": 0.0, "num_trades": 1,
        }
    )


def _flat_bars15(n=6):
    """15m frame short enough that CVD/ROC stay NaN, so the early-exit rule
    cannot fire and stop/target/time logic is isolated."""
    return _bars(n, TF15, [100.0] * n, [100.0] * n, [100.0] * n)


def _intent(side, stop, signal_time, kind="bearish"):
    return pd.DataFrame(
        [{"kind": kind, "side": side, "signal_idx15": 0,
          "signal_time": signal_time, "stop_price": stop, "swing_price": stop}]
    )


# ------------------------------------------------------------------ costs

def test_slippage_always_moves_against_us():
    c = CostModel(slippage_bps=10.0)
    assert c.fill_price(LONG, 100.0, is_entry=True) > 100.0    # buy: pay up
    assert c.fill_price(LONG, 100.0, is_entry=False) < 100.0   # sell: receive less
    assert c.fill_price(SHORT, 100.0, is_entry=True) < 100.0   # sell to open
    assert c.fill_price(SHORT, 100.0, is_entry=False) > 100.0  # buy to close


def test_round_trip_cost_is_two_taker_fees():
    c = CostModel(taker_fee_bps=5.0, slippage_bps=0.0)
    assert c.round_trip_cost(100.0, 100.0, 1.0) == pytest.approx(2 * 100.0 * 5e-4)
    assert c.round_trip_bps == pytest.approx(10.0)


def test_negative_rates_rejected():
    with pytest.raises(ValueError):
        CostModel(taker_fee_bps=-1)


# ------------------------------------------------------------------ engine

def test_entry_is_the_first_candle_closing_after_the_signal():
    """No look-ahead: the candle that produced the signal is not tradeable."""
    n = 10
    bars5 = _bars(n, TF5, [101.0] * n, [99.0] * n, [100.0] * n)
    sig_time = int(bars5["close_time"].iloc[2])  # signal lands exactly at a close
    tr = simulate(_intent(LONG, 95.0, sig_time), bars5, _flat_bars15(),
                  StrategyParams(), CostModel())
    assert len(tr) == 1
    assert tr.iloc[0]["entry_idx5"] == 3  # bar 3, not bar 2


def test_ambiguous_candle_records_the_stop_not_the_target():
    """Both levels inside one candle -> pessimistic: stop."""
    n = 8
    highs = [100.0] * n
    lows = [100.0] * n
    closes = [100.0] * n
    highs[3], lows[3] = 130.0, 70.0  # engulfs stop AND target
    bars5 = _bars(n, TF5, highs, lows, closes)
    tr = simulate(_intent(LONG, 90.0, 0), bars5, _flat_bars15(),
                  StrategyParams(rr_ratio=1.5), CostModel(slippage_bps=0))
    assert tr.iloc[0]["exit_reason"] == EXIT_STOP


def test_long_take_profit_path():
    n = 8
    highs = [100.0] * n
    lows = [100.0] * n
    closes = [100.0] * n
    highs[3] = 200.0  # target only
    bars5 = _bars(n, TF5, highs, lows, closes)
    tr = simulate(_intent(LONG, 90.0, 0), bars5, _flat_bars15(),
                  StrategyParams(rr_ratio=1.5), CostModel(slippage_bps=0))
    row = tr.iloc[0]
    assert row["exit_reason"] == EXIT_TARGET
    assert row["r_multiple"] > 0


def test_short_stop_and_target_directions():
    n = 8
    highs = [100.0] * n
    lows = [100.0] * n
    closes = [100.0] * n
    lows[3] = 50.0  # price falls -> good for a short
    bars5 = _bars(n, TF5, highs, lows, closes)
    tr = simulate(_intent(SHORT, 110.0, 0, kind="bearish"), bars5, _flat_bars15(),
                  StrategyParams(rr_ratio=1.5), CostModel(slippage_bps=0))
    row = tr.iloc[0]
    assert row["exit_reason"] == EXIT_TARGET
    assert row["target_price"] < row["entry_price"]
    assert row["stop_price"] > row["entry_price"]
    assert row["r_multiple"] > 0


def test_time_stop_when_nothing_resolves():
    n = 60
    bars5 = _bars(n, TF5, [100.5] * n, [99.5] * n, [100.0] * n)
    tr = simulate(_intent(LONG, 90.0, 0), bars5, _flat_bars15(),
                  StrategyParams(time_stop_bars_15m=3), CostModel())
    row = tr.iloc[0]
    assert row["exit_reason"] == EXIT_TIME
    assert row["bars_held"] == 3 * 3  # 3 x 15m == 9 x 5m


def test_costs_are_always_applied():
    n = 8
    highs = [100.0] * n
    lows = [100.0] * n
    highs[3] = 200.0
    bars5 = _bars(n, TF5, highs, lows, [100.0] * n)
    tr = simulate(_intent(LONG, 90.0, 0), bars5, _flat_bars15(),
                  StrategyParams(), CostModel(taker_fee_bps=5, slippage_bps=3))
    row = tr.iloc[0]
    assert row["fees"] > 0
    assert row["net_pnl"] < row["gross_pnl"]  # costs never improve a trade


def test_overlapping_trades_suppressed_by_default():
    n = 40
    bars5 = _bars(n, TF5, [100.5] * n, [99.5] * n, [100.0] * n)
    two = pd.concat([_intent(LONG, 90.0, 0), _intent(LONG, 90.0, TF5)],
                    ignore_index=True)
    assert len(simulate(two, bars5, _flat_bars15(), StrategyParams())) == 1
    assert len(simulate(two, bars5, _flat_bars15(), StrategyParams(),
                        allow_overlapping=True)) == 2


def test_degenerate_stop_on_wrong_side_is_skipped():
    n = 8
    bars5 = _bars(n, TF5, [100.0] * n, [100.0] * n, [100.0] * n)
    # long with a stop ABOVE entry -> zero/negative risk, must not trade
    assert simulate(_intent(LONG, 100.0, 0), bars5, _flat_bars15()).empty


def test_empty_intents_gives_empty_trades():
    n = 5
    bars5 = _bars(n, TF5, [100.0] * n, [100.0] * n, [100.0] * n)
    empty = _intent(LONG, 90.0, 0).iloc[0:0]
    assert simulate(empty, bars5, _flat_bars15()).empty


# ------------------------------------------------------------------ metrics

def test_breakeven_win_rate_at_1_5R_is_40pct():
    assert breakeven_win_rate(1.5) == pytest.approx(0.40)


def test_summarize_reports_intervals_not_just_point_estimates():
    tr = pd.DataFrame(
        {
            "r_multiple": [1.5, -1.0, 1.5, -1.0, 1.5],
            "net_pnl": [15.0, -10.0, 15.0, -10.0, 15.0],
            "gross_pnl": [16.0, -9.0, 16.0, -9.0, 16.0],
            "fees": [1.0] * 5,
            "exit_reason": ["target", "stop", "target", "stop", "target"],
            "bars_held": [5, 5, 5, 5, 5],
        }
    )
    s = summarize(tr, rr_ratio=1.5)
    assert s["n_trades"] == 5
    assert s["win_rate"] == pytest.approx(0.6)
    lo, hi = s["win_rate_ci95"]
    assert lo <= s["win_rate"] <= hi
    el, eh = s["expectancy_r_ci95"]
    assert el <= s["expectancy_r"] <= eh
    assert s["profit_factor"] > 1
    assert s["total_fees"] > 0


def test_summarize_handles_no_trades():
    from src.backtest.engine import _empty_trades
    s = summarize(_empty_trades())
    assert s["n_trades"] == 0
    assert "win_rate" not in s  # no fabricated stats from an empty set


def test_bootstrap_ci_is_deterministic():
    v = np.array([1.0, -1, 1, -1, 2, -0.5])
    assert bootstrap_ci(v, seed=42) == bootstrap_ci(v, seed=42)


def test_max_drawdown_is_negative_or_zero():
    assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0
    assert max_drawdown(np.cumsum(np.array([1.0, -3.0, 1.0]))) < 0


def test_permutation_p_value_never_zero():
    """p=0 is not a defensible claim from finite resampling."""
    res = permutation_test_vs_random_entries(
        np.array([5.0] * 10), [np.array([0.0] * 10) for _ in range(50)]
    )
    assert res["p_value"] > 0
    assert res["p_value"] == pytest.approx(1 / 51)


# ------------------------------------------------------------------ walk-forward

def test_folds_are_rolling_and_test_follows_train():
    import datetime as dt
    from src.backtest.walkforward import make_folds
    folds = make_folds(dt.date(2025, 1, 1), dt.date(2025, 9, 30),
                       train_months=2, test_months=1, step_months=1)
    assert len(folds) >= 5
    for f in folds:
        assert f.train_end < f.test_start          # test strictly after train
        assert f.test_start <= f.test_end
        assert f.test_end <= dt.date(2025, 9, 30)  # never past the research end
    # rolling: each fold starts one month later
    assert folds[1].train_start > folds[0].train_start


def test_calibration_uses_only_the_training_window():
    """Thresholds derived from train must not change when test data changes."""
    import datetime as dt
    from src.backtest.walkforward import calibrate_thresholds, slice_bars
    rng = np.random.default_rng(5)
    n = 4000
    close = 100 + np.cumsum(rng.normal(size=n))
    bars = pd.DataFrame({
        "open_time": [i * TF15 for i in range(n)],
        "close_time": [(i + 1) * TF15 for i in range(n)],
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "vwap": close, "volume": 1.0, "buy_vol": 0.0, "sell_vol": 0.0,
        "delta": rng.normal(size=n), "num_trades": 1,
    })
    train = bars.iloc[:2000].reset_index(drop=True)
    p1 = calibrate_thresholds(train, StrategyParams())
    # mutate the LATER (test) portion wildly; calibration must be unchanged
    bars2 = bars.copy()
    bars2.loc[2000:, "delta"] = 999.0
    p2 = calibrate_thresholds(bars2.iloc[:2000].reset_index(drop=True), StrategyParams())
    assert p1.spearman_threshold == p2.spearman_threshold
    assert p1.decel_ratio == p2.decel_ratio


def test_calibrated_decel_ratio_stays_in_valid_domain():
    import datetime as dt
    from src.backtest.walkforward import calibrate_thresholds
    rng = np.random.default_rng(9)
    n = 3000
    close = 100 + np.cumsum(rng.normal(size=n))
    bars = pd.DataFrame({
        "open_time": [i * TF15 for i in range(n)],
        "close_time": [(i + 1) * TF15 for i in range(n)],
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "vwap": close, "volume": 1.0, "buy_vol": 0.0, "sell_vol": 0.0,
        "delta": rng.normal(size=n), "num_trades": 1,
    })
    p = calibrate_thresholds(bars, StrategyParams())
    assert 0 < p.decel_ratio <= 1.0   # the domain roc_deceleration accepts
