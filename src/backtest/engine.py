"""Backtest engine: signal assembly + trade simulation.

Follows HYPOTHESIS.md v1 entry/exit exactly:
  entry trigger  Layers 1+2+3 confirm in the same window AND funding does not veto
  entry timing   confirm on 15m, execute on the next CLOSED 5m candle
  stop loss      structural — beyond the swing that formed the divergence
  take profit    R:R 1:1.5 baseline
  early exit     CVD ROC re-accelerates in the original trend direction
  time stop      N 15m candles without resolution

Two honesty rules are baked in and worth stating plainly:

1. **Pessimistic intrabar resolution.** If a candle's range contains BOTH the
   stop and the target, we record the STOP. Without tick-level replay we cannot
   know which printed first, and assuming the favourable one is exactly how
   backtests get flattered. (We do hold the aggTrades to resolve this exactly;
   that is a deliberate future refinement, not a silent assumption.)

2. **Costs always.** Every trade is priced through ``CostModel`` — there is no
   code path that produces a pre-cost PnL.

Component attribution (methodology rule 11) is supported by ``enabled_layers``:
each layer can be switched off so its individual contribution is measurable
before the combined signal is judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import structlog

from src.core.cvd import DEFAULT_CVD_WINDOW, rolling_cvd
from src.strategy.divergence import (
    BEARISH,
    BULLISH,
    DEFAULT_DECEL_RATIO,
    DEFAULT_PEAK_LOOKBACK,
    DEFAULT_ROC_WINDOW,
    DEFAULT_SPEARMAN_THRESHOLD,
    DEFAULT_SPEARMAN_WINDOW,
    SwingParams,
    detect_regular_divergence,
    find_swing_points,
    roc_deceleration,
    spearman_breakdown,
)
from src.strategy.funding_filter import FundingGateParams, gate_divergences

from .costs import LONG, SHORT, CostModel

log = structlog.get_logger(__name__)

LAYER_SWING = "swing"
LAYER_SPEARMAN = "spearman"
LAYER_ROC = "roc"
LAYER_FUNDING = "funding"
ALL_LAYERS = (LAYER_SWING, LAYER_SPEARMAN, LAYER_ROC, LAYER_FUNDING)

EXIT_STOP = "stop"
EXIT_TARGET = "target"
EXIT_TIME = "time_stop"
EXIT_EARLY = "early_roc"
EXIT_EOD = "end_of_data"


@dataclass(frozen=True)
class StrategyParams:
    """All tunables in one place.

    These are PARAMETERS, never module constants, so research-phase calibration
    is an argument change logged in ITERATION_LOG.md rather than an edit to
    src/ (which would break the holdout's code-freeze gate).
    """

    swing: SwingParams = SwingParams()
    cvd_window: int = DEFAULT_CVD_WINDOW
    spearman_window: int = DEFAULT_SPEARMAN_WINDOW
    spearman_threshold: float = DEFAULT_SPEARMAN_THRESHOLD
    roc_window: int = DEFAULT_ROC_WINDOW
    peak_lookback: int = DEFAULT_PEAK_LOOKBACK
    decel_ratio: float = DEFAULT_DECEL_RATIO
    # Early exit fires only when CVD ROC RE-ACCELERATES in the original
    # trend direction: |ROC| >= reaccel_ratio * prior-peak |ROC|, AND the
    # sign matches. Defined as the mirror of decel_ratio so it is a
    # magnitude test, not a sign test. A bare sign test fires on ~50% of
    # bars (CVD ROC is a zero-mean oscillator) and terminates trades at
    # random -- which is exactly what it did before this was corrected.
    reaccel_ratio: float = 1.0
    funding: FundingGateParams = FundingGateParams()
    rr_ratio: float = 1.5           # take profit at 1.5R
    time_stop_bars_15m: int = 10    # HYPOTHESIS.md: 8-12
    max_bars_between_swings: int | None = 100
    enabled_layers: tuple[str, ...] = ALL_LAYERS
    stop_buffer_bps: float = 0.0    # optional pad beyond the structural swing
    # Minimum stop distance, expressed as a multiple of round-trip cost.
    # 1.0 = risk must at least cover fees+slippage; below that a trade
    # cannot be profitable at any R.
    min_risk_cost_multiple: float = 1.0
    # Position size in base units. Was an unexamined hardcoded 1.0, which for
    # BTC is ~$100k notional and ~333x the median trade -- large enough that
    # market impact would be real and is NOT in the cost model. Kept at 1.0 so
    # existing results are unchanged, but it is now a visible choice.
    position_qty: float = 1.0

    def with_layers(self, *layers: str) -> "StrategyParams":
        return replace(self, enabled_layers=tuple(layers))


@dataclass
class SignalContext:
    """15m analytical frame plus the derived per-bar layer states."""

    bars15: pd.DataFrame
    divergences: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_signals(
    bars15: pd.DataFrame,
    funding: pd.DataFrame,
    params: StrategyParams = StrategyParams(),
) -> pd.DataFrame:
    """Assemble trade intents from the enabled layers.

    Returns one row per accepted divergence with the fields the simulator needs:
    direction, structural stop, and the 15m bar index/time at which the signal
    became actionable.
    """
    b = rolling_cvd(bars15, _infer_tf(bars15), params.cvd_window)
    b = spearman_breakdown(
        b, window=params.spearman_window, threshold=params.spearman_threshold
    )
    b = roc_deceleration(
        b,
        roc_window=params.roc_window,
        peak_lookback=params.peak_lookback,
        decel_ratio=params.decel_ratio,
    )

    swings = find_swing_points(b, params.swing)
    div = detect_regular_divergence(
        b, swings, max_bars_between=params.max_bars_between_swings
    )
    if div.empty:
        return _empty_intents()

    # Layer 2 / 3 must hold AT the confirmation bar (not at the swing bar).
    conf_idx = div["confirmed_at_idx"].to_numpy()
    if LAYER_SPEARMAN in params.enabled_layers:
        div = div[b["spearman_breakdown"].to_numpy()[conf_idx]]
        conf_idx = div["confirmed_at_idx"].to_numpy()
    if LAYER_ROC in params.enabled_layers:
        div = div[b["roc_decelerating"].to_numpy()[conf_idx]]
    if div.empty:
        return _empty_intents()

    if LAYER_FUNDING in params.enabled_layers:
        div = gate_divergences(div.reset_index(drop=True), funding, params.funding)
        div = div[div["funding_gate_open"]]
    if div.empty:
        return _empty_intents()

    div = div.reset_index(drop=True)
    # Structural stop: beyond the swing that formed the divergence.
    is_bear = div["kind"] == BEARISH
    pad = params.stop_buffer_bps * 1e-4
    stop = np.where(
        is_bear, div["price_curr"] * (1 + pad), div["price_curr"] * (1 - pad)
    )
    return pd.DataFrame(
        {
            "kind": div["kind"],
            "side": np.where(is_bear, SHORT, LONG),
            "signal_idx15": div["confirmed_at_idx"].astype("int64"),
            "signal_time": div["confirmed_at_time"].astype("int64"),
            "stop_price": stop.astype(float),
            "swing_price": div["price_curr"].astype(float),
        }
    )


def _empty_intents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "signal_idx15": pd.Series(dtype="int64"),
            "signal_time": pd.Series(dtype="int64"),
            "stop_price": pd.Series(dtype="float64"),
            "swing_price": pd.Series(dtype="float64"),
        }
    )


def _infer_tf(bars: pd.DataFrame) -> str:
    step = int(bars["close_time"].iloc[0] - bars["open_time"].iloc[0])
    return {60_000: "1m", 300_000: "5m", 900_000: "15m", 3_600_000: "1h"}[step]


def simulate(
    intents: pd.DataFrame,
    bars5: pd.DataFrame,
    bars15: pd.DataFrame,
    params: StrategyParams = StrategyParams(),
    costs: CostModel = CostModel(),
    *,
    allow_overlapping: bool = False,
) -> pd.DataFrame:
    """Simulate each intent on the 5m series. Returns one row per trade.

    Entry is the CLOSE of the first 5m candle that closes strictly after the 15m
    confirmation time — i.e. we never trade on information from the candle that
    produced the signal.
    """
    if intents.empty:
        return _empty_trades()

    o5 = bars5["open_time"].to_numpy()
    c5 = bars5["close_time"].to_numpy()
    hi5 = bars5["high"].to_numpy(dtype=float)
    lo5 = bars5["low"].to_numpy(dtype=float)
    cl5 = bars5["close"].to_numpy(dtype=float)

    # 15m ROC series for the early-exit rule, mapped onto 5m bars by "last
    # CLOSED 15m bar at or before this 5m bar's close".
    roc15, peak15 = _roc15_lookup(bars15, bars5, params)

    bars_per_15m = 3
    max_hold = params.time_stop_bars_15m * bars_per_15m

    trades = []
    busy_until = -1
    skipped_invalid_stop = 0
    skipped_risk_too_small = 0
    for _, it in intents.iterrows():
        # first 5m candle CLOSING after the signal
        e = int(np.searchsorted(c5, it["signal_time"], side="right"))
        if e >= len(c5):
            continue
        if not allow_overlapping and e < busy_until:
            continue

        side = it["side"]
        entry_ref = cl5[e]
        entry_px = costs.fill_price(side, entry_ref, is_entry=True)
        stop_px = float(it["stop_price"])

        # The structural stop must sit on the correct side of the ACTUAL fill.
        # If price already ran past the swing before we could enter, the setup
        # is void — not a trade with a tiny stop.
        if (side == LONG and stop_px >= entry_px) or (
            side == SHORT and stop_px <= entry_px
        ):
            skipped_invalid_stop += 1
            continue

        risk = abs(entry_px - stop_px)
        # A stop closer than the round-trip cost cannot produce a profitable
        # trade at ANY R, and would emit an explosive R-multiple (tiny
        # denominator) that distorts expectancy. Drop it rather than let a
        # handful of such trades dominate the statistics.
        min_risk = entry_px * costs.round_trip_bps * 1e-4 * params.min_risk_cost_multiple
        if risk < min_risk:
            skipped_risk_too_small += 1
            continue
        target_px = (
            entry_px + params.rr_ratio * risk
            if side == LONG
            else entry_px - params.rr_ratio * risk
        )

        exit_idx, exit_ref, reason = _walk_to_exit(
            e, side, stop_px, target_px, hi5, lo5, cl5, roc15, peak15,
            max_hold, it["kind"], params.reaccel_ratio
        )
        exit_px = costs.fill_price(side, exit_ref, is_entry=False)

        qty = params.position_qty
        gross = (exit_px - entry_px) if side == LONG else (entry_px - exit_px)
        fees = costs.round_trip_cost(entry_px, exit_px, qty)
        net = gross * qty - fees

        trades.append(
            {
                "kind": it["kind"],
                "side": side,
                "signal_time": int(it["signal_time"]),
                "entry_idx5": e,
                "entry_time": int(c5[e]),
                "entry_price": entry_px,
                "stop_price": stop_px,
                "target_price": target_px,
                "risk_per_unit": risk,
                "exit_idx5": exit_idx,
                "exit_time": int(c5[exit_idx]),
                "exit_price": exit_px,
                "exit_reason": reason,
                "gross_pnl": gross * qty,
                "fees": fees,
                "net_pnl": net,
                # Divide by risk*qty, not risk: net_pnl scales with size, so
                # dividing by per-unit risk would make R scale with position
                # size. Identical at qty=1.0 (all results to date), wrong above it.
                "r_multiple": net / (risk * qty),
                "bars_held": exit_idx - e,
            }
        )
        busy_until = exit_idx

    if skipped_invalid_stop or skipped_risk_too_small:
        log.info(
            "backtest.intents_skipped",
            invalid_stop=skipped_invalid_stop,
            risk_below_cost=skipped_risk_too_small,
            taken=len(trades),
            total_intents=len(intents),
        )
    if not trades:
        return _empty_trades()
    return pd.DataFrame(trades)


def _walk_to_exit(
    e: int,
    side: str,
    stop_px: float,
    target_px: float,
    hi5: np.ndarray,
    lo5: np.ndarray,
    cl5: np.ndarray,
    roc15: np.ndarray,
    peak15: np.ndarray,
    max_hold: int,
    kind: str,
    reaccel_ratio: float,
) -> tuple[int, float, str]:
    """Advance bar by bar to the exit. Pessimistic on ambiguous bars."""
    n = len(cl5)
    last = min(e + max_hold, n - 1)
    for i in range(e + 1, last + 1):
        hit_stop = lo5[i] <= stop_px if side == LONG else hi5[i] >= stop_px
        hit_tgt = hi5[i] >= target_px if side == LONG else lo5[i] <= target_px
        if hit_stop and hit_tgt:
            # Ambiguous bar: record the STOP. Never assume the good fill.
            return i, stop_px, EXIT_STOP
        if hit_stop:
            return i, stop_px, EXIT_STOP
        if hit_tgt:
            return i, target_px, EXIT_TARGET
        # early exit: CVD ROC RE-ACCELERATING in the ORIGINAL trend direction.
        # Magnitude AND direction, per HYPOTHESIS.md. Sign alone is not
        # re-acceleration -- CVD ROC changes sign every ~6 bars on average.
        r, pk = roc15[i], peak15[i]
        if not np.isnan(r) and not np.isnan(pk) and pk > 0:
            strong = abs(r) >= reaccel_ratio * pk
            if strong and kind == BEARISH and r > 0:
                return i, cl5[i], EXIT_EARLY
            if strong and kind == BULLISH and r < 0:
                return i, cl5[i], EXIT_EARLY
    if last <= e:
        return min(e + 1, n - 1), cl5[min(e + 1, n - 1)], EXIT_EOD
    return last, cl5[last], EXIT_TIME if last == e + max_hold else EXIT_EOD


def _roc15_lookup(
    bars15: pd.DataFrame, bars5: pd.DataFrame, params: StrategyParams
) -> tuple[np.ndarray, np.ndarray]:
    """Map each 5m bar to the ROC and prior-peak ROC of the last 15m bar
    CLOSED at or before it.

    The shift is what keeps this causal: a 5m bar may only see 15m information
    that had already closed.
    """
    b = rolling_cvd(bars15, _infer_tf(bars15), params.cvd_window)
    b = roc_deceleration(
        b,
        roc_window=params.roc_window,
        peak_lookback=params.peak_lookback,
        decel_ratio=params.decel_ratio,
    )
    roc = b["cvd_roc"].to_numpy(dtype=float)
    peak = b["prior_peak_roc"].to_numpy(dtype=float)
    c15 = b["close_time"].to_numpy()
    c5 = bars5["close_time"].to_numpy()
    pos = np.searchsorted(c15, c5, side="right") - 1
    out = np.full(len(c5), np.nan)
    out_pk = np.full(len(c5), np.nan)
    ok = pos >= 0
    out[ok] = roc[pos[ok]]
    out_pk[ok] = peak[pos[ok]]
    return out, out_pk


def _empty_trades() -> pd.DataFrame:
    cols = {
        "kind": "object", "side": "object", "signal_time": "int64",
        "entry_idx5": "int64", "entry_time": "int64", "entry_price": "float64",
        "stop_price": "float64", "target_price": "float64",
        "risk_per_unit": "float64", "exit_idx5": "int64", "exit_time": "int64",
        "exit_price": "float64", "exit_reason": "object", "gross_pnl": "float64",
        "fees": "float64", "net_pnl": "float64", "r_multiple": "float64",
        "bars_held": "int64",
    }
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})
