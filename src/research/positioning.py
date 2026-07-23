"""Positioning-divergence signal (retail vs top-trader cohorts).

Implements hypothesis/SCREEN_SPEC_positioning_divergence.md EXACTLY. That spec
was committed before any positioning data was examined; this module must not
drift from it. Any change here is a new logged combination, not a tweak.

Signal:
    D_t = ln(global_account_ratio) - ln(top_account_ratio)

positive => retail more long than top traders.

Extremity is the percentile rank of D_t within a trailing 30-day window,
tie-aware (average ranks). Threshold is the top/bottom decile.

The episode rule is the load-bearing part. Raw 5-minute data yields tens of
thousands of threshold-exceeding rows from a few hundred genuinely independent
episodes; treating those as independent samples gives spuriously tight intervals
and a FALSE PASS. So a signal fires only on the CROSSING into an extreme, cannot
re-arm until the series returns to a neutral band, and is additionally rate
limited per direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BAR_MS = 300_000  # metrics are 5-minute snapshots

# --- pre-registered constants (SCREEN_SPEC section 3/4/9) -------------------
LOOKBACK_DAYS = 30
BARS_PER_DAY = 288
LOOKBACK_BARS = LOOKBACK_DAYS * BARS_PER_DAY  # 8640
EXTREME_PCT = 0.10          # decile
NEUTRAL_LO, NEUTRAL_HI = 0.25, 0.75   # hysteresis re-arm band
MIN_GAP_MS = 24 * 3600 * 1000         # same-direction rate limit
PUBLICATION_LAG_MS = BAR_MS           # one bar; create_time semantics unverified


@dataclass(frozen=True)
class PositioningParams:
    lookback_bars: int = LOOKBACK_BARS
    extreme_pct: float = EXTREME_PCT
    neutral_lo: float = NEUTRAL_LO
    neutral_hi: float = NEUTRAL_HI
    min_gap_ms: int = MIN_GAP_MS
    publication_lag_ms: int = PUBLICATION_LAG_MS


def divergence(metrics: pd.DataFrame) -> pd.Series:
    """D_t = ln(retail) - ln(top). NaN where either ratio is missing/non-positive."""
    retail = pd.to_numeric(metrics["global_account_ratio"], errors="coerce")
    top = pd.to_numeric(metrics["top_account_ratio"], errors="coerce")
    ok = (retail > 0) & (top > 0)
    return pd.Series(
        np.where(ok, np.log(retail.where(ok)) - np.log(top.where(ok)), np.nan),
        index=metrics.index, name="D",
    )


def trailing_pct_rank(d: pd.Series, lookback_bars: int) -> pd.Series:
    """Percentile rank of each value within its own trailing window.

    Strictly trailing (window ends at the current bar) and tie-aware: pandas
    rolling rank with method="average" is the midrank, matching the funding
    gate's tie handling. Ties matter — positioning ratios repeat.
    """
    return d.rolling(lookback_bars, min_periods=lookback_bars).rank(
        pct=True, method="average"
    )


def build_episodes(
    metrics: pd.DataFrame, params: PositioningParams = PositioningParams()
) -> pd.DataFrame:
    """Reduce raw 5-min rows to independent signal EPISODES.

    Returns one row per episode: ``signal_time`` (already lagged for
    publication), ``direction``, ``pct_rank``, ``D``.

    direction: -1 when retail is unusually long vs top traders (fade retail),
               +1 when retail is unusually short.
    """
    m = metrics.sort_values("create_time").reset_index(drop=True)
    d = divergence(m)
    pr = trailing_pct_rank(d, params.lookback_bars)
    t = m["create_time"].to_numpy()

    hi_thr = 1.0 - params.extreme_pct
    lo_thr = params.extreme_pct
    prv = pr.to_numpy()

    rows = []
    armed = True          # may a new episode start?
    last_time = {-1: -np.inf, 1: -np.inf}

    for i in range(len(prv)):
        p = prv[i]
        if np.isnan(p):
            continue
        in_neutral = params.neutral_lo <= p <= params.neutral_hi
        if in_neutral:
            armed = True          # hysteresis: returned to neutral, can re-fire
            continue
        if not armed:
            continue              # still inside an extreme zone from a prior fire
        if p >= hi_thr:
            direction = -1
        elif p <= lo_thr:
            direction = 1
        else:
            continue              # between neutral band and extreme: no fire, stay armed

        ts = int(t[i])
        if ts - last_time[direction] < params.min_gap_ms:
            armed = False         # rate limited, but consume the crossing
            continue
        rows.append({
            "signal_time": ts + params.publication_lag_ms,
            "raw_time": ts,
            "direction": direction,
            "pct_rank": float(p),
            "D": float(d.iloc[i]),
        })
        last_time[direction] = ts
        armed = False             # must return to neutral before firing again

    if not rows:
        return pd.DataFrame({
            "signal_time": pd.Series(dtype="int64"),
            "raw_time": pd.Series(dtype="int64"),
            "direction": pd.Series(dtype="int64"),
            "pct_rank": pd.Series(dtype="float64"),
            "D": pd.Series(dtype="float64"),
        })
    return pd.DataFrame(rows)


def funding_pct_rank_at(
    funding: pd.DataFrame, times_ms: np.ndarray, lookback_days: int = 90
) -> np.ndarray:
    """Funding percentile rank knowable at each time (for the proxy check).

    Reuses the funding gate's definition so the two signals are compared on
    like terms.
    """
    from src.strategy.funding_filter import FundingGateParams, evaluate_funding_gate

    st = evaluate_funding_gate(
        funding, times_ms, FundingGateParams(lookback_days=lookback_days, min_obs=30)
    )
    return st["pct_rank"].to_numpy()
