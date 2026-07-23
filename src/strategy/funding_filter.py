"""Funding rate filter — a GATE, not a signal (HYPOTHESIS.md, layer 4).

Entry is permitted only when funding is at an extreme of its own recent
distribution: top/bottom ~10% of the trailing 90-day window, measured **per
pair**, because pairs differ in typical funding range (an absolute cutoff like
0.01% means very different things on BTC vs a new listing).

Direction matters. The hypothesis exploits a crowded side being forced out:
  - bearish setup (price made a higher high) -> the crowd is LONG -> require
    funding at a HIGH extreme (longs paying shorts)
  - bullish setup (price made a lower low)   -> the crowd is SHORT -> require
    funding at a LOW extreme (shorts paying longs)
A funding extreme in the wrong direction does not gate the trade open.

**No look-ahead.** At decision time T only funding events already SETTLED
(``funding_time <= T``) exist. Binance also publishes a continuously-updating
*predicted* rate, but that is not a settled fact and is not used here. The
trailing distribution likewise contains only settled events in [T-90d, T].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .divergence import BEARISH, BULLISH

MS_PER_DAY = 86_400_000

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_EXTREME_PCT = 0.10  # top/bottom decile
# Minimum settled funding events required before the gate will judge an extreme.
# 90d at 8h cadence is ~270; requiring a solid fraction avoids ruling on a
# distribution built from a handful of points (matters for newly listed pairs).
DEFAULT_MIN_OBS = 60


@dataclass(frozen=True)
class FundingGateParams:
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    extreme_pct: float = DEFAULT_EXTREME_PCT
    min_obs: int = DEFAULT_MIN_OBS

    def __post_init__(self) -> None:
        if not 0 < self.extreme_pct < 0.5:
            raise ValueError("extreme_pct must be in (0, 0.5)")
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")


def load_funding(symbol: str, data_root: Path) -> pd.DataFrame:
    """Load settled funding history for a symbol, sorted by settlement time."""
    path = data_root / "raw" / symbol / "funding" / f"{symbol}-fundingRate.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no funding data for {symbol}: {path}")
    df = pd.read_parquet(path)
    return df.sort_values("funding_time").reset_index(drop=True)


def evaluate_funding_gate(
    funding: pd.DataFrame,
    decision_times_ms: np.ndarray | list[int],
    params: FundingGateParams = FundingGateParams(),
) -> pd.DataFrame:
    """Evaluate the funding state that was knowable at each decision time.

    Returns one row per decision time:
      ``decision_time``     the input timestamp (epoch ms)
      ``last_funding_time`` settlement time of the most recent SETTLED event
      ``last_funding_rate`` that event's rate — the only rate known at T
      ``pct_rank``          its rank within the trailing window, in [0, 1]
      ``n_obs``             settled events in the trailing window
      ``is_high_extreme``   pct_rank >= 1 - extreme_pct  (crowded longs)
      ``is_low_extreme``    pct_rank <= extreme_pct      (crowded shorts)

    Rows where nothing has settled yet, or the window is too thin
    (``n_obs < min_obs``), get NaN/False — the gate abstains rather than
    guessing, so a thin early-listing window cannot manufacture an "extreme".
    """
    times = np.asarray(decision_times_ms, dtype="int64")
    ft = funding["funding_time"].to_numpy(dtype="int64")
    fr = funding["funding_rate"].to_numpy(dtype=float)
    lookback_ms = params.lookback_days * MS_PER_DAY

    # index of the last event with funding_time <= T  (settled, hence knowable)
    last_idx = np.searchsorted(ft, times, side="right") - 1
    # first index inside the trailing window
    first_idx = np.searchsorted(ft, times - lookback_ms, side="left")

    n = len(times)
    out_rate = np.full(n, np.nan)
    out_time = np.full(n, -1, dtype="int64")
    out_rank = np.full(n, np.nan)
    out_nobs = np.zeros(n, dtype="int64")

    for k in range(n):
        li = last_idx[k]
        if li < 0:
            continue  # nothing settled yet at this decision time
        fi = first_idx[k]
        window = fr[fi : li + 1]
        out_rate[k] = fr[li]
        out_time[k] = ft[li]
        out_nobs[k] = window.size
        if window.size >= params.min_obs:
            # MIDRANK, not a plain "<=" fraction. Funding data is full of exact
            # ties (Binance parks at clamped values like 0.0001 for long
            # stretches); counting ties as "at or below" would rank the single
            # most COMMON value at 1.0 and hold the gate permanently open.
            # Midrank puts a fully tied distribution at 0.5 — correctly "not
            # extreme" — while a true max still lands ~1.0.
            below = float((window < fr[li]).sum())
            equal = float((window == fr[li]).sum())
            out_rank[k] = (below + 0.5 * equal) / window.size

    res = pd.DataFrame(
        {
            "decision_time": times,
            "last_funding_time": out_time,
            "last_funding_rate": out_rate,
            "pct_rank": out_rank,
            "n_obs": out_nobs,
        }
    )
    res["is_high_extreme"] = res["pct_rank"] >= (1.0 - params.extreme_pct)
    res["is_low_extreme"] = res["pct_rank"] <= params.extreme_pct
    return res


def gate_divergences(
    divergences: pd.DataFrame,
    funding: pd.DataFrame,
    params: FundingGateParams = FundingGateParams(),
) -> pd.DataFrame:
    """Attach the funding gate to divergence events.

    Adds ``funding_gate_open``: True only when the funding extreme points the
    SAME way as the crowded side the setup intends to exploit
    (bearish -> high extreme, bullish -> low extreme).
    """
    if divergences.empty:
        return divergences.assign(funding_gate_open=pd.Series(dtype=bool))

    state = evaluate_funding_gate(
        funding, divergences["confirmed_at_time"].to_numpy(), params
    )
    out = divergences.reset_index(drop=True).join(
        state.drop(columns=["decision_time"])
    )
    is_bear = out["kind"] == BEARISH
    is_bull = out["kind"] == BULLISH
    out["funding_gate_open"] = (is_bear & out["is_high_extreme"]) | (
        is_bull & out["is_low_extreme"]
    )
    return out
