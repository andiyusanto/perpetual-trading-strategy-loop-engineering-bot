"""Three-layer CVD divergence detection (HYPOTHESIS.md, Layers 1-3).

Layer 1  structural swing  - fractal swing highs/lows, regular divergence
Layer 2  Spearman breakdown - rolling rank-correlation between price and CVD
Layer 3  CVD ROC deceleration - flow decelerating vs its prior peak

**Look-ahead discipline is the whole game here.** A fractal swing at bar *i* is
not knowable until ``right`` further bars have closed. Every swing and every
divergence therefore carries an explicit ``confirmed_at_idx`` /
``confirmed_at_time``, which is the earliest moment the event could have been
acted on. Nothing in this module lets a caller see a swing at the bar where it
occurred. Layers 2 and 3 use strictly trailing windows for the same reason.

Thresholds for Layers 2 and 3 are parameters, NOT hardcoded constants: HYPOTHESIS
.md marks them "to be calibrated from the research-set distribution". They are
left explicit so calibration is a visible, logged act rather than a magic number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Layer 1 - structural swings (fractals) and regular divergence
# ---------------------------------------------------------------------------

SWING_HIGH = "high"
SWING_LOW = "low"


@dataclass(frozen=True)
class SwingParams:
    """Fractal swing definition. HYPOTHESIS.md baseline: 3 bars either side
    (sensitivity set 2/3/5, research-set only)."""

    left: int = 3
    right: int = 3


def find_swing_points(
    bars: pd.DataFrame, params: SwingParams = SwingParams()
) -> pd.DataFrame:
    """Locate fractal swing highs/lows.

    A swing high at *i* requires ``high[i]`` to be strictly greater than the
    ``left`` highs before and the ``right`` highs after it (strict inequality on
    both sides, so flat plateaus do not register as swings). Swing lows mirror
    this on ``low``.

    Returns one row per swing with:
      ``idx``               bar index where the swing occurred
      ``kind``              "high" | "low"
      ``price``             the extreme price of the swing
      ``open_time``         open time of the swing bar
      ``confirmed_at_idx``  idx + right — the FIRST bar at which this swing is
                            knowable
      ``confirmed_at_time`` close_time of that confirming bar; the earliest
                            timestamp a decision may use this swing
    """
    left, right = params.left, params.right
    if left < 1 or right < 1:
        raise ValueError("left and right must both be >= 1")
    n = len(bars)
    if n < left + right + 1:
        return _empty_swings()

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    open_time = bars["open_time"].to_numpy()
    close_time = bars["close_time"].to_numpy()

    # Candidate centres are indices that have `left` bars behind and `right` ahead.
    centre = np.arange(left, n - right)
    win_h = sliding_window_view(high, left + right + 1)  # rows indexed by centre-left
    win_l = sliding_window_view(low, left + right + 1)

    centre_h = win_h[:, left]
    centre_l = win_l[:, left]
    # strictly greater than every neighbour (exclude the centre column itself)
    neighbours = np.r_[np.arange(0, left), np.arange(left + 1, left + right + 1)]
    is_high = (centre_h[:, None] > win_h[:, neighbours]).all(axis=1)
    is_low = (centre_l[:, None] < win_l[:, neighbours]).all(axis=1)

    rows = []
    for mask, kind, price_arr in (
        (is_high, SWING_HIGH, high),
        (is_low, SWING_LOW, low),
    ):
        idxs = centre[mask]
        for i in idxs:
            rows.append(
                {
                    "idx": int(i),
                    "kind": kind,
                    "price": float(price_arr[i]),
                    "open_time": int(open_time[i]),
                    "confirmed_at_idx": int(i + right),
                    "confirmed_at_time": int(close_time[i + right]),
                }
            )

    if not rows:
        return _empty_swings()
    out = pd.DataFrame(rows).sort_values(["idx", "kind"]).reset_index(drop=True)
    return out


def _empty_swings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "idx": pd.Series(dtype="int64"),
            "kind": pd.Series(dtype="object"),
            "price": pd.Series(dtype="float64"),
            "open_time": pd.Series(dtype="int64"),
            "confirmed_at_idx": pd.Series(dtype="int64"),
            "confirmed_at_time": pd.Series(dtype="int64"),
        }
    )


BEARISH = "bearish"  # price higher high + CVD lower high -> expect down reversal
BULLISH = "bullish"  # price lower low  + CVD higher low  -> expect up reversal


def detect_regular_divergence(
    bars: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    cvd_col: str = "cvd",
    max_bars_between: int | None = None,
) -> pd.DataFrame:
    """Regular divergence between CONSECUTIVE same-kind swings (v1 = regular only).

    bearish: price HH (``high2 > high1``) while CVD makes a lower high
             (``cvd2 < cvd1``)
    bullish: price LL (``low2 < low1``)  while CVD makes a higher low
             (``cvd2 > cvd1``)

    CVD is read at the swing bar itself, but the event is stamped
    ``confirmed_at_time`` = the confirmation time of the LATER swing, since that
    is the first moment both swings are knowable.

    ``max_bars_between`` optionally rejects pairs whose swings are far apart
    (comparing swings hundreds of bars apart is a different claim than comparing
    adjacent structure). ``None`` = no limit.
    """
    if cvd_col not in bars.columns:
        raise KeyError(f"bars has no {cvd_col!r} column - compute rolling CVD first")
    if swings.empty:
        return _empty_divergences()

    cvd = bars[cvd_col].to_numpy(dtype=float)
    rows = []
    for kind, price_cmp, cvd_cmp, label in (
        (SWING_HIGH, lambda a, b: b > a, lambda a, b: b < a, BEARISH),
        (SWING_LOW, lambda a, b: b < a, lambda a, b: b > a, BULLISH),
    ):
        sub = swings[swings["kind"] == kind].sort_values("idx").reset_index(drop=True)
        for k in range(1, len(sub)):
            s1, s2 = sub.iloc[k - 1], sub.iloc[k]
            i1, i2 = int(s1["idx"]), int(s2["idx"])
            if max_bars_between is not None and (i2 - i1) > max_bars_between:
                continue
            c1, c2 = cvd[i1], cvd[i2]
            if np.isnan(c1) or np.isnan(c2):
                continue  # inside CVD warm-up; not comparable
            if price_cmp(s1["price"], s2["price"]) and cvd_cmp(c1, c2):
                rows.append(
                    {
                        "kind": label,
                        "idx_prev": i1,
                        "idx_curr": i2,
                        "price_prev": float(s1["price"]),
                        "price_curr": float(s2["price"]),
                        "cvd_prev": float(c1),
                        "cvd_curr": float(c2),
                        "confirmed_at_idx": int(s2["confirmed_at_idx"]),
                        "confirmed_at_time": int(s2["confirmed_at_time"]),
                    }
                )
    if not rows:
        return _empty_divergences()
    return (
        pd.DataFrame(rows)
        .sort_values(["confirmed_at_idx", "kind"])
        .reset_index(drop=True)
    )


def _empty_divergences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": pd.Series(dtype="object"),
            "idx_prev": pd.Series(dtype="int64"),
            "idx_curr": pd.Series(dtype="int64"),
            "price_prev": pd.Series(dtype="float64"),
            "price_curr": pd.Series(dtype="float64"),
            "cvd_prev": pd.Series(dtype="float64"),
            "cvd_curr": pd.Series(dtype="float64"),
            "confirmed_at_idx": pd.Series(dtype="int64"),
            "confirmed_at_time": pd.Series(dtype="int64"),
        }
    )


# ---------------------------------------------------------------------------
# Layer 2 - rolling Spearman correlation breakdown
# ---------------------------------------------------------------------------

DEFAULT_SPEARMAN_WINDOW = 20      # HYPOTHESIS.md baseline (sensitivity 14/20/30)
DEFAULT_SPEARMAN_THRESHOLD = 0.3  # STARTING POINT - calibrate on research set


def rolling_spearman(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Trailing rolling Spearman rank correlation between two series.

    Strictly trailing: the value at *i* uses bars [i-window+1 .. i] only.
    Emits NaN for the warm-up and for any window where either side is constant
    (rank correlation is undefined with zero variance) — never a fabricated 0.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    n = len(xv)
    out = np.full(n, np.nan)
    if n < window:
        return pd.Series(out, index=x.index)

    wx = sliding_window_view(xv, window)
    wy = sliding_window_view(yv, window)
    valid = ~(np.isnan(wx).any(axis=1) | np.isnan(wy).any(axis=1))

    # average ranks (handles ties correctly), then Pearson on the ranks
    rx = rankdata(wx, axis=1).astype(float)
    ry = rankdata(wy, axis=1).astype(float)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    num = (rx * ry).sum(axis=1)
    den = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where((den > 0) & valid, num / den, np.nan)
    out[window - 1:] = corr
    return pd.Series(out, index=x.index)


def spearman_breakdown(
    bars: pd.DataFrame,
    *,
    price_col: str = "close",
    cvd_col: str = "cvd",
    window: int = DEFAULT_SPEARMAN_WINDOW,
    threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> pd.DataFrame:
    """Layer 2: flag bars where price/CVD rank correlation has broken down.

    ``threshold`` is a calibration target, not settled dogma (HYPOTHESIS.md).
    """
    corr = rolling_spearman(bars[price_col], bars[cvd_col], window)
    out = bars.copy()
    out["spearman"] = corr
    out["spearman_breakdown"] = corr < threshold  # NaN compares False -> no signal
    return out


# ---------------------------------------------------------------------------
# Layer 3 - CVD rate-of-change deceleration
# ---------------------------------------------------------------------------

DEFAULT_ROC_WINDOW = 7        # HYPOTHESIS.md: 5-10 candle window
DEFAULT_PEAK_LOOKBACK = 20    # bars over which "prior peak ROC" is measured
DEFAULT_DECEL_RATIO = 0.5     # STARTING POINT - calibrate on research set


def cvd_roc(cvd: pd.Series, window: int = DEFAULT_ROC_WINDOW) -> pd.Series:
    """Rate of change of CVD: average delta-of-CVD per bar over a trailing window.

    Uses an absolute difference (not percent change): CVD crosses zero, so a
    percentage would explode near the crossing and is not meaningful here.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    return (cvd - cvd.shift(window)) / float(window)


def roc_deceleration(
    bars: pd.DataFrame,
    *,
    cvd_col: str = "cvd",
    roc_window: int = DEFAULT_ROC_WINDOW,
    peak_lookback: int = DEFAULT_PEAK_LOOKBACK,
    decel_ratio: float = DEFAULT_DECEL_RATIO,
) -> pd.DataFrame:
    """Layer 3: flag bars where |CVD ROC| has decelerated vs its prior peak.

    ``prior_peak_roc`` is the max |ROC| over the trailing ``peak_lookback`` bars
    ENDING AT THE PREVIOUS BAR (shifted by 1), so the current bar never
    contributes to the peak it is being compared against.

    Fires when ``|roc| < decel_ratio * prior_peak_roc``.
    """
    if not 0 < decel_ratio <= 1:
        raise ValueError("decel_ratio must be in (0, 1]")
    out = bars.copy()
    roc = cvd_roc(out[cvd_col], roc_window)
    out["cvd_roc"] = roc
    prior_peak = (
        roc.abs().rolling(peak_lookback, min_periods=peak_lookback).max().shift(1)
    )
    out["prior_peak_roc"] = prior_peak
    out["roc_decelerating"] = roc.abs() < (decel_ratio * prior_peak)
    return out
