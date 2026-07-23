"""Walk-forward evaluation (CLAUDE.md methodology rule 4).

A single backtest window over the whole research set would let a threshold that
happens to suit that particular stretch look like an edge. Instead the research
set is cut into rolling folds — train N months, test the next M, slide — and
every reported number comes from TEST folds only.

Calibration honesty: HYPOTHESIS.md leaves the Layer 2/3 thresholds "to be
calibrated from the research-set distribution". Here that calibration happens
**per fold, on the training window only**. A threshold derived from data the
fold is then scored on would be circular, and would quietly manufacture an edge.
``calibrate_thresholds`` therefore never sees the test window.

Because calibration is data-derived rather than hand-picked, each distinct
calibration RULE (not each resulting number) is what counts against the
multiple-testing budget in KILL_CRITERIA.md — log it in ITERATION_LOG.md.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import structlog

from src.core.cvd import rolling_cvd
from src.strategy.divergence import roc_deceleration, rolling_spearman

from .costs import CostModel
from .engine import StrategyParams, build_signals, simulate
from .metrics import summarize

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    def __str__(self) -> str:
        return (
            f"train {self.train_start}..{self.train_end} "
            f"test {self.test_start}..{self.test_end}"
        )


def _add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod((d.year * 12 + d.month - 1) + n, 12)
    return dt.date(y, m + 1, 1)


def make_folds(
    start: dt.date,
    end: dt.date,
    *,
    train_months: int = 2,
    test_months: int = 1,
    step_months: int = 1,
) -> list[Fold]:
    """Rolling train/test folds across [start, end]. Test never precedes train."""
    folds: list[Fold] = []
    anchor = dt.date(start.year, start.month, 1)
    while True:
        tr_s = anchor
        te_s = _add_months(tr_s, train_months)
        te_e = _add_months(te_s, test_months) - dt.timedelta(days=1)
        if te_s > end:
            break
        folds.append(
            Fold(tr_s, te_s - dt.timedelta(days=1), te_s, min(te_e, end))
        )
        if te_e >= end:
            break
        anchor = _add_months(anchor, step_months)
    return folds


def _ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _end_ms(d: dt.date) -> int:
    return _ms(d + dt.timedelta(days=1))


def slice_bars(bars: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    lo, hi = _ms(start), _end_ms(end)
    out = bars[(bars["open_time"] >= lo) & (bars["close_time"] <= hi)]
    return out.reset_index(drop=True)


def calibrate_thresholds(
    train_bars15: pd.DataFrame,
    params: StrategyParams,
    *,
    spearman_q: float = 25.0,
    decel_q: float = 25.0,
) -> StrategyParams:
    """Derive Layer 2/3 thresholds from the TRAIN window's own distributions.

    - spearman_threshold := the ``spearman_q``-th percentile of observed rolling
      correlation, i.e. "breakdown" means the bottom quartile of how coupled
      price and CVD normally are on this instrument.
    - decel_ratio := the ``decel_q``-th percentile of |ROC| / prior-peak-|ROC|.

    Both are relative to the instrument's own behaviour, which is the point:
    an absolute 0.3 means different things on BTC and on a new listing.
    """
    tf = _infer_tf(train_bars15)
    b = rolling_cvd(train_bars15, tf, params.cvd_window)

    corr = rolling_spearman(b["close"], b["cvd"], params.spearman_window).dropna()
    sp = float(np.percentile(corr, spearman_q)) if len(corr) else params.spearman_threshold

    r = roc_deceleration(
        b,
        roc_window=params.roc_window,
        peak_lookback=params.peak_lookback,
        decel_ratio=params.decel_ratio,
    )
    ratio = (r["cvd_roc"].abs() / r["prior_peak_roc"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    dr = float(np.percentile(ratio, decel_q)) if len(ratio) else params.decel_ratio
    dr = float(min(max(dr, 1e-6), 1.0))  # keep inside the validated domain

    return replace(params, spearman_threshold=sp, decel_ratio=dr)


def _infer_tf(bars: pd.DataFrame) -> str:
    step = int(bars["close_time"].iloc[0] - bars["open_time"].iloc[0])
    return {60_000: "1m", 300_000: "5m", 900_000: "15m", 3_600_000: "1h"}[step]


def run_walkforward(
    bars15: pd.DataFrame,
    bars5: pd.DataFrame,
    funding: pd.DataFrame,
    folds: list[Fold],
    params: StrategyParams = StrategyParams(),
    costs: CostModel = CostModel(),
    *,
    calibrate: bool = True,
    warmup_days: int = 90,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run every fold; return (all test-fold trades, per-fold summaries).

    Signals for a test fold are built on bars that include backward warm-up
    (so indicators are primed) but the fold's trades are restricted to entries
    inside the test window — warm-up never contributes trades.
    """
    all_trades = []
    fold_reports: list[dict] = []

    for f in folds:
        train15 = slice_bars(bars15, f.train_start, f.train_end)
        if train15.empty:
            continue
        p = calibrate_thresholds(train15, params) if calibrate else params

        # test window + backward warm-up for indicator priming
        wu_start = f.test_start - dt.timedelta(days=warmup_days)
        ctx15 = slice_bars(bars15, wu_start, f.test_end)
        ctx5 = slice_bars(bars5, wu_start, f.test_end)
        if ctx15.empty or ctx5.empty:
            continue

        intents = build_signals(ctx15, funding, p)
        # keep only signals that fire INSIDE the test window
        lo, hi = _ms(f.test_start), _end_ms(f.test_end)
        intents = intents[
            (intents["signal_time"] >= lo) & (intents["signal_time"] < hi)
        ].reset_index(drop=True)

        trades = simulate(intents, ctx5, ctx15, p, costs)
        if not trades.empty:
            trades = trades.assign(
                fold=str(f), test_start=str(f.test_start), test_end=str(f.test_end)
            )
            all_trades.append(trades)

        rep = {
            "fold": str(f),
            "spearman_threshold": p.spearman_threshold,
            "decel_ratio": p.decel_ratio,
            "n_intents": len(intents),
            **summarize(trades, rr_ratio=p.rr_ratio),
        }
        fold_reports.append(rep)
        log.info("walkforward.fold", fold=str(f), intents=len(intents),
                 trades=len(trades))

    combined = (
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    )
    return combined, fold_reports
