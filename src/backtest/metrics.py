"""Performance metrics with uncertainty attached.

KILL_CRITERIA.md is explicit that a point estimate is not evidence: win rate
must be reported "with a margin sufficient to survive the confidence interval
width at n~100+ trades — report the actual CI, don't just compare point
estimates". So ``summarize`` returns intervals alongside every headline number,
and there is no function here that returns a bare win rate on its own.

All PnL inputs are already NET of fees and slippage (see engine/costs) — there
is no pre-cost path in this codebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Breakeven win rate for a given reward:risk. At R:R 1.5 this is 40%.
def breakeven_win_rate(rr_ratio: float) -> float:
    return 1.0 / (1.0 + rr_ratio)


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    stat=np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Deterministic given ``seed`` (backtests must
    reproduce exactly)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    dist = stat(v[idx], axis=1)
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown of a cumulative equity curve (same units)."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def summarize(
    trades: pd.DataFrame,
    *,
    rr_ratio: float = 1.5,
    seed: int = 12345,
) -> dict:
    """Headline metrics with confidence intervals.

    ``effective_n`` is deliberately NOT computed here: KILL_CRITERIA.md requires
    clustering correction across correlated pairs, which needs the multi-pair
    trade set, not a single symbol's trades. Reporting raw n as if it were the
    effective sample size is exactly the substitution that rule warns against.
    """
    n = len(trades)
    out: dict = {"n_trades": n, "breakeven_win_rate": breakeven_win_rate(rr_ratio)}
    if n == 0:
        return out

    r = trades["r_multiple"].to_numpy(dtype=float)
    wins = r > 0
    out["win_rate"] = float(wins.mean())
    out["win_rate_ci95"] = bootstrap_ci(wins.astype(float), seed=seed)
    out["expectancy_r"] = float(r.mean())
    out["expectancy_r_ci95"] = bootstrap_ci(r, seed=seed)
    out["median_r"] = float(np.median(r))

    gross_win = float(trades.loc[wins, "net_pnl"].sum())
    gross_loss = float(-trades.loc[~wins, "net_pnl"].sum())
    out["profit_factor"] = (
        float("inf") if gross_loss == 0 else gross_win / gross_loss
    )

    out["total_net_pnl"] = float(trades["net_pnl"].sum())
    out["total_fees"] = float(trades["fees"].sum())
    out["total_gross_pnl"] = float(trades["gross_pnl"].sum())
    # How much of the raw edge the costs consumed - the number that usually kills
    # a strategy that "worked" before costs.
    out["fees_as_pct_of_gross"] = (
        float("inf")
        if out["total_gross_pnl"] == 0
        else abs(out["total_fees"] / out["total_gross_pnl"]) * 100.0
    )

    sd = r.std(ddof=1) if n > 1 else 0.0
    out["trade_sharpe"] = float(r.mean() / sd) if sd > 0 else float("nan")
    out["max_drawdown_r"] = max_drawdown(np.cumsum(r))
    out["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()
    out["avg_bars_held"] = float(trades["bars_held"].mean())
    return out


def permutation_test_vs_random_entries(
    actual_r: np.ndarray | pd.Series,
    random_r_samples: list[np.ndarray],
    *,
    stat=np.mean,
) -> dict:
    """Compare the strategy's statistic against randomized-entry baselines.

    ``random_r_samples`` is a list of R-multiple arrays produced by running the
    SAME risk management on randomized entry timing (KILL_CRITERIA.md). The
    p-value is the fraction of random runs that matched or beat the real one.
    """
    a = stat(np.asarray(actual_r, dtype=float))
    null = np.array([stat(np.asarray(s, dtype=float)) for s in random_r_samples])
    if null.size == 0:
        return {"actual": float(a), "p_value": float("nan"), "n_null": 0}
    p = float((null >= a).sum() + 1) / float(null.size + 1)  # +1: never report p=0
    return {
        "actual": float(a),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "n_null": int(null.size),
    }


def format_summary(s: dict) -> str:
    if s.get("n_trades", 0) == 0:
        return "no trades"
    wl, wh = s["win_rate_ci95"]
    el, eh = s["expectancy_r_ci95"]
    return "\n".join(
        [
            f"trades            {s['n_trades']}",
            f"win rate          {s['win_rate']*100:.1f}%  CI95 [{wl*100:.1f}%, {wh*100:.1f}%]"
            f"   (breakeven {s['breakeven_win_rate']*100:.1f}%)",
            f"expectancy        {s['expectancy_r']:+.4f} R  CI95 [{el:+.4f}, {eh:+.4f}]",
            f"profit factor     {s['profit_factor']:.3f}",
            f"trade sharpe      {s['trade_sharpe']:.3f}",
            f"max drawdown      {s['max_drawdown_r']:.2f} R",
            f"net PnL           {s['total_net_pnl']:+.2f} (fees {s['total_fees']:.2f}"
            f" = {s['fees_as_pct_of_gross']:.1f}% of gross)",
            f"avg bars held     {s['avg_bars_held']:.1f}",
            f"exits             {s['exit_reasons']}",
        ]
    )
