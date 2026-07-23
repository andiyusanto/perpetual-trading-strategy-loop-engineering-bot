"""Signal screening: does a candidate carry information worth more than costs?

WHY THIS EXISTS
---------------
The CVD-divergence cycle built a full backtest engine — entries, stops, targets,
time stops, walk-forward, costs — before asking whether the signal contained any
directional information at all. It didn't (forward returns |t| <= 0.66 at every
horizon from 30min to 5h). That question needs none of the machinery and takes
seconds to answer.

So: screen first, build second.

WHAT IT MEASURES
----------------
One question only: given a signal timestamp and an intended direction, does
price move favourably more than chance AND by more than the round-trip cost?

Deliberately absent: stops, targets, R:R, time stops, position sizing. Those are
choices, and in the last cycle they *generated* the result — the exit rule caused
72% of outcomes and the R:R choice made the breakeven look like 40% when it was
really ~57%. Removing them isolates the signal itself.

WHAT PASSING DOES *NOT* MEAN
----------------------------
Necessary, not sufficient. A positive screen says a signal has a pulse; it says
nothing about path dependency, drawdown, or whether a stop survives the noise.
Promotion to a hypothesis still requires the full gate battery.

MULTIPLE-TESTING DISCIPLINE
---------------------------
Cheap screening makes p-hacking cheap. Every screen run is appended to
``results/screening_log.jsonl`` automatically — passes and failures alike — so
that if a candidate is later promoted, the number of screens it beat is on the
record and the significance bar can be raised accordingly (KILL_CRITERIA.md
multiple-testing rule).

NO LOOK-AHEAD
-------------
Entry price is the close of the last bar CLOSED at or before the signal time.
The forward price is the close of the last bar closed at or before
``signal_time + horizon``. Nothing reads a bar that had not finished.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCREENING_LOG = _PROJECT_ROOT / "results" / "screening_log.jsonl"

MS_PER_MIN = 60_000

# Binance USD-M taker round trip (2 x 5bps) + 2 x 3bps slippage. A signal whose
# edge does not clear this is real-but-untradeable as a taker.
DEFAULT_COST_BPS = 16.0

DEFAULT_HORIZONS_MIN = (30, 60, 120, 240, 480, 1440)


@dataclass(frozen=True)
class SignalSet:
    """A candidate signal: when it fired, and which way it pointed.

    ``times_ms``  decision timestamps (epoch ms) — the moment the signal was
                  knowable, NOT the bar it describes.
    ``direction`` +1 expects price up, -1 expects price down.
    """

    name: str
    times_ms: np.ndarray
    direction: np.ndarray
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.times_ms) != len(self.direction):
            raise ValueError("times_ms and direction must be the same length")
        bad = set(np.unique(self.direction)) - {-1, 1}
        if bad:
            raise ValueError(f"direction must be +1/-1, got {sorted(bad)}")

    def __len__(self) -> int:
        return len(self.times_ms)


def forward_returns(
    bars: pd.DataFrame,
    times_ms: np.ndarray,
    direction: np.ndarray,
    horizon_min: int,
) -> np.ndarray:
    """Direction-signed forward return in bps for each signal, at one horizon.

    NaN where the horizon runs past the end of the data (those signals are
    excluded from statistics rather than silently treated as zero).
    """
    close_t = bars["close_time"].to_numpy()
    close_p = bars["close"].to_numpy(dtype=float)

    entry_i = np.searchsorted(close_t, times_ms, side="right") - 1
    exit_i = (
        np.searchsorted(close_t, times_ms + horizon_min * MS_PER_MIN, side="right") - 1
    )

    out = np.full(len(times_ms), np.nan)
    ok = (entry_i >= 0) & (exit_i > entry_i) & (exit_i < len(close_p))
    if not ok.any():
        return out
    ep = close_p[entry_i[ok]]
    xp = close_p[exit_i[ok]]
    out[ok] = direction[ok] * (xp / ep - 1.0) * 1e4
    return out


def _block_bootstrap_ci(
    x: np.ndarray, *, block: int = 20, n_boot: int = 5000, alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """CI that respects serial correlation.

    Signals cluster in time, so an i.i.d. bootstrap would understate the
    interval. Resampling contiguous blocks keeps local dependence intact.
    """
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n <= block:
        block = max(1, n // 2) or 1
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    idx = np.clip(idx[:, :n], 0, n - 1)
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _permutation_baseline(
    bars: pd.DataFrame,
    n_signals: int,
    direction: np.ndarray,
    horizon_min: int,
    *,
    n_perm: int = 200,
    seed: int = 11,
) -> np.ndarray:
    """Null distribution of mean forward return from RANDOM entry times.

    Same count, same mix of long/short, same horizon — only the timing is
    random. This is the honest baseline: "could picking moments at random have
    done as well?"
    """
    rng = np.random.default_rng(seed)
    close_t = bars["close_time"].to_numpy()
    lo, hi = 0, len(close_t) - 1
    out = np.empty(n_perm)
    for k in range(n_perm):
        picks = rng.integers(lo, hi, size=n_signals)
        t = close_t[picks]
        d = rng.permutation(direction)
        r = forward_returns(bars, t, d, horizon_min)
        out[k] = np.nanmean(r) if not np.all(np.isnan(r)) else np.nan
    return out


def screen_signal(
    bars: pd.DataFrame,
    signal: SignalSet,
    *,
    horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN,
    cost_bps: float = DEFAULT_COST_BPS,
    n_perm: int = 200,
    n_subperiods: int = 4,
    log: bool = True,
    segment: str = "research",
) -> dict:
    """Screen one candidate. Returns a report dict and appends it to the log."""
    report: dict = {
        "signal": signal.name,
        "params": signal.params,
        "segment": segment,
        "n_signals": len(signal),
        "cost_bps": cost_bps,
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "horizons": {},
    }
    if len(signal) == 0:
        report["verdict"] = "NO SIGNALS"
        if log:
            _append_log(report)
        return report

    # how clustered are these signals? raw n overstates independence
    days = np.unique((signal.times_ms // 86_400_000)).size
    report["distinct_days"] = int(days)
    report["signals_per_day"] = round(len(signal) / max(days, 1), 2)

    any_tradeable = False
    for h in horizons_min:
        r = forward_returns(bars, signal.times_ms, signal.direction, h)
        v = r[~np.isnan(r)]
        if v.size < 2:
            continue
        mean = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(v.size))
        t = mean / se if se > 0 else float("nan")
        lo, hi = _block_bootstrap_ci(v)
        null = _permutation_baseline(bars, len(signal), signal.direction, h, n_perm=n_perm)
        null = null[~np.isnan(null)]
        p = (float((null >= mean).sum()) + 1) / (null.size + 1) if null.size else float("nan")

        # The decisive test: does the edge clear the cost of taking it?
        clears = (lo > cost_bps)
        any_tradeable |= bool(clears)
        report["horizons"][f"{h}m"] = {
            "n": int(v.size),
            "mean_bps": round(mean, 3),
            "t_stat": round(float(t), 3),
            "ci95_bps": [round(lo, 3), round(hi, 3)],
            "frac_positive": round(float((v > 0).mean()), 4),
            "perm_p": round(float(p), 4),
            "null_mean_bps": round(float(null.mean()), 3) if null.size else None,
            "clears_cost": bool(clears),
        }

    # stability: is any effect present across sub-periods, or one window only?
    if n_subperiods > 1 and horizons_min:
        h0 = horizons_min[len(horizons_min) // 2]
        r = forward_returns(bars, signal.times_ms, signal.direction, h0)
        edges = np.quantile(signal.times_ms, np.linspace(0, 1, n_subperiods + 1))
        sub = []
        for i in range(n_subperiods):
            m = (signal.times_ms >= edges[i]) & (signal.times_ms <= edges[i + 1])
            vv = r[m]
            vv = vv[~np.isnan(vv)]
            sub.append({"n": int(vv.size),
                        "mean_bps": round(float(vv.mean()), 3) if vv.size else None})
        report["subperiods"] = {"horizon": f"{h0}m", "buckets": sub}

    report["verdict"] = "TRADEABLE EDGE" if any_tradeable else "NO TRADEABLE EDGE"
    if log:
        _append_log(report)
    return report


def _append_log(report: dict) -> None:
    """Append-only record of EVERY screen — passes and failures alike."""
    SCREENING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SCREENING_LOG.open("a") as f:
        f.write(json.dumps(report, default=str) + "\n")


def screens_run() -> int:
    """How many screens have ever been run (for multiple-testing accounting)."""
    if not SCREENING_LOG.exists():
        return 0
    return sum(1 for _ in SCREENING_LOG.open())


def format_report(rep: dict) -> str:
    lines = [
        f"signal   : {rep['signal']}   [{rep.get('segment')}]",
        f"n        : {rep['n_signals']} signals over {rep.get('distinct_days','?')} days "
        f"({rep.get('signals_per_day','?')}/day)",
        f"cost bar : {rep['cost_bps']} bps round trip",
        "",
        f"{'horizon':>8} {'mean bps':>10} {'t':>7} {'CI95 bps':>20} {'>0':>7} {'perm p':>8} {'clears cost':>12}",
    ]
    for h, d in rep.get("horizons", {}).items():
        ci = f"[{d['ci95_bps'][0]:.2f}, {d['ci95_bps'][1]:.2f}]"
        lines.append(
            f"{h:>8} {d['mean_bps']:>10.2f} {d['t_stat']:>7.2f} {ci:>20} "
            f"{d['frac_positive']:>7.3f} {d['perm_p']:>8.3f} {str(d['clears_cost']):>12}"
        )
    if "subperiods" in rep:
        b = ", ".join(str(x["mean_bps"]) for x in rep["subperiods"]["buckets"])
        lines.append(f"\nsub-period means ({rep['subperiods']['horizon']}): [{b}]")
    lines.append(f"\nVERDICT  : {rep['verdict']}")
    return "\n".join(lines)
