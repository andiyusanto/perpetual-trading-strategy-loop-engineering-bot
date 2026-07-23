"""CVD reconstruction from tick-level aggTrades — FOUNDATION (bar aggregation).

This module builds the deterministic, look-ahead-free foundation the CVD
divergence layers sit on: it turns raw aggTrades into fixed-grid OHLCV bars with
per-bar **signed volume delta** (buy volume − sell volume).

What is fully specified (and implemented here):
  - Taker side (CLAUDE.md): is_buyer_maker=True  -> taker was SELLER -> sell vol
                            is_buyer_maker=False -> taker was BUYER  -> buy vol
  - Bars live on a UTC-aligned grid (5m/15m divide evenly into the day and into
    every month boundary, so bars never straddle a month → monthly bar sets
    concatenate cleanly).
  - delta = buy_vol − sell_vol, per bar.

Anchoring ("rolling window reset"), decided 2026-07-23 and specified in
HYPOTHESIS.md: CVD is a **trailing N-bar rolling sum of delta** (see
``rolling_cvd``). The window slides so old flow drops out, which keeps CVD
bounded and comparable across time without an arbitrary session boundary in a
24/7 market. N is a real parameter subject to the sensitivity gate.

Determinism: every function here is a pure function of the input trades —
same trades → same bars, every time (required for valid backtests).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import structlog

log = structlog.get_logger(__name__)

# Supported signal timeframes -> milliseconds. (Day-trade: 15m confirm, 5m entry.)
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}

# Columns actually needed for bar aggregation (avoids loading trade ids).
_NEEDED_COLS = ["timestamp", "price", "quantity", "is_buyer_maker"]

# Trades per streaming batch. ~2M rows x ~25B/row ≈ 50MB, so peak memory stays
# bounded regardless of month size (largest month is 84M trades).
_BATCH_ROWS = 2_000_000

# v1 rolling-CVD window, in bars. Chosen to match Layer 2's rolling 20-candle
# Spearman lookback so both layers observe the same horizon. This is a real
# parameter: it counts against the multiple-testing budget and is subject to the
# ±20%/±40% sensitivity gate in KILL_CRITERIA.md (sensitivity set: 14/20/30).
DEFAULT_CVD_WINDOW = 20

# Output bar schema (column order).
BAR_COLUMNS = [
    "open_time",   # bar start, epoch ms (UTC-aligned)
    "close_time",  # bar end (exclusive boundary), epoch ms = open_time + tf_ms
    "open", "high", "low", "close",
    "vwap",
    "volume",      # total base-asset volume in the bar
    "buy_vol",     # taker-buy base volume
    "sell_vol",    # taker-sell base volume
    "delta",       # buy_vol - sell_vol
    "num_trades",
]


def _partial_bars(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Per-chunk partial bar aggregates (still combinable across chunks).

    Keeps raw sums (``price_qty``) rather than vwap so partials from different
    chunks of the same bar can be merged exactly.
    """
    ts = df["timestamp"].to_numpy()
    if len(ts) and not (ts[:-1] <= ts[1:]).all():
        # Defensive: the aggregations below assume time order for open/close.
        df = df.sort_values("timestamp", kind="stable")

    qty = df["quantity"]
    is_maker = df["is_buyer_maker"]
    # taker BUY when is_buyer_maker is False; taker SELL when True.
    work = pd.DataFrame(
        {
            "open_time": (df["timestamp"] // tf) * tf,
            "price": df["price"],
            "quantity": qty,
            "buy_qty": qty.where(~is_maker, 0.0),
            "sell_qty": qty.where(is_maker, 0.0),
            "price_qty": df["price"] * qty,
        }
    )
    return work.groupby("open_time", sort=True).agg(
        open=("price", "first"),   # time-sorted -> first == earliest trade
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),   # last == latest trade in the chunk's bar
        volume=("quantity", "sum"),
        buy_vol=("buy_qty", "sum"),
        sell_vol=("sell_qty", "sum"),
        price_qty=("price_qty", "sum"),
        num_trades=("price", "size"),
    ).reset_index()


def _finalize(partials: list[pd.DataFrame], tf: int) -> pd.DataFrame:
    """Merge chronologically-ordered partials into final bars.

    Chunks arrive in time order, so for a bar split across chunks ``first`` picks
    the earliest chunk's open and ``last`` the latest chunk's close.
    """
    if not partials:
        return pd.DataFrame(columns=BAR_COLUMNS)
    cat = pd.concat(partials, ignore_index=True)
    bars = cat.groupby("open_time", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        price_qty=("price_qty", "sum"),
        num_trades=("num_trades", "sum"),
    ).reset_index()
    bars["vwap"] = bars["price_qty"] / bars["volume"]
    bars["delta"] = bars["buy_vol"] - bars["sell_vol"]
    bars["close_time"] = bars["open_time"] + tf
    return bars[BAR_COLUMNS]


def bars_from_trades(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate chronological aggTrades into fixed-grid OHLCV+delta bars.

    ``df`` must contain ``_NEEDED_COLS`` and be sorted by ``timestamp`` ascending
    (raw aggTrades are chronological by construction). Returns one row per bar
    that had at least one trade; empty (no-trade) bars are not emitted here.
    Pure function: same trades -> same bars.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unknown timeframe {timeframe!r}; known: {list(TIMEFRAME_MS)}")
    tf = TIMEFRAME_MS[timeframe]
    return _finalize([_partial_bars(df, tf)], tf)


def reindex_to_grid(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Insert missing (no-trade) bars so the series is contiguous on the grid.

    Empty bars are not emitted by aggregation, so on illiquid symbols a "20-bar"
    window could silently span far more than 20 bars of wall-clock time. Filling
    the gaps makes a trailing N-bar window a fixed N*timeframe horizon.

    Inserted bars carry zero flow (volume/buy/sell/delta/num_trades = 0) and a
    flat price carried forward from the previous close, which is the honest
    representation of "nothing traded". Leading gaps (before the first trade)
    cannot be filled and are left absent.
    """
    tf = TIMEFRAME_MS[timeframe]
    if bars.empty:
        return bars
    full = pd.RangeIndex(
        int(bars["open_time"].iloc[0]), int(bars["open_time"].iloc[-1]) + tf, tf
    )
    out = bars.set_index("open_time").reindex(full)
    out.index.name = "open_time"

    filled = out["close"].isna()
    for col in ("volume", "buy_vol", "sell_vol", "delta", "num_trades"):
        out[col] = out[col].fillna(0.0)
    # flat synthetic candle at the last known close
    prev_close = out["close"].ffill()
    for col in ("open", "high", "low", "close", "vwap"):
        out[col] = out[col].where(~filled, prev_close)
    out = out.reset_index()
    out["close_time"] = out["open_time"] + tf
    return out[BAR_COLUMNS]


def _assert_contiguous(bars: pd.DataFrame, timeframe: str) -> None:
    tf = TIMEFRAME_MS[timeframe]
    steps = bars["open_time"].diff().dropna()
    if len(steps) and (steps != tf).any():
        n_gaps = int((steps != tf).sum())
        raise ValueError(
            f"bar series has {n_gaps} gap(s) on the {timeframe} grid, so a trailing "
            f"N-bar window would not be a fixed N*{timeframe} horizon. Call "
            f"reindex_to_grid(bars, {timeframe!r}) first if no-trade bars should "
            f"count as zero flow."
        )


def rolling_cvd(
    bars: pd.DataFrame,
    timeframe: str,
    window: int = DEFAULT_CVD_WINDOW,
    *,
    out_col: str = "cvd",
) -> pd.DataFrame:
    """Rolling-window CVD: sum of per-bar ``delta`` over the trailing ``window`` bars.

    This is the anchoring mechanism chosen for v1 (see HYPOTHESIS.md, CVD
    reconstruction spec): the window slides forward so old flow drops out,
    keeping CVD bounded and comparable across time without an arbitrary session
    boundary.

    No look-ahead: the window is strictly trailing (current bar + N-1 prior).
    The first ``window - 1`` bars are NaN rather than partial sums — a partial
    window is a different statistic and must not be silently compared against
    full ones. Downstream layers skip NaN rows.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    _assert_contiguous(bars, timeframe)
    out = bars.copy()
    out[out_col] = (
        out["delta"].rolling(window=window, min_periods=window).sum()
    )
    return out


def _month_parquet(symbol: str, year: int, month: int, data_root: Path) -> Path:
    return (
        data_root / "raw" / symbol / "aggTrades"
        / f"{symbol}-aggTrades-{year:04d}-{month:02d}.parquet"
    )


def bars_for_month(
    symbol: str,
    year: int,
    month: int,
    timeframe: str,
    data_root: Path,
    *,
    batch_rows: int = _BATCH_ROWS,
) -> pd.DataFrame:
    """Build bars for one raw month, streaming in batches.

    A single month can be 85M+ trades; materialising one as a DataFrame costs
    several GB. Instead we stream row batches, reduce each to partial bars
    (a few thousand rows), and merge — so peak memory tracks the batch size,
    not the month size. Batches are chronological, which ``_finalize`` relies on
    for correct open/close on bars that straddle a batch boundary.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unknown timeframe {timeframe!r}; known: {list(TIMEFRAME_MS)}")
    tf = TIMEFRAME_MS[timeframe]
    path = _month_parquet(symbol, year, month, data_root)
    if not path.exists():
        raise FileNotFoundError(path)

    pf = pq.ParquetFile(path)
    partials: list[pd.DataFrame] = []
    n_trades = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=_NEEDED_COLS):
        chunk = batch.to_pandas()
        n_trades += len(chunk)
        partials.append(_partial_bars(chunk, tf))
        del chunk, batch

    bars = _finalize(partials, tf)
    log.info("cvd.bars_month", symbol=symbol, month=f"{year:04d}-{month:02d}",
             timeframe=timeframe, bars=len(bars), trades=n_trades)
    return bars


def bars_cache_path(
    symbol: str, year: int, month: int, timeframe: str, data_root: Path
) -> Path:
    """Where a month of derived bars is cached.

    Bars are DERIVED data (reproducible from data/raw/), so they live under
    data/interim/ and are safe to delete. Only bars are cached, not CVD: bars
    are expensive (aggregated from ~869M trades) while rolling CVD is a trivial
    rolling sum over ~50k rows, and caching it would bake in a parameter (N).
    """
    return (
        data_root / "interim" / symbol / "bars" / timeframe
        / f"{symbol}-bars-{timeframe}-{year:04d}-{month:02d}.parquet"
    )


def load_bars(
    symbol: str,
    months: list[tuple[int, int]],
    timeframe: str,
    data_root: Path,
) -> pd.DataFrame:
    """Load cached bars for the given months and concatenate.

    Raises if a month has not been built yet (rather than silently returning a
    short series, which would quietly change any statistic computed on it).
    """
    parts = []
    for (y, m) in months:
        p = bars_cache_path(symbol, y, m, timeframe, data_root)
        if not p.exists():
            raise FileNotFoundError(
                f"no cached bars for {symbol} {y:04d}-{m:02d} {timeframe}: {p}. "
                f"Run scripts/build_bars.py first."
            )
        parts.append(pd.read_parquet(p))
    out = pd.concat(parts, ignore_index=True).sort_values("open_time")
    out = out.reset_index(drop=True)
    if out["open_time"].duplicated().any():
        raise AssertionError("duplicate bar open_time across months")
    return out


def build_bars(
    symbol: str,
    months: list[tuple[int, int]],
    timeframe: str,
    data_root: Path,
) -> pd.DataFrame:
    """Build a continuous bar series over the given months (memory-safe).

    Month boundaries fall on the bar grid, so monthly bar sets are disjoint and
    concatenate without overlap. Result is sorted by ``open_time``.
    """
    parts = [
        bars_for_month(symbol, y, m, timeframe, data_root) for (y, m) in months
    ]
    out = pd.concat(parts, ignore_index=True).sort_values("open_time")
    out = out.reset_index(drop=True)
    if out["open_time"].duplicated().any():  # would signal a straddling-bar bug
        raise AssertionError("duplicate bar open_time across months — grid misaligned")
    return out
