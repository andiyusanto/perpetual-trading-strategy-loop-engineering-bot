#!/usr/bin/env python3
"""Build and cache OHLCV+delta bars from raw aggTrades.

Bars are DERIVED data (reproducible from data/raw/) and are cached per month so
the build is resumable: an interrupted run picks up where it stopped.

    data/interim/<SYMBOL>/bars/<tf>/<SYMBOL>-bars-<tf>-YYYY-MM.parquet

Only bars are cached, not CVD — rolling CVD is a cheap rolling sum and caching
it would bake the window parameter N into stored data.

Examples:
    python scripts/build_bars.py --symbol BTCUSDT --start 2025-01 --end 2026-06
    python scripts/build_bars.py --symbol BTCUSDT --start 2025-01 --end 2026-06 \
        --timeframes 15m,5m
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import structlog  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.core.cvd import (  # noqa: E402
    TIMEFRAME_MS,
    bars_cache_path,
    bars_for_month,
)
from src.ingest.archive import month_range  # noqa: E402

log = structlog.get_logger("build_bars")


def _parse_month(s: str) -> tuple[int, int]:
    dt = datetime.strptime(s, "%Y-%m")
    return dt.year, dt.month


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="first month YYYY-MM")
    ap.add_argument("--end", required=True, help="last month YYYY-MM")
    ap.add_argument("--timeframes", default="15m,5m",
                    help="comma-separated (default: 15m,5m)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else settings.data_root
    symbol = args.symbol.upper()
    sy, sm = _parse_month(args.start)
    ey, em = _parse_month(args.end)
    months = month_range(sy, sm, ey, em)
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    for tf in tfs:
        if tf not in TIMEFRAME_MS:
            raise SystemExit(f"unknown timeframe {tf!r}; known: {list(TIMEFRAME_MS)}")

    print(f"\n=== build bars: {symbol} {args.start}..{args.end} tf={tfs} ===")

    failures: list[str] = []
    for tf in tfs:
        print(f"\n--- {tf} ---")
        total = 0
        for (y, m) in months:
            dest = bars_cache_path(symbol, y, m, tf, data_root)
            if dest.exists() and not args.overwrite:
                import pandas as pd
                n = len(pd.read_parquet(dest, columns=["open_time"]))
                total += n
                print(f"  {y:04d}-{m:02d}  bars={n:>7,}  [skip(exists)]")
                continue
            try:
                bars = bars_for_month(symbol, y, m, tf, data_root)
            except FileNotFoundError:
                failures.append(f"{tf} {y:04d}-{m:02d} (no raw aggTrades)")
                print(f"  {y:04d}-{m:02d}  [SKIP - raw month not downloaded]")
                continue
            except Exception as exc:  # noqa: BLE001 - keep going, re-run retries
                failures.append(f"{tf} {y:04d}-{m:02d} ({exc})")
                print(f"  {y:04d}-{m:02d}  [FAILED] {exc}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".parquet.tmp")
            bars.to_parquet(tmp, compression="zstd", index=False)
            tmp.replace(dest)
            total += len(bars)
            print(f"  {y:04d}-{m:02d}  bars={len(bars):>7,}  "
                  f"{_iso(int(bars.open_time.iloc[0]))} .. "
                  f"{_iso(int(bars.open_time.iloc[-1]))}  [ok]")
        print(f"  {tf} total bars: {total:,}")

    if failures:
        print(f"\n[warn] {len(failures)} month(s) not built: {failures}")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
