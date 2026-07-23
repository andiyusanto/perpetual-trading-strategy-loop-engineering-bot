#!/usr/bin/env python3
"""Download long-history funding + klines for SCREENING (not for backtesting).

Screening needs only close prices, so klines (tens of KB per month) replace
aggTrades (hundreds of MB per month). That makes years of history cheap. Full
aggTrades are only required once a signal needs order flow, e.g. CVD.

Storage (kept separate from data/raw/, which holds the segmented backtest data):
    data/screening/<SYMBOL>/klines/<SYMBOL>-<interval>-YYYY-MM.parquet
    data/screening/<SYMBOL>/funding/<SYMBOL>-fundingRate-YYYY-MM.parquet

Why a separate pool: the 2025-01..2026-06 window is already split into
research/validation/holdout. Screening on OLDER data leaves all three of those
segments untouched, so anything that survives screening can still be tested
against genuinely unseen data.

    python scripts/download_history.py --symbols BTCUSDT,ETHUSDT,SOLUSDT \
        --start 2020-01 --end 2024-12
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.ingest.archive import (  # noqa: E402
    download_funding_month,
    download_klines_month,
    month_range,
)


def _parse_month(s: str) -> tuple[int, int]:
    d = datetime.strptime(s, "%Y-%m")
    return d.year, d.month


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else get_settings().data_root
    tmp = root / "raw" / ".tmp"
    sy, sm = _parse_month(args.start)
    ey, em = _parse_month(args.end)
    months = month_range(sy, sm, ey, em)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"\n=== screening history: {symbols} {args.start}..{args.end} "
          f"({args.interval} klines + funding) ===")

    for sym in symbols:
        kdir = root / "screening" / sym / "klines"
        fdir = root / "screening" / sym / "funding"
        got_k = got_f = miss = 0
        for (y, m) in months:
            for kind, fn, d in (
                ("klines", lambda: download_klines_month(sym, y, m, args.interval, kdir, tmp), kdir),
                ("funding", lambda: download_funding_month(sym, y, m, fdir, tmp), fdir),
            ):
                try:
                    fn()
                    if kind == "klines":
                        got_k += 1
                    else:
                        got_f += 1
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        miss += 1  # month predates listing - expected
                    else:
                        print(f"  {sym} {y}-{m:02d} {kind}: {e}")
                except Exception as e:  # noqa: BLE001
                    print(f"  {sym} {y}-{m:02d} {kind}: {type(e).__name__}: {e}")
        print(f"  {sym}: klines {got_k}/{len(months)}, funding {got_f}/{len(months)}"
              f"{f', {miss} month(s) before listing' if miss else ''}")

    print("\ndone\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
