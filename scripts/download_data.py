#!/usr/bin/env python3
"""Download raw aggTrades (archive) + funding rate (REST via DoH) to data/raw/.

This is the research-phase data acquisition entrypoint. It only WRITES to
data/raw/ (the immutable source layer). Segmentation into research/validation/
holdout happens later in scripts/segment_data.py -- this script deliberately
knows nothing about those cutoffs.

Storage layout:
    data/raw/<SYMBOL>/aggTrades/<SYMBOL>-aggTrades-YYYY-MM.parquet
    data/raw/<SYMBOL>/funding/<SYMBOL>-fundingRate.parquet   (whole requested span)

Examples:
    # 18 months of BTC (aggTrades + funding)
    python scripts/download_data.py --symbol BTCUSDT --start 2025-01 --end 2026-06

    # just one month, to smoke-test the pipeline
    python scripts/download_data.py --symbol BTCUSDT --start 2026-06 --end 2026-06
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a plain script (no editable install needed).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import structlog  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.ingest import archive, funding  # noqa: E402

log = structlog.get_logger("download")


def _parse_month(s: str) -> tuple[int, int]:
    dt = datetime.strptime(s, "%Y-%m")
    return dt.year, dt.month


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _month_start_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _month_end_ms(year: int, month: int) -> int:
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return int(datetime(ny, nm, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--start", required=True, help="first month, YYYY-MM (inclusive)")
    ap.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    ap.add_argument("--data-root", default=None, help="override data/ root")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-download months that already have a parquet")
    ap.add_argument("--skip-aggtrades", action="store_true")
    ap.add_argument("--skip-funding", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else settings.data_root
    symbol = args.symbol.upper()
    sy, sm = _parse_month(args.start)
    ey, em = _parse_month(args.end)

    raw_symbol_dir = data_root / "raw" / symbol
    agg_dir = raw_symbol_dir / "aggTrades"
    fund_dir = raw_symbol_dir / "funding"
    tmp_dir = data_root / "raw" / ".tmp"

    print(f"\n=== download: {symbol}  {args.start} .. {args.end} ===")
    print(f"data root: {data_root}")

    agg_results: list[archive.MonthResult] = []
    if not args.skip_aggtrades:
        available = set(archive.list_available_months(symbol))
        requested = archive.month_range(sy, sm, ey, em)
        missing = [ym for ym in requested if ym not in available]
        if missing:
            print(f"\n[warn] {len(missing)} requested month(s) not in archive "
                  f"(skipped): {['%04d-%02d' % ym for ym in missing]}")
        to_get = [ym for ym in requested if ym in available]

        print(f"\naggTrades: {len(to_get)} month(s) to process\n")
        failed_months: list[tuple[int, int]] = []
        for (y, m) in to_get:
            try:
                res = archive.download_aggtrades_month(
                    symbol, y, m, agg_dir, tmp_dir, overwrite=args.overwrite
                )
            except Exception as exc:  # noqa: BLE001 - keep going; re-run picks it up
                failed_months.append((y, m))
                log.error("aggtrades.month_failed", month=f"{y:04d}-{m:02d}", err=str(exc))
                print(f"  {y:04d}-{m:02d}  [FAILED] {exc}")
                continue
            agg_results.append(res)
            tag = "skip(exists)" if res.skipped else "ok"
            print(f"  {y:04d}-{m:02d}  rows={res.rows:>12,}  "
                  f"{_ms_to_iso(res.ts_min_ms)} .. {_ms_to_iso(res.ts_max_ms)}  [{tag}]")
        if failed_months:
            print(f"\n[warn] {len(failed_months)} month(s) FAILED after retries "
                  f"(re-run to retry): {['%04d-%02d' % ym for ym in failed_months]}")

    fund_rows = 0
    fund_min = fund_max = None
    if not args.skip_funding:
        start_ms = _month_start_ms(sy, sm)
        end_ms = _month_end_ms(ey, em)
        print(f"\nfunding: {symbol} {_ms_to_iso(start_ms)} .. {_ms_to_iso(end_ms)}")
        fdf = funding.fetch_funding(symbol, start_ms, end_ms)
        fund_path = fund_dir / f"{symbol}-fundingRate.parquet"
        fund_rows = funding.write_funding_parquet(fdf, fund_path)
        if fund_rows:
            fund_min = int(fdf["funding_time"].min())
            fund_max = int(fdf["funding_time"].max())
        print(f"  funding events={fund_rows:,}  -> {fund_path}")

    # ---- summary ----
    print("\n=== SUMMARY ===")
    if agg_results:
        total = sum(r.rows for r in agg_results)
        tmin = min(r.ts_min_ms for r in agg_results)
        tmax = max(r.ts_max_ms for r in agg_results)
        print(f"aggTrades months : {len(agg_results)}")
        print(f"aggTrades rows   : {total:,}")
        print(f"aggTrades range  : {_ms_to_iso(tmin)} .. {_ms_to_iso(tmax)}")
    if not args.skip_funding:
        rng = (f"{_ms_to_iso(fund_min)} .. {_ms_to_iso(fund_max)}"
               if fund_min is not None else "(none)")
        print(f"funding events   : {fund_rows:,}")
        print(f"funding range    : {rng}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
