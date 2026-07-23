#!/usr/bin/env python3
"""Measure realised perp-quarterly basis carry. An accounting exercise, not a screen.

THE IDENTITY
------------
Hold to expiry: short the quarterly at F, long the perp at P. Both converge to
the index S at delivery.

    P&L = (F - S)          short quarterly
        + (S - P)          long perp
        - sum(funding)     the long perp leg pays funding each settlement
        - fees
        = (F - P) - sum(funding) - fees
        =  entry basis  -  funding carry  -  costs

The price path cancels. This is delta-neutral and contractual, not predictive:
a delivery future MUST converge at expiry. So the question "is this profitable"
is decidable from data on disk, with no forecast and no statistical inference.

(Backwardation simply flips the legs; the identity is symmetric.)

COSTS
-----
Modelled as 4 taker fills (open both legs, close the perp, settle the quarterly)
= 4 x 5 bps = 20 bps, deliberately pessimistic. Quarterly liquidity is far
thinner than the perp, so the one-tick spread measured on the perp does NOT
transfer to that leg -- an extra allowance is included and flagged.

    python scripts/measure_basis.py --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.config import get_settings  # noqa: E402

TAKER_BPS = 5.0
N_FILLS = 4
QUARTERLY_SPREAD_ALLOWANCE_BPS = 5.0   # thin book on the delivery leg
COST_BPS = N_FILLS * TAKER_BPS + QUARTERLY_SPREAD_ALLOWANCE_BPS  # 25 bps


def _load_klines(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def _expiry_ms(contract: str) -> int:
    yymmdd = contract.rsplit("_", 1)[1]
    d = dt.datetime.strptime(yymmdd, "%y%m%d").replace(
        hour=8, tzinfo=dt.timezone.utc)  # Binance delivers 08:00 UTC
    return int(d.timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--entry-days-before", type=int, default=60,
                    help="evaluate an entry this many days before expiry")
    args = ap.parse_args()

    root = get_settings().data_root / "screening"
    sym = args.symbol.upper()

    perp = _load_klines(root / sym / "klines")
    if perp.empty:
        print("no perp klines"); return 1
    # Funding must span EVERY contract window. The screening pull covers
    # 2020-2024 only; 2025-2026 lives in data/raw/ from the original REST
    # download. Loading just one source silently scores the funding cost as
    # ZERO for uncovered contracts -- which flattered exactly the most recent,
    # most profitable-looking contracts before this was caught.
    fparts = [pd.read_parquet(f) for f in sorted((root / sym / "funding").glob("*.parquet"))]
    raw_f = get_settings().data_root / "raw" / sym / "funding" / f"{sym}-fundingRate.parquet"
    if raw_f.exists():
        fparts.append(pd.read_parquet(raw_f)[["symbol", "funding_time", "funding_rate", "mark_price"]])
    fund = (pd.concat(fparts, ignore_index=True)
            .drop_duplicates("funding_time")
            .sort_values("funding_time").reset_index(drop=True))
    f_lo, f_hi = int(fund.funding_time.min()), int(fund.funding_time.max())
    print(f"funding coverage: {pd.to_datetime(f_lo,unit='ms',utc=True).date()} .. "
          f"{pd.to_datetime(f_hi,unit='ms',utc=True).date()}")

    contracts = sorted(p.name for p in root.glob(f"{sym}_*") if p.is_dir())
    print(f"\n=== perp-quarterly basis carry: {sym} ===")
    print(f"perp klines {len(perp):,} bars | funding {len(fund):,} settlements "
          f"| {len(contracts)} quarterly contracts")
    print(f"cost model: {N_FILLS} taker fills x {TAKER_BPS}bps + "
          f"{QUARTERLY_SPREAD_ALLOWANCE_BPS}bps quarterly-book allowance = {COST_BPS} bps\n")

    perp_t = perp.open_time.to_numpy()
    perp_c = perp.close.to_numpy(dtype=float)
    ft = fund.funding_time.to_numpy()
    fr = fund.funding_rate.to_numpy(dtype=float)

    rows = []
    skipped: list[tuple[str, str]] = []
    for c in contracts:
        q = _load_klines(root / c / "klines")
        if q.empty:
            continue
        exp = _expiry_ms(c)
        entry_t = exp - args.entry_days_before * 86_400_000
        # nearest available bar at/after the intended entry, on BOTH legs
        qi = np.searchsorted(q.open_time.to_numpy(), entry_t, side="left")
        pi = np.searchsorted(perp_t, entry_t, side="left")
        if qi >= len(q) or pi >= len(perp_t):
            continue
        if q.open_time.iloc[qi] > exp or abs(int(q.open_time.iloc[qi]) - int(perp_t[pi])) > 3_600_000:
            continue
        F = float(q.close.iloc[qi]); P = float(perp_c[pi])
        t0 = int(q.open_time.iloc[qi])
        days = (exp - t0) / 86_400_000
        if days <= 1:
            continue
        basis_bps = (F / P - 1.0) * 1e4
        # funding paid by the long perp leg between entry and expiry
        # Refuse to score a window the funding data does not fully cover:
        # summing an empty slice yields 0.0, i.e. a free trade.
        if t0 < f_lo or exp > f_hi:
            skipped.append((c, "funding data does not cover the holding window"))
            continue
        m = (ft >= t0) & (ft <= exp)
        n_settle = int(m.sum())
        expected = days * 3.0          # 8h cadence
        if n_settle < 0.8 * expected:
            skipped.append((c, f"only {n_settle} settlements, expected ~{expected:.0f}"))
            continue
        carry_bps = float(np.nansum(fr[m])) * 1e4
        net_bps = basis_bps - carry_bps - COST_BPS
        ann = net_bps / days * 365
        rows.append({
            "contract": c, "entry": pd.to_datetime(t0, unit="ms", utc=True).date(),
            "days": round(days, 1), "basis_bps": round(basis_bps, 1),
            "carry_bps": round(carry_bps, 1), "settles": n_settle,
            "net_bps": round(net_bps, 1), "net_ann_pct": round(ann / 100, 2),
        })

    if skipped:
        print(f"EXCLUDED {len(skipped)} contract(s) - not scored rather than scored wrongly:")
        for c, why in skipped:
            print(f"  {c}: {why}")
        print()
    if not rows:
        print("no contract had both legs available at that entry offset")
        return 0
    df = pd.DataFrame(rows)
    print(f"Entry {args.entry_days_before}d before expiry, held to delivery:\n")
    print(f"{'contract':16} {'entry':>11} {'days':>6} {'basis':>8} {'funding':>9} "
          f"{'net':>8} {'net ann':>8}")
    for _, r in df.iterrows():
        print(f"{r.contract:16} {str(r.entry):>11} {r.days:>6.0f} {r.basis_bps:>8.1f} "
              f"{r.carry_bps:>9.1f} {r.net_bps:>8.1f} {r.net_ann_pct:>7.1f}%")

    print(f"\n--- summary over {len(df)} contracts ---")
    print(f"  median entry basis   : {df.basis_bps.median():>8.1f} bps")
    print(f"  median funding carry : {df.carry_bps.median():>8.1f} bps  (paid by long perp leg)")
    print(f"  median NET           : {df.net_bps.median():>8.1f} bps after {COST_BPS} bps costs")
    print(f"  median net annualised: {df.net_ann_pct.median():>8.2f}%")
    print(f"  profitable contracts : {int((df.net_bps>0).sum())}/{len(df)}")
    print(f"  worst / best net     : {df.net_bps.min():.1f} / {df.net_bps.max():.1f} bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
