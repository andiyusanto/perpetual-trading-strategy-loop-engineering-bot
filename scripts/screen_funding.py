#!/usr/bin/env python3
"""Screen funding-rate extremes as a PRIMARY signal, on the long-history pool.

Pre-registered signal definition (forced-flow thesis):
  - funding in the TOP decile of its own trailing 90d distribution
    => positioning crowded LONG, longs paying to hold  => expect reversal DOWN
  - funding in the BOTTOM decile  => crowded SHORT     => expect reversal UP
  - signal timestamp = the funding settlement time (knowable at that instant)

Runs on data/screening/ (2020-2024), NOT on the 2025-2026 segments — so
research, validation and holdout all stay untouched for anything that survives.

Reported per-pair AND pooled. Pooled n overstates independence because funding
extremes tend to coincide across correlated pairs, so the per-pair rows are the
honest evidence and the pooled row is an upper bound on precision.

    python scripts/screen_funding.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.research.screen import (  # noqa: E402
    SignalSet,
    format_report,
    screen_multi,
    screen_signal,
    screens_run,
)
from src.strategy.funding_filter import (  # noqa: E402
    FundingGateParams,
    evaluate_funding_gate,
)

SWING_HORIZONS = (240, 480, 1440, 2880, 4320, 10080)  # 4h 8h 1d 2d 3d 7d


def _load(root: Path, symbol: str, kind: str) -> pd.DataFrame:
    d = root / "screening" / symbol / kind
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} for {symbol} in {d}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    key = "open_time" if kind == "klines" else "funding_time"
    return df.sort_values(key).drop_duplicates(key).reset_index(drop=True)


def build_signal(funding: pd.DataFrame, params: FundingGateParams, label: str) -> SignalSet:
    st = evaluate_funding_gate(funding, funding["funding_time"].to_numpy(), params)
    hi, lo = st["is_high_extreme"].to_numpy(), st["is_low_extreme"].to_numpy()
    t = np.concatenate([st["decision_time"].to_numpy()[hi],
                        st["decision_time"].to_numpy()[lo]])
    d = np.concatenate([-np.ones(int(hi.sum()), int), np.ones(int(lo.sum()), int)])
    order = np.argsort(t)
    return SignalSet(
        name=f"funding_extreme_decile_reversal[{label}]",
        times_ms=t[order], direction=d[order],
        params={"lookback_days": params.lookback_days,
                "extreme_pct": params.extreme_pct, "pool": label,
                "thesis": "forced-flow reversal", "history": "2020-2024 screening pool"},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--extreme-pct", type=float, default=0.10)
    ap.add_argument("--lookback-days", type=int, default=90)
    args = ap.parse_args()

    root = get_settings().data_root
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    gp = FundingGateParams(lookback_days=args.lookback_days,
                           extreme_pct=args.extreme_pct)

    pairs = []
    print(f"\n{'='*90}\nFUNDING-EXTREME SCREEN — long-history pool (segments untouched)\n{'='*90}")

    for sym in symbols:
        bars = _load(root, sym, "klines")
        fund = _load(root, sym, "funding")
        sig = build_signal(fund, gp, sym)
        span = (pd.to_datetime(bars.open_time.iloc[0], unit="ms", utc=True).date(),
                pd.to_datetime(bars.open_time.iloc[-1], unit="ms", utc=True).date())
        print(f"\n--- {sym}  klines {len(bars):,} bars {span[0]}..{span[1]} | "
              f"funding {len(fund):,} settlements | extremes {len(sig)} ---")
        print(format_report(screen_signal(bars, sig, horizons_min=SWING_HORIZONS,
                                          segment="screening_pool_2020_2024")))
        pairs.append((bars, sig))

    print(f"\n{'='*90}\nPOOLED ACROSS PAIRS (forward returns computed per-pair, then concatenated)\n{'='*90}")
    print("Concatenated in signal-time order, so the block bootstrap treats")
    print("near-simultaneous cross-pair events as ONE cluster — funding extremes")
    print("fire together on correlated pairs, so naive pooled n would overstate")
    print("precision.\n")
    rep = screen_multi(pairs, name="funding_extreme_decile_reversal[POOLED]",
                       horizons_min=SWING_HORIZONS,
                       segment="screening_pool_2020_2024")
    print(format_report(rep))

    print(f"\nscreens ever run: {screens_run()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
