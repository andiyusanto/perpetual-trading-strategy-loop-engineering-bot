#!/usr/bin/env python3
"""Run the walk-forward backtest on a data segment.

Lives in scripts/ rather than src/ on purpose: src/ is frozen by the
hypothesis-v1 tag (the holdout gate refuses to open if src/ changed), so the
runner must not live there. All strategy logic is in src/ and untouched here.

Component attribution (methodology rule 11 / KILL_CRITERIA.md) is the default:
each layer is evaluated on its own before the combined signal is judged, so we
can tell which component — if any — is contributing.

    python scripts/run_backtest.py --symbol BTCUSDT --segment research
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from scripts import segment_data as sd  # noqa: E402
from src.backtest.costs import CostModel  # noqa: E402
from src.backtest.engine import (  # noqa: E402
    LAYER_FUNDING,
    LAYER_ROC,
    LAYER_SPEARMAN,
    LAYER_SWING,
    StrategyParams,
)
from src.backtest.metrics import format_summary, summarize  # noqa: E402
from src.backtest.walkforward import make_folds, run_walkforward  # noqa: E402
from src.core.cvd import reindex_to_grid  # noqa: E402
from src.core.config import get_settings  # noqa: E402

# The 5 configurations KILL_CRITERIA.md requires, and no more. Each extra
# combination spends the multiple-testing budget (max 8 in v1).
CONFIGS: dict[str, tuple[str, ...]] = {
    "1_cvd_only":        (LAYER_SWING,),
    "2_cvd_spearman":    (LAYER_SWING, LAYER_SPEARMAN),
    "3_cvd_roc":         (LAYER_SWING, LAYER_ROC),
    "4_cvd_funding":     (LAYER_SWING, LAYER_FUNDING),
    "5_full_combined":   (LAYER_SWING, LAYER_SPEARMAN, LAYER_ROC, LAYER_FUNDING),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--segment", default=sd.RESEARCH,
                    choices=[sd.RESEARCH, sd.VALIDATION, sd.HOLDOUT])
    ap.add_argument("--train-months", type=int, default=2)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    st = get_settings()
    costs = CostModel(
        taker_fee_bps=st.taker_fee_bps,
        maker_fee_bps=st.maker_fee_bps,
        slippage_bps=st.slippage_bps,
    )
    sym, seg = args.symbol.upper(), args.segment

    print(f"\n=== walk-forward backtest: {sym} / {seg} ===")
    print(f"costs: taker {costs.taker_fee_bps}bps, maker {costs.maker_fee_bps}bps, "
          f"slippage {costs.slippage_bps}bps -> round trip ~{costs.round_trip_bps:.1f}bps")

    # Segment loaders enforce isolation; validation/holdout also log the open.
    loader = {sd.RESEARCH: sd.load_research,
              sd.VALIDATION: sd.load_validation,
              sd.HOLDOUT: sd.load_holdout}[seg]
    bars15 = loader(sym, "15m")
    bars5 = sd.load_segment_bars(sym, seg, "5m")
    funding = sd.load_segment_funding(sym, seg)

    # 5m can contain no-trade gaps (BTC had one during an Aug-2025 outage);
    # fill them as explicit zero-flow so bar-count-based holds mean fixed time.
    before = len(bars5)
    bars5 = reindex_to_grid(bars5.drop(columns=["is_warmup"]), "5m")
    if len(bars5) != before:
        print(f"[note] filled {len(bars5)-before} empty 5m bar(s) as zero-flow")

    b = sd.SEGMENT_BOUNDS[sym][seg]
    folds = make_folds(b.start, b.end,
                       train_months=args.train_months, test_months=args.test_months)
    print(f"segment {b.start}..{b.end} | bars15={len(bars15):,} bars5={len(bars5):,} "
          f"| funding={len(funding):,} | folds={len(folds)}")
    for f in folds:
        print(f"   {f}")

    results = {}
    for name, layers in CONFIGS.items():
        params = StrategyParams().with_layers(*layers)
        trades, reports = run_walkforward(
            bars15.drop(columns=["is_warmup"]), bars5, funding, folds, params, costs
        )
        s = summarize(trades, rr_ratio=params.rr_ratio) if not trades.empty else {"n_trades": 0}
        results[name] = {"layers": list(layers), "summary": _jsonable(s),
                         "folds": [_jsonable(r) for r in reports]}
        print(f"\n--- {name}  layers={list(layers)} ---")
        print(format_summary(s) if s.get("n_trades") else "no trades")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.out}")
    print("\nNOTE: these are RESEARCH-set numbers produced in the same pass that "
          "built the engine. Per methodology rule 9 they are NOT a verdict; an "
          "adversarial review is a separate pass.")
    return 0


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            out[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
        elif isinstance(v, dict):
            out[k] = {str(kk): (int(vv) if hasattr(vv, "__int__") else vv)
                      for kk, vv in v.items()}
        elif isinstance(v, float) and pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    raise SystemExit(main())
