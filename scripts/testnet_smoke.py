#!/usr/bin/env python3
"""Authenticated Binance USD-M TESTNET smoke test — run this with testnet keys.

Verifies the parts the unit tests cannot: that the adapter talks to a real venue
correctly and that the kill switch actually flattens a real position.

This is the step that must pass before anything touches real money. It is
deliberately conservative and self-cleaning:

  1. read account state (fail-loud margin check on live data)
  2. open a MINIMUM-size position
  3. read it back and confirm the reconcile sees it
  4. fire the REAL kill switch (RiskLoop) -> reduce-only flatten
  5. confirm the account is flat again

TESTNET ONLY. Get keys at https://testnet.binancefuture.com (they are separate
from mainnet keys and control only play money). Set them in the environment:

    export BINANCE_TESTNET_KEY=...
    export BINANCE_TESTNET_SECRET=...
    python scripts/testnet_smoke.py --symbol SOLUSDT

Never put mainnet keys here. Never commit keys.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.binance_um import BinanceUMExchange  # noqa: E402
from src.risk.monitor import RiskLimits, RiskLoop  # noqa: E402


async def run(symbol: str) -> int:
    key = os.environ.get("BINANCE_TESTNET_KEY", "")
    secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
    if not key or not secret:
        print("Set BINANCE_TESTNET_KEY and BINANCE_TESTNET_SECRET (testnet only).")
        return 2

    ex = BinanceUMExchange(key, secret, testnet=True)
    try:
        print("=== 1. read account state (live testnet) ===")
        st = await ex.fetch_account_state()
        print(f"  equity {st.equity:.2f} | wallet {st.wallet_balance:.2f} | "
              f"maint {st.maintenance_margin:.4f} | margin_ratio {st.margin_ratio:.4f}")
        print(f"  open positions: {[(p.symbol, p.qty) for p in st.positions]}")

        unified = await ex._unified(symbol)
        market = ex._ex.market(unified)
        min_amt = market["limits"]["amount"]["min"]
        min_cost = market["limits"]["cost"]["min"] or 0
        ticker = await ex._ex.fetch_ticker(unified)
        px = ticker["last"]
        # smallest amount that clears BOTH the lot-size min and the notional min
        amount = max(min_amt, (min_cost / px) * 1.05 if min_cost else min_amt)
        amount = float(ex._ex.amount_to_precision(unified, amount))
        print(f"\n=== 2. open a minimum LONG on {symbol}: {amount} @ ~{px} "
              f"(~{amount*px:.2f} USDT) ===")
        await ex._ex.create_order(unified, "market", "buy", amount,
                                  params={"newClientOrderId": "smoke-open"})

        print("\n=== 3. reconcile: does the venue show the position? ===")
        st = await ex.fetch_account_state()
        held = {p.symbol: p.qty for p in st.positions}
        print(f"  positions now: {held}")
        assert held.get(symbol, 0) > 0, "opened position not visible on reconcile!"

        print("\n=== 4. fire the REAL kill switch (equity_floor above equity) ===")
        # force a trip by setting the floor above current equity
        loop = RiskLoop(exchange=ex,
                        limits=RiskLimits(equity_floor=st.equity + 1e9),
                        peak_equity=st.equity)
        d = await loop.check_once()
        print(f"  halted={loop.halted} breaches={[b.value for b in d.breaches]}")

        print("\n=== 5. confirm flat ===")
        st = await ex.fetch_account_state()
        held = {p.symbol: p.qty for p in st.positions if not p.is_flat}
        print(f"  open positions: {held}")
        if held:
            print("  !! NOT FLAT — investigate before going further")
            return 1
        print("\nSMOKE TEST PASSED: open -> reconcile -> kill-switch flatten all worked.")
        return 0
    finally:
        await ex.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="SOLUSDT",
                    help="a $5-min-notional pair keeps the test cheap")
    args = ap.parse_args()
    return asyncio.run(run(args.symbol.upper()))


if __name__ == "__main__":
    raise SystemExit(main())
