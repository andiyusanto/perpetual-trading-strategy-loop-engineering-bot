"""Binance USD-M Futures adapter implementing the ``Exchange`` protocol via CCXT.

This is the bridge between the tested risk core and a real venue. The risk loop
was validated against a simulator; this adapter is where the reduce-only,
client-order-id, and reconcile contracts meet Binance, so those contracts are
enforced here explicitly rather than assumed.

Safety choices, stated because they are the whole point:

- ``fetch_account_state`` reads maintenance margin from the venue and RAISES if
  it cannot be read. A risk system that silently guesses its margin is worse
  than one that stops. Fail loud.
- ``close_position`` reads the CURRENT position from the venue and closes
  exactly that, reduce-only. It never trusts a passed-in size. Reduce-only is
  set on every close so the order can only shrink/flatten, never open or flip —
  which also makes closing twice equal closing once (effect-level idempotency),
  the backstop behind the client-order-id.
- ``client_order_id`` is passed to Binance as ``newClientOrderId`` so a retry
  after a timeout does not double up. A duplicate-id rejection is treated as
  success: it means the order already reached the venue.

Auth: keys are read from the environment / .env by the caller and passed in.
Never hard-code or log them. Testnet is selected with ``testnet=True``.

NOTE: the authenticated paths (balance, positions, orders) require API keys and
have been verified only against MOCKED CCXT responses here plus the public
testnet transport. The real authenticated round-trip is scripts/testnet_smoke.py,
run by a human with testnet keys, before this touches real money.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.ingest.binance_dns import enable_binance_doh

from .interface import AccountState, OrderResult, Position

log = structlog.get_logger(__name__)

# Binance rejects a reused clientOrderId with one of these; that means the order
# already landed, so a retry should treat it as success, not failure.
_DUPLICATE_MARKERS = ("duplicate", "-4015", "clientorderid")


class BinanceUMExchange:
    """USD-M futures adapter. Construct with keys, or inject ``client`` in tests."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        *,
        testnet: bool = True,
        client: Any = None,
    ) -> None:
        self.testnet = testnet
        if client is not None:
            self._ex = client
        else:
            enable_binance_doh()  # mainnet fapi is DNS-hijacked on some networks
            import ccxt.async_support as ccxta

            self._ex = ccxta.binanceusdm({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            if testnet:
                self._ex.set_sandbox_mode(True)
        self._id_to_unified: dict[str, str] = {}
        self._unified_to_id: dict[str, str] = {}

    # -- symbol mapping (BTCUSDT <-> BTC/USDT:USDT) -------------------------

    async def _ensure_markets(self) -> None:
        if self._id_to_unified:
            return
        markets = await self._ex.load_markets()
        for unified, m in markets.items():
            if m.get("id"):
                self._id_to_unified[m["id"]] = unified
                self._unified_to_id[unified] = m["id"]

    async def _unified(self, symbol: str) -> str:
        await self._ensure_markets()
        return self._id_to_unified.get(symbol, symbol)

    def _to_id(self, unified: str, raw: dict | None = None) -> str:
        if unified in self._unified_to_id:
            return self._unified_to_id[unified]
        if raw and raw.get("symbol"):
            return raw["symbol"]
        return unified

    # -- Exchange protocol --------------------------------------------------

    async def fetch_account_state(self) -> AccountState:
        await self._ensure_markets()
        bal = await self._ex.fetch_balance()
        info = bal.get("info", {}) or {}
        try:
            equity = float(info["totalMarginBalance"])
            wallet = float(info["totalWalletBalance"])
            maint = float(info["totalMaintMargin"])
        except (KeyError, TypeError, ValueError) as exc:
            # A risk system must NOT proceed on a guessed margin figure.
            raise RuntimeError(
                "could not read margin fields from fetch_balance().info "
                f"(keys present: {sorted(info)[:12]}...): {exc}"
            ) from exc

        raw_positions = await self._ex.fetch_positions()
        positions = tuple(
            p for p in (self._to_position(rp) for rp in raw_positions)
            if p is not None and not p.is_flat
        )
        return AccountState(
            equity=equity, wallet_balance=wallet,
            maintenance_margin=maint, positions=positions,
        )

    def _to_position(self, rp: dict) -> Position | None:
        raw = rp.get("info", {}) or {}
        # positionAmt is signed (+ long / - short) and authoritative.
        amt_str = raw.get("positionAmt")
        if amt_str is not None:
            qty = float(amt_str)
        else:
            contracts = float(rp.get("contracts") or 0.0)
            qty = contracts if rp.get("side") == "long" else -contracts
        sym = self._to_id(rp.get("symbol", ""), raw)
        entry = float(rp.get("entryPrice") or raw.get("entryPrice") or 0.0)
        mark = float(rp.get("markPrice") or raw.get("markPrice") or 0.0)
        return Position(symbol=sym, qty=qty, entry_price=entry, mark_price=mark)

    async def close_position(
        self, symbol: str, *, client_order_id: str
    ) -> OrderResult:
        unified = await self._unified(symbol)
        # Read the CURRENT position; never trust a cached size.
        poss = await self._ex.fetch_positions([unified])
        qty = 0.0
        for rp in poss:
            p = self._to_position(rp)
            if p and p.symbol == symbol:
                qty = p.qty
                break
        if qty == 0.0:
            return OrderResult(ok=True, symbol=symbol, filled_qty=0.0,
                               client_order_id=client_order_id)

        side = "sell" if qty > 0 else "buy"   # opposite of the position
        amount = abs(qty)
        try:
            order = await self._ex.create_order(
                unified, "market", side, amount,
                params={"reduceOnly": True, "newClientOrderId": client_order_id},
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(mk in msg for mk in _DUPLICATE_MARKERS):
                # The id already landed -> the close is in flight; treat as ok.
                log.info("binance.close_duplicate_ok", symbol=symbol,
                         client_order_id=client_order_id)
                return OrderResult(ok=True, symbol=symbol, filled_qty=0.0,
                                   client_order_id=client_order_id)
            log.error("binance.close_failed", symbol=symbol, err=str(exc))
            return OrderResult(ok=False, symbol=symbol, error=str(exc))

        filled = float(order.get("filled") or amount)
        signed = -filled if qty > 0 else filled
        return OrderResult(ok=True, symbol=symbol, filled_qty=signed,
                           client_order_id=order.get("clientOrderId", client_order_id))

    async def cancel_all_orders(self, symbol: str | None = None) -> None:
        if symbol is not None:
            await self._ex.cancel_all_orders(await self._unified(symbol))
            return
        # Binance requires a symbol per cancel; sweep symbols with open orders.
        open_orders = await self._ex.fetch_open_orders()
        for unified in {o["symbol"] for o in open_orders}:
            await self._ex.cancel_all_orders(unified)

    async def close(self) -> None:
        if hasattr(self._ex, "close"):
            await self._ex.close()
