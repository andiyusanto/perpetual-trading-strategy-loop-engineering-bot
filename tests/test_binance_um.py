"""Tests for the Binance USD-M adapter against a fake CCXT client.

Covers everything that does NOT require live keys: the mapping from CCXT
structures to our types, the reduce-only + client-order-id contract on closes,
fail-loud on missing margin, and — most importantly — that the adapter plugs
into the already-tested RiskLoop and flattens a real kill path.

The authenticated round-trip against Binance testnet is scripts/testnet_smoke.py.
"""

from __future__ import annotations

import pytest

from src.execution.binance_um import BinanceUMExchange
from src.execution.interface import Exchange
from src.risk.monitor import Breach, RiskLimits, RiskLoop

MARKETS = {
    "BTC/USDT:USDT": {"id": "BTCUSDT"},
    "ETH/USDT:USDT": {"id": "ETHUSDT"},
    "SOL/USDT:USDT": {"id": "SOLUSDT"},
}


class FakeCCXT:
    """Mimics ccxt.async_support.binanceusdm at the surface the adapter uses."""

    def __init__(self, *, balance_info, positions):
        self._balance_info = balance_info
        self._positions = positions          # keyed by unified symbol
        self.created: list[dict] = []
        self.cancelled: list[str] = []
        self.open_orders: list[dict] = []
        self.raise_on_create: Exception | None = None

    async def load_markets(self):
        return MARKETS

    async def fetch_balance(self):
        return {"info": self._balance_info}

    async def fetch_positions(self, symbols=None):
        if symbols is None:
            return list(self._positions.values())
        return [self._positions[s] for s in symbols if s in self._positions]

    async def create_order(self, symbol, type, side, amount, params=None):
        if self.raise_on_create is not None:
            raise self.raise_on_create
        self.created.append({"symbol": symbol, "type": type, "side": side,
                             "amount": amount, "params": params or {}})
        # simulate the fill: position goes flat
        if symbol in self._positions:
            self._positions[symbol]["info"]["positionAmt"] = "0.0"
        return {"filled": amount, "clientOrderId": (params or {}).get("newClientOrderId")}

    async def cancel_all_orders(self, symbol):
        self.cancelled.append(symbol)

    async def fetch_open_orders(self):
        return self.open_orders

    def set_sandbox_mode(self, on):  # pragma: no cover - unused with injected client
        pass

    async def close(self):
        pass


def _pos(unified, amt, entry=100.0, mark=100.0):
    return {"symbol": unified, "entryPrice": entry, "markPrice": mark,
            "contracts": abs(amt), "side": "long" if amt > 0 else "short",
            "info": {"positionAmt": str(amt)}}


def _balance(equity, wallet, maint):
    return {"totalMarginBalance": str(equity), "totalWalletBalance": str(wallet),
            "totalMaintMargin": str(maint)}


def _adapter(balance_info, positions):
    fake = FakeCCXT(balance_info=balance_info, positions=positions)
    return BinanceUMExchange(client=fake), fake


# ------------------------------------------------------------- mapping

async def test_adapter_satisfies_exchange_protocol():
    ad, _ = _adapter(_balance(100, 100, 2), {})
    assert isinstance(ad, Exchange)


async def test_fetch_account_state_maps_margin_and_positions():
    ad, _ = _adapter(
        _balance(equity=105.0, wallet=100.0, maint=3.5),
        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.2, 100, 110),
         "SOL/USDT:USDT": _pos("SOL/USDT:USDT", -1.0, 50, 48)},
    )
    st = await ad.fetch_account_state()
    assert st.equity == 105.0 and st.wallet_balance == 100.0
    assert st.maintenance_margin == 3.5
    syms = {p.symbol: p.qty for p in st.positions}
    assert syms == {"ETHUSDT": 0.2, "SOLUSDT": -1.0}   # signed, raw ids
    assert st.margin_ratio == pytest.approx(3.5 / 105.0)


async def test_flat_positions_are_filtered_out():
    ad, _ = _adapter(
        _balance(100, 100, 1),
        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.0),
         "BTC/USDT:USDT": _pos("BTC/USDT:USDT", 0.001)},
    )
    st = await ad.fetch_account_state()
    assert {p.symbol for p in st.positions} == {"BTCUSDT"}


async def test_fetch_account_state_fails_loud_on_missing_margin():
    """A risk system must stop, not guess, if it cannot read maintenance margin."""
    ad, _ = _adapter({"totalWalletBalance": "100"}, {})   # no totalMaintMargin
    with pytest.raises(RuntimeError, match="margin fields"):
        await ad.fetch_account_state()


# ------------------------------------------------------------- close contract

async def test_close_sends_reduce_only_market_order_opposite_side():
    ad, fake = _adapter(
        _balance(100, 100, 2),
        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.2)},
    )
    res = await ad.close_position("ETHUSDT", client_order_id="k-1")
    assert res.ok
    assert len(fake.created) == 1
    o = fake.created[0]
    assert o["symbol"] == "ETH/USDT:USDT"
    assert o["type"] == "market"
    assert o["side"] == "sell"                 # closing a long
    assert o["amount"] == pytest.approx(0.2)
    assert o["params"]["reduceOnly"] is True   # THE contract
    assert o["params"]["newClientOrderId"] == "k-1"
    assert res.filled_qty == pytest.approx(-0.2)


async def test_close_short_buys_back():
    ad, fake = _adapter(_balance(100, 100, 2),
                        {"SOL/USDT:USDT": _pos("SOL/USDT:USDT", -3.0)})
    res = await ad.close_position("SOLUSDT", client_order_id="k-2")
    assert fake.created[0]["side"] == "buy"    # closing a short
    assert fake.created[0]["amount"] == pytest.approx(3.0)
    assert res.filled_qty == pytest.approx(3.0)


async def test_close_flat_position_sends_no_order():
    ad, fake = _adapter(_balance(100, 100, 1),
                        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.0)})
    res = await ad.close_position("ETHUSDT", client_order_id="k-3")
    assert res.ok and res.filled_qty == 0.0
    assert fake.created == []                   # nothing to close


async def test_duplicate_client_order_id_is_treated_as_success():
    """Retry after timeout: a reused id means the order already landed."""
    ad, fake = _adapter(_balance(100, 100, 2),
                        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.2)})
    fake.raise_on_create = Exception("APIError: Duplicate order sent. code=-4015")
    res = await ad.close_position("ETHUSDT", client_order_id="k-4")
    assert res.ok                                # not a failure


async def test_non_duplicate_error_is_a_failure():
    ad, fake = _adapter(_balance(100, 100, 2),
                        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.2)})
    fake.raise_on_create = Exception("APIError: insufficient margin code=-2019")
    res = await ad.close_position("ETHUSDT", client_order_id="k-5")
    assert not res.ok and "insufficient" in res.error


async def test_cancel_all_sweeps_symbols_with_open_orders():
    ad, fake = _adapter(_balance(100, 100, 1), {})
    fake.open_orders = [{"symbol": "ETH/USDT:USDT"}, {"symbol": "ETH/USDT:USDT"},
                        {"symbol": "SOL/USDT:USDT"}]
    await ad.cancel_all_orders()
    assert set(fake.cancelled) == {"ETH/USDT:USDT", "SOL/USDT:USDT"}


# ------------------------------------------------ end-to-end through the risk loop

async def test_risk_loop_flattens_through_the_real_adapter():
    """The tested kill switch, driven through the actual Binance adapter (fake
    transport). Near-liquidation -> reduce-only closes on the real code path."""
    ad, fake = _adapter(
        _balance(equity=100, wallet=100, maint=60),   # margin_ratio 0.6 > 0.5
        {"ETH/USDT:USDT": _pos("ETH/USDT:USDT", 0.2),
         "SOL/USDT:USDT": _pos("SOL/USDT:USDT", -1.0)},
    )
    loop = RiskLoop(exchange=ad, limits=RiskLimits(equity_floor=50), peak_equity=100.0)
    await loop.check_once()
    assert loop.halted and Breach.MARGIN_RATIO in loop.halt_reason
    # both closed, both reduce-only
    assert {o["symbol"] for o in fake.created} == {"ETH/USDT:USDT", "SOL/USDT:USDT"}
    assert all(o["params"]["reduceOnly"] is True for o in fake.created)
    st = await ad.fetch_account_state()
    assert st.open_positions == ()
