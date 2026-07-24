"""A controllable in-memory exchange for testing the risk loop.

It is NOT a market simulator — it is a puppet. A test can put it into any state
(deep drawdown, near-liquidation, exposure mismatch) and can make any call fail
on demand, so every kill path can be triggered deliberately and asserted.

It enforces the two invariants that matter for safety:
  - close_position is REDUCE-ONLY: it can only move a position toward zero,
    never past it and never open one. If the simulator ever let a "close" flip a
    position, the kill-switch tests would pass against a lie.
  - client_order_id is idempotent: the same id applied twice closes once.
"""

from __future__ import annotations

from .interface import AccountState, OrderResult, Position


class SimulatedExchange:
    def __init__(
        self,
        *,
        wallet_balance: float = 100.0,
        maintenance_margin: float = 0.0,
        positions: list[Position] | None = None,
    ) -> None:
        self.wallet_balance = wallet_balance
        self.maintenance_margin = maintenance_margin
        self._positions: dict[str, Position] = {
            p.symbol: p for p in (positions or [])
        }
        # fault injection
        self.fail_fetch = False
        self.fail_close_for: set[str] = set()
        self.fail_cancel = False
        # audit trail for assertions
        self.close_calls: list[tuple[str, str]] = []      # (symbol, client_order_id)
        self.cancel_calls: list[str | None] = []
        self._applied_ids: set[str] = set()

    # -- state the tests manipulate directly --------------------------------

    def set_position(self, pos: Position) -> None:
        self._positions[pos.symbol] = pos

    def set_mark(self, symbol: str, mark: float) -> None:
        p = self._positions[symbol]
        self._positions[symbol] = Position(p.symbol, p.qty, p.entry_price, mark)

    @property
    def equity(self) -> float:
        return self.wallet_balance + sum(
            p.unrealized_pnl for p in self._positions.values()
        )

    def _state(self) -> AccountState:
        return AccountState(
            equity=self.equity,
            wallet_balance=self.wallet_balance,
            maintenance_margin=self.maintenance_margin,
            positions=tuple(self._positions.values()),
        )

    # -- Exchange protocol --------------------------------------------------

    async def fetch_account_state(self) -> AccountState:
        if self.fail_fetch:
            raise ConnectionError("simulated fetch failure")
        return self._state()

    async def close_position(
        self, symbol: str, *, client_order_id: str
    ) -> OrderResult:
        self.close_calls.append((symbol, client_order_id))
        if symbol in self.fail_close_for:
            return OrderResult(ok=False, symbol=symbol, error="simulated close failure")
        if client_order_id in self._applied_ids:
            # idempotent: already applied, report success without re-acting
            return OrderResult(ok=True, symbol=symbol, filled_qty=0.0,
                               client_order_id=client_order_id)
        pos = self._positions.get(symbol)
        if pos is None or pos.is_flat:
            return OrderResult(ok=True, symbol=symbol, filled_qty=0.0,
                               client_order_id=client_order_id)
        # REDUCE-ONLY: move to exactly flat, never past zero.
        closed = -pos.qty
        self._positions[symbol] = Position(symbol, 0.0, pos.entry_price, pos.mark_price)
        self._applied_ids.add(client_order_id)
        return OrderResult(ok=True, symbol=symbol, filled_qty=closed,
                           client_order_id=client_order_id)

    async def cancel_all_orders(self, symbol: str | None = None) -> None:
        self.cancel_calls.append(symbol)
        if self.fail_cancel:
            raise ConnectionError("simulated cancel failure")
