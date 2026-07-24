"""Exchange-agnostic types and protocol.

Deliberately minimal: the risk loop only needs to (1) see the account's true
state, (2) close positions reduce-only, and (3) cancel resting orders. Anything
the risk loop does not need is not here, so the surface a real adapter must get
right — and that a bug could hide in — stays small.

Sign convention: ``qty`` is signed. qty > 0 is long, qty < 0 is short, qty == 0
is flat. This is the ONE convention the whole system uses; mixing it up is how a
"close" becomes a "double up", so it is stated once, here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

LONG = "long"
SHORT = "short"
FLAT = "flat"


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float           # signed base units; + long, - short, 0 flat
    entry_price: float
    mark_price: float

    @property
    def side(self) -> str:
        if self.qty > 0:
            return LONG
        if self.qty < 0:
            return SHORT
        return FLAT

    @property
    def notional(self) -> float:
        """Absolute mark-to-market exposure in quote currency (USDT)."""
        return abs(self.qty) * self.mark_price

    @property
    def unrealized_pnl(self) -> float:
        return self.qty * (self.mark_price - self.entry_price)

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


@dataclass(frozen=True)
class AccountState:
    """A single consistent snapshot of the account.

    ``equity`` is the margin balance: wallet balance + unrealized PnL, in USDT.
    ``maintenance_margin`` is what the exchange requires to avoid liquidation.
    """

    equity: float
    wallet_balance: float
    maintenance_margin: float
    positions: tuple[Position, ...] = field(default_factory=tuple)

    @property
    def gross_notional(self) -> float:
        return sum(p.notional for p in self.positions)

    @property
    def open_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions if not p.is_flat)

    @property
    def margin_ratio(self) -> float:
        """maintenance_margin / equity. Liquidation risk rises toward 1.0.

        Bankrupt/zero equity returns +inf so every downstream threshold trips.
        """
        if self.equity <= 0:
            return math.inf
        return self.maintenance_margin / self.equity

    @property
    def effective_leverage(self) -> float:
        if self.equity <= 0:
            return math.inf
        return self.gross_notional / self.equity


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    symbol: str
    filled_qty: float = 0.0
    client_order_id: str = ""
    error: str = ""


@runtime_checkable
class Exchange(Protocol):
    """The only exchange surface the risk loop depends on."""

    async def fetch_account_state(self) -> AccountState:
        """Return the account's TRUE current state, read from the venue.

        Must reflect live positions, never a locally cached belief — the risk
        loop trusts this over any internal state (reconcile-on-read)."""
        ...

    async def close_position(
        self, symbol: str, *, client_order_id: str
    ) -> OrderResult:
        """Flatten ``symbol`` with a REDUCE-ONLY market order.

        Reduce-only is mandatory: it guarantees the order can only shrink or
        close the position, never open or flip one. ``client_order_id`` makes
        the call idempotent — a retry after a timeout must not double up."""
        ...

    async def cancel_all_orders(self, symbol: str | None = None) -> None:
        """Cancel resting orders (all, or for one symbol). Resting orders can
        re-open exposure after a flatten, so the kill switch cancels them."""
        ...
