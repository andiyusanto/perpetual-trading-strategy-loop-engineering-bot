"""Trading cost model — applied from the FIRST backtest run, never bolted on.

CLAUDE.md methodology rule 7: every backtest includes real Binance USD-M futures
fees and a conservative slippage model. There is no "clean" pre-cost number
reported anywhere in this codebase, because a pre-cost edge is not evidence.

Fee schedule (Binance USD-M Futures, VIP 0): maker 0.0200%, taker 0.0500%.
These are the DEFAULTS in .env.example and must be re-verified against the
current published schedule before any result is treated as final — fee tiers
change, and BNB/VIP discounts would make them lower (i.e. these defaults are the
conservative side, which is the right direction to be wrong in).

Conservatism choices, all deliberately pessimistic:
  - Both entry and exit are modelled as TAKER fills. The strategy enters on a
    confirmed signal, which in practice means crossing the spread; assuming
    maker fills would be assuming free optionality we have not demonstrated.
  - Slippage is charged on BOTH sides, always against us.

MEASURED 2026-07-23 (from 869M BTCUSDT aggTrades already on disk, not assumed):
the quoted spread is ONE TICK 89.2% of the time. Tick is $0.10, so at ~$70k
that is **0.014 bps** — the original 3.0 bps/side slippage default was ~200x too
pessimistic on the spread component.

``slippage_bps`` is therefore reinterpreted: it is an allowance for MARKET
IMPACT, not for the spread. Impact depends on order size relative to resting
depth, which aggTrades cannot measure (it carries no resting-order sizes). The
default below is a modest allowance for small size; **it must be raised for
large orders**, and settling it properly needs order-book depth data.

For scale: the median BTCUSDT trade is ~0.003 BTC. A 1.0 BTC order is ~333x
that, and its impact is NOT captured by this default.

Fees dominate regardless: 2 x 5bps taker = 10 bps is the floor no matter how
tight the book is.
"""

from __future__ import annotations

from dataclasses import dataclass

BPS = 1e-4

LONG = "long"
SHORT = "short"


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost model. All rates in basis points."""

    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    # Impact allowance, NOT spread (measured spread is ~0.014 bps). Raise this
    # for size; see module docstring.
    slippage_bps: float = 0.5

    def __post_init__(self) -> None:
        for name in ("taker_fee_bps", "maker_fee_bps", "slippage_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    # -- price impact -------------------------------------------------------

    def fill_price(self, side: str, ref_price: float, *, is_entry: bool) -> float:
        """Apply slippage against us.

        Entering long or exiting short means buying -> we pay UP.
        Entering short or exiting long means selling -> we receive DOWN.
        """
        buying = (side == LONG) == is_entry
        adj = 1.0 + self.slippage_bps * BPS if buying else 1.0 - self.slippage_bps * BPS
        return ref_price * adj

    # -- fees ---------------------------------------------------------------

    def fee(self, notional: float, *, taker: bool = True) -> float:
        rate = self.taker_fee_bps if taker else self.maker_fee_bps
        return abs(notional) * rate * BPS

    def round_trip_cost(self, entry_px: float, exit_px: float, qty: float) -> float:
        """Total fees for a round trip (both legs taker by default)."""
        return self.fee(entry_px * qty) + self.fee(exit_px * qty)

    @property
    def round_trip_bps(self) -> float:
        """Approximate all-in round-trip cost in bps, for sanity checks.

        Two taker fees plus slippage on both sides — the hurdle every trade must
        clear before it is worth anything.
        """
        return 2 * self.taker_fee_bps + 2 * self.slippage_bps


def default_cost_model() -> CostModel:
    return CostModel()
