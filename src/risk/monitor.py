"""Risk evaluation + the kill switch loop.

Design stance: FAIL SAFE. Every ambiguous condition resolves toward flattening,
not toward hoping. If the loop cannot see the account (repeated fetch failures),
that is itself a kill condition — a bot that cannot verify it is safe must not
assume it is.

The kill switch is:
  - LATCHED: once tripped it stays tripped. Recovery is a human decision, never
    automatic, because the condition that tripped it (a bug, a crash, a market
    break) is not something the bot should decide has passed.
  - IDEMPOTENT + PERSISTENT: it re-issues reduce-only closes every cycle until
    the book is verified flat, so a single failed close does not leave a
    position stranded.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field

import structlog

from src.execution.interface import AccountState, Exchange, Position

log = structlog.get_logger(__name__)


class Breach(enum.Enum):
    EQUITY_FLOOR = "equity_floor"          # equity fell below the hard floor
    MARGIN_RATIO = "margin_ratio"          # too close to liquidation
    DRAWDOWN = "drawdown"                  # peak-to-now drawdown too large
    LEVERAGE = "leverage"                  # gross notional exceeds the cap
    EXPOSURE_MISMATCH = "exposure_mismatch"  # live != intended (bug signature)
    CONNECTIVITY_LOSS = "connectivity_loss"  # cannot see the account


@dataclass(frozen=True)
class RiskLimits:
    """All hard limits in one place. Chosen for a small, unlevered account.

    Defaults encode the tf-v1 design: a 1x book on majors is structurally
    unliquidatable, so these thresholds should essentially never trip in normal
    operation — if they do, something is wrong, which is exactly when to flatten.
    """

    equity_floor: float                    # absolute USDT; kill below this
    max_margin_ratio: float = 0.50         # flatten at half-way to liquidation
    max_drawdown_pct: float = 0.40         # matches stated tolerance; kill beyond
    max_leverage: float = 1.10             # 1x book + a little slack for marks
    exposure_tolerance_qty: float = 1e-8   # any real live/intended divergence
    max_fetch_failures: int = 5            # consecutive; then assume danger

    def __post_init__(self) -> None:
        if not 0 < self.max_margin_ratio < 1:
            raise ValueError("max_margin_ratio must be in (0, 1)")
        if not 0 < self.max_drawdown_pct <= 1:
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")


@dataclass(frozen=True)
class RiskDecision:
    breaches: tuple[Breach, ...]
    reasons: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        return not self.breaches


def evaluate(
    state: AccountState,
    limits: RiskLimits,
    *,
    peak_equity: float,
    intended: dict[str, float] | None = None,
) -> RiskDecision:
    """Pure function: given a state snapshot, which limits are breached?

    ``peak_equity`` is the high-water mark for drawdown. ``intended`` maps symbol
    -> signed target qty; any live position that disagrees beyond tolerance is an
    EXPOSURE_MISMATCH (the signature of an execution bug — e.g. five positions
    opened where one was intended).
    """
    breaches: list[Breach] = []
    reasons: list[str] = []

    if state.equity < limits.equity_floor:
        breaches.append(Breach.EQUITY_FLOOR)
        reasons.append(f"equity {state.equity:.2f} < floor {limits.equity_floor:.2f}")

    if state.margin_ratio > limits.max_margin_ratio:
        breaches.append(Breach.MARGIN_RATIO)
        reasons.append(
            f"margin_ratio {state.margin_ratio:.3f} > {limits.max_margin_ratio:.3f}"
        )

    if peak_equity > 0:
        dd = 1.0 - state.equity / peak_equity
        if dd > limits.max_drawdown_pct:
            breaches.append(Breach.DRAWDOWN)
            reasons.append(f"drawdown {dd:.1%} > {limits.max_drawdown_pct:.1%}")

    if state.effective_leverage > limits.max_leverage:
        breaches.append(Breach.LEVERAGE)
        reasons.append(
            f"leverage {state.effective_leverage:.2f}x > {limits.max_leverage:.2f}x"
        )

    if intended is not None:
        live = {p.symbol: p.qty for p in state.positions}
        symbols = set(live) | set(intended)
        for sym in sorted(symbols):
            diff = abs(live.get(sym, 0.0) - intended.get(sym, 0.0))
            if diff > limits.exposure_tolerance_qty:
                breaches.append(Breach.EXPOSURE_MISMATCH)
                reasons.append(
                    f"{sym}: live {live.get(sym, 0.0)} != intended "
                    f"{intended.get(sym, 0.0)}"
                )
                break  # one mismatch is enough to halt; do not spam

    return RiskDecision(tuple(breaches), tuple(reasons))


@dataclass
class RiskLoop:
    exchange: Exchange
    limits: RiskLimits
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: tuple[Breach, ...] = ()
    _fetch_failures: int = 0
    _halted_at: float = 0.0
    on_halt: object = None   # optional callable(decision) for alerting

    def intended_positions(self) -> dict[str, float] | None:
        """What the signal loop currently wants, or None if intent is not yet
        known. Injected/overridden by the orchestrator.

        The default is None, NOT {} — and the difference is safety-critical.
        {} means "I expect zero positions", which would flatten legitimate
        positions on every restart before the signal loop has declared its
        intent. None means "intent unknown, skip reconciliation", so
        reconcile-on-start reads the real positions and leaves them alone until
        the signal loop asserts what it wants. Exposure reconciliation only
        engages once intent is actually declared."""
        return None

    async def check_once(self) -> RiskDecision:
        """One risk evaluation cycle. Trips the kill switch on any breach."""
        try:
            state = await self.exchange.fetch_account_state()
            self._fetch_failures = 0
        except Exception as exc:  # noqa: BLE001 - any fetch failure is a risk event
            self._fetch_failures += 1
            log.warning("risk.fetch_failed", consecutive=self._fetch_failures,
                        err=str(exc))
            if self._fetch_failures >= self.limits.max_fetch_failures:
                decision = RiskDecision(
                    (Breach.CONNECTIVITY_LOSS,),
                    (f"{self._fetch_failures} consecutive fetch failures",),
                )
                await self._trip(decision, state=None)
                return decision
            return RiskDecision(())  # transient; keep watching

        self.peak_equity = max(self.peak_equity, state.equity)
        decision = evaluate(
            state, self.limits,
            peak_equity=self.peak_equity,
            intended=self.intended_positions(),
        )
        if not decision.safe:
            await self._trip(decision, state=state)
        elif self.halted:
            # Latched: even a now-"safe" reading keeps flattening until verified
            # flat, and never un-halts on its own.
            await self._flatten(state)
        return decision

    async def _trip(self, decision: RiskDecision, state: AccountState | None) -> None:
        if not self.halted:
            self.halted = True
            self.halt_reason = decision.breaches
            self._halted_at = time.time()
            log.error("risk.kill_switch_tripped",
                      breaches=[b.value for b in decision.breaches],
                      reasons=list(decision.reasons))
            if callable(self.on_halt):
                try:
                    self.on_halt(decision)
                except Exception as exc:  # noqa: BLE001 - alerting must not crash the kill
                    log.error("risk.on_halt_failed", err=str(exc))
        if state is not None:
            await self._flatten(state)

    async def _flatten(self, state: AccountState) -> None:
        """Reduce-only close every open position, then cancel resting orders.

        Idempotent per position (client_order_id derived from symbol + halt time),
        and re-run each cycle by the latch until the book is flat, so a single
        failed close is retried rather than abandoned."""
        open_positions = state.open_positions
        for p in open_positions:
            coid = f"kill-{int(self._halted_at)}-{p.symbol}"
            res = await self.exchange.close_position(p.symbol, client_order_id=coid)
            if not res.ok:
                log.error("risk.close_failed", symbol=p.symbol, err=res.error)
            else:
                log.info("risk.closed", symbol=p.symbol, filled=res.filled_qty)
        try:
            await self.exchange.cancel_all_orders()
        except Exception as exc:  # noqa: BLE001 - retried next cycle
            log.error("risk.cancel_failed", err=str(exc))

    async def run(self, interval_s: float = 60.0, *, max_cycles: int | None = None) -> None:
        """Run the risk loop forever (or for max_cycles, for tests)."""
        n = 0
        while True:
            await self.check_once()
            n += 1
            if max_cycles is not None and n >= max_cycles:
                return
            await asyncio.sleep(interval_s)
