"""Tests for the risk loop and kill switch.

This file IS the "test the kill switch by deliberate trigger" requirement from
HYPOTHESIS_trend_following.md section 9. Every kill condition is forced and the
flatten is asserted. An untested kill switch is a comment; these are the tests
that make it real.
"""

from __future__ import annotations

import pytest

from src.execution.interface import LONG, SHORT, AccountState, Position
from src.execution.simulator import SimulatedExchange
from src.risk.monitor import Breach, RiskLimits, RiskLoop, evaluate


def _pos(sym, qty, entry=100.0, mark=100.0):
    return Position(sym, qty, entry, mark)


def _limits(**kw):
    base = dict(equity_floor=50.0, max_margin_ratio=0.5, max_drawdown_pct=0.4,
                max_leverage=1.1)
    base.update(kw)
    return RiskLimits(**base)


# --------------------------------------------------------------- pure evaluate

def test_healthy_1x_book_is_safe():
    """The tf-v1 base case: 1x on majors, comfortable margin -> no breach."""
    st = AccountState(equity=100, wallet_balance=100, maintenance_margin=2,
                      positions=(_pos("ETHUSDT", 0.2, 100, 100),))  # $20 notional
    d = evaluate(st, _limits(), peak_equity=100, intended={"ETHUSDT": 0.2})
    assert d.safe


def test_equity_floor_breach():
    st = AccountState(equity=40, wallet_balance=40, maintenance_margin=1)
    d = evaluate(st, _limits(equity_floor=50), peak_equity=100)
    assert Breach.EQUITY_FLOOR in d.breaches


def test_margin_ratio_breach():
    st = AccountState(equity=100, wallet_balance=100, maintenance_margin=60)
    d = evaluate(st, _limits(), peak_equity=100)
    assert Breach.MARGIN_RATIO in d.breaches  # 0.6 > 0.5


def test_zero_equity_is_infinite_margin_ratio():
    st = AccountState(equity=0, wallet_balance=0, maintenance_margin=5)
    d = evaluate(st, _limits(equity_floor=1), peak_equity=100)
    assert Breach.MARGIN_RATIO in d.breaches
    assert Breach.EQUITY_FLOOR in d.breaches


def test_drawdown_breach():
    st = AccountState(equity=55, wallet_balance=55, maintenance_margin=1)
    d = evaluate(st, _limits(), peak_equity=100)  # 45% dd > 40%
    assert Breach.DRAWDOWN in d.breaches


def test_leverage_breach_catches_oversized_book():
    # gross notional 150 on 100 equity = 1.5x > 1.1x cap
    st = AccountState(equity=100, wallet_balance=100, maintenance_margin=3,
                      positions=(_pos("ETHUSDT", 1.5, 100, 100),))
    d = evaluate(st, _limits(), peak_equity=100)
    assert Breach.LEVERAGE in d.breaches


def test_exposure_mismatch_catches_the_extra_position_bug():
    """The 'opened 5 positions instead of 1' signature."""
    st = AccountState(equity=100, wallet_balance=100, maintenance_margin=2,
                      positions=(_pos("ETHUSDT", 0.2), _pos("SOLUSDT", 1.0)))
    d = evaluate(st, _limits(), peak_equity=100, intended={"ETHUSDT": 0.2})
    assert Breach.EXPOSURE_MISMATCH in d.breaches


def test_exposure_mismatch_on_wrong_size_and_wrong_sign():
    st = AccountState(equity=100, wallet_balance=100, maintenance_margin=2,
                      positions=(_pos("ETHUSDT", -0.2),))  # short, intended long
    d = evaluate(st, _limits(), peak_equity=100, intended={"ETHUSDT": 0.2})
    assert Breach.EXPOSURE_MISMATCH in d.breaches


def test_matching_exposure_is_safe():
    st = AccountState(equity=100, wallet_balance=100, maintenance_margin=2,
                      positions=(_pos("ETHUSDT", 0.2),))
    d = evaluate(st, _limits(), peak_equity=100, intended={"ETHUSDT": 0.2})
    assert d.safe


def test_limits_reject_nonsense_config():
    with pytest.raises(ValueError):
        RiskLimits(equity_floor=50, max_margin_ratio=1.5)
    with pytest.raises(ValueError):
        RiskLimits(equity_floor=50, max_drawdown_pct=0)


# --------------------------------------------------- kill switch flattens (async)

async def _tripped(ex, limits, **kw):
    loop = RiskLoop(exchange=ex, limits=limits, peak_equity=100.0, **kw)
    d = await loop.check_once()
    return loop, d


async def test_kill_switch_flattens_on_margin_breach():
    ex = SimulatedExchange(wallet_balance=100, maintenance_margin=60,
                           positions=[_pos("ETHUSDT", 0.2), _pos("SOLUSDT", 1.0)])
    loop, d = await _tripped(ex, _limits())
    assert not d.safe and loop.halted
    # BOTH positions were closed reduce-only
    assert {c[0] for c in ex.close_calls} == {"ETHUSDT", "SOLUSDT"}
    st = await ex.fetch_account_state()
    assert all(p.is_flat for p in st.positions)
    assert ex.cancel_calls, "resting orders must be cancelled on kill"


async def test_kill_switch_flattens_on_drawdown():
    ex = SimulatedExchange(wallet_balance=55, maintenance_margin=1,
                           positions=[_pos("ETHUSDT", 0.1)])
    loop, d = await _tripped(ex, _limits())
    assert Breach.DRAWDOWN in loop.halt_reason
    assert (await ex.fetch_account_state()).open_positions == ()


async def test_kill_switch_flattens_on_exposure_mismatch():
    ex = SimulatedExchange(wallet_balance=100, maintenance_margin=2,
                           positions=[_pos("ETHUSDT", 0.2), _pos("SOLUSDT", 5.0)])

    class L(RiskLoop):
        def intended_positions(self):
            return {"ETHUSDT": 0.2}  # SOL position is a bug

    loop = L(exchange=ex, limits=_limits(), peak_equity=100.0)
    await loop.check_once()
    assert loop.halted and Breach.EXPOSURE_MISMATCH in loop.halt_reason
    assert (await ex.fetch_account_state()).open_positions == ()


async def test_halt_is_latched_and_never_self_recovers():
    ex = SimulatedExchange(wallet_balance=40, maintenance_margin=1,
                           positions=[_pos("ETHUSDT", 0.1)])
    loop, _ = await _tripped(ex, _limits(equity_floor=50))
    assert loop.halted
    # now the account looks healthy again
    ex.wallet_balance = 100
    d2 = await loop.check_once()
    assert loop.halted, "kill switch must stay latched; recovery is a human act"
    assert d2.safe  # the reading is safe, but the latch holds


async def test_failed_close_is_retried_until_flat():
    ex = SimulatedExchange(wallet_balance=100, maintenance_margin=60,
                           positions=[_pos("ETHUSDT", 0.2)])
    ex.fail_close_for = {"ETHUSDT"}          # first attempt fails
    loop, _ = await _tripped(ex, _limits())
    assert loop.halted
    assert not (await ex.fetch_account_state()).open_positions == ()  # still open
    ex.fail_close_for = set()                 # venue recovers
    await loop.check_once()                    # latch re-attempts the flatten
    assert (await ex.fetch_account_state()).open_positions == ()


async def test_connectivity_loss_trips_after_threshold():
    ex = SimulatedExchange(wallet_balance=100, positions=[_pos("ETHUSDT", 0.1)])
    ex.fail_fetch = True
    loop = RiskLoop(exchange=ex, limits=_limits(max_fetch_failures=3),
                    peak_equity=100.0)
    d1 = await loop.check_once()
    assert d1.safe and not loop.halted        # transient, still watching
    await loop.check_once()
    d3 = await loop.check_once()
    assert Breach.CONNECTIVITY_LOSS in d3.breaches and loop.halted


async def test_transient_fetch_failure_recovers_without_halting():
    ex = SimulatedExchange(wallet_balance=100, positions=[_pos("ETHUSDT", 0.1)])
    ex.fail_fetch = True
    loop = RiskLoop(exchange=ex, limits=_limits(max_fetch_failures=5),
                    peak_equity=100.0)
    await loop.check_once()
    await loop.check_once()
    ex.fail_fetch = False                      # recovers before the threshold
    d = await loop.check_once()
    assert d.safe and not loop.halted
    assert loop._fetch_failures == 0


async def test_on_halt_callback_fires_once():
    fired = []
    ex = SimulatedExchange(wallet_balance=40, maintenance_margin=1,
                           positions=[_pos("ETHUSDT", 0.1)])
    loop = RiskLoop(exchange=ex, limits=_limits(equity_floor=50),
                    peak_equity=100.0, on_halt=lambda d: fired.append(d))
    await loop.check_once()
    await loop.check_once()   # still halted, but alert must not re-fire
    assert len(fired) == 1


# ------------------------------------------------ reduce-only / idempotency

async def test_close_is_reduce_only_never_flips():
    """A 'close' must never open the opposite position, even if called twice."""
    ex = SimulatedExchange(positions=[_pos("ETHUSDT", 0.2)])
    r1 = await ex.close_position("ETHUSDT", client_order_id="k1")
    assert r1.filled_qty == pytest.approx(-0.2)          # closed the long
    st = await ex.fetch_account_state()
    assert st.positions[0].is_flat
    # a duplicate id must not re-open or double
    r2 = await ex.close_position("ETHUSDT", client_order_id="k1")
    assert r2.ok and r2.filled_qty == 0.0
    assert (await ex.fetch_account_state()).positions[0].is_flat


async def test_run_stops_after_max_cycles():
    ex = SimulatedExchange(wallet_balance=100, positions=[_pos("ETHUSDT", 0.1)])
    loop = RiskLoop(exchange=ex, limits=_limits(), peak_equity=100.0)
    await loop.run(interval_s=0, max_cycles=3)
    assert not loop.halted  # healthy account, ran clean


async def test_unknown_intent_does_not_flatten_legitimate_positions():
    """Restart safety: with intent not yet declared (None), a live position must
    NOT be treated as a bug. Flattening legitimate positions on every restart
    would be catastrophic; reconciliation only engages once intent is known."""
    ex = SimulatedExchange(wallet_balance=100, maintenance_margin=2,
                           positions=[_pos("ETHUSDT", 0.2)])
    loop = RiskLoop(exchange=ex, limits=_limits(), peak_equity=100.0)
    assert loop.intended_positions() is None
    d = await loop.check_once()
    assert d.safe and not loop.halted
    assert ex.close_calls == [], "must not flatten a position when intent is unknown"

    # once intent is declared and it disagrees, THEN it trips
    class L(RiskLoop):
        def intended_positions(self):
            return {}   # signal loop now says: hold nothing
    loop2 = L(exchange=ex, limits=_limits(), peak_equity=100.0)
    d2 = await loop2.check_once()
    assert Breach.EXPOSURE_MISMATCH in d2.breaches
