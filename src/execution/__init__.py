"""Execution layer: exchange-agnostic interface + a simulator for testing.

Real venues (Binance USD-M via CCXT, later Bybit/OKX/Bitget) implement the same
``Exchange`` protocol. The risk loop and kill switch are written against that
protocol only, so they can be exhaustively tested against a controllable
``SimulatedExchange`` that can be driven into any dangerous state on demand.
"""

from .interface import AccountState, Exchange, OrderResult, Position

__all__ = ["AccountState", "Exchange", "OrderResult", "Position"]
