"""Risk loop and kill switch.

Independent of and higher-priority than the signal loop (CLAUDE.md architecture).
It runs on a fast cadence, can flatten the book at any time without asking the
signal loop, and is the component that decides whether a bug costs $5 or the
whole account.
"""

from .monitor import Breach, RiskDecision, RiskLimits, RiskLoop, evaluate

__all__ = ["Breach", "RiskDecision", "RiskLimits", "RiskLoop", "evaluate"]
