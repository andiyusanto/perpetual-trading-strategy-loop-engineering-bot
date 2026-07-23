"""Cheap signal screening — kill candidates before building machinery for them."""

from .screen import SignalSet, screen_signal, forward_returns

__all__ = ["SignalSet", "screen_signal", "forward_returns"]
