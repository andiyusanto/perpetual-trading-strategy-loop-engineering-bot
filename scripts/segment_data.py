"""Data segmentation with hard technical isolation.

The isolation rules (CLAUDE.md methodology rules 2, 5, 6):

- Each segment has a HARDCODED date range. Requesting data outside it raises;
  it never silently clips.
- **Forward isolation is absolute.** No loader will ever return a bar or a
  funding event dated after its segment's end. That is the boundary that
  protects against look-ahead.
- **Backward warm-up is permitted and bounded.** Indicators need history to be
  correct at a segment's first bar (funding gate: trailing 90d; rolling CVD:
  N bars). Reading *earlier* data is not look-ahead — it is past data — so
  loaders may reach back ``WARMUP_DAYS`` before the segment start. Those rows
  come back flagged ``is_warmup=True`` and MUST NOT produce trades or feed
  performance statistics; they exist only to prime indicator state.
- Validation may be opened a limited number of times; holdout exactly once,
  behind a git-tag gate. Every open is logged with timestamp + commit hash.

Segment boundaries are oldest -> newest, split 50/25/25, per pair (pairs are NOT
pooled; HYPEUSDT has its own window because it listed 2025-05-30).
"""

from __future__ import annotations

import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.cvd import bars_cache_path  # noqa: E402
from src.strategy.funding_filter import load_funding  # noqa: E402

RESEARCH = "research"
VALIDATION = "validation"
HOLDOUT = "holdout"

# Days of history a segment may reach BACKWARD for indicator warm-up.
# Sized by the longest trailing window in the strategy: the funding gate's
# 90-day percentile distribution. Rolling CVD (20 bars) is far shorter.
WARMUP_DAYS = 90

# KILL_CRITERIA.md: validation may be opened 2-3 times, holdout exactly once.
MAX_VALIDATION_OPENS = 3
MAX_HOLDOUT_OPENS = 1


@dataclass(frozen=True)
class Bounds:
    start: dt.date
    end: dt.date


# ---------------------------------------------------------------------------
# HARDCODED segment boundaries. Derived once from the actually-downloaded data
# range and then frozen — see tests/test_segment_data.py, which re-derives the
# 50/25/25 split and asserts these constants still match.
#
# BTC/ETH/SOL: downloaded window 2025-01-01 .. 2026-06-30 (546d)
# HYPE:        listed 2025-05-30T10:30Z; first 14d of new-listing artifact
#              (thin liquidity, 4h funding, price discovery) dropped, giving
#              2025-06-13 .. 2026-06-30 (383d)
# ---------------------------------------------------------------------------
SEGMENT_BOUNDS: dict[str, dict[str, Bounds]] = {
    "BTCUSDT": {
        RESEARCH:   Bounds(dt.date(2025, 1, 1),  dt.date(2025, 9, 30)),
        VALIDATION: Bounds(dt.date(2025, 10, 1), dt.date(2026, 2, 13)),
        HOLDOUT:    Bounds(dt.date(2026, 2, 14), dt.date(2026, 6, 30)),
    },
    "ETHUSDT": {
        RESEARCH:   Bounds(dt.date(2025, 1, 1),  dt.date(2025, 9, 30)),
        VALIDATION: Bounds(dt.date(2025, 10, 1), dt.date(2026, 2, 13)),
        HOLDOUT:    Bounds(dt.date(2026, 2, 14), dt.date(2026, 6, 30)),
    },
    "SOLUSDT": {
        RESEARCH:   Bounds(dt.date(2025, 1, 1),  dt.date(2025, 9, 30)),
        VALIDATION: Bounds(dt.date(2025, 10, 1), dt.date(2026, 2, 13)),
        HOLDOUT:    Bounds(dt.date(2026, 2, 14), dt.date(2026, 6, 30)),
    },
    "HYPEUSDT": {
        RESEARCH:   Bounds(dt.date(2025, 6, 13), dt.date(2025, 12, 21)),
        VALIDATION: Bounds(dt.date(2025, 12, 22), dt.date(2026, 3, 27)),
        HOLDOUT:    Bounds(dt.date(2026, 3, 28), dt.date(2026, 6, 30)),
    },
}

DATA_ROOT = _PROJECT_ROOT / "data"
LOG_PATH = _PROJECT_ROOT / "logs" / "access.log"


class SegmentIsolationError(RuntimeError):
    """Raised when code tries to load data outside its segment's allowed range."""


def _bounds(symbol: str, segment: str) -> Bounds:
    try:
        return SEGMENT_BOUNDS[symbol][segment]
    except KeyError as exc:
        raise SegmentIsolationError(
            f"no hardcoded bounds for symbol={symbol!r} segment={segment!r}"
        ) from exc


def _assert_in_range(date: dt.date, start: dt.date, end: dt.date, segment: str) -> None:
    if not (start <= date <= end):
        raise SegmentIsolationError(
            f"Attempted to load {date} for segment '{segment}', "
            f"but allowed range is [{start}, {end}]. This is blocked by design — "
            f"see CLAUDE.md methodology rule 2."
        )


def _log_access(segment: str, note: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_PROJECT_ROOT
    ).stdout.strip() or "no-git"
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    with LOG_PATH.open("a") as f:
        f.write(f"{ts}\t{segment}\t{commit}\t{note}\n")


def count_opens(segment: str) -> int:
    """How many times a segment has been opened, per logs/access.log."""
    if not LOG_PATH.exists():
        return 0
    n = 0
    for line in LOG_PATH.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == segment:
            n += 1
    return n


def _to_ms(d: dt.date) -> int:
    return int(
        dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


def _end_ms_exclusive(d: dt.date) -> int:
    """End of day *d* (exclusive) in epoch ms — the hard forward boundary."""
    nxt = d + dt.timedelta(days=1)
    return _to_ms(nxt)


def _months_spanning(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def load_segment_bars(
    symbol: str,
    segment: str,
    timeframe: str,
    *,
    with_warmup: bool = True,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Load bars for a segment, optionally including backward warm-up.

    Rows before the segment start are returned with ``is_warmup=True``: they
    prime indicator state and must be excluded from trade generation and
    statistics. Nothing after the segment end is ever returned.
    """
    b = _bounds(symbol, segment)
    load_start = b.start - dt.timedelta(days=WARMUP_DAYS) if with_warmup else b.start
    lo_ms = _to_ms(load_start)
    seg_start_ms = _to_ms(b.start)
    hi_ms = _end_ms_exclusive(b.end)  # exclusive

    frames = []
    for (y, m) in _months_spanning(load_start, b.end):
        p = bars_cache_path(symbol, y, m, timeframe, data_root)
        if not p.exists():
            continue  # warm-up months may predate available data; that's fine
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(
            f"no cached bars for {symbol} {segment} {timeframe} — run scripts/build_bars.py"
        )

    bars = pd.concat(frames, ignore_index=True).sort_values("open_time")
    # HARD forward boundary: a bar may not even partially extend past segment end.
    bars = bars[(bars["open_time"] >= lo_ms) & (bars["close_time"] <= hi_ms)]
    bars = bars.reset_index(drop=True)
    bars["is_warmup"] = bars["open_time"] < seg_start_ms

    if not bars.empty:
        last = bars["close_time"].max()
        if last > hi_ms:  # belt-and-braces; the filter above already guarantees it
            raise SegmentIsolationError(
                f"bar past segment end leaked into {segment} for {symbol}"
            )
    return bars


def load_segment_funding(
    symbol: str,
    segment: str,
    *,
    with_warmup: bool = True,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Load settled funding for a segment (plus backward warm-up).

    The funding gate needs a trailing 90-day distribution, so warm-up matters
    here more than anywhere else: without it the first ~90 days of validation
    and holdout would have no usable gate at all.
    """
    b = _bounds(symbol, segment)
    load_start = b.start - dt.timedelta(days=WARMUP_DAYS) if with_warmup else b.start
    f = load_funding(symbol, data_root)
    lo, hi = _to_ms(load_start), _end_ms_exclusive(b.end)
    out = f[(f["funding_time"] >= lo) & (f["funding_time"] < hi)].reset_index(drop=True)
    out["is_warmup"] = out["funding_time"] < _to_ms(b.start)
    return out


# ---------------------------------------------------------------------------
# Public per-segment loaders
# ---------------------------------------------------------------------------


def load_research(symbol: str, timeframe: str = "15m", **kw) -> pd.DataFrame:
    """Research segment — free to explore, no access accounting."""
    return load_segment_bars(symbol, RESEARCH, timeframe, **kw)


def load_validation(symbol: str, timeframe: str = "15m", **kw) -> pd.DataFrame:
    """Validation segment. Each call counts as an OPEN against a hard budget
    (KILL_CRITERIA.md: 2-3 total). Every open is logged."""
    opens = count_opens(VALIDATION)
    if opens >= MAX_VALIDATION_OPENS:
        raise SegmentIsolationError(
            f"validation has already been opened {opens} times "
            f"(limit {MAX_VALIDATION_OPENS}); see logs/access.log. Each open "
            f"costs statistical validity — this is not a bug, it's the point."
        )
    _log_access(VALIDATION, f"symbol={symbol} timeframe={timeframe} open#{opens + 1}")
    return load_segment_bars(symbol, VALIDATION, timeframe, **kw)


def _holdout_unlocked() -> tuple[bool, str]:
    """
    Refuse to load holdout unless:
    1. hypothesis/HYPOTHESIS.md and hypothesis/KILL_CRITERIA.md are committed
    2. A git tag matching 'hypothesis-v*' exists
    3. No changes to hypothesis/ or src/ since that tag
    This is intentionally strict. Loosen only with explicit human sign-off,
    never automatically.
    """
    tags = subprocess.run(
        ["git", "tag", "--list", "hypothesis-v*"],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    ).stdout.strip().splitlines()
    if not tags:
        return False, "no 'hypothesis-v*' git tag exists"
    latest_tag = sorted(tags)[-1]
    # ITERATION_LOG.md is deliberately exempt: it is an append-only record of
    # what was TRIED, not part of the hypothesis specification. Freezing it
    # would mean the audit trail could not be written without locking the
    # holdout — the opposite of what rule 6 is protecting.
    diff = subprocess.run(
        [
            "git", "diff", latest_tag, "--",
            "hypothesis/", "src/", ":(exclude)hypothesis/ITERATION_LOG.md",
        ],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    ).stdout.strip()
    if diff:
        return False, f"hypothesis/ or src/ changed since tag {latest_tag}"

    untracked = untracked_in_frozen_paths()
    if untracked:
        return False, (
            f"{len(untracked)} untracked file(s) in hypothesis/ or src/ — commit "
            f"or remove them; uncommitted code is still code"
        )
    return True, latest_tag


def untracked_in_frozen_paths() -> list[str]:
    """Untracked files sitting in the frozen paths.

    `git diff` does NOT report untracked files, so without this an entire new
    strategy module could sit uncommitted in src/ while the gate still reported
    "unlocked" — bypassing the freeze completely. This was verified to be
    exploitable before the check was added.
    """
    out = subprocess.run(
        [
            "git", "ls-files", "--others", "--exclude-standard", "--",
            "hypothesis/", "src/",
        ],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    ).stdout.strip()
    return out.splitlines() if out else []


def load_holdout(symbol: str, timeframe: str = "15m", **kw) -> pd.DataFrame:
    """Holdout segment. Opens exactly once, at the very end, no re-tuning after.
    See KILL_CRITERIA.md and CLAUDE.md methodology rule 6."""
    unlocked, reason = _holdout_unlocked()
    if not unlocked:
        raise SegmentIsolationError(
            f"Holdout is locked: {reason}. Requires a clean git tag 'hypothesis-v*' "
            f"with no changes to hypothesis/ or src/ since. This is not a bug — "
            f"it's the point."
        )
    opens = count_opens(HOLDOUT)
    if opens >= MAX_HOLDOUT_OPENS:
        raise SegmentIsolationError(
            f"holdout has already been opened {opens} time(s); it opens exactly "
            f"once and there is no re-tuning after. See logs/access.log."
        )
    _log_access(HOLDOUT, f"symbol={symbol} timeframe={timeframe} tag={reason}")
    return load_segment_bars(symbol, HOLDOUT, timeframe, **kw)


def describe() -> str:
    lines = [f"warm-up allowance: {WARMUP_DAYS}d backward (forward: none)"]
    for sym, segs in SEGMENT_BOUNDS.items():
        lines.append(f"\n{sym}:")
        for name in (RESEARCH, VALIDATION, HOLDOUT):
            b = segs[name]
            n = (b.end - b.start).days + 1
            lines.append(f"  {name:11} {b.start} .. {b.end}  ({n}d)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print(f"\nvalidation opens so far: {count_opens(VALIDATION)}/{MAX_VALIDATION_OPENS}")
    print(f"holdout opens so far:    {count_opens(HOLDOUT)}/{MAX_HOLDOUT_OPENS}")
    ok, why = _holdout_unlocked()
    print(f"holdout unlocked: {ok} ({why})")
