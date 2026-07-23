"""Tests for segment isolation.

These tests must never consume the real validation/holdout budget, so anything
that logs an access redirects LOG_PATH to a tmp file first.
"""

from __future__ import annotations

import datetime as dt

import pytest

import scripts.segment_data as sd

PAIRS_STANDARD = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _redirect_log(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "LOG_PATH", tmp_path / "access.log")


# ------------------------------------------------- boundary constants

@pytest.mark.parametrize("symbol,total_start,total_end", [
    ("BTCUSDT", dt.date(2025, 1, 1), dt.date(2026, 6, 30)),
    ("HYPEUSDT", dt.date(2025, 6, 13), dt.date(2026, 6, 30)),
])
def test_hardcoded_bounds_match_a_50_25_25_split(symbol, total_start, total_end):
    """Re-derive the split and assert the frozen constants still match it."""
    segs = sd.SEGMENT_BOUNDS[symbol]
    n = (total_end - total_start).days + 1
    n_r, n_v = round(n * 0.50), round(n * 0.25)

    assert segs[sd.RESEARCH].start == total_start
    assert segs[sd.RESEARCH].end == total_start + dt.timedelta(days=n_r - 1)
    assert segs[sd.VALIDATION].start == segs[sd.RESEARCH].end + dt.timedelta(days=1)
    assert segs[sd.VALIDATION].end == segs[sd.VALIDATION].start + dt.timedelta(days=n_v - 1)
    assert segs[sd.HOLDOUT].start == segs[sd.VALIDATION].end + dt.timedelta(days=1)
    assert segs[sd.HOLDOUT].end == total_end


@pytest.mark.parametrize("symbol", list(sd.SEGMENT_BOUNDS))
def test_segments_are_contiguous_and_non_overlapping(symbol):
    segs = sd.SEGMENT_BOUNDS[symbol]
    r, v, h = segs[sd.RESEARCH], segs[sd.VALIDATION], segs[sd.HOLDOUT]
    assert r.start < r.end < v.start < v.end < h.start < h.end
    assert v.start == r.end + dt.timedelta(days=1)
    assert h.start == v.end + dt.timedelta(days=1)


def test_holdout_is_the_most_recent_segment():
    """Oldest -> newest: holdout must be the newest data, never the oldest."""
    for symbol, segs in sd.SEGMENT_BOUNDS.items():
        assert segs[sd.HOLDOUT].end >= segs[sd.VALIDATION].end
        assert segs[sd.VALIDATION].end >= segs[sd.RESEARCH].end


def test_hype_window_starts_after_its_listing_date():
    """HYPE listed 2025-05-30T10:30Z; its research window must start after that
    with the new-listing artifact days dropped."""
    assert sd.SEGMENT_BOUNDS["HYPEUSDT"][sd.RESEARCH].start > dt.date(2025, 5, 30)


def test_unknown_symbol_or_segment_raises():
    with pytest.raises(sd.SegmentIsolationError):
        sd._bounds("DOGEUSDT", sd.RESEARCH)
    with pytest.raises(sd.SegmentIsolationError):
        sd._bounds("BTCUSDT", "nonsense")


def test_assert_in_range_blocks_out_of_range_dates():
    b = sd.SEGMENT_BOUNDS["BTCUSDT"][sd.RESEARCH]
    sd._assert_in_range(b.start, b.start, b.end, sd.RESEARCH)  # ok
    with pytest.raises(sd.SegmentIsolationError, match="blocked by design"):
        sd._assert_in_range(
            b.end + dt.timedelta(days=1), b.start, b.end, sd.RESEARCH
        )


# ------------------------------------------------- forward isolation (real data)

@pytest.mark.parametrize("segment", [sd.RESEARCH])
def test_no_bar_past_segment_end_on_real_data(segment):
    bars = sd.load_segment_bars("BTCUSDT", segment, "15m")
    b = sd.SEGMENT_BOUNDS["BTCUSDT"][segment]
    hi = sd._end_ms_exclusive(b.end)
    assert bars["close_time"].max() <= hi
    assert not bars.empty


def test_warmup_rows_are_flagged_and_precede_segment_start():
    bars = sd.load_segment_bars("BTCUSDT", sd.VALIDATION, "15m", with_warmup=True)
    b = sd.SEGMENT_BOUNDS["BTCUSDT"][sd.VALIDATION]
    seg_start = sd._to_ms(b.start)
    assert bars["is_warmup"].any(), "expected warm-up rows before validation start"
    assert (bars.loc[bars.is_warmup, "open_time"] < seg_start).all()
    assert (bars.loc[~bars.is_warmup, "open_time"] >= seg_start).all()
    # warm-up must not reach further back than allowed
    earliest_allowed = sd._to_ms(b.start - dt.timedelta(days=sd.WARMUP_DAYS))
    assert bars["open_time"].min() >= earliest_allowed


def test_warmup_can_be_disabled():
    bars = sd.load_segment_bars("BTCUSDT", sd.VALIDATION, "15m", with_warmup=False)
    assert not bars["is_warmup"].any()


def test_warmup_never_extends_forward():
    """Warm-up reaches backward only; the forward edge is identical either way."""
    with_w = sd.load_segment_bars("BTCUSDT", sd.VALIDATION, "15m", with_warmup=True)
    without = sd.load_segment_bars("BTCUSDT", sd.VALIDATION, "15m", with_warmup=False)
    assert with_w["close_time"].max() == without["close_time"].max()


def test_research_and_validation_evaluation_rows_do_not_overlap():
    r = sd.load_segment_bars("BTCUSDT", sd.RESEARCH, "15m", with_warmup=False)
    v = sd.load_segment_bars("BTCUSDT", sd.VALIDATION, "15m", with_warmup=False)
    assert r["close_time"].max() <= v["open_time"].min()


# ------------------------------------------------- access gating

def test_holdout_is_locked_without_a_git_tag(monkeypatch, tmp_path):
    _redirect_log(monkeypatch, tmp_path)
    unlocked, reason = sd._holdout_unlocked()
    if not unlocked:
        with pytest.raises(sd.SegmentIsolationError, match="Holdout is locked"):
            sd.load_holdout("BTCUSDT")
    else:  # a tag exists and tree is clean - then the open-budget still applies
        assert isinstance(reason, str)


def test_validation_budget_is_enforced(monkeypatch, tmp_path):
    _redirect_log(monkeypatch, tmp_path)
    assert sd.count_opens(sd.VALIDATION) == 0
    for i in range(sd.MAX_VALIDATION_OPENS):
        sd.load_validation("BTCUSDT")
        assert sd.count_opens(sd.VALIDATION) == i + 1
    with pytest.raises(sd.SegmentIsolationError, match="already been opened"):
        sd.load_validation("BTCUSDT")


def test_every_validation_open_is_logged_with_commit(monkeypatch, tmp_path):
    _redirect_log(monkeypatch, tmp_path)
    sd.load_validation("BTCUSDT")
    line = sd.LOG_PATH.read_text().strip().splitlines()[0]
    ts, segment, commit, note = line.split("\t")
    assert segment == sd.VALIDATION
    assert len(commit) >= 7  # a real commit hash (or 'no-git')
    assert "BTCUSDT" in note
    dt.datetime.fromisoformat(ts)  # parses
