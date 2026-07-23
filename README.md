# Perpetual Trading Strategy — Loop Engineering Bot

Multi-strategy (scalping / day trade / swing) perpetual futures trading bot,
built on a nested control-loop architecture. See `CLAUDE.md` for the full
project brief, methodology rules, and build status — read that file first,
every session.

**Current phase: research.** No live execution code exists, and none should be
written until a hypothesis passes the gates in `hypothesis/KILL_CRITERIA.md`
against a holdout set.

## Status — what has actually been tested

| Mechanism | Verdict | Where |
|---|---|---|
| CVD divergence + funding gate (HYPOTHESIS v1.1) | **DISPROVEN** — zero forward information at every horizon 30min–5h (\|t\| ≤ 0.66) | `hypothesis/ITERATION_LOG.md` |
| Funding-rate extremes as a primary signal | **Not tradeable** — real effect on BTC (perm p 0.005) but never clears the cost bar, and inverts on SOL | `results/screening_log.jsonl` |
| Positioning divergence (retail vs top trader) | **Shelved at power** — independent of funding, but ~288 episodes available vs ~515 needed | `hypothesis/SCREEN_SPEC_positioning_divergence.md` |

**Validation and holdout have never been opened** (0/3 and 0/1). Three mechanisms
were examined without spending any out-of-sample budget. That is the point.

## The research order (learned the hard way)

The first cycle built a complete backtest engine — entries, stops, walk-forward,
costs — before asking whether the signal carried any information. It didn't, and
the question took seconds to answer once asked properly.

**So: screen first, build second.**

1. `src/research/screen.py` — does the signal move price more than chance AND
   more than it costs to trade? No stops, no R:R, no exits; those are choices
   that can manufacture a result.
2. Only if a signal passes: pre-register a spec, lock it, then build.

A screen reports a minimum detectable effect alongside every number, so a null
is only called a rejection when the test could actually have *seen* an effect.
Every screen ever run is appended to `results/screening_log.jsonl` — passes and
failures alike — because cheap screening makes p-hacking cheap.

## Known constraints (measured, not assumed)

- **Round-trip cost ≈ 11 bps** = 2×5 bps taker + 2×0.5 bps impact allowance.
  The quoted BTCUSDT spread is one tick 89.2% of the time (0.014 bps), so fees
  dominate. Any candidate edge must clear this.
- **Impact at size is unmeasured.** `slippage_bps` is an allowance, not a
  measurement; aggTrades carries no resting-order sizes. A 1.0 BTC order is
  ~333× the median trade.
- **The archive beats the REST API.** Full-history funding, klines, and 5-min
  OI/positioning metrics are all archived; the 30-day limit applies only to the
  REST endpoints. See the data table in `CLAUDE.md`.
- **`fapi.binance.com` may be DNS-blocked** on some networks. `src/ingest/binance_dns.py`
  resolves it out-of-band via DoH; TLS verification stays intact.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest -q                       # 121 tests
```

## Workflows

```bash
# 1. raw tick data for backtesting (large: ~6 GB per pair-18mo)
python scripts/download_data.py --symbol BTCUSDT --start 2025-01 --end 2026-06

# 2. derive OHLCV+delta bars from aggTrades (cached, resumable)
python scripts/build_bars.py --symbol BTCUSDT --start 2025-01 --end 2026-06 \
    --timeframes 15m,5m

# 3. cheap long-history data for SCREENING (klines+funding, ~7 MB for 5y x 3 pairs)
python scripts/download_history.py --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --start 2020-01 --end 2024-12

# 4. screen a candidate signal before building anything
python scripts/screen_funding.py

# 5. walk-forward backtest with component attribution (only after a screen passes)
python scripts/run_backtest.py --symbol BTCUSDT --segment research

# segment boundaries and access budget
python scripts/segment_data.py
```

## Structure

```
├── CLAUDE.md                     # read first, every session
├── hypothesis/
│   ├── HYPOTHESIS.md             # locked + git-tagged before any backtest
│   ├── KILL_CRITERIA.md          # numeric gates
│   ├── ITERATION_LOG.md          # every combination tried, especially failures
│   └── SCREEN_SPEC_*.md          # pre-registered screen specs
├── data/
│   ├── raw/                      # immutable aggTrades + funding (gitignored)
│   ├── interim/                  # derived bars, reproducible (gitignored)
│   ├── screening/                # long-history klines/funding/metrics (gitignored)
│   ├── research/                 # oldest 50% — free to explore
│   ├── validation/               # next 25% — max 3 opens, metered
│   └── holdout/                  # most recent 25% — opens once, git-tag gated
├── scripts/
│   ├── download_data.py          # aggTrades + funding -> parquet
│   ├── download_history.py       # klines + funding for screening
│   ├── build_bars.py             # aggTrades -> OHLCV+delta bars
│   ├── screen_funding.py         # example screen runner
│   ├── run_backtest.py           # walk-forward + component attribution
│   └── segment_data.py           # hard segment isolation + access metering
├── src/
│   ├── ingest/                   # archive client, DoH pin, funding REST
│   ├── core/                     # config, CVD reconstruction
│   ├── research/                 # signal screening harness
│   ├── strategy/                 # divergence layers, funding gate
│   ├── backtest/                 # costs, engine, walk-forward, metrics
│   ├── risk/ execution/          # future phase — intentionally empty
│   └── persistence/
├── results/
│   └── screening_log.jsonl       # append-only record of every screen
├── tests/
└── logs/
    └── access.log                # validation/holdout opens, with commit hash
```

## Methodology guarantees enforced in code

Not conventions — these fail loudly:

- Segment loaders **raise** on out-of-range dates. Forward isolation is absolute;
  backward warm-up is bounded to 90 days and flagged.
- The holdout gate refuses to open unless a `hypothesis-v*` tag exists with no
  changes to `hypothesis/` or `src/` since — **including untracked files**
  (verified bypassable before that check was added).
- Validation opens are metered and logged with a commit hash.
- No code path produces a pre-cost PnL.
- Rolling CVD **refuses** to compute over a gapped bar series rather than
  silently mis-windowing.
- Look-ahead is tested, not asserted: truncation-invariance tests confirm no
  layer changes its past output when future data is appended.
