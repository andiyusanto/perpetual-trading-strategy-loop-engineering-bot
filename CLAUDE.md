# Perpetual Trading Strategy — Loop Engineering Bot — Claude Code Project Instructions

## What this project is

A multi-strategy (scalping / day trade / swing) perpetual futures trading bot
built on a **nested control-loop architecture** (not a single flat loop, and
not an LLM-driven decision loop). This is a **separate project** from any
prior CVD Divergence bot design — no legacy code exists to migrate.

**Phase history:**
- *Day-trade research (CONCLUDED)*: the CVD-divergence + funding day-trade
  hypothesis (v1/v1.1) was **disproven** on the research set — zero forward
  information at every horizon. Funding-extreme, positioning-divergence, and
  perp-quarterly-basis candidates were also screened out. See
  `hypothesis/ITERATION_LOG.md`. Validation/holdout were never opened.
- *Current phase: TREND-FOLLOWING live build.* Pivoted to a documented,
  cross-asset-validated trend premium (`hypothesis/HYPOTHESIS_trend_following.md`,
  tf-v1). Because our own data cannot statistically gate a multi-week strategy
  (the "power wall" — see ITERATION_LOG), the backtest here is an implementation
  check, NOT a profitability gate; real validation is forward (shadow → live
  $100). We ARE now writing execution/risk code — deliberately **risk loop and
  kill switch FIRST**, before signal code, because at $100 that is what decides
  whether a bug costs $5 or the account.

The methodology rules below still apply in full (no look-ahead, real costs, no
self-scoring, pre-registration before locking). Only the "no live code yet" gate
is lifted, and only because the research phase it gated has concluded.

---

## Architecture (target — build toward this, don't build it all at once)

Five loops, each with its own cadence, reading/writing a shared state store:

1. **Market data loop** — websocket/REST ingest, writes to shared state. Event-driven.
2. **Signal loop** — one per strategy timeframe (scalp / day trade / swing), each
   reads shared state at its own cadence. Day trade signal = 15m confirmation,
   5m entry timing.
3. **Risk loop** — independent, higher priority than signal loops. Has veto power
   over execution regardless of signal quality. Runs on its own cadence checking
   drawdown, exposure, kill-switch conditions.
4. **Execution loop** — receives trade intents, handles order placement, retry,
   slippage guard. Exchange-agnostic interface (CCXT), even though we only
   implement Binance USD-M Futures for now.
5. **Feedback loop** — slowest cadence (weekly). Trade journal analysis, rolling
   win-rate/expectancy tracking, edge decay detection. Outputs one of: reweight /
   pause / kill. Never auto-reoptimizes parameters without the change being
   logged as a new hypothesis version.

**Deterministic core.** No LLM reasoning in the signal, risk, or execution loops.
Every entry/exit decision must be reproducible: same inputs → same output, every
time, so backtests are valid. If a "smarter" filter is ever proposed, it must be
implemented as a deterministic, backtestable function — not a model call.
An LLM-based macro/news regime layer MAY be explored later as a parameter
*modifier* (e.g. tighten thresholds ahead of a known macro event), but never
as the entry/exit decision-maker itself, and never in a way that breaks
backtest reproducibility.

---

## Language & stack

- **Python 3.11+**, `asyncio` for loop concurrency
- `pandas` / `numpy` for CVD reconstruction, correlation, rolling windows
- `scipy.stats` for bootstrap/permutation significance testing
- `pydantic` (v2) for config (`BaseSettings`, overridable via `.env`)
- `ccxt` (async) — only for future exchange-agnostic execution, not needed yet
  for the research phase
- `pyarrow` / parquet for local tick-data storage (not raw CSV — too large)
- `structlog` for structured logging
- `pytest` for tests

---

## Scope for this phase

- **Exchange**: Binance USD-M Futures only. Generalize to Bybit/Bitget/OKX/MEXC
  later, after day trade strategy is validated.
- **Pairs**: BTCUSDT as the base case (cleanest liquidity, longest history).
  HYPEUSDT / SOLUSDT / ETHUSDT run in parallel as a robustness check — NOT
  pooled naively with BTC for sample-size padding. Log entry timestamps per
  pair so cross-pair time-overlap can be corrected for in significance testing
  (correlated pairs are not independent samples).
- **IMPORTANT**: Verify HYPEUSDT's actual futures listing date on Binance before
  assuming it has the full 12-18 month window. BTC/ETH/SOL are confirmed to have
  sufficient history; HYPE was not confirmed and must be checked via `exchangeInfo`
  or the earliest available file in the data.binance.vision archive before it's
  used in the research/validation/holdout split.
- **Data range**: 12–18 months, split research 50% / validation 25% / holdout 25%,
  oldest to newest (holdout = most recent).

## Data sources — confirmed availability (do not assume otherwise)

| Data | Source | Range | Use |
|---|---|---|---|
| aggTrades | `data.binance.vision` archive (USD-M Futures) | Full history | CVD reconstruction — primary |
| Funding rate | `GET /fapi/v1/fundingRate` | Full history | Funding filter |
| Klines | `data.binance.vision` archive | Full history | Reference/validation only |
| Open Interest + positioning metrics | `data.binance.vision` archive, `data/futures/um/daily/metrics/` | **Full history** (BTCUSDT from 2020-09-01), 5-min granularity, daily files only | **CORRECTED 2026-07-23** — verified from a real file. Carries `sum_open_interest`, `count_toptrader_long_short_ratio`, `sum_toptrader_long_short_ratio`, `count_long_short_ratio`, `sum_taker_long_short_vol_ratio`. The 30-day limit applies to the REST endpoint ONLY, not the archive. |
| Open Interest (REST) | `GET /futures/data/openInterestHist` | Last 30 days only | Superseded by the archive row above for backtesting |
| Liquidation feed (`!forceOrder`) | Live WebSocket only | **No historical archive exists** | NOT usable for backtest — scalping strategy (future phase) will need a third-party historical source or forward-only shadow validation |

aggTrades schema (USD-M Futures): `agg_trade_id, price, quantity, first_trade_id,
last_trade_id, timestamp, is_buyer_maker`. CVD convention: `is_buyer_maker=True`
means the taker was the seller → count as sell volume. `is_buyer_maker=False`
means the taker was the buyer → count as buy volume. CVD = running sum of
(buy volume − sell volume), resampled to the signal timeframe with the rolling
window reset strategy already designed (see HYPOTHESIS.md for the mechanism).

---

## Methodology rules — hard constraints, do not deviate

These exist to prevent data-snooping and self-delusion. Follow them exactly.

1. **Hypothesis and kill criteria are locked before touching research data.**
   `hypothesis/HYPOTHESIS.md` and `hypothesis/KILL_CRITERIA.md` must be committed
   to git before any backtest is run. If either changes after seeing results,
   it must be added as a NEW dated entry, never silently edited.

2. **Segment isolation is technical, not just intentional.** Each of
   `data/research/`, `data/validation/`, `data/holdout/` has its own data loader
   with a hardcoded allowed date range. Loading out-of-range data must raise an
   exception, not silently succeed. See `scripts/segment_data.py` for the pattern.

3. **No look-ahead bias.** Only use fully closed candles / settled funding data
   at the timestamp a decision would have been made. Never let a "confirmation"
   layer read data that would not have existed yet at entry time.

4. **Walk-forward, not a single backtest window.** Within the research set, use
   rolling windows (e.g. 2-month train / 1-month test, sliding) before touching
   validation.

5. **Validation set may be opened a limited number of times** (2–3 max, tracked
   in `logs/access.log`). Each open counts against statistical validity — treat
   it as a scarce resource, not a free re-check.

6. **Holdout set opens exactly once, at the very end, with no re-tuning after.**
   Gate this in code: `scripts/segment_data.py`'s holdout loader should refuse
   to run unless hypothesis/kill-criteria files are git-tagged and no code has
   changed since that tag. Log every access with timestamp + git commit hash.

7. **Realistic costs from the first backtest, not added later.** Every backtest
   run must include real Binance taker/maker fees and a conservative slippage
   model. Do not report a "clean" edge number before costs and call it progress.

8. **Multiple-testing discipline.** If more than ~8 parameter/indicator
   combinations are tried, note this explicitly in results and raise the bar for
   statistical significance accordingly. Log every combination tried in
   `hypothesis/ITERATION_LOG.md` (create this file when the first iteration happens),
   even failed ones — especially failed ones.

9. **No self-scoring.** Do not generate a "strategy passed, ready for live" verdict
   in the same pass that built or tuned the strategy. If asked to audit results,
   explicitly look for reasons the result might be invalid (look-ahead bias,
   overfit parameters, fragile thresholds, synthetic-data-like artifacts) before
   reporting a pass. A gate that technically failed (e.g. parameter sensitivity)
   must not be glossed over in a final verdict.

10. **Never validate against synthetic/generated data and report it as if it were
    a live-market validation.** All backtest claims must trace to real historical
    aggTrades/funding data pulled from the sources above.

11. **Component attribution before combining.** Test each of the 3 CVD divergence
    layers (structural swing / Spearman breakdown / CVD ROC) and the funding filter
    independently before evaluating the combined signal, so we know which
    component (if any) is contributing.

---

## Current status

- [x] Verify HYPEUSDT listing date / available history
      (onboardDate 2025-05-30T10:30Z, ~13mo; per-pair window, drop first days)
- [x] Download aggTrades + funding rate for BTCUSDT (12–18mo)
      (2025-01→2026-06, 869.2M aggTrades + 1,638 funding events, in data/raw/)
- [x] Implement CVD reconstruction from aggTrades (tick-level, rolling window reset)
      (src/core/cvd.py; rolling-window anchoring N=20 bars, spec'd in HYPOTHESIS.md
       2026-07-23; 18 unit tests; validated against raw on real BTC data)
- [x] Implement 3-layer divergence detection (structural swing / Spearman / ROC)
      (src/strategy/divergence.py; explicit confirmed_at_* look-ahead guards,
       22 tests incl. truncation-invariance; L2/L3 thresholds left UNCALIBRATED)
- [x] Implement funding rate filter
      (src/strategy/funding_filter.py; per-pair trailing-90d percentile gate,
       tie-aware midrank, direction-aware, abstains on thin windows; 14 tests)
- [x] Implement `scripts/segment_data.py` with hard isolation
      (hardcoded per-pair 50/25/25 bounds; absolute forward isolation +
       bounded 90d BACKWARD warm-up only; validation budget 3, holdout 1 +
       git-tag gate; 18 tests)
- [x] Lock `hypothesis/HYPOTHESIS.md` + `hypothesis/KILL_CRITERIA.md`, commit + tag
      (tags hypothesis-v1, then hypothesis-v1.1 after the early-exit defect fix)
- [x] Backtest on research set (walk-forward)
      (BTCUSDT, 7 folds, 5 attribution configs — all fail. Best 32.3% WR,
       expectancy -0.413R. NB: the 40% breakeven printed by the run is
       pre-cost and does NOT apply; measured random baseline is 27.2%/-0.388R
       and cost-adjusted breakeven is ~57%. Every layer is zero-information
       (zero-cost expectancy -0.004 to +0.020 R). See ITERATION_LOG.md for
       claims withdrawn after review. Budget used 5/8.)
- [ ] Validate once/twice on validation set  <-- NOT opened: nothing passed research
- [x] Adversarial review pass (separate from the building pass)
      (independent pass, 2026-07-23: verdict SOUND — mechanism carries zero
       forward information at 30min-5h, |t|<=0.66. No look-ahead found. It
       also WITHDREW 3 incorrect claims from the building pass — see
       hypothesis/ITERATION_LOG.md)
- [ ] Open holdout once — final verdict  <-- DELIBERATELY NOT OPENED.
      v1.1 is disproven on the research set; there is no candidate to test.
      The holdout stays sealed (0/1) for a future hypothesis version.

Do not jump ahead of this list. If asked to "build the bot," start from the top
unchecked item.
