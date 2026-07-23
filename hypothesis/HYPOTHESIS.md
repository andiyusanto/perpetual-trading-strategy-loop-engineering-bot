# HYPOTHESIS — Day Trade: CVD Divergence + Funding Rate Filter

**Version**: v1
**Status**: DRAFT — lock this (commit + git tag `hypothesis-v1`) before any
backtest is run. Do not edit in place after locking; add a new version below.

---

## Core mechanism (why this should work, structurally)

When cumulative volume delta (CVD) diverges from price at a structural swing
point — price makes a new high/low but CVD fails to confirm it — the side
pushing price in that direction is running out of aggressive participation.
Combined with an extreme funding rate (positioning crowded in the direction
price has been moving), this indicates the crowded side is vulnerable to being
forced out, which should produce a reversal that can be captured.

**Participant being exploited**: traders crowded into a funding-extreme
position whose flow is decelerating relative to price — a structural /
mechanical edge category (forced-flow), not behavioral or informational.

This is a REGULAR divergence only in v1 (not hidden divergence — that is a
separate hypothesis for a later version if v1 fails or partially works).

---

## CVD reconstruction spec

*Added 2026-07-23. This mechanism was previously referenced in CLAUDE.md as
"the rolling window reset strategy already designed" but was never actually
written down. It is specified here BEFORE any backtest is run, while this
document is still DRAFT/unlocked. It is a specification of a gap, not a change
to a tested claim.*

1. **Taker side.** `is_buyer_maker=False` → the taker bought → **buy volume**.
   `is_buyer_maker=True` → the taker sold → **sell volume**. Volumes are in
   base-asset units (e.g. BTC).

2. **Per-bar delta.** `delta = buy_vol − sell_vol`, aggregated from tick-level
   aggTrades onto a UTC-aligned bar grid (15m confirmation, 5m entry timing).
   A bar contains exactly the trades with `bar_open <= timestamp < bar_open + tf`.

3. **Anchoring — rolling window (the "reset").** CVD at bar *t* is the
   **trailing N-bar sum of delta**:

       CVD_t = Σ delta over bars [t-N+1 .. t]

   The window slides forward, so old flow drops out of the sum. This keeps CVD
   bounded and comparable across time and avoids an arbitrary session boundary
   in a 24/7 market. There is no daily/session reset and no unbounded
   cumulative sum from the dataset start.

4. **N (window length).** Baseline **N = 20 bars**, chosen to match Layer 2's
   rolling 20-candle Spearman lookback so both layers observe the same horizon.
   N is a real parameter: it counts against the multiple-testing budget in
   KILL_CRITERIA.md and is subject to the ±20%/±40% sensitivity gate
   (sensitivity set: 14 / 20 / 30).

5. **Warm-up.** The first N−1 bars have an incomplete window and are emitted as
   NaN, never as partial sums — a partial window is a different statistic and
   must not be compared against full ones. Downstream layers skip NaN rows.

6. **No look-ahead.** The window is strictly trailing (current bar + N−1 prior).
   CVD at bar *t* is unchanged by any later bar.

7. **No-trade bars.** Bars with no trades are filled as zero-flow
   (`delta = 0`, price flat at the prior close) before rolling, so an N-bar
   window is always a fixed N × timeframe horizon. On illiquid symbols an
   unfilled gap would silently stretch the window; the implementation raises
   rather than rolling over a gapped series.

## Signal layers

1. **Structural swing (Layer 1)** — fractal swing high/low, 15m timeframe,
   minimum 3 candles either side (test 2/3/5 as a sensitivity check within
   research set only). Regular divergence: price HH + CVD LH (or price LL +
   CVD HL).

2. **Spearman correlation breakdown (Layer 2)** — rolling 20-candle window
   (test 14/20/30 as sensitivity check). Breakdown threshold: correlation
   drops below 0.3 (calibrate from the actual historical distribution in the
   research set — 0.3 is a starting point, not fixed dogma; document the
   calibrated value here once determined).

3. **CVD ROC deceleration (Layer 3)** — rate of change of CVD over a 5–10
   candle window, must show deceleration relative to the prior peak ROC
   (threshold to be calibrated from research-set distribution, documented here
   once determined).

4. **Funding rate filter (gate, not a signal)** — entry only permitted when
   funding rate is in the extreme percentile (top/bottom ~10% of the trailing
   90-day distribution, per-pair — not an absolute fixed number, since pairs
   differ in typical funding range).

## Entry / exit (v1 baseline — adjust only via a new hypothesis version)

- Entry trigger: Layers 1+2+3 all confirm within the same window AND funding
  filter does not veto.
- Entry timing: confirm at 15m, execute at the next closed 5m candle.
- Stop loss: structural — beyond the swing point that formed the divergence.
- Take profit: R:R 1:1.5 baseline.
- Early exit: CVD ROC re-accelerates in the direction of the original
  (pre-reversal) trend before TP is hit.
- Time stop: 8–12 candles (15m) with no significant movement.

## Pairs and independence note

BTCUSDT is the base case. ETHUSDT/SOLUSDT/HYPEUSDT are robustness checks, not
sample-size padding — log entry timestamps per pair so time-correlated entries
across pairs can be treated as clustered events, not independent trades, in
any significance test.

---

## Falsification — what would prove this wrong

See `KILL_CRITERIA.md` for the numeric gates. In prose: if, after testing each
layer independently and the combined signal across multiple research-set
windows, there is no combination that clears the win-rate floor after real
costs, with parameter sensitivity that isn't a fragile knife-edge — the
mechanism does not hold on this pair/timeframe/exchange, and should be
recorded as a disproven hypothesis, not iterated indefinitely.
