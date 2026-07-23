# ITERATION LOG

Every parameter/indicator combination tried, including — especially — the ones
that failed. CLAUDE.md methodology rule 8: the v1 budget is **8 combinations**;
exhausting it without a robust pass is a valid negative result, not a licence to
keep searching within the same hypothesis family.

"Combination" counts the *rule*, not each number a data-derived rule produces.
Per-fold calibrated thresholds under one fixed calibration rule = 1 combination;
changing the percentile, the statistic, or the window = a new one.

| # | Date | Segment | Configuration | Rationale | Outcome |
|---|------|---------|---------------|-----------|---------|
| 1 | 2026-07-23 | research | `cvd_only` — Layer 1 structural swing divergence alone | Component attribution baseline (KILL_CRITERIA: "CVD-only" must be measured before combining) | **INVALID — exit defect** (see below) |
| 2 | 2026-07-23 | research | `cvd + spearman` — Layers 1+2 | Isolate Layer 2's marginal contribution | **INVALID — exit defect** |
| 3 | 2026-07-23 | research | `cvd + roc` — Layers 1+3 | Isolate Layer 3's marginal contribution | **INVALID — exit defect** |
| 4 | 2026-07-23 | research | `cvd + funding` — Layer 1 + funding gate | Isolate the funding gate's marginal contribution | **INVALID — exit defect** |
| 5 | 2026-07-23 | research | `full_combined` — Layers 1+2+3 + funding gate | The v1 hypothesis as specified | **INVALID — exit defect** |

**Budget used: 5 / 8** — but runs 1-5 are VOID (see below), so they are not
counted as spent search. Re-running the identical 5 configurations after a
defect fix is the same 5 combinations, not 10.

## Runs 1-5 VOIDED — 2026-07-23

All five research runs are invalid as tests of the hypothesis. The early-exit
rule was implemented as a SIGN test on CVD ROC, while HYPOTHESIS.md v1
specifies "re-accelerates" (a magnitude condition). CVD ROC is a zero-mean
oscillator on this data (49.7% of bars positive, mean same-sign run 5.88 bars),
so the exit fired on ~72% of trades within ~1 hour and only 39/575 trades ever
reached target. The exit rule produced the outcome, not the signal.

Recorded numbers, for the record (all far below the 40% breakeven, but NOT
evidence about the hypothesis):

| Config | Trades | Win rate | Expectancy | PF |
|---|--:|--:|--:|--:|
| 1 cvd_only | 575 | 19.1% | -0.428 R | 0.26 |
| 2 +spearman | 278 | 16.2% | -0.504 R | 0.21 |
| 3 +roc | 152 | 13.8% | -0.465 R | 0.18 |
| 4 +funding | 65 | 21.5% | -0.376 R | 0.53 |
| 5 full | 7 | 14.3% | -0.884 R | 0.15 |

Corrected under HYPOTHESIS.md v1.1 (magnitude+direction early exit,
`reaccel_ratio` calibrated per fold at the 75th percentile, mirroring the
locked 25th-percentile decel rule). Re-running the same 5 configurations.

| # | Date | Segment | Configuration | Rationale | Outcome |
|---|------|---------|---------------|-----------|---------|
| 1r-5r | 2026-07-23 | research | same 5 attribution configs, under v1.1 | Re-test after defect fix | pending |

Fixed across all five (not separate combinations — they are the locked v1
baseline from HYPOTHESIS.md): swing 3/3, CVD window 20, Spearman window 20,
ROC window 7, peak lookback 20, R:R 1.5, time stop 10x15m, funding
trailing-90d/decile, walk-forward 2m train / 1m test sliding 1m, thresholds
calibrated per fold on the training window at the 25th percentile.

Costs applied from run #1: taker 5bps, maker 2bps, slippage 3bps (round trip
~16bps). No pre-cost number is recorded anywhere.

## Notes / observations

- Layer 3's *placeholder* `decel_ratio=0.5` fired on ~64% of bars in a
  pre-lock functional check — far too permissive to filter anything. This is
  why calibration is percentile-based per fold rather than a fixed constant.
- In the same pre-lock check the funding gate opened on 0/138 bearish vs
  22/120 bullish divergences (Q1 2025 BTC). Flagged for the adversarial review
  pass; NOT tuned, since that would be fitting to research data.
- These entries were written when the runs were launched, before results were
  read, so the log records what was *tried*, not a post-hoc selection.
