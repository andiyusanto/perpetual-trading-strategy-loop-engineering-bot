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
| 1r | 2026-07-23 | research | `cvd_only` under v1.1 | attribution baseline | FAIL — 560 trades, 32.3% WR [28.6,36.2], -0.413R, PF 0.41 |
| 2r | 2026-07-23 | research | `cvd + spearman` | Layer 2 contribution | FAIL — 274, 31.8% [26.3,37.2], -0.452R, PF 0.42 |
| 3r | 2026-07-23 | research | `cvd + roc` | Layer 3 contribution | FAIL — 150, 31.3% [24.0,38.7], -0.485R, PF 0.32 |
| 4r | 2026-07-23 | research | `cvd + funding` | funding gate contribution | FAIL — 65, 32.3% [21.5,44.6], -0.387R, PF 0.56 |
| 5r | 2026-07-23 | research | `full_combined` | v1 hypothesis as specified | FAIL — 7 trades (below 100 minimum), 14.3%, -1.272R, PF 0.07 |

### Result of runs 1r-5r (valid test, HYPOTHESIS v1.1)

The defect fix was material and worked as intended: `early_roc` exits fell from
72% to 19%, mean hold rose 8.7 -> 17.8 bars, win rate rose 19.1% -> 32.3%. The
test is now a fair one — and the hypothesis fails it.

Gates failed (KILL_CRITERIA.md):
- **Win-rate floor**: all configs below the 40% breakeven; 1r-3r have 95% CIs
  entirely below 40%.
- **Profit factor**: 0.07-0.56, all below the 1.10 trigger.
- **Component attribution**: layers SUBTRACT value. CVD-only (-0.413R) beats the
  full stack (-1.272R), and the full stack collapses the sample to 7 trades.
  Per KILL_CRITERIA this alone says "simplify rather than keep them".
- **Sample size**: 5r's 7 trades is below the 100-trade minimum.
- **Cost realism**: fees are 113-193% of gross PnL. No configuration's raw edge
  covers the ~16bps round trip.

**Open question for the adversarial review (NOT acted on here):** at R:R 1.5 a
random entry wins ~40% of the time. Observed 32.3% is materially WORSE than
random, which is a signature worth explaining before the hypothesis is buried.
Candidate causes: (a) the mechanism is genuinely anti-predictive on this
pair/timeframe; (b) entry is ~50 minutes after the swing (3 bars to confirm the
fractal + next 5m close), so the structural stop sits unusually close to entry
by fill time. Deliberately not "fixed" here — changing entry timing after seeing
results would be result-chasing, and would need a new hypothesis version.

**Budget used: 5 / 8.** Validation and holdout remain UNOPENED (0/3 and 0/1) —
correctly, since nothing passed research.

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

---

## Adversarial review — 2026-07-23 (independent pass)

Conducted by a separate reviewer that did not build the engine, per methodology
rule 9. It reproduced the run exactly (identical skip counters and trade counts)
before analysing, and modified nothing in `src/` or `hypothesis/`.

### VERDICT: the negative result is SOUND. The mechanism is disproven.

Machinery-independent test — forward return from all 705 cvd_only signals,
signed by intended direction, with no exit rules, no costs, no R:R involved:

| horizon | mean | t | frac > 0 |
|---|--:|--:|--:|
| 30 min | -0.01 bps | -0.01 | 0.495 |
| 60 min | -0.50 bps | -0.28 | 0.493 |
| 150 min | -0.29 bps | -0.10 | 0.488 |
| 300 min | -2.44 bps | -0.66 | 0.505 |

CVD divergence at 15m fractal swings on BTCUSDT carries **zero directional
information** at every horizon tested, against a 16 bps round trip. This is
independent of every engine choice, so the failure is a property of the
mechanism, not of the implementation. **No look-ahead or leakage was found**
(truncation-invariance 186/186, 265/265, 340/340 exact; 15m->5m mapping max
lookahead 0 ms; per-fold calibration confirmed isolated).

### CLAIMS WITHDRAWN — earlier entries in this log were wrong

1. **"Materially worse than random" — WITHDRAWN.** The permutation test that
   KILL_CRITERIA requires had never actually been run (`run_backtest.py` never
   called `permutation_test_vs_random_entries`). Measured over 200 reps through
   the same simulate() machinery, the random-entry baseline is **27.2% win /
   -0.388 R**, not 40%. The strategy BEATS random on win rate (p = 0.005) and
   TIES it on expectancy. The error was comparing a post-cost win rate against a
   pre-cost breakeven — an inconsistency that also exists in the code
   (`metrics.breakeven_win_rate` has no cost term but is printed beside a net
   win rate).

2. **"Layers SUBTRACT value" — WITHDRAWN.** That rested on comparing -0.413 R
   (n=560) with -1.272 R (n=7). n=7 is uninterpretable. The supportable
   statement is that every layer is zero-information: zero-cost expectancy
   -0.004, -0.013, -0.057, +0.020 R for configs 1-4.

3. **"Fees are 113-193% of gross PnL" — WITHDRAWN as meaningless.** The ratio
   explodes as gross -> 0 (it reads 6667% for config 4). The stable statement is
   **cost 0.42 R/trade against a gross edge of 0.00 R/trade** (fees 0.263 R +
   slippage 0.158 R).

4. **Funding-gate asymmetry (0/138 bearish) — CLOSED, does not generalise.**
   Over the full research set the gate opens on 45/454 bearish (9.9%) and
   46/402 bullish (11.4%).

5. Earlier candidate cause "the mechanism is anti-predictive" — **rejected**:
   inverting the direction also loses (net -0.443 R, zero-cost -0.040 R). It is
   noise, not inversion. The ~50-minute entry lag IS confirmed as the operative
   mechanism, but its consequence is cost domination, not a bad signal.

### Gates NOT evaluated (previously left ambiguous)

These KILL_CRITERIA requirements were never run and must not be read as passed
or failed: (a) +/-20%/+/-40% parameter sensitivity, (b) regime consistency
(trending vs ranging), (c) permutation test — now run during this review,
(d) effective sample size / cross-pair clustering correction.

They are moot for the verdict: a signal with zero forward information cannot be
rescued by parameter choice.

### Engine defects recorded (NOT fixed — would require a new hypothesis version)

- `min_risk_cost_multiple = 1.0` admits trades where round-trip cost consumes
  up to 100% of risk. A 5-10x floor would have surfaced the cost problem as a
  "no trades" result instead of a mysterious win rate. 98/705 dropped.
- 30 intents (4.3%) dropped by overlap suppression with no log line; only
  `invalid_stop` and `risk_below_cost` are logged. Immaterial to the result
  (`allow_overlapping=True` gives 31.9% / -0.424 R).
- Pessimistic-intrabar rule never fires (0/560 bars contain both stop and
  target; target sits ~4.7 ATR away). The claimed conservatism is inert.
- CVD anchoring spec deviation: a trailing 20-bar rolling sum makes
  `CVD_t2 - CVD_t1` equal net flow between swings MINUS flow in the 20 bars
  before swing 1, not the classic "cumulative CVD failed to confirm"
  (correlation +0.723, flag agreement 74.7%). Both definitions give the same
  null result.
- Time stop applied unconditionally; HYPOTHESIS.md says "with no significant
  movement". Immaterial (widening the time stop reveals no edge).

### Conclusion

**HYPOTHESIS v1.1 is DISPROVEN on BTCUSDT / 15m+5m / Binance USD-M.** Not a
marginal miss, and not a cost artifact that a cheaper venue would fix: the
signal is a coin flip that pays 16 bps a round trip against a 42 bps structural
stop. Per HYPOTHESIS.md's own falsification clause this is recorded as a
disproven hypothesis and NOT iterated further within the same family.

Budget used: 5/8 — the remaining 3 are deliberately left unspent. Validation
(0/3) and holdout (0/1) were never opened.
