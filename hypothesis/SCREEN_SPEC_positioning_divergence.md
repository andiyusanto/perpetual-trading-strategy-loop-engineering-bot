# SCREEN PRE-REGISTRATION — Positioning divergence (retail vs top trader)

**Written 2026-07-23, BEFORE any positioning data was examined.** The only thing
inspected so far is the schema and granularity of a single metrics file (column
names, row count) — no ratio values, no returns, no relationship to price. Every
choice below is fixed now so none of it can be retrofitted after seeing results.

This is a SCREEN spec, not a hypothesis. Passing it earns the right to write
HYPOTHESIS v2; it does not authorise a backtest.

---

## 1. Mechanism

Binance publishes two positioning cohorts. When the retail-heavy cohort is
positioned very differently from the top-trader cohort, the informed side and
the uninformed side disagree. The claim: price subsequently follows the
informed side.

**Edge category**: informational (informed vs uninformed disagreement).

**Honest caveat about category**: the operational thesis — "the crowd is
positioned wrong, fade it" — is the same one behind the funding-extreme screen
that just failed to replicate. This may be a second proxy for one idea rather
than a genuinely independent family. Section 8 tests that explicitly, and the
answer determines whether this gets a fresh budget or inherits the funding
lineage.

## 2. Data source (VERIFIED, not assumed)

`data.binance.vision`, `data/futures/um/daily/metrics/<SYMBOL>/<SYMBOL>-metrics-YYYY-MM-DD.zip`

Verified by downloading and opening one real file (BTCUSDT 2024-06-10):

| Fact | Value |
|---|---|
| Columns | `create_time, symbol, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio` |
| Granularity | **5 minutes** (288 rows/day) |
| BTCUSDT history | from **2020-09-01** |
| File size | ~12 KB/day zipped |
| Rollup | **daily files only** — no monthly |

This contradicts CLAUDE.md's data table, which says Open Interest is "last 30
days only / NOT usable". That is true of the REST endpoint but false for the
archive. CLAUDE.md must be corrected.

**Practical cost**: daily-only means ~1,580 files for the screening window. It
is latency-bound, not bandwidth-bound (~19 MB total), so expect ~1-2 hours.

## 3. Signal construction — EXACT

Two top-trader columns exist and they are different signals. This run uses
**account-based vs account-based** so the two cohorts are measured the same way:

```
retail_t = count_long_short_ratio              (global accounts, retail-heavy)
top_t    = count_toptrader_long_short_ratio    (top trader accounts)

D_t = ln(retail_t) - ln(top_t)
```

Log difference because these are ratios: it is symmetric about 0, and `D_t = 0`
means both cohorts are positioned identically. `D_t > 0` means retail is more
long than top traders.

`sum_toptrader_long_short_ratio` (position/notional-weighted) is **deliberately
not used** in this run. It is a different signal and would be a separate logged
combination.

**Extremity**: percentile rank of `D_t` within a **trailing 30-day window**
(~8,640 observations at 5-min granularity — ample for a percentile, and
positioning regimes shift faster than the funding basis). Rank uses the same
tie-aware midrank already implemented for funding.

**Threshold**: top/bottom **decile** (>=90th / <=10th percentile) — matching the
funding-gate convention already in the codebase rather than inventing a new one.

## 4. Episode de-duplication — MANDATORY

5-minute data yields roughly 62,000 threshold-exceeding rows from a few hundred
genuinely independent episodes. Treating those as independent samples would
produce spuriously tight confidence intervals and a **false pass**. So:

1. A signal fires only on the **crossing INTO** an extreme zone.
2. It does not re-arm until `D_t` exits to the neutral band (percentile rank
   between the 25th and 75th) and then re-enters. (Hysteresis, so chattering
   around the threshold cannot generate signals.)
3. Additionally, no two signals **in the same direction within 24 hours**.

The screen must report episode count alongside raw row count.

## 5. Direction convention

| Condition | Reading | Direction |
|---|---|---|
| `D_t` >= 90th pct | retail unusually long *relative to* top traders | **-1 (short)** |
| `D_t` <= 10th pct | retail unusually short *relative to* top traders | **+1 (long)** |

`D_t` is relative, so absolute levels do not create ambiguity: if both cohorts
are long but retail far more so, `D_t` is high and the signal is short. That is
the intended reading.

## 6. Horizons

`60m, 240m, 480m, 1440m, 2880m` (1h, 4h, 8h, 1d, 2d).

7d is excluded on purpose: in the funding screen it was underpowered at every
pair and contributed nothing but a wide interval.

## 7. Costs — and funding carry

- **Cost bar: 16 bps** round trip (2 x 5bps taker + 2 x 3bps slippage), same as
  every other screen.
- **Funding carry is INCLUDED for horizons >= 8h.** A perp held across
  settlements has real carry, and for horizons of 1-2 days it can exceed the
  cost bar. Omitting it measures price return, not P&L.

```
carry_bps = -direction * sum(funding_rate over settlements in (t, t+h]) * 1e4
```

(A long pays when the rate is positive; a short receives.) Forward return
reported to the screen is `price_return_bps + carry_bps`.

## 8. Independence check vs the funding mechanism — RUN FIRST

Before this is treated as a new family, measure overlap with the funding-extreme
signal on the same timestamps:

- fraction of positioning episodes falling within +/-8h of a funding-extreme event
- Spearman correlation between the `D_t` percentile and the funding percentile

**Pre-committed**: if overlap > 50%, this is a proxy for the funding mechanism.
It then inherits that lineage and does NOT get a fresh 8-combination budget.

## 9. Look-ahead safeguard

`create_time` semantics are unverified: it is unknown whether the 00:05 row was
published at 00:05 or computed over the prior interval and released later. Until
demonstrated otherwise:

**A signal stamped at T is actionable only at T + 5 minutes (one bar lag).**

Removing that lag later requires evidence and counts as a new run.

## 10. Pass / fail — PRE-COMMITTED

**PASS** requires ALL of, on BTCUSDT:
1. at least one pre-registered horizon whose block-bootstrap CI95 **lower bound
   exceeds 16 bps**, and
2. that horizon is **powered** (MDE <= 16 bps, cluster-aware SE), and
3. permutation **p < 0.01** at that horizon, and
4. sign consistent in **>= 2 of 3** sub-periods.

**FAIL — record as disproven, stop:** BTC is powered at >= 1 horizon and no
horizon clears the cost bar. Do not then test ETH/SOL/HYPE; a mechanism that
fails on the deepest, cleanest venue is not rescued by a thinner one.

**INCONCLUSIVE — stop and report:** no horizon powered. Any remedy (more
history, wider threshold) is a NEW logged combination and a human decision, not
a silent retry.

## 11. What will NOT be done

- No alternative percentile, window, or metric variant without logging it as a
  new combination in ITERATION_LOG.md.
- No other pair until BTCUSDT yields a decision.
- No backtest engine, entry rules, stops or R:R unless the screen PASSES.
- No use of the position-weighted top-trader column in this run.

## 12. Scope and budget

- **Screening window: 2020-09-01 .. 2024-12-31**, using `data/screening/`. The
  2025-2026 research/validation/holdout segments stay untouched, so anything
  that survives can still be tested against genuinely unseen data.
- This is **combination 1** of a new family, budget 8 — *subject to section 8*,
  which may make it an extension of the funding lineage instead.
- Screens run to date: 9 (all logged in `results/screening_log.jsonl`). A future
  significance claim must account for that search.

---

# OUTCOME — 2026-07-23 (appended after the run; nothing above was edited)

Section 8 proxy check: **INDEPENDENT** (0/11 episodes at a funding extreme,
Spearman -0.127, p=0.71).

Power pre-check on a 181-day sample: 11 episodes, i.e. ~96 for BTC over the full
window and ~288 across three pairs, against the ~515+ needed to reach
MDE <= the cost bar. **Result: SHELVED — untestable at power on available data.**

The full download was not made. The pre-registered threshold and
de-duplication rule were NOT relaxed. See ITERATION_LOG.md for detail.
