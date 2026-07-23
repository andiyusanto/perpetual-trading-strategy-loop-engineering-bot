# KILL CRITERIA — Day Trade: CVD Divergence + Funding Rate Filter

**Version**: v1 — LOCKED 2026-07-23 alongside `HYPOTHESIS.md` v1 as git tag
`hypothesis-v1`, before any backtest was run. Committed + tagged before
running any backtest. Do not edit after locking; add a new dated version if
criteria genuinely need to change (and say why).

---

## Sample size

- Minimum 100 trades in the research set before any conclusion is drawn.
- Cross-pair trades are logged with entry timestamps; if effective independent
  sample size (after clustering correction) falls below 100, more data/time is
  needed before concluding — do not substitute raw trade count for effective
  sample size.

## Win rate floor

- With baseline R:R 1:1.5, breakeven win rate ≈ 40%. Strategy must exceed this
  with a margin sufficient to survive the confidence interval width at n≈100+
  trades — report the actual CI, don't just compare point estimates.

## Cost realism

- Every backtest run includes real Binance taker/maker fees and a conservative
  slippage model from the first run — never reported "before costs."

## Regime consistency

- Edge must appear in more than one market regime (trending AND ranging/chop)
  within the research set. A strategy that is only profitable in one regime
  that happens to dominate the backtest window is not validated.

## Component attribution

- Each of: CVD-only, +Spearman, +ROC, +funding filter must be tested
  separately. If the full combination doesn't outperform the best individual
  component, the extra layers are adding complexity without contribution —
  simplify rather than keep them.

## Parameter sensitivity / robustness

- Vary each parameter ±20% and ±40%. A parameter causing >40% expectancy
  degradation on a ±20% perturbation is FRAGILE — this must be reported as a
  fail for that parameter even if the headline backtest numbers look good.
  (See the reviewed APFTS v3 audit — this exact failure mode occurred there:
  a fragile `composite_threshold` was reported as an acceptable "Path B"
  strategy despite the sensitivity gate failing outright. Do not repeat that.)

## Multiple-testing discipline

- Maximum 8 parameter/indicator combinations tested in v1 research. If this
  budget is exhausted without a robust pass, that is a valid negative result —
  log it in `ITERATION_LOG.md` and treat as informative, not as a reason to
  keep searching indefinitely within the same hypothesis family.

## Statistical significance

- Bootstrap or permutation test against randomized entry timing, same risk
  management. Report the confidence interval and p-value — a positive point
  estimate without this is not sufficient.

## Validation set access

- Maximum 2–3 opens of the validation set, logged in `logs/access.log` with
  timestamp and git commit hash.

## Holdout

- Opens exactly once, at the end, only after hypothesis + kill criteria are
  git-tagged and no code has changed since. No re-tuning after holdout result,
  regardless of outcome. A holdout failure is final for this hypothesis
  version.

---

## Falsification triggers (post-deployment, if this reaches shadow/live)

- Trade-level Sharpe < 1.5 for 2 consecutive weeks (≥30 trades each)
- Win rate < 40% for 100 consecutive trades
- Profit factor < 1.10 for 50 trades
- Max drawdown > 8% in any 30-day window

## Early warning triggers (if this reaches shadow/live)

- 3 consecutive stop-loss exits
- Composite/divergence signal strength averaging below calibrated floor over
  last 20 bars
- Any single pair's win rate diverging >15pp from the aggregate — investigate
  before continuing to trade that pair
