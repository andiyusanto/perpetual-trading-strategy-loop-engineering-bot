# HYPOTHESIS — Cross-Sectional Trend Following (crypto perpetuals)

**Family**: trend (new family, unrelated to the disproven CVD divergence v1)
**Version**: tf-v1
**Status**: DRAFT — lock (commit + git tag `trend-v1`) before any live or
shadow trading. Do not edit in place after locking; add a new dated version.

Every parameter below is fixed NOW, before any trend backtest has been run, and
is taken from the published literature rather than fitted to our data. That is
deliberate — see section 3.

---

## 1. Mechanism — why a trend premium exists

Trend following is long an option-like payoff on persistent directional moves.
It is the single most out-of-sample-validated systematic strategy in existence:
documented across ~25 years and 58 instruments spanning equities, bonds, FX and
commodities (Moskowitz, Ooi & Pedersen, *Time Series Momentum*, 2012), and
robust in data going back a century.

Why the premium survives arbitrage — it is NOT an inefficiency someone patches:
- **It is compensation for holding risk.** Trend followers lose slowly and often
  (many small losers) and win rarely and large (few huge winners). Most
  participants cannot psychologically hold that distribution, so they are paid
  to. A bot has no such aversion — this is a structural edge a disciplined
  machine genuinely has over a human.
- **It is a crisis hedge / long-volatility profile.** It tends to pay in
  sustained dislocations, which makes it a diversifier institutions want to
  hold even at modest standalone Sharpe — sustaining demand for the trade.

The participant being "exploited" is not a counterparty; it is the collective
preference for smooth returns. That preference does not go away.

## 2. The edge is NOT in the indicator

Measured on 5 years of BTC/ETH daily data (2020–2024): SMA price-cross,
Donchian breakout, and time-series-momentum sign agree on direction **82–93% of
the time**. They are one signal in different clothing. MACD is the outlier
(63–82% agreement) because it is a short-horizon oscillator, not a trend filter,
and is excluded.

The return, if any, lives in three things, in order of importance:
**(1) the timeframe, (2) volatility-scaled sizing, (3) operational reliability.**
The choice of trend indicator is a rounding error next to these.

## 3. What OUR data can and cannot establish — READ THIS

This hypothesis is validated DIFFERENTLY from CVD v1, and pretending otherwise
would be dishonest.

Our power analysis (see `ITERATION_LOG.md`, "the power wall") showed that with
18 months of data an 11 bps edge is only statistically resolvable out to ~7-hour
horizons. A trend strategy holds for **weeks**. We therefore **cannot** prove or
disprove its profitability from our own data — the confidence intervals at a
multi-week horizon are far wider than any plausible edge.

Consequences, committed to up front:

- **The backtest is an IMPLEMENTATION CHECK, not a profitability gate.** It
  verifies: no look-ahead, cost drag within budget, sizing keeps the book
  unliquidatable, and that realised behaviour is *qualitatively consistent* with
  the known trend profile (low win rate, fat right tail, positive skew). A good
  backtest Sharpe here is NOT evidence and will NOT be reported as a pass.
- **Parameters are NOT optimised on our data.** They are pre-registered from the
  literature (section 5). We do not search lookbacks for the best Sharpe — that
  search is exactly the overfitting the power wall guarantees would fool us.
- **Real validation is forward**: shadow mode, then live at $100, judged on
  operational reliability and consistency-with-profile, not on a backtest number.
- The research/validation/holdout segments are **not** consumed by this
  hypothesis. There is nothing to tune, so there is nothing to hold out.

Honest limitation: crypto-only trend following is **under-diversified**.
Traditional managed futures earns its Sharpe by combining dozens of *uncorrelated*
markets; all crypto is effectively one correlated bet, so expect a lower and
lumpier Sharpe than the literature's cross-asset numbers. And crypto trend has
been visibly degrading as the market matures. This is a real headwind, not a
detail.

## 4. Universe

Liquid majors only. Rationale is risk, not preference:
- At 1x notional (section 6) a long is effectively unliquidatable but a **short**
  is only safe if a +100% rally is implausible on the horizon. That is true of
  BTC/ETH, NOT of small alts that can double in a day. Trend goes short, so the
  universe must exclude anything that can rocket.
- Liquidity: our measured spreads are ~0.01 bps (BTC) to ~0.26 bps (HYPE) — all
  negligible vs fees — but a real cascade drains thin books. Majors only.

**Core universe: BTCUSDT, ETHUSDT, SOLUSDT.** These have multi-year history and
deep books on all target venues.

**HYPEUSDT excluded from tf-v1**: only ~13 months of history and a newly-listed
short side that can gap violently. Revisit once it has ≥ 18 months.

**$100 capital constraint (measured against live exchange rules):** min notional
is BTC $50, ETH $20, SOL $5. With a 1x cap of $100 split into balanced positions
(~$20 each), **BTC's $50 minimum makes it too concentrated to hold in a balanced
book** — it is effectively excluded by size until the account reaches ~$500. So
the PRACTICAL starting universe at $100 is **ETH + SOL + additional liquid
$5-min majors** (e.g. add 2-3 of: BNB, XRP, DOGE, LTC — all deep books), giving
~4-5 positions. Vol-targeting at $100 is necessarily coarse (positions are only
1-4 minimum units); precise sizing needs more capital. Another reason $100 is
for proving the system, not for return.

## 5. Signal — EXACT

**Time-series momentum sign, ensembled over three lookbacks.**

For each instrument, on each daily close (00:00 UTC):

```
for L in (50, 100, 200):                       # calendar days
    s_L = sign( close_today - close_[L days ago] )     # +1 / -1
raw_signal = mean(s_50, s_100, s_200)          # in {-1, -1/3, +1/3, +1}
```

- Lookbacks 50/100/200 are the standard managed-futures set, pre-registered, NOT
  fitted here.
- The ensemble makes positions change gradually (the three sub-signals rarely
  flip together), which both reduces single-parameter fragility (KILL_CRITERIA
  ±20%/±40% concern) and cuts turnover.
- `sign(x)=0` (exactly flat) carries forward the previous sub-signal rather than
  going to zero.
- **No look-ahead**: uses only closed daily bars; today's decision uses data
  through today's 00:00 UTC close and is executed on the next available bar.

## 6. Sizing — the actual risk control

Stops are not the primary risk control; SIZE is (section 7 explains why).

1. **Volatility scaling.** Target-weight each instrument inversely to its recent
   realised volatility (e.g. 30-day daily-return stdev), so each contributes
   similar risk. Without this, SOL (measured 2× BTC's vol) would dominate the
   book's risk entirely.

   ```
   target_weight_i  ∝  raw_signal_i / vol_i
   ```

2. **Hard 1x notional cap — the no-liquidation guarantee.** Scale the whole book
   so `sum(|position_notional|) ≤ equity`. At 1x: a long liquidates only on a
   −99.6% move, a short on +99.6%. Crypto does ±30%, not ±99%. **The book is
   structurally unliquidatable at 1x on majors.** This cap is inviolable; the
   strategy never uses it to "size up a conviction".

3. **Isolated margin per position.** A worst-case single-position loss is capped
   at that position's margin (~20% of the account across 5 positions), never the
   whole balance. No cross-margin.

4. **Rebalance deadband.** Only send an order when target vs current position
   differs by more than a threshold (e.g. 20% of the target unit). Prevents
   trivial signal/vol drift from silently rebuilding turnover.

## 7. Exits — no take-profit, by design

**Take profit: NONE.** Measured on the core universe, a 100-day trend produced a
~18% win rate: the return is carried entirely by a few very large winners. A
fixed TP (e.g. the 1.5R that suited mean-reverting CVD) would cap exactly the
fat-tail winner that pays for the year and mathematically guarantee losses.
Exit design must match signal shape: trend = fat right tail = let winners run.

**Primary exit: the signal flipping.** When `raw_signal` changes sign, the
position reverses/closes on the next daily bar. The entry statistic is the exit
statistic. This is adaptive and does not fight the strategy's own noise.

**Disaster stop only: ~6 × daily ATR (≈ 25-30% on majors), ATR-scaled per
instrument.** Measured adverse excursion of *winning* trades is 6-12% (median 6%,
95th pct 16.5%), so a 6×ATR stop fires on ~0-2% of winners — it essentially
never interferes with normal trades. Its sole job is structural breaks the slow
signal cannot react to in time (a token collapse, a delisting, a 100-day SMA
that takes weeks to cross while price halves). ATR-scaled, not a fixed %, because
25% is ~7σ on BTC but ~4σ on SOL.

Note: stop-loss ORDERS do not reliably fill at their trigger in a cascade (thin
book). The disaster stop is a backstop, not a promise; sizing (section 6) is what
actually protects capital when liquidity vanishes.

## 8. Costs and turnover budget

- Round-trip cost **≈ 11 bps** (measured: 2× 5 bps taker + ~1 bp
  spread/impact). Passive/maker entry could approach ~7 bps but is NOT assumed.
- Cost budget: keep drag under ~20% of a ~10%/yr gross target ⇒ ~200 bps/yr ⇒
  **≤ ~18 round trips per instrument per year.**
- The 50/100/200 ensemble flips ~10×/yr blended (measured), giving ~110 bps/yr
  drag — inside budget with headroom.
- **No pre-cost number is ever reported as a result** (methodology rule 7).

## 9. Risk architecture — non-negotiable

Per CLAUDE.md's loop design, the risk loop is INDEPENDENT of and higher-priority
than the signal loop, and runs on a far faster cadence:

- **Signal loop: once daily** (decides target positions).
- **Risk loop: every ~60 s** (can veto or flatten at any time, without the
  signal loop). Monitors: margin ratio, account drawdown, and **exposure
  reconciliation** — does live exchange position == intended position? The
  exposure check catches the bug that opens 5 positions instead of 1, a more
  likely way to lose $100 than any market move.
- **Kill switch** flattens everything (reduce-only) on: equity < floor, margin
  ratio < threshold, or exposure mismatch. It must be **tested by deliberate
  triggering on testnet** before live — an untested kill switch is a comment.
- **Idempotent execution**: `clientOrderId` on every order so a retry after a
  timeout cannot double a position; **reduce-only** on all closes so a sign bug
  cannot flip flat into a new position.
- **Reconcile-on-start**: on boot, read actual positions from the exchange and
  reconcile; never trust persisted local state. Restart-with-open-position is a
  top operational risk.

## 10. Falsification — what would disprove this (FORWARD, not backtest)

Because our data cannot gate profitability (section 3), the kill criteria are
behavioural and operational, evaluated in shadow then live:

- **Profile inconsistency**: if realised trades show a HIGH win rate with small
  winners (the mean-reversion profile), the implementation is wrong or the
  premium is absent on these instruments — trend should show low win rate + fat
  right tail. Report the win-rate/skew, not just PnL.
- **Cost blowout**: realised turnover > ~25 round trips/yr per instrument ⇒ the
  deadband/ensemble is not working; drag will dominate.
- **No trend regime**: an extended chop-only market produces steady bleed; that
  is expected and NOT a falsification by itself, but bleed **beyond** the
  historical max drawdown of a trend proxy (be explicit about the number before
  going live) is.
- **Any operational incident that a correct risk loop should have caught**
  (unreconciled restart, duplicate order, kill switch that did not fire) halts
  live trading until fixed — regardless of PnL.

## 11. Go-live and scaling gates — PRE-COMMITTED

1. **Backtest = implementation check only.** Pass = no look-ahead, cost drag in
   budget, book provably unliquidatable, behaviour qualitatively trend-shaped.
   NOT a profitability verdict.
2. **Shadow mode ≥ 2-3 weeks**: full pipeline, simulated fills, zero real orders.
   Pass = zero operational incidents and signals/sizes look sane.
3. **Live at $100.** Judged on **operational reliability**, not PnL — $100 of
   PnL over a month is indistinguishable from luck.
4. **Scale only after ~3 months live with ZERO operational incidents** — not
   after a profitable month. Reliability is the measurable thing at this size.

## 12. What will NOT be done

- No lookback/parameter optimisation on our data (section 3).
- No take-profit (section 7).
- No leverage above 1x total notional (section 6).
- No small/illiquid alts, no HYPE in tf-v1 (section 4).
- No live trading before the kill switch is tested by deliberate trigger.
- No scaling on the basis of a profitable backtest or a single good month.
- The disproven CVD family is not revisited here.
