# Perpetual Trading Strategy — Loop Engineering Bot

Multi-strategy (scalping / day trade / swing) perpetual futures trading bot,
built on a nested control-loop architecture. See `CLAUDE.md` for the full
project brief, methodology rules, and current build status — read that file
first, every session.

**Current phase**: Day trade strategy research (CVD Divergence + Funding Rate
Filter on Binance USD-M Futures). Not live-trading yet.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env with the pairs/date range you're targeting
```

## Structure

```
apfts-loop/
├── CLAUDE.md                  # read this first, every session
├── hypothesis/
│   ├── HYPOTHESIS.md          # locked before any backtest — see methodology rule 1
│   └── KILL_CRITERIA.md       # locked numeric gates
├── data/
│   ├── raw/                   # immutable downloaded aggTrades/funding
│   ├── research/               # oldest 50% — free to explore
│   ├── validation/              # next 25% — max 2-3 opens
│   └── holdout/                 # most recent 25% — opens once, sealed
├── scripts/
│   └── segment_data.py         # hard technical isolation between segments
├── src/
│   ├── core/                   # CVD reconstruction, shared utils
│   ├── strategy/                # signal layers, divergence detection
│   ├── risk/                    # risk loop, kill switch (future phase)
│   ├── execution/                # order execution (future phase)
│   ├── backtest/                 # backtest engine, walk-forward
│   └── persistence/               # trade store
├── tests/
└── logs/
    └── access.log               # validation/holdout access is logged here
```
