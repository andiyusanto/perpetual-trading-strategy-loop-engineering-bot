"""Data acquisition (research phase: historical batch download).

This is the batch/historical counterpart to the future live "market data loop"
(architecture loop #1). For the day-trade research phase we only need historical
aggTrades + funding rate pulled from public Binance sources:

- aggTrades:  data.binance.vision archive (monthly zip files)  -- NOT geo-blocked
- funding:    GET /fapi/v1/fundingRate                          -- reached via DoH pin

See ``binance_dns`` for why the DoH pin is necessary in some networks.
"""
