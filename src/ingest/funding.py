"""Funding-rate history via GET /fapi/v1/fundingRate (reached through the DoH pin).

Returns *realized* 8h funding events -- the same data the archive's fundingRate
dataset carries -- with fields: symbol, fundingTime (epoch ms), fundingRate,
markPrice. Paginated by startTime; the endpoint is weight 1, so even years of
history costs only a handful of requests against the 2400/min budget.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import structlog

from .binance_dns import enable_binance_doh

log = structlog.get_logger(__name__)

FAPI_BASE = "https://fapi.binance.com"
_FUNDING_PATH = "/fapi/v1/fundingRate"
_PAGE_LIMIT = 1000  # endpoint max
_WEIGHT_LIMIT = 2400  # REQUEST_WEIGHT / 1m (confirmed from exchangeInfo)
_PAGE_PAUSE_S = 0.25  # polite spacing between pages
_MAX_ATTEMPTS = 8
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 30
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _get_page(sess: requests.Session, params: dict) -> requests.Response:
    """One funding page with retry/backoff on transient network errors."""
    last: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = sess.get(
                FAPI_BASE + _FUNDING_PATH,
                params=params,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            resp.raise_for_status()
            return resp
        except _TRANSIENT as exc:  # noqa: PERF203
            last = exc
            if attempt < _MAX_ATTEMPTS:
                wait = min(30.0, 2.0 ** attempt)
                log.warning("funding.retry", attempt=attempt, backoff_s=wait, err=str(exc))
                time.sleep(wait)
    raise RuntimeError(f"funding page failed after {_MAX_ATTEMPTS} attempts: {last}")

FUNDING_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("funding_time", pa.int64()),  # epoch ms
        ("funding_rate", pa.float64()),
        ("mark_price", pa.float64()),
    ]
)


def _respect_weight(resp: requests.Response) -> None:
    used = resp.headers.get("x-mbx-used-weight-1m")
    if used is not None and int(used) > _WEIGHT_LIMIT * 0.8:
        log.warning("funding.weight_backoff", used_weight_1m=used)
        time.sleep(5.0)


def fetch_funding(
    symbol: str, start_ms: int, end_ms: int, *, session: requests.Session | None = None
) -> pd.DataFrame:
    """Fetch all funding events for *symbol* in [start_ms, end_ms] (inclusive)."""
    enable_binance_doh()
    sess = session or requests.Session()
    rows: list[dict] = []
    cursor = start_ms
    seen: set[int] = set()

    while cursor <= end_ms:
        resp = _get_page(
            sess,
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _PAGE_LIMIT,
            },
        )
        _respect_weight(resp)
        page = resp.json()
        if not page:
            break

        new = [r for r in page if r["fundingTime"] not in seen]
        for r in new:
            seen.add(r["fundingTime"])
        rows.extend(new)

        last_ft = page[-1]["fundingTime"]
        log.info("funding.page", symbol=symbol, got=len(page),
                 through=pd.Timestamp(last_ft, unit="ms", tz="UTC").isoformat())
        if len(page) < _PAGE_LIMIT:
            break  # last partial page
        next_cursor = last_ft + 1
        if next_cursor <= cursor:  # safety against a stuck cursor
            break
        cursor = next_cursor
        time.sleep(_PAGE_PAUSE_S)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            {c.name: pd.Series(dtype="object") for c in FUNDING_SCHEMA}
        )
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype("string"),
            "funding_time": df["fundingTime"].astype("int64"),
            "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce"),
            # markPrice can be "" for some historical rows -> NaN
            "mark_price": pd.to_numeric(df.get("markPrice"), errors="coerce"),
        }
    )
    out = out.sort_values("funding_time").reset_index(drop=True)
    out = out[out["funding_time"].between(start_ms, end_ms)]
    return out


def write_funding_parquet(df: pd.DataFrame, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=FUNDING_SCHEMA, preserve_index=False)
    pq.write_table(table, dest, compression="zstd")
    return len(df)
