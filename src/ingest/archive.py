"""data.binance.vision archive client (USD-M Futures monthly aggTrades).

Downloads monthly aggTrades zips, verifies them against the published SHA256
CHECKSUM, and streams the CSV inside straight to a compressed parquet file
without ever holding the whole (multi-GB uncompressed) month in memory.

The archive host is NOT DNS-hijacked, so this module needs no DoH pin.

aggTrades archive schema (USD-M futures), normalised to canonical column names:
    agg_trade_id, price, quantity, first_trade_id, last_trade_id,
    transact_time (-> ``timestamp``, epoch ms), is_buyer_maker

CVD convention (applied later, not here): is_buyer_maker=True  -> taker was the
seller -> SELL volume; is_buyer_maker=False -> taker was the buyer -> BUY volume.
This module stores raw trades faithfully; it does not compute CVD.
"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import structlog

log = structlog.get_logger(__name__)

# Direct download host (CloudFront, not blocked). Listing uses the S3 REST host.
ARCHIVE_DL_BASE = "https://data.binance.vision"
ARCHIVE_S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

_MARKET_PREFIX = "data/futures/um/monthly/aggTrades"

# Canonical parquet schema for stored aggTrades.
AGGTRADES_SCHEMA = pa.schema(
    [
        ("agg_trade_id", pa.int64()),
        ("price", pa.float64()),
        ("quantity", pa.float64()),
        ("first_trade_id", pa.int64()),
        ("last_trade_id", pa.int64()),
        ("timestamp", pa.int64()),  # epoch ms
        ("is_buyer_maker", pa.bool_()),
    ]
)

# Archive header -> canonical column name.
_ARCHIVE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]
_RENAME = {"transact_time": "timestamp"}

_CHUNK_ROWS = 4_000_000  # ~4M rows/chunk keeps peak memory modest for BTC months


class ChecksumMismatch(RuntimeError):
    """Downloaded zip did not match the published SHA256 checksum."""


@dataclass(frozen=True)
class AggTradesMonth:
    symbol: str
    year: int
    month: int

    @property
    def filename(self) -> str:
        return f"{self.symbol}-aggTrades-{self.year:04d}-{self.month:02d}.zip"

    @property
    def url(self) -> str:
        return f"{ARCHIVE_DL_BASE}/{_MARKET_PREFIX}/{self.symbol}/{self.filename}"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


def month_range(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from start to end."""
    out: list[tuple[int, int]] = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def list_available_months(symbol: str) -> list[tuple[int, int]]:
    """Return sorted (year, month) tuples that exist in the archive for *symbol*."""
    resp = requests.get(
        ARCHIVE_S3_LIST,
        params={"delimiter": "/", "prefix": f"{_MARKET_PREFIX}/{symbol}/"},
        timeout=30,
    )
    resp.raise_for_status()
    import re

    months = sorted(
        {
            (int(y), int(mo))
            for y, mo in re.findall(
                rf"{symbol}-aggTrades-(\d{{4}})-(\d{{2}})\.zip<", resp.text
            )
        }
    )
    return months


# Network resilience knobs. The archive link from some locations is slow AND
# flaky (~0.6 MB/s with occasional stalls), so transient timeouts are expected,
# not exceptional. (connect_timeout, read_timeout): read_timeout is max seconds
# with NO bytes arriving before we treat the socket as stalled and retry.
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 120
_MAX_ATTEMPTS = 10

# Exceptions that mean "transient, retry" rather than "give up".
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _backoff_seconds(attempt: int) -> float:
    return min(60.0, 2.0 ** attempt)  # 2, 4, 8, ... capped at 60s


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    """Small GET (e.g. CHECKSUM) with retry/backoff."""
    last: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=(_CONNECT_TIMEOUT, timeout))
            resp.raise_for_status()
            return resp.content
        except _TRANSIENT as exc:  # noqa: PERF203
            last = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_backoff_seconds(attempt))
    raise RuntimeError(f"GET failed after {_MAX_ATTEMPTS} attempts: {url}: {last}")


def _download_to_file(url: str, dest: Path) -> None:
    """Resumable, retrying download to *dest*.

    Streams to ``<dest>.part``; on a transient failure it retries with backoff
    and RESUMES via an HTTP Range request from the bytes already on disk, so a
    dropped connection on this slow link never restarts a ~650 MB month from 0.
    Integrity is guaranteed by the SHA256 check the caller runs afterwards.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with requests.get(
                url, stream=True, headers=headers,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            ) as resp:
                # If we asked to resume but the server ignores Range (200 instead
                # of 206), start over from byte 0 to avoid a corrupt concatenation.
                if pos and resp.status_code == 200:
                    part.unlink(missing_ok=True)
                    pos = 0
                elif resp.status_code not in (200, 206):
                    resp.raise_for_status()
                with part.open("ab" if pos else "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB
                        if chunk:
                            fh.write(chunk)
            part.replace(dest)
            return
        except _TRANSIENT as exc:
            last = exc
            have = part.stat().st_size if part.exists() else 0
            if attempt < _MAX_ATTEMPTS:
                wait = _backoff_seconds(attempt)
                log.warning("download.retry", url=url, attempt=attempt,
                            have_bytes=have, backoff_s=wait, err=str(exc))
                time.sleep(wait)
    raise RuntimeError(
        f"download failed after {_MAX_ATTEMPTS} attempts: {url}: {last}"
    )


def _expected_sha256(checksum_url: str) -> str:
    # CHECKSUM format: "<sha256>  <filename>"
    text = _download_bytes(checksum_url, timeout=60).decode().strip()
    return text.split()[0].lower()


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().lower()


def _has_header(sample: bytes) -> bool:
    first_line = sample.split(b"\n", 1)[0]
    return b"agg_trade_id" in first_line or b"price" in first_line


def _normalise_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns=_RENAME)
    # is_buyer_maker arrives as the strings "true"/"false" in futures files
    # (occasionally already bool). Map explicitly rather than trust truthiness.
    bm = chunk["is_buyer_maker"]
    if bm.dtype != bool:
        chunk["is_buyer_maker"] = (
            bm.astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )
    return chunk[[f.name for f in AGGTRADES_SCHEMA]]


def _csv_to_parquet(csv_stream: io.BufferedReader, has_header: bool, dest: Path) -> int:
    """Stream a (possibly huge) aggTrades CSV to parquet. Returns row count."""
    reader = pd.read_csv(
        csv_stream,
        header=0 if has_header else None,
        names=None if has_header else _ARCHIVE_COLUMNS,
        chunksize=_CHUNK_ROWS,
        dtype={
            "agg_trade_id": "int64",
            "price": "float64",
            "quantity": "float64",
            "first_trade_id": "int64",
            "last_trade_id": "int64",
            "transact_time": "int64",
            "is_buyer_maker": "string",
        },
    )
    rows = 0
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    writer = pq.ParquetWriter(tmp_dest, AGGTRADES_SCHEMA, compression="zstd")
    try:
        for chunk in reader:
            norm = _normalise_chunk(chunk)
            if norm["is_buyer_maker"].isna().any():
                raise ValueError("Unparseable is_buyer_maker value in aggTrades chunk")
            table = pa.Table.from_pandas(
                norm, schema=AGGTRADES_SCHEMA, preserve_index=False
            )
            writer.write_table(table)
            rows += len(norm)
    finally:
        writer.close()
    tmp_dest.replace(dest)
    return rows


@dataclass
class MonthResult:
    month: AggTradesMonth
    parquet_path: Path
    rows: int
    ts_min_ms: int
    ts_max_ms: int
    skipped: bool = False
    reason: str = ""


def download_aggtrades_month(
    symbol: str,
    year: int,
    month: int,
    out_dir: Path,
    tmp_dir: Path,
    *,
    overwrite: bool = False,
) -> MonthResult:
    """Download + verify + convert one month of aggTrades to parquet.

    Idempotent: if the parquet already exists and ``overwrite`` is False, it is
    reused (and its row/timestamp stats are read back from the file).
    """
    m = AggTradesMonth(symbol, year, month)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{symbol}-aggTrades-{year:04d}-{month:02d}.parquet"

    if parquet_path.exists() and not overwrite:
        meta = pq.read_metadata(parquet_path)
        stats = _parquet_ts_bounds(parquet_path)
        log.info("aggtrades.skip_existing", file=parquet_path.name, rows=meta.num_rows)
        return MonthResult(m, parquet_path, meta.num_rows, stats[0], stats[1],
                           skipped=True, reason="already downloaded")

    zip_path = tmp_dir / m.filename
    log.info("aggtrades.download", url=m.url)
    _download_to_file(m.url, zip_path)

    expected = _expected_sha256(m.checksum_url)
    actual = _sha256_of(zip_path)
    if expected != actual:
        zip_path.unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"{m.filename}: expected {expected}, got {actual}"
        )
    log.info("aggtrades.checksum_ok", file=m.filename)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as raw:
                sample = raw.read(256)
            with zf.open(inner) as raw:
                # re-open to read from the top after peeking
                has_hdr = _has_header(sample)
                rows = _csv_to_parquet(raw, has_hdr, parquet_path)
    finally:
        zip_path.unlink(missing_ok=True)  # never keep the multi-GB raw zip

    ts_min, ts_max = _parquet_ts_bounds(parquet_path)
    log.info("aggtrades.written", file=parquet_path.name, rows=rows)
    return MonthResult(m, parquet_path, rows, ts_min, ts_max)


def _parquet_ts_bounds(path: Path) -> tuple[int, int]:
    col = pq.read_table(path, columns=["timestamp"])["timestamp"]
    return pa.compute.min(col).as_py(), pa.compute.max(col).as_py()


# ---------------------------------------------------------------------------
# Other monthly datasets: klines and fundingRate.
#
# Screening only needs close prices, and klines are megabytes where aggTrades
# are gigabytes — so years of history become cheap. aggTrades stay necessary
# only for order-flow signals (CVD).
# ---------------------------------------------------------------------------

_KLINES_PREFIX = "data/futures/um/monthly/klines"
_FUNDING_PREFIX = "data/futures/um/monthly/fundingRate"

# USD-M futures monthly klines CSV layout.
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
    "ignore",
]

# fundingRate CSV layout (verified against a downloaded file).
_FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

_INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _fetch_monthly_csv(url: str, tmp_dir: Path, columns: list[str]) -> pd.DataFrame:
    """Download + checksum-verify a monthly zip and return its CSV as a frame."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    zpath = tmp_dir / name
    _download_to_file(url, zpath)
    expected = _expected_sha256(url + ".CHECKSUM")
    actual = _sha256_of(zpath)
    if expected != actual:
        zpath.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{name}: expected {expected}, got {actual}")
    try:
        with zipfile.ZipFile(zpath) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as raw:
                head = raw.read(256)
            has_hdr = any(c.encode() in head.split(b"\n", 1)[0] for c in columns)
            with zf.open(inner) as raw:
                df = pd.read_csv(
                    raw, header=0 if has_hdr else None,
                    names=None if has_hdr else columns,
                )
    finally:
        zpath.unlink(missing_ok=True)
    return df


def download_klines_month(
    symbol: str, year: int, month: int, interval: str, out_dir: Path, tmp_dir: Path,
    *, overwrite: bool = False,
) -> Path:
    """One month of klines -> parquet with this project's bar column names.

    ``close_time`` is normalised to ``open_time + interval`` (Binance publishes
    ``open_time + interval - 1ms``) so it matches the aggTrades-derived bars:
    a bar is knowable at the instant it closes.
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"unsupported interval {interval!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{symbol}-{interval}-{year:04d}-{month:02d}.parquet"
    if dest.exists() and not overwrite:
        return dest
    url = (f"{ARCHIVE_DL_BASE}/{_KLINES_PREFIX}/{symbol}/{interval}/"
           f"{symbol}-{interval}-{year:04d}-{month:02d}.zip")
    df = _fetch_monthly_csv(url, tmp_dir, _KLINE_COLUMNS)
    step = _INTERVAL_MS[interval]
    out = pd.DataFrame({
        "open_time": df["open_time"].astype("int64"),
        "close_time": df["open_time"].astype("int64") + step,
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "num_trades": df["count"].astype("int64"),
    }).sort_values("open_time").reset_index(drop=True)
    out.to_parquet(dest, compression="zstd", index=False)
    log.info("klines.written", file=dest.name, rows=len(out))
    return dest


def download_funding_month(
    symbol: str, year: int, month: int, out_dir: Path, tmp_dir: Path,
    *, overwrite: bool = False,
) -> Path:
    """One month of settled funding -> parquet, matching the REST schema.

    Archive column ``calc_time`` is the settlement timestamp and maps to
    ``funding_time``; ``last_funding_rate`` maps to ``funding_rate``. The
    archive carries no mark price, so ``mark_price`` is NaN.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{symbol}-fundingRate-{year:04d}-{month:02d}.parquet"
    if dest.exists() and not overwrite:
        return dest
    url = (f"{ARCHIVE_DL_BASE}/{_FUNDING_PREFIX}/{symbol}/"
           f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip")
    df = _fetch_monthly_csv(url, tmp_dir, _FUNDING_COLUMNS)
    out = pd.DataFrame({
        "symbol": symbol,
        "funding_time": df["calc_time"].astype("int64"),
        "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce"),
        "mark_price": float("nan"),
    }).sort_values("funding_time").reset_index(drop=True)
    out.to_parquet(dest, compression="zstd", index=False)
    log.info("funding.written", file=dest.name, rows=len(out))
    return dest
