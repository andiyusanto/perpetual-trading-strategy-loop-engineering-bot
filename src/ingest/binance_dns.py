"""DNS-over-HTTPS pin for Binance API hosts blocked by DNS hijacking.

Why this exists
---------------
On some networks (observed here: an Indonesian ISP running "Internet Positif")
``fapi.binance.com`` is blocked at the DNS layer -- the name resolves to the
ISP's block-page server (e.g. ``internetpositif.ioh.co.id``) instead of Binance.
The block is *purely* DNS hijacking: the host is actually served by CloudFront
(``d2ukl3c6tymv7q.cloudfront.net``), and connecting to the real CloudFront IPs
with the correct SNI succeeds (verified: HTTP 200 on /fapi/v1/ping). There is no
SNI/DPI filtering on the path.

So the fix is: resolve the real IPs out-of-band via DNS-over-HTTPS (querying a
resolver by IP, which the hijack can't touch) and make ``socket.getaddrinfo``
hand those IPs to ``requests``/``urllib`` while the hostname -- and therefore SNI
and TLS certificate verification -- stays intact. Certificates are still fully
verified against ``fapi.binance.com``; we only override name->IP resolution.

``data.binance.vision`` (the archive) is NOT hijacked and needs no pin.

Usage
-----
    from src.ingest.binance_dns import enable_binance_doh
    enable_binance_doh()          # idempotent; call once at startup
    requests.get("https://fapi.binance.com/fapi/v1/ping")  # now works
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field

import requests

# Binance API hostnames that get DNS-hijacked and must be pinned. The archive
# host (data.binance.vision) is intentionally NOT here -- it resolves correctly.
PINNED_HOSTS: frozenset[str] = frozenset(
    {
        "fapi.binance.com",
        "fapi1.binance.com",
        "fapi2.binance.com",
        "fapi3.binance.com",
        "api.binance.com",
        "dapi.binance.com",
    }
)

# DoH resolvers addressed BY IP so the hijack can't intercept the lookup itself.
# (Cloudflare uses the RFC-8484 JSON form at ?name=; Google uses /resolve.)
_DOH_ENDPOINTS: tuple[str, ...] = (
    "https://1.1.1.1/dns-query",
    "https://8.8.8.8/resolve",
)

_CACHE_TTL_SECONDS = 300.0
_DOH_TIMEOUT = 15.0


@dataclass
class _CacheEntry:
    ips: list[str]
    fetched_at: float = field(default_factory=time.monotonic)

    def fresh(self) -> bool:
        return (time.monotonic() - self.fetched_at) < _CACHE_TTL_SECONDS


_cache: dict[str, _CacheEntry] = {}
_original_getaddrinfo = None  # set when the patch is installed


def _doh_query(host: str) -> list[str]:
    """Resolve A records for *host* via DoH. Raises if all resolvers fail."""
    last_err: Exception | None = None
    for endpoint in _DOH_ENDPOINTS:
        try:
            resp = requests.get(
                endpoint,
                params={"name": host, "type": "A"},
                headers={"accept": "application/dns-json"},
                timeout=_DOH_TIMEOUT,
            )
            resp.raise_for_status()
            answers = resp.json().get("Answer", [])
            # type 1 == A record. CNAME chains (type 5) are followed by the
            # resolver, so the A answers already point at the final IPs.
            ips = [a["data"] for a in answers if a.get("type") == 1]
            if ips:
                return ips
            last_err = RuntimeError(f"DoH {endpoint} returned no A records for {host}")
        except Exception as exc:  # noqa: BLE001 - try the next resolver
            last_err = exc
    raise RuntimeError(f"DoH resolution failed for {host}: {last_err}")


def resolve(host: str) -> list[str]:
    """Return cached (or freshly fetched) real IPs for a pinned *host*."""
    entry = _cache.get(host)
    if entry is None or not entry.fresh():
        entry = _CacheEntry(ips=_doh_query(host))
        _cache[host] = entry
    return entry.ips


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    if host in PINNED_HOSTS:
        results = []
        for ip in resolve(host):
            # AF_INET / SOCK_STREAM only -- Binance API is IPv4 HTTPS.
            results.append(
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
            )
        if results:
            return results
    return _original_getaddrinfo(host, port, family, type, proto, flags)


def enable_binance_doh() -> None:
    """Install the getaddrinfo pin. Idempotent and safe to call repeatedly."""
    global _original_getaddrinfo
    if _original_getaddrinfo is not None:
        return
    _original_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = _patched_getaddrinfo  # type: ignore[assignment]


def disable_binance_doh() -> None:
    """Restore the original resolver (mainly for tests)."""
    global _original_getaddrinfo
    if _original_getaddrinfo is not None:
        socket.getaddrinfo = _original_getaddrinfo  # type: ignore[assignment]
        _original_getaddrinfo = None
