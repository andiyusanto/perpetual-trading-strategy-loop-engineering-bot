#!/usr/bin/env python3
"""Download perp + quarterly (delivery) futures klines for basis measurement.

Unlike every signal we have screened, the perp-quarterly basis is not a
prediction: a delivery contract MUST converge to the index at expiry. This
pulls both legs so the realised carry can be measured as an accounting exercise
rather than a statistical one.

Quarterly contracts are named <SYMBOL>_<YYMMDD> (expiry date) and each lists for
roughly 7 months before it expires.

    python scripts/download_basis.py --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.ingest.archive import ARCHIVE_S3_LIST, download_klines_month  # noqa: E402


def list_quarterly_contracts(symbol: str) -> list[str]:
    r = requests.get(ARCHIVE_S3_LIST, params={
        "delimiter": "/", "prefix": f"data/futures/um/monthly/klines/{symbol}_",
    }, timeout=40)
    r.raise_for_status()
    return sorted(set(re.findall(rf"({symbol}_\d{{6}})/", r.text)))


def list_months(contract: str, interval: str) -> list[tuple[int, int]]:
    r = requests.get(ARCHIVE_S3_LIST, params={
        "delimiter": "/",
        "prefix": f"data/futures/um/monthly/klines/{contract}/{interval}/",
        "max-keys": "1000",
    }, timeout=40)
    r.raise_for_status()
    return sorted({(int(y), int(m)) for y, m in
                   re.findall(rf"{contract}-{interval}-(\d{{4}})-(\d{{2}})\.zip<", r.text)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    args = ap.parse_args()

    root = get_settings().data_root
    tmp = root / "raw" / ".tmp"
    sym, iv = args.symbol.upper(), args.interval

    contracts = list_quarterly_contracts(sym)
    print(f"\n=== basis data: {sym} perp + {len(contracts)} quarterly contracts ===")

    total = 0
    for c in contracts:
        months = list_months(c, iv)
        if not months:
            print(f"  {c}: no {iv} data")
            continue
        out = root / "screening" / c / "klines"
        got = 0
        for (y, m) in months:
            try:
                download_klines_month(c, y, m, iv, out, tmp)
                got += 1
            except requests.exceptions.HTTPError:
                pass
            except Exception as e:  # noqa: BLE001
                print(f"    {c} {y}-{m:02d}: {type(e).__name__}")
        total += got
        print(f"  {c}: {got}/{len(months)} months "
              f"({months[0][0]}-{months[0][1]:02d} .. {months[-1][0]}-{months[-1][1]:02d})")

    # perp months not already held (we have 2020-01..2024-12 from the screening pull)
    perp_out = root / "screening" / sym / "klines"
    missing = [(y, m) for y in (2025, 2026) for m in range(1, 13) if (y, m) <= (2026, 6)]
    got = 0
    for (y, m) in missing:
        try:
            download_klines_month(sym, y, m, iv, perp_out, tmp)
            got += 1
        except requests.exceptions.HTTPError:
            pass
        except Exception:  # noqa: BLE001
            pass
    print(f"  {sym} perp: +{got} months (2025-01..2026-06)")
    print(f"\ntotal quarterly month-files: {total}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
