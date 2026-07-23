"""Project configuration (pydantic-settings, overridable via .env).

Only the fields needed by the current research phase are defined. Execution/risk
settings will be added when those loops are built -- not before.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Data acquisition
    data_symbols: str = Field(default="BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT")
    data_root: Path = Field(default=_PROJECT_ROOT / "data")

    # Backtest costs (verify against the current Binance USD-M fee schedule before
    # using in a backtest -- these are placeholders carried from .env.example).
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 3.0

    log_level: str = "INFO"

    @property
    def symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.data_symbols.split(",") if s.strip()]

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"


def get_settings() -> Settings:
    return Settings()
