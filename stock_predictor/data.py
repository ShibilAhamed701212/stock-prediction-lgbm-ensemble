"""Market data loading (Yahoo Finance with local CSV cache) and synthetic data for offline use."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(data_dir: str, ticker: str, start: str, end: Optional[str]) -> str:
    safe = ticker.replace("^", "IDX_").replace("/", "_").replace("=", "_")
    return os.path.join(data_dir, f"{safe}_{start}_{end or 'latest'}.csv")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[OHLCV].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()
    df = df[(df["Close"] > 0)]
    return df.astype(float)


def download_prices(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    data_dir: str = "data",
    use_cache: bool = True,
    max_cache_age_hours: float = 12.0,
) -> pd.DataFrame:
    """Download split/dividend-adjusted daily OHLCV data, cached as CSV."""
    path = _cache_path(data_dir, ticker, start, end)
    if use_cache and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if end is not None or age_h < max_cache_age_hours:
            log.info("Loading %s from cache (%s)", ticker, path)
            return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so offline/synthetic mode works without it

    log.info("Downloading %s from Yahoo Finance ...", ticker)
    raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol / network.")
    df = _clean(raw)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(path)
    return df


def synthetic_prices(
    n: int = 2500,
    seed: int = 0,
    start: str = "2014-01-01",
    drift: float = 0.0003,
    base_vol: float = 0.015,
) -> pd.DataFrame:
    """Generate realistic-looking OHLCV data (GARCH-like volatility + weak momentum).

    Useful for demos, CI and tests without network access.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = np.empty(n)
    var = base_vol**2
    prev = 0.0
    for i in range(n):
        var = 0.05 * base_vol**2 + 0.88 * var + 0.07 * prev**2
        prev = drift + 0.04 * prev + np.sqrt(var) * rng.standard_normal()
        rets[i] = prev
    close = 100 * np.exp(np.cumsum(rets))
    prev_close = np.r_[100.0, close[:-1]]
    open_ = prev_close * (1 + rng.normal(0, base_vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, base_vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, base_vol / 2, n)))
    volume = rng.lognormal(15, 0.3, n) * (1 + 20 * np.abs(rets))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.Index(dates, name="Date"),
    )


def load_data(cfg: Config, synthetic: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Return (asset_prices, market_prices_or_None)."""
    if synthetic:
        seed = abs(hash(cfg.ticker)) % (2**31)
        prices = synthetic_prices(seed=seed)
        market = synthetic_prices(seed=seed + 1, drift=0.0002, base_vol=0.01) if cfg.market_ticker else None
        return prices, market

    prices = download_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir)
    market = None
    if cfg.market_ticker and cfg.market_ticker.upper() != cfg.ticker.upper():
        try:
            market = download_prices(cfg.market_ticker, cfg.start, cfg.end, cfg.data_dir)
        except Exception as exc:  # market features are optional
            log.warning("Could not load market ticker %s (%s); continuing without it.", cfg.market_ticker, exc)
    return prices, market
