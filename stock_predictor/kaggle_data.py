"""Large-scale panel data: S&P 500 daily prices from Kaggle (andrewmvd/sp-500-stocks).

~1.77M rows, 503 stocks, 2010 → late 2024. Downloaded anonymously via `kagglehub`
and cached as Parquet. Optionally refreshed with recent bars from Yahoo Finance so
live predictions use today's data.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

KAGGLE_DATASET = "andrewmvd/sp-500-stocks"
# Newer versions of this dataset (>= ~1000) are partially broken (only ~175 symbols populated).
DEFAULT_VERSION = 950


def download_kaggle(version: Optional[int] = DEFAULT_VERSION) -> str:
    import kagglehub

    handle = f"{KAGGLE_DATASET}/versions/{version}" if version else KAGGLE_DATASET
    log.info("Fetching Kaggle dataset %s ...", handle)
    return kagglehub.dataset_download(handle)


def _clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Long format → adjusted OHLCV, MultiIndex (Date, Symbol)."""
    df = df.dropna(subset=["Close", "Adj Close"])
    df = df[(df["Close"] > 0) & (df["Adj Close"] > 0)]
    adj = df["Adj Close"] / df["Close"]  # split + dividend adjustment factor
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]),
        "Symbol": df["Symbol"].astype(str),
        "Open": df["Open"] * adj,
        "High": df["High"] * adj,
        "Low": df["Low"] * adj,
        "Close": df["Adj Close"],
        "Volume": df["Volume"].fillna(0),
    })
    out = out.drop_duplicates(["Date", "Symbol"], keep="last")
    out = out.set_index(["Date", "Symbol"]).sort_index()
    # Fix occasional bad OHLC rows
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    return out.astype("float32")


def load_panel(
    data_dir: str = "data",
    version: Optional[int] = DEFAULT_VERSION,
    min_history: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices[MultiIndex Date,Symbol], companies[Symbol → Sector, Industry, ...])."""
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"sp500_panel_v{version}.parquet")
    comp_cache = os.path.join(data_dir, f"sp500_companies_v{version}.parquet")
    if os.path.exists(cache) and os.path.exists(comp_cache):
        prices, companies = pd.read_parquet(cache), pd.read_parquet(comp_cache)
    else:
        path = download_kaggle(version)
        raw = pd.read_csv(os.path.join(path, "sp500_stocks.csv"))
        prices = _clean_panel(raw)
        companies = pd.read_csv(os.path.join(path, "sp500_companies.csv"))
        companies = companies[["Symbol", "Shortname", "Sector", "Industry", "Marketcap"]].set_index("Symbol")
        prices.to_parquet(cache)
        companies.to_parquet(comp_cache)

    counts = prices.groupby(level="Symbol").size()
    keep = counts[counts >= min_history].index
    prices = prices[prices.index.get_level_values("Symbol").isin(keep)]
    log.info("Panel: %d rows, %d symbols, %s → %s", len(prices), len(keep),
             prices.index.get_level_values(0).min().date(), prices.index.get_level_values(0).max().date())
    return prices, companies


def fetch_recent_yahoo(symbols, lookback_days: int = 550) -> pd.DataFrame:
    """Download recent adjusted OHLCV for many symbols from Yahoo (for live predictions)."""
    import yfinance as yf

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    yf_syms = [s.replace(".", "-") for s in symbols]  # BRK.B → BRK-B
    log.info("Downloading recent data for %d symbols from Yahoo ...", len(yf_syms))
    raw = yf.download(yf_syms, start=start, auto_adjust=True, group_by="column",
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo download returned no data.")
    long = raw.stack(level=1, future_stack=True).reset_index()
    long.columns = [str(c) for c in long.columns]
    long = long.rename(columns={"Ticker": "Symbol", "level_1": "Symbol"})
    long["Symbol"] = long["Symbol"].str.replace("-", ".", regex=False)
    long["Date"] = pd.to_datetime(long["Date"]).dt.tz_localize(None).dt.normalize()
    long = long.dropna(subset=["Close"])
    long = long[long["Close"] > 0]
    out = long.set_index(["Date", "Symbol"])[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    return out.astype("float32")
