"""Feature engineering: ~60 causal (no look-ahead) technical & market-context features.

Every feature at date t uses only information available at the close of day t.
Features are expressed as ratios / normalized values so they are roughly stationary
and comparable across tickers and price levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Indicator helpers
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr = true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame(
        {"adx": dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), "plus_di": plus_di, "minus_di": minus_di}
    )


# ----------------------------------------------------------------------------
# Feature builder
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame, market: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    o, h, l, c, v = (df[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    f = pd.DataFrame(index=df.index)
    logret = np.log(c).diff()
    ret1 = c.pct_change()

    # Momentum / returns
    for n in (1, 2, 3, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = c.pct_change(n)

    # Volatility
    for n in (5, 10, 20, 60):
        f[f"vol_{n}"] = logret.rolling(n).std()
    f["vol_ratio_5_20"] = f["vol_5"] / f["vol_20"]
    f["vol_ratio_20_60"] = f["vol_20"] / f["vol_60"]
    # Parkinson (high-low) volatility estimator
    f["parkinson_vol_20"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)))

    # Trend: distance from moving averages
    for n in (5, 10, 20, 50, 100, 200):
        f[f"sma_dist_{n}"] = c / c.rolling(n).mean() - 1
    f["sma_cross_20_50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    f["sma_cross_50_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1

    # MACD (normalized by price)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd
    f["macd_signal"] = macd_signal
    f["macd_hist"] = macd - macd_signal

    # Oscillators
    f["rsi_14"] = rsi(c, 14) / 100
    f["rsi_5"] = rsi(c, 5) / 100
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_k.rolling(3).mean()
    tp = (h + l + c) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan)) / 100

    # Bollinger Bands
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    f["bb_width"] = 4 * s20 / m20

    # Range / ATR / ADX
    atr14 = true_range(h, l, c).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr_14"] = atr14 / c
    adx_df = adx(h, l, c, 14)
    f["adx_14"] = adx_df["adx"] / 100
    f["di_diff"] = (adx_df["plus_di"] - adx_df["minus_di"]) / 100
    f["hl_range"] = (h - l) / c
    f["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    f["gap"] = o / c.shift(1) - 1
    f["intraday_ret"] = c / o - 1

    # Volume
    vmean20, vstd20 = v.rolling(20).mean(), v.rolling(20).std()
    f["volume_z_20"] = (v - vmean20) / vstd20.replace(0, np.nan)
    f["volume_ratio_5_20"] = v.rolling(5).mean() / vmean20
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope_20"] = (obv - obv.shift(20)) / v.rolling(20).sum()
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    f["mfi_14"] = 1 - 1 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    # Distribution shape
    f["skew_20"] = logret.rolling(20).skew()
    f["kurt_60"] = logret.rolling(60).kurt()
    f["up_days_20"] = (ret1 > 0).astype(float).rolling(20).mean()

    # 52-week positioning
    f["dist_52w_high"] = c / c.rolling(252).max() - 1
    f["dist_52w_low"] = c / c.rolling(252).min() - 1

    # Calendar
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month

    # Market context
    if market is not None and not market.empty:
        mc = market["Close"].reindex(df.index).ffill()
        mret = mc.pct_change()
        f["mkt_ret_1"] = mret
        f["mkt_ret_5"] = mc.pct_change(5)
        f["mkt_ret_20"] = mc.pct_change(20)
        f["mkt_vol_20"] = np.log(mc).diff().rolling(20).std()
        f["mkt_sma_dist_200"] = mc / mc.rolling(200).mean() - 1
        f["rel_strength_20"] = f["ret_20"] - f["mkt_ret_20"]
        f["rel_strength_60"] = f["ret_60"] - mc.pct_change(60)
        f["beta_60"] = ret1.rolling(60).cov(mret) / mret.rolling(60).var()
        f["corr_60"] = ret1.rolling(60).corr(mret)

    return f.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    X: pd.DataFrame            # labeled, feature-complete rows (for training / validation)
    y: pd.Series               # 1 = price up more than threshold over horizon
    future_ret: pd.Series      # realized forward return over horizon
    next_ret: pd.Series        # next-day return (used by the daily backtest)
    X_live: pd.DataFrame       # all feature-complete rows incl. most recent unlabeled ones
    close: pd.Series


def make_target(close: pd.Series, horizon: int, threshold: float = 0.0):
    future_ret = close.shift(-horizon) / close - 1
    y = (future_ret > threshold).astype(float).where(future_ret.notna())
    return future_ret, y


def build_dataset(
    prices: pd.DataFrame,
    market: Optional[pd.DataFrame],
    horizon: int,
    threshold: float = 0.0,
) -> Dataset:
    feats = build_features(prices, market)
    future_ret, y = make_target(prices["Close"], horizon, threshold)
    next_ret = prices["Close"].shift(-1) / prices["Close"] - 1

    complete = feats.notna().all(axis=1)
    X_live = feats[complete]
    labeled = complete & y.notna()
    if labeled.sum() < 300:
        raise ValueError(
            f"Only {int(labeled.sum())} usable rows after feature warm-up; need more history (≥ ~2.5 years)."
        )
    return Dataset(
        X=feats[labeled],
        y=y[labeled].astype(int),
        future_ret=future_ret[labeled],
        next_ret=next_ret[labeled],
        X_live=X_live,
        close=prices["Close"],
    )
