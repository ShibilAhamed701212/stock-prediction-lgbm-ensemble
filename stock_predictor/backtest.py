"""Vectorized daily backtest of a probability-driven strategy, with transaction costs."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def positions_from_proba(
    p: np.ndarray,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    allow_short: bool = False,
    sizing: str = "binary",
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if sizing == "binary":
        pos = np.where(p >= long_threshold, 1.0, 0.0)
        if allow_short:
            pos = np.where(p <= short_threshold, -1.0, pos)
    elif sizing == "scaled":
        # linear in confidence: 0 at p=0.5, full size at p=long_threshold
        width = max(long_threshold - 0.5, 1e-6)
        pos = np.clip((p - 0.5) / width, -1.0 if allow_short else 0.0, 1.0)
    else:
        raise ValueError("sizing must be 'binary' or 'scaled'")
    return pos


def perf_metrics(returns: pd.Series, positions: pd.Series | None = None) -> Dict[str, float]:
    r = returns.fillna(0.0)
    n = len(r)
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1) if n else 0.0
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 and not np.isnan(cagr) else float("nan")
    out = {
        "total_return": total, "cagr": cagr, "ann_vol": vol, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
    }
    if positions is not None:
        active = positions.abs() > 0
        out["exposure"] = float(positions.abs().mean())
        out["hit_rate"] = float((r[active] > 0).mean()) if active.any() else float("nan")
        out["n_trades"] = int((positions.diff().abs().fillna(positions.abs()) > 0).sum())
    return out


def run_backtest(
    proba: pd.Series,
    next_ret: pd.Series,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    allow_short: bool = False,
    sizing: str = "binary",
    cost_bps: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Position decided at close t (using P(up) known at t) earns the return from t to t+1."""
    pos = pd.Series(
        positions_from_proba(proba.values, long_threshold, short_threshold, allow_short, sizing),
        index=proba.index,
    )
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * next_ret - turnover * cost_bps / 1e4
    bt = pd.DataFrame({
        "proba": proba, "position": pos, "next_ret": next_ret, "strategy_ret": strat,
        "equity": (1 + strat).cumprod(), "buy_hold_equity": (1 + next_ret).cumprod(),
    })
    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1
    metrics = {
        "strategy": perf_metrics(strat, pos),
        "buy_and_hold": perf_metrics(next_ret, pd.Series(1.0, index=next_ret.index)),
    }
    return bt, metrics
