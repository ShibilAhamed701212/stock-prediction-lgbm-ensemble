"""Static report charts (matplotlib)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def save_all_plots(bt: pd.DataFrame, oos: pd.DataFrame, importance: pd.Series, out_dir: str, ticker: str):
    os.makedirs(out_dir, exist_ok=True)

    # Equity curve + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(bt.index, bt["equity"], label="ML strategy", lw=1.6)
    ax1.plot(bt.index, bt["buy_hold_equity"], label="Buy & hold", lw=1.2, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_title(f"{ticker} — out-of-sample walk-forward backtest")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.fill_between(bt.index, bt["drawdown"], 0, color="tab:red", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=120)
    plt.close(fig)

    # Feature importance
    if not importance.empty:
        top = importance.head(20)[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top.index, top.values, color="tab:blue")
        ax.set_title("Top 20 features (ensemble-averaged importance)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=120)
        plt.close(fig)

    # Calibration: predicted probability bucket vs realized frequency
    p, y = oos["ensemble"].values, oos["y"].values
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
    dfc = pd.DataFrame({"p": p, "y": y, "b": idx}).groupby("b").mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.3, 0.7], [0.3, 0.7], "k--", lw=1, label="Perfect calibration")
    ax.plot(dfc["p"], dfc["y"], "o-", label="Ensemble")
    ax.set_xlabel("Predicted P(up)")
    ax.set_ylabel("Realized frequency of up moves")
    ax.set_title("Calibration (deciles)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=120)
    plt.close(fig)
