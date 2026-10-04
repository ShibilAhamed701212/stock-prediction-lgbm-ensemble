"""Command-line interface.

Examples
--------
    python main.py train   --ticker AAPL --horizon 5
    python main.py predict --ticker AAPL --horizon 5
    python main.py scan    --tickers AAPL MSFT NVDA TSLA
    python main.py train   --ticker DEMO --synthetic     # offline demo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from stock_predictor.config import DEFAULT_MODELS, Config
from stock_predictor.pipeline import predict_latest, run_training


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days (default 5)")
    p.add_argument("--start", default="2010-01-01", help="History start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="History end date (default: today)")
    p.add_argument("--market", default="SPY", help="Market/benchmark ticker for context features ('none' to disable)")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Subset of: logreg rf et xgb lgbm")
    p.add_argument("--splits", type=int, default=8, help="Walk-forward folds")
    p.add_argument("--long-threshold", type=float, default=0.55)
    p.add_argument("--short-threshold", type=float, default=0.45)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--sizing", choices=["binary", "scaled"], default="binary")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost per unit turnover (bps)")
    p.add_argument("--synthetic", action="store_true", help="Use generated data (no internet needed)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")


def _cfg(args, ticker: str) -> Config:
    return Config(
        ticker=ticker, start=args.start, end=args.end,
        market_ticker=None if str(args.market).lower() == "none" else args.market,
        horizon=args.horizon, models=tuple(args.models), n_splits=args.splits,
        long_threshold=args.long_threshold, short_threshold=args.short_threshold,
        allow_short=args.allow_short, sizing=args.sizing, cost_bps=args.cost_bps,
    )


def _fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:6.2f}%"


def print_report(res: dict):
    m = pd.DataFrame(res["metrics"]).T[["accuracy", "auc", "log_loss", "brier", "precision_at_0.55", "coverage_at_0.55"]]
    print("\n=== Out-of-sample classification metrics (walk-forward) ===")
    print(m.round(4).to_string())

    print("\n=== Backtest (out-of-sample, after costs) ===")
    bt = pd.DataFrame(res["backtest"]).T
    wanted = ["total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "exposure", "hit_rate", "n_trades"]
    cols = [c for c in wanted if c in bt]
    print(bt[cols].round(3).to_string())

    print("\n=== Top 10 features ===")
    print(res["importance"].head(10).round(4).to_string())
    print_prediction(res["prediction"])
    print(f"\nModel saved to : {res['model_path']}")
    print(f"Reports saved to: {res['report_dir']}/")


def print_prediction(p: dict):
    print(f"\n=== Prediction for {p['ticker']} (as of {p['as_of']}, close {p['last_close']:.2f}) ===")
    print(f"P(up over next {p['horizon_days']} days): {p['prob_up'] * 100:.1f}%   →  {p['signal']}")
    print("Per model: " + ", ".join(f"{k}={v:.3f}" for k, v in p["per_model"].items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ensemble ML stock direction predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Walk-forward evaluate, backtest, and train the final model")
    p_train.add_argument("--ticker", required=True)
    _add_common(p_train)

    p_pred = sub.add_parser("predict", help="Predict with a previously trained model")
    p_pred.add_argument("--ticker", required=True)
    p_pred.add_argument("--json", action="store_true", help="Output JSON")
    _add_common(p_pred)

    p_scan = sub.add_parser("scan", help="Train + rank several tickers by P(up)")
    p_scan.add_argument("--tickers", nargs="+", required=True)
    _add_common(p_scan)

    p_panel_train = sub.add_parser("panel-train", help="Train cross-sectional S&P 500 panel model")
    p_panel_train.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days")
    p_panel_train.add_argument("--tune-trials", type=int, default=25, help="Optuna trials (0 to skip)")
    p_panel_train.add_argument("--cost-bps", type=float, default=5.0)
    p_panel_train.add_argument("--no-plots", action="store_true")
    p_panel_train.add_argument("-v", "--verbose", action="store_true")

    p_panel_pred = sub.add_parser("panel-predict", help="Predict all S&P 500 stocks with panel model")
    p_panel_pred.add_argument("--horizon", type=int, default=5)
    p_panel_pred.add_argument("--no-refresh", action="store_true", help="Don't download recent Yahoo data")
    p_panel_pred.add_argument("--json", action="store_true")
    p_panel_pred.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "train":
        res = run_training(_cfg(args, args.ticker), synthetic=args.synthetic, make_plots=not args.no_plots)
        print_report(res)
    elif args.cmd == "predict":
        pred = predict_latest(_cfg(args, args.ticker), synthetic=args.synthetic)
        if args.json:
            print(json.dumps(pred, indent=2))
        else:
            print_prediction(pred)
    elif args.cmd == "scan":
        rows = []
        for t in args.tickers:
            try:
                res = run_training(_cfg(args, t), synthetic=args.synthetic, make_plots=not args.no_plots)
                p, s, e = res["prediction"], res["backtest"]["strategy"], res["metrics"]["ensemble"]
                rows.append({
                    "ticker": t.upper(), "as_of": p["as_of"], "prob_up": round(p["prob_up"], 4),
                    "signal": p["signal"], "oos_auc": round(e["auc"], 4), "oos_acc": round(e["accuracy"], 4),
                    "bt_sharpe": round(s["sharpe"], 2), "bh_sharpe": round(res["backtest"]["buy_and_hold"]["sharpe"], 2),
                    "bt_max_dd": round(s["max_drawdown"], 3),
                })
                print(f"✓ {t}", file=sys.stderr)
            except Exception as exc:
                print(f"✗ {t}: {exc}", file=sys.stderr)
        if rows:
            df = pd.DataFrame(rows).sort_values("prob_up", ascending=False)
            print("\n=== Scan results (sorted by P(up)) ===")
            print(df.to_string(index=False))
    elif args.cmd == "panel-train":
        from stock_predictor.panel import PanelConfig, run_panel
        cfg = PanelConfig(horizon=args.horizon, n_trials=args.tune_trials, cost_bps=args.cost_bps)
        res = run_panel(cfg, make_plots=not args.no_plots)
        print(f"\n=== Panel model (H={args.horizon}) ===")
        print(f"Tuned params: {res['params']}")
        print("\nHoldout metrics (2020+):")
        for k, v in res["metrics"]["holdout"].items():
            if not isinstance(v, dict) and not isinstance(v, list):
                print(f"  {k}: {v}")
        print("\nHoldout long/short portfolio (annualized):")
        for k, v in res["metrics"]["holdout"]["portfolio"]["long_short_net"].items():
            print(f"  {k}: {v}")
        print(f"\nSaved to: {res['report_dir']}/")
    elif args.cmd == "panel-predict":
        from stock_predictor.panel import PanelConfig, predict_panel
        cfg = PanelConfig(horizon=args.horizon)
        df = predict_panel(cfg, refresh=not args.no_refresh).reset_index()
        if args.json:
            print(df.to_json(orient="records", indent=2))
        else:
            print(f"\n=== S&P 500 Live Ranking (H={args.horizon}d, as of {df['as_of'].iloc[0]}) ===")
            cols = ["as_of", "Symbol", "Shortname", "Sector", "close", "target_price", "prob_beat_median", "signal"]
            print("\nTop 10 (Long / Buy):")
            print(df.head(10)[cols].to_string(index=False))
            print("\nBottom 10 (Short / Sell):")
            print(df.tail(10)[cols].to_string(index=False))

if __name__ == "__main__":
    main()
