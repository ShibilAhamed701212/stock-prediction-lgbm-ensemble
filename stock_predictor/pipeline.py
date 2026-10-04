"""High-level workflows: evaluate → backtest → train final model → predict."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np

from .backtest import run_backtest
from .config import Config
from .data import load_data
from .features import build_dataset
from .models import EnsembleClassifier
from .validation import walk_forward_evaluate

log = logging.getLogger(__name__)


def _model_path(cfg: Config) -> str:
    return os.path.join(cfg.model_dir, f"{cfg.ticker.upper()}_h{cfg.horizon}.joblib")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def run_training(cfg: Config, synthetic: bool = False, make_plots: bool = True) -> dict:
    """Full pipeline: walk-forward evaluation, backtest, final fit on all data, save artifacts."""
    prices, market = load_data(cfg, synthetic=synthetic)
    ds = build_dataset(prices, market, cfg.horizon, cfg.threshold)
    log.info("%s: %d labeled samples, %d features (%s → %s)", cfg.ticker, len(ds.X), ds.X.shape[1],
             ds.X.index[0].date(), ds.X.index[-1].date())

    # 1) Honest out-of-sample evaluation
    oos, metrics, folds = walk_forward_evaluate(ds, cfg)

    # 2) Backtest on out-of-sample probabilities
    bt, bt_metrics = run_backtest(
        oos["ensemble"], oos["next_ret"], cfg.long_threshold, cfg.short_threshold,
        cfg.allow_short, cfg.sizing, cfg.cost_bps,
    )

    # 3) Final model trained on ALL labeled data → used for live predictions
    final = EnsembleClassifier(cfg.models, cfg.random_state).fit(ds.X, ds.y)
    importance = final.feature_importance()
    meta = {
        "config": cfg.to_dict(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_start": ds.X.index[0].date().isoformat(),
        "train_end": ds.X.index[-1].date().isoformat(),
        "n_samples": len(ds.X),
        "models": final.model_names,
        "oos_metrics": metrics,
        "backtest": bt_metrics,
        "synthetic": synthetic,
    }
    final.save(_model_path(cfg), meta)

    # 4) Reports
    out_dir = os.path.join(cfg.report_dir, f"{cfg.ticker.upper()}_h{cfg.horizon}")
    os.makedirs(out_dir, exist_ok=True)
    oos.to_csv(os.path.join(out_dir, "oos_predictions.csv"))
    bt.to_csv(os.path.join(out_dir, "backtest.csv"))
    folds.to_csv(os.path.join(out_dir, "folds.csv"), index=False)
    importance.rename("importance").to_csv(os.path.join(out_dir, "feature_importance.csv"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"classification": metrics, "backtest": bt_metrics, "meta": meta}, fh,
                  indent=2, default=_json_default)
    if make_plots:
        try:
            from .plots import save_all_plots
            save_all_plots(bt, oos, importance, out_dir, cfg.ticker)
        except Exception as exc:
            log.warning("Plotting failed: %s", exc)

    prediction = predict_latest(cfg, model=final, ds=ds)
    return {
        "metrics": metrics, "backtest": bt_metrics, "folds": folds, "importance": importance,
        "oos": oos, "bt": bt, "prediction": prediction, "report_dir": out_dir,
        "model_path": _model_path(cfg),
    }


def predict_latest(
    cfg: Config,
    model: Optional[EnsembleClassifier] = None,
    ds=None,
    synthetic: bool = False,
) -> dict:
    """Predict P(price up over the next `horizon` days) using the most recent data."""
    meta = {}
    if model is None:
        path = _model_path(cfg)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained model at {path}. Run `train` first.")
        model, meta = EnsembleClassifier.load(path)
        synthetic = synthetic or meta.get("synthetic", False)
    if ds is None:
        prices, market = load_data(cfg, synthetic=synthetic)
        ds = build_dataset(prices, market, cfg.horizon, cfg.threshold)

    missing = [c for c in model.feature_names_ if c not in ds.X_live.columns]
    if missing:
        raise ValueError(
            f"Model expects {len(missing)} feature(s) not available now (e.g. {missing[:3]}). "
            "Use the same --market setting as at training time, or retrain."
        )
    x = ds.X_live.iloc[[-1]]
    each = model.predict_proba_each(x)
    p = float(model.combine(each)[0])
    if p >= cfg.long_threshold:
        signal = "BUY / LONG"
    elif cfg.allow_short and p <= cfg.short_threshold:
        signal = "SELL / SHORT"
    elif p <= cfg.short_threshold:
        signal = "AVOID / CASH"
    else:
        signal = "HOLD / NEUTRAL"
    return {
        "ticker": cfg.ticker.upper(),
        "as_of": x.index[0].date().isoformat(),
        "last_close": float(ds.close.loc[x.index[0]]),
        "horizon_days": cfg.horizon,
        "prob_up": p,
        "per_model": {k: float(v) for k, v in each.iloc[0].items()},
        "signal": signal,
        "model_trained_at": meta.get("trained_at"),
    }
