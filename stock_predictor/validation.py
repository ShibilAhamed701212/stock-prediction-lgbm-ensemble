"""Walk-forward (expanding window) validation with a purge gap to prevent label leakage."""
from __future__ import annotations

import logging
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .config import Config
from .features import Dataset
from .models import EnsembleClassifier

log = logging.getLogger(__name__)


def walk_forward_splits(
    n: int, n_splits: int, min_train_size: int, gap: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for an expanding-window walk-forward scheme.

    `gap` rows are purged between train and test: a label at row i depends on prices up to
    i + horizon, so with gap = horizon no training label overlaps the test period.
    """
    test_size = (n - min_train_size) // n_splits
    if test_size < 20:
        raise ValueError(f"Not enough data for {n_splits} splits (n={n}, min_train={min_train_size}).")
    for k in range(n_splits):
        test_start = min_train_size + k * test_size
        test_end = n if k == n_splits - 1 else test_start + test_size
        train_end = test_start - gap
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    # precision of high-confidence "up" calls
    hi = p >= 0.55
    out["precision_at_0.55"] = float(y[hi].mean()) if hi.any() else float("nan")
    out["coverage_at_0.55"] = float(hi.mean())
    return out


def walk_forward_evaluate(ds: Dataset, cfg: Config):
    """Train/test across folds. Returns (oos_predictions_df, metrics_by_model, fold_table)."""
    n = len(ds.X)
    min_train = min(cfg.min_train_size, int(n * 0.4))
    frames, fold_rows = [], []
    for fold, (tr, te) in enumerate(walk_forward_splits(n, cfg.n_splits, min_train, cfg.horizon)):
        Xtr, ytr = ds.X.iloc[tr], ds.y.iloc[tr]
        Xte = ds.X.iloc[te]
        model = EnsembleClassifier(cfg.models, cfg.random_state).fit(Xtr, ytr)
        each = model.predict_proba_each(Xte)
        each["ensemble"] = model.combine(each)
        each["y"] = ds.y.iloc[te].values
        each["future_ret"] = ds.future_ret.iloc[te].values
        each["next_ret"] = ds.next_ret.iloc[te].values
        each["fold"] = fold
        frames.append(each)

        m = classification_metrics(each["y"], each["ensemble"])
        fold_rows.append({
            "fold": fold,
            "train_start": Xtr.index[0].date(), "train_end": Xtr.index[-1].date(),
            "test_start": Xte.index[0].date(), "test_end": Xte.index[-1].date(),
            "n_train": len(tr), "n_test": len(te),
            "base_rate_up": float(each["y"].mean()),
            **{k: m[k] for k in ("accuracy", "auc", "log_loss")},
        })
        log.info("Fold %d/%d  test %s→%s  acc=%.3f auc=%.3f",
                 fold + 1, cfg.n_splits, Xte.index[0].date(), Xte.index[-1].date(), m["accuracy"], m["auc"])

    oos = pd.concat(frames)
    model_cols = [c for c in oos.columns if c not in ("y", "future_ret", "next_ret", "fold")]
    metrics = {c: classification_metrics(oos["y"], oos[c]) for c in model_cols}
    # Naive baselines for honest comparison
    base = float(oos["y"].mean())
    metrics["baseline_always_up"] = classification_metrics(oos["y"], np.full(len(oos), 0.5 + 1e-3))
    metrics["baseline_always_up"]["accuracy"] = base
    metrics["baseline_prior"] = classification_metrics(oos["y"], np.full(len(oos), base))
    return oos, metrics, pd.DataFrame(fold_rows)
